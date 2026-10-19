//! Entity resolution: fuzzy lookup, duplicate detection and merging.
//!
//! LLM extraction produces many surface forms for one real-world entity
//! (`Microsoft` / `Microsoft Corp.` / `マイクロソフト`). Without resolution
//! the graph fragments and multi-hop retrieval breaks. Canonical keys catch
//! width / punctuation / legal-form variants at insert time; this module
//! finds the remaining near-duplicates (string + embedding similarity) and
//! merges them while preserving provenance and temporal validity.

use std::collections::{HashMap, HashSet};

use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::json;

use super::common::{exec, get, graph_id, one, one_mut, rows, rows_mut, Vis};
use crate::text::{self, canonical_key, char_ngrams, name_similarity};
use crate::vector;

#[derive(Clone, Debug)]
pub struct Match {
    pub query_idx: usize,
    pub entity_id: i64,
    pub name: String,
    pub score: f64,
}

fn best_name_score(query: &str, name: &str, aliases: &[String]) -> f64 {
    let mut best = name_similarity(query, name);
    for a in aliases {
        best = best.max(name_similarity(query, a));
    }
    best
}

/// Fuzzy-match names (e.g. entities found in a question by an LLM) against
/// the graph, using the bigram GIN index for candidate generation.
pub fn match_names(
    gid: i32,
    vis: &Vis,
    names: &[String],
    min_score: f64,
    per_name: usize,
) -> Vec<Match> {
    let mut out = Vec::new();
    let sql = format!(
        "SELECT e.id, e.name, e.aliases FROM graphrag.entity e
          WHERE e.graph_id = $1 AND (e.canonical_key = $2 OR e.key_grams && $3) AND {}
          ORDER BY (e.canonical_key = $2) DESC, e.id
          LIMIT 2000",
        vis.entity("e")
    );
    for (i, name) in names.iter().enumerate() {
        let key = canonical_key(name);
        if key.is_empty() {
            continue;
        }
        let grams = char_ngrams(&key, 2);
        let cands = rows(
            &sql,
            &[gid.into(), key.as_str().into(), grams.into()],
            |row| {
                Ok((
                    get::<i64>(row, 1)?.unwrap(),
                    get::<String>(row, 2)?.unwrap_or_default(),
                    get::<Vec<String>>(row, 3)?.unwrap_or_default(),
                ))
            },
        );
        let mut scored: Vec<Match> = cands
            .into_iter()
            .map(|(id, n, aliases)| Match {
                query_idx: i,
                entity_id: id,
                score: best_name_score(name, &n, &aliases),
                name: n,
            })
            .filter(|m| m.score >= min_score)
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap()
                .then(a.entity_id.cmp(&b.entity_id))
        });
        scored.truncate(per_name);
        out.extend(scored);
    }
    out
}

/// Entities whose name (or alias) occurs in free text such as a question,
/// plus entities similar to the question's keyphrases. LLM-free query NER.
pub fn match_query_text(gid: i32, vis: &Vis, query: &str, limit: usize) -> Vec<(i64, String, f64)> {
    let normalized = text::normalize(query);
    let qkey = canonical_key(query);
    let words: Vec<String> = normalized
        .split(|c: char| !c.is_alphanumeric() || text::classify(c) != text::CharClass::Word)
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect();
    let mut word_ngrams: HashSet<String> = HashSet::new();
    for n in 1..=4 {
        for w in words.windows(n) {
            word_ngrams.insert(w.concat());
        }
    }
    let phrases: Vec<String> = text::extract_keyphrases(query, 10)
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let mut grams: Vec<String> = char_ngrams(&qkey, 2);
    for p in &phrases {
        grams.extend(char_ngrams(&canonical_key(p), 2));
    }
    grams.sort();
    grams.dedup();
    if grams.is_empty() {
        return Vec::new();
    }
    let sql = format!(
        "SELECT e.id, e.name, e.canonical_key, e.aliases FROM graphrag.entity e
          WHERE e.graph_id = $1 AND e.key_grams && $2 AND {}
          LIMIT 5000",
        vis.entity("e")
    );
    let cands = rows(&sql, &[gid.into(), grams.into()], |row| {
        Ok((
            get::<i64>(row, 1)?.unwrap(),
            get::<String>(row, 2)?.unwrap_or_default(),
            get::<String>(row, 3)?.unwrap_or_default(),
            get::<Vec<String>>(row, 4)?.unwrap_or_default(),
        ))
    });
    let mut out: Vec<(i64, String, f64)> = Vec::new();
    for (id, name, key, aliases) in cands {
        let mut keys = vec![key];
        keys.extend(aliases.iter().map(|a| canonical_key(a)));
        let mut score: f64 = 0.0;
        for k in &keys {
            let n = k.chars().count();
            if n < 2 {
                continue;
            }
            let ascii_word = k
                .chars()
                .all(|c| text::classify(c) == text::CharClass::Word);
            let occurs = if ascii_word {
                word_ngrams.contains(k)
            } else {
                qkey.contains(k.as_str())
            };
            if occurs {
                score = score.max(if n >= 3 { 1.0 } else { 0.8 });
            }
        }
        for p in &phrases {
            let s = best_name_score(p, &name, &aliases);
            if s >= 0.8 {
                score = score.max(s * 0.95);
            }
        }
        if score > 0.0 {
            out.push((id, name, score));
        }
    }
    out.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap()
            .then(b.1.chars().count().cmp(&a.1.chars().count()))
    });
    out.truncate(limit);
    out
}

/// Fuzzy entity lookup.
#[pg_extern(schema = "graphrag")]
fn match_entities(
    graph: &str,
    names: Vec<String>,
    min_score: default!(f64, 0.6),
    per_name: default!(i32, 3),
) -> TableIterator<
    'static,
    (
        name!(query, String),
        name!(entity_id, i64),
        name!(name, String),
        name!(score, f64),
    ),
> {
    let gid = graph_id(graph);
    let vis = Vis::for_graph(gid);
    let matches = match_names(gid, &vis, &names, min_score, per_name.max(1) as usize);
    TableIterator::new(
        matches
            .into_iter()
            .map(move |m| (names[m.query_idx].clone(), m.entity_id, m.name, m.score)),
    )
}

/// Candidate duplicate entities. Candidates share at least one character
/// bigram (blocking); the score mixes name similarity with embedding cosine
/// similarity when both entities have embeddings. Pairs whose types disagree
/// are penalised.
#[pg_extern(schema = "graphrag")]
fn find_duplicates(
    graph: &str,
    threshold: default!(f64, 0.85),
    embedding_weight: default!(f64, 0.3),
    max_pairs: default!(i32, 100),
) -> TableIterator<
    'static,
    (
        name!(entity_a, i64),
        name!(name_a, String),
        name!(entity_b, i64),
        name!(name_b, String),
        name!(score, f64),
        name!(name_score, f64),
        name!(embedding_score, Option<f64>),
    ),
> {
    let gid = graph_id(graph);
    let vis = Vis::for_graph(gid);
    let ents = rows(
        &format!(
            "SELECT e.id, e.name, e.canonical_key, e.aliases, e.entity_type, e.embedding
               FROM graphrag.entity e WHERE e.graph_id = $1 AND {} ORDER BY e.id",
            vis.entity("e")
        ),
        &[gid.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
                get::<String>(row, 3)?.unwrap_or_default(),
                get::<Vec<String>>(row, 4)?.unwrap_or_default(),
                get::<String>(row, 5)?,
                get::<Vec<f32>>(row, 6)?,
            ))
        },
    );
    let mut index: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, e) in ents.iter().enumerate() {
        let grams: HashSet<String> = char_ngrams(&e.2, 2).into_iter().collect();
        for g in grams {
            index.entry(g).or_default().push(i);
        }
    }
    let mut pairs: HashSet<(usize, usize)> = HashSet::new();
    for list in index.values() {
        if list.len() > 300 {
            continue; // too common to be discriminative
        }
        for a in 0..list.len() {
            for b in (a + 1)..list.len() {
                pairs.insert((list[a], list[b]));
            }
        }
    }
    let w = embedding_weight.clamp(0.0, 1.0);
    let mut out = Vec::new();
    for (a, b) in pairs {
        let (ea, eb) = (&ents[a], &ents[b]);
        let mut ns = best_name_score(&ea.1, &eb.1, &eb.3);
        for al in &ea.3 {
            ns = ns.max(best_name_score(al, &eb.1, &eb.3));
        }
        let es = match (&ea.5, &eb.5) {
            (Some(x), Some(y)) => vector::cosine(x, y),
            _ => None,
        };
        let mut score = match es {
            Some(c) => (1.0 - w) * ns + w * c,
            None => ns,
        };
        if let (Some(ta), Some(tb)) = (&ea.4, &eb.4) {
            if ta != tb {
                score *= 0.9;
            }
        }
        if score >= threshold {
            out.push((ea.0, ea.1.clone(), eb.0, eb.1.clone(), score, ns, es));
        }
    }
    out.sort_by(|x, y| {
        y.4.partial_cmp(&x.4)
            .unwrap()
            .then(x.0.cmp(&y.0))
            .then(x.2.cmp(&y.2))
    });
    out.truncate(max_pairs.max(0) as usize);
    TableIterator::new(out)
}

/// Merge `merge_ids` into `keep_id`: mentions, relations (with temporal
/// validity and evidence), aliases and embeddings are carried over;
/// self-loops created by the merge are dropped; affected reports become stale.
#[pg_extern(schema = "graphrag")]
fn merge_entities(keep_id: i64, merge_ids: Vec<i64>) -> JsonB {
    let gid = one::<i32>(
        "SELECT graph_id FROM graphrag.entity WHERE id = $1",
        &[keep_id.into()],
    )
    .unwrap_or_else(|| error!("graphrag: entity {keep_id} does not exist"));
    let mut merged = Vec::new();
    let mut moved = 0;
    let mut combined = 0;
    let mut loops = 0;
    for &m in merge_ids.iter().filter(|m| **m != keep_id) {
        let Some(mg) = one::<i32>(
            "SELECT graph_id FROM graphrag.entity WHERE id = $1",
            &[m.into()],
        ) else {
            continue;
        };
        if mg != gid {
            error!("graphrag: entity {m} belongs to a different graph than {keep_id}");
        }
        // 1. mentions
        exec(
            "INSERT INTO graphrag.mention AS t (entity_id, chunk_id, description, frequency)
             SELECT $1, chunk_id, description, frequency FROM graphrag.mention WHERE entity_id = $2
             ON CONFLICT (entity_id, chunk_id) DO UPDATE SET
                frequency = t.frequency + EXCLUDED.frequency,
                description = concat_ws(E'\\n', t.description, EXCLUDED.description)",
            &[keep_id.into(), m.into()],
        );
        // 2. relations
        let rels = rows(
            "SELECT id, source_id, target_id, rel_type, weight, valid_during::text
               FROM graphrag.relation WHERE source_id = $1 OR target_id = $1 ORDER BY id",
            &[m.into()],
            |row| {
                Ok((
                    get::<i64>(row, 1)?.unwrap(),
                    get::<i64>(row, 2)?.unwrap(),
                    get::<i64>(row, 3)?.unwrap(),
                    get::<String>(row, 4)?.unwrap(),
                    get::<f64>(row, 5)?.unwrap(),
                    get::<String>(row, 6)?.unwrap(),
                ))
            },
        );
        for (rid, s, t, rtype, w, during) in rels {
            let ns = if s == m { keep_id } else { s };
            let nt = if t == m { keep_id } else { t };
            if ns == nt {
                exec("DELETE FROM graphrag.relation WHERE id = $1", &[rid.into()]);
                loops += 1;
                continue;
            }
            let twin = one::<i64>(
                "SELECT id FROM graphrag.relation
                  WHERE graph_id = $1 AND source_id = $2 AND target_id = $3 AND rel_type = $4
                    AND valid_during && $5::tstzrange AND id <> $6
                  ORDER BY id LIMIT 1",
                &[
                    gid.into(),
                    ns.into(),
                    nt.into(),
                    rtype.as_str().into(),
                    during.as_str().into(),
                    rid.into(),
                ],
            );
            match twin {
                Some(tw) => {
                    exec(
                        "INSERT INTO graphrag.relation_evidence (relation_id, chunk_id, description)
                         SELECT $1, chunk_id, description FROM graphrag.relation_evidence WHERE relation_id = $2
                         ON CONFLICT DO NOTHING",
                        &[tw.into(), rid.into()],
                    );
                    exec(
                        "UPDATE graphrag.relation SET weight = weight + $2,
                                is_curated = is_curated OR (SELECT is_curated FROM graphrag.relation WHERE id = $3)
                          WHERE id = $1",
                        &[tw.into(), w.into(), rid.into()],
                    );
                    exec("DELETE FROM graphrag.relation WHERE id = $1", &[rid.into()]);
                    combined += 1;
                }
                None => {
                    exec(
                        "UPDATE graphrag.relation SET source_id = $2, target_id = $3 WHERE id = $1",
                        &[rid.into(), ns.into(), nt.into()],
                    );
                    moved += 1;
                }
            }
        }
        // 3. names, aliases, curation, embedding, metadata
        exec(
            "UPDATE graphrag.entity k SET
                aliases = ARRAY(SELECT DISTINCT a FROM unnest(k.aliases || d.aliases || ARRAY[d.name]) AS a
                                 WHERE graphrag.canonical_key(a) <> k.canonical_key ORDER BY a),
                is_curated = k.is_curated OR d.is_curated,
                embedding = coalesce(k.embedding, d.embedding),
                entity_type = coalesce(k.entity_type, d.entity_type),
                metadata = d.metadata || k.metadata
               FROM graphrag.entity d
              WHERE k.id = $1 AND d.id = $2",
            &[keep_id.into(), m.into()],
        );
        exec(
            "UPDATE graphrag.report SET stale = true
              WHERE entity_id = $1
                 OR community_id IN (SELECT community_id FROM graphrag.community_member WHERE entity_id IN ($1, $2))",
            &[keep_id.into(), m.into()],
        );
        exec(
            "INSERT INTO graphrag.community_member (community_id, entity_id)
             SELECT community_id, $1 FROM graphrag.community_member WHERE entity_id = $2
             ON CONFLICT DO NOTHING",
            &[keep_id.into(), m.into()],
        );
        let _ = rows_mut(
            "DELETE FROM graphrag.entity WHERE id = $1 RETURNING id",
            &[m.into()],
            |row| get::<i64>(row, 1),
        );
        merged.push(m);
    }
    let name = one_mut::<String>(
        "SELECT name FROM graphrag.entity WHERE id = $1",
        &[keep_id.into()],
    );
    JsonB(json!({
        "kept": keep_id,
        "name": name,
        "merged": merged,
        "relations_moved": moved,
        "relations_combined": combined,
        "self_loops_removed": loops,
    }))
}
