//! Ingestion: graphs, documents, chunks, entities, relations.
//!
//! The extension does not call LLMs. The application extracts entities and
//! relations (with any model) and hands the result to `ingest_extraction`, or
//! uses `lazy_index` to build a keyphrase co-occurrence graph without any LLM
//! tokens (LazyGraphRAG-style, for cost-sensitive deployments).

use pgrx::datum::TimestampWithTimeZone;
use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::{json, Value};

use super::common::{exec, get, graph_id, normalize_rel_type, one, one_mut, rows, rows_mut};
use crate::text;

/// Create a graph (namespace). Idempotent: returns the existing id.
#[pg_extern(schema = "graphrag")]
fn create_graph(name: &str, config: default!(Option<JsonB>, "NULL")) -> i32 {
    one_mut::<i32>(
        "INSERT INTO graphrag.graph (name, config) VALUES ($1, coalesce($2, '{}'::jsonb))
         ON CONFLICT (name) DO UPDATE SET config = coalesce($2, graphrag.graph.config)
         RETURNING id",
        &[name.into(), config.into()],
    )
    .expect("graph id")
}

/// Drop a graph and everything in it.
#[pg_extern(schema = "graphrag")]
fn drop_graph(name: &str) -> bool {
    one_mut::<i32>(
        "DELETE FROM graphrag.graph WHERE name = $1 RETURNING id",
        &[name.into()],
    )
    .is_some()
}

/// Register a document. When `content` is given it is split into chunks
/// (sentence aware, Japanese punctuation included). `acl` lists the roles
/// allowed to read it (NULL = public); chunks inherit it.
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn add_document(
    graph: &str,
    content: default!(Option<&str>, "NULL"),
    external_id: default!(Option<&str>, "NULL"),
    title: default!(Option<&str>, "NULL"),
    metadata: default!(Option<JsonB>, "NULL"),
    acl: default!(Option<Vec<String>>, "NULL"),
    source_uri: default!(Option<&str>, "NULL"),
    chunk_tokens: default!(i32, 400),
    chunk_overlap: default!(i32, 40),
) -> i64 {
    let gid = graph_id(graph);
    let doc_id = one_mut::<i64>(
        "INSERT INTO graphrag.document (graph_id, external_id, title, metadata, acl, source_uri)
         VALUES ($1, $2, $3, coalesce($4, '{}'::jsonb), $5, $6) RETURNING id",
        &[
            gid.into(),
            external_id.into(),
            title.into(),
            metadata.into(),
            acl.into(),
            source_uri.into(),
        ],
    )
    .expect("document id");
    if let Some(content) = content {
        let chunks = text::chunk_text(
            content,
            chunk_tokens.max(16) as usize,
            chunk_overlap.max(0) as usize,
        );
        for (i, c) in chunks.iter().enumerate() {
            exec(
                "INSERT INTO graphrag.chunk (document_id, graph_id, ord, content) VALUES ($1, $2, $3, $4)",
                &[doc_id.into(), gid.into(), (i as i32).into(), c.as_str().into()],
            );
        }
    }
    doc_id
}

/// Add one chunk to a document (for callers that chunk themselves).
#[pg_extern(schema = "graphrag")]
fn add_chunk(
    document_id: i64,
    content: &str,
    ord: default!(Option<i32>, "NULL"),
    embedding: default!(Option<Vec<f32>>, "NULL"),
) -> i64 {
    let gid = one::<i32>(
        "SELECT graph_id FROM graphrag.document WHERE id = $1",
        &[document_id.into()],
    )
    .unwrap_or_else(|| error!("graphrag: document {document_id} does not exist"));
    one_mut::<i64>(
        "INSERT INTO graphrag.chunk (document_id, graph_id, ord, content, embedding)
         VALUES ($1, $2, coalesce($3, (SELECT coalesce(max(ord) + 1, 0) FROM graphrag.chunk WHERE document_id = $1)), $4, $5)
         RETURNING id",
        &[document_id.into(), gid.into(), ord.into(), content.into(), embedding.into()],
    )
    .expect("chunk id")
}

fn chunk_graph(chunk_id: i64) -> i32 {
    one::<i32>(
        "SELECT graph_id FROM graphrag.chunk WHERE id = $1",
        &[chunk_id.into()],
    )
    .unwrap_or_else(|| error!("graphrag: chunk {chunk_id} does not exist"))
}

#[derive(Default)]
pub struct EntityInput<'a> {
    pub name: &'a str,
    pub entity_type: Option<&'a str>,
    pub description: Option<&'a str>,
    pub chunk_id: Option<i64>,
    pub embedding: Option<Vec<f32>>,
    pub aliases: Vec<String>,
    pub metadata: Option<Value>,
}

/// Insert or merge an entity and record its mention in `chunk_id`.
///
/// With a chunk, what the extraction says about the entity (aliases, a
/// differing surface form, description, embedding, metadata) is stored on
/// the mention and is therefore visible exactly to the roles that may read
/// the chunk. Only the name (on first insert) and the type label reach the
/// entity row, which every role that can see the entity at all can read.
/// Without a chunk the input is curated and merged into the entity row.
pub fn upsert_entity_internal(gid: i32, e: &EntityInput) -> i64 {
    let name = e.name.trim();
    if name.is_empty() {
        error!("graphrag: entity name must not be empty");
    }
    if let Some(cid) = e.chunk_id {
        if chunk_graph(cid) != gid {
            error!("graphrag: chunk {cid} belongs to a different graph");
        }
    }
    let mut metadata = e.metadata.clone().unwrap_or_else(|| json!({}));
    if e.chunk_id.is_none() {
        if let (Some(d), Some(obj)) = (e.description, metadata.as_object_mut()) {
            obj.insert("description".into(), Value::String(d.to_string()));
        }
    }
    let entity_type = e
        .entity_type
        .map(|t| t.trim().to_uppercase())
        .filter(|t| !t.is_empty());
    let Some(cid) = e.chunk_id else {
        return one_mut::<i64>(
            "INSERT INTO graphrag.entity AS e (graph_id, name, canonical_key, entity_type, aliases, embedding, metadata, is_curated)
             VALUES ($1, $2, graphrag.canonical_key($2), $3, $4, $5, $6, true)
             ON CONFLICT (graph_id, canonical_key) DO UPDATE SET
                entity_type = coalesce(e.entity_type, EXCLUDED.entity_type),
                aliases = ARRAY(SELECT DISTINCT a FROM unnest(e.aliases || EXCLUDED.aliases
                                  || CASE WHEN EXCLUDED.name <> e.name THEN ARRAY[EXCLUDED.name] ELSE '{}'::text[] END) AS a
                                ORDER BY a),
                embedding = coalesce(EXCLUDED.embedding, e.embedding),
                metadata = e.metadata || EXCLUDED.metadata,
                updated_at = now()
             RETURNING id",
            &[
                gid.into(),
                name.into(),
                entity_type.into(),
                e.aliases.clone().into(),
                e.embedding.clone().into(),
                JsonB(metadata).into(),
            ],
        )
        .expect("entity id");
    };
    let (id, stored_name) = rows_mut(
        "INSERT INTO graphrag.entity AS e (graph_id, name, canonical_key, entity_type)
         VALUES ($1, $2, graphrag.canonical_key($2), $3)
         ON CONFLICT (graph_id, canonical_key) DO UPDATE SET
            entity_type = coalesce(e.entity_type, EXCLUDED.entity_type),
            updated_at = now()
         RETURNING id, name",
        &[gid.into(), name.into(), entity_type.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    )
    .into_iter()
    .next()
    .expect("entity id");
    let mut aliases = e.aliases.clone();
    if name != stored_name {
        aliases.push(name.to_string());
    }
    exec(
        "INSERT INTO graphrag.mention AS m (entity_id, chunk_id, description, aliases, embedding, metadata)
         VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (entity_id, chunk_id) DO UPDATE SET
            frequency = m.frequency + 1,
            description = CASE
                WHEN EXCLUDED.description IS NULL THEN m.description
                WHEN m.description IS NULL THEN EXCLUDED.description
                WHEN position(EXCLUDED.description IN m.description) > 0 THEN m.description
                ELSE m.description || E'\\n' || EXCLUDED.description END,
            aliases = ARRAY(SELECT DISTINCT a FROM unnest(m.aliases || EXCLUDED.aliases) AS a ORDER BY a),
            embedding = coalesce(EXCLUDED.embedding, m.embedding),
            metadata = m.metadata || EXCLUDED.metadata",
        &[
            id.into(),
            cid.into(),
            e.description.into(),
            aliases.into(),
            e.embedding.clone().into(),
            JsonB(metadata).into(),
        ],
    );
    id
}

pub struct RelationInput<'a> {
    pub source: &'a str,
    pub target: &'a str,
    pub rel_type: &'a str,
    pub description: Option<&'a str>,
    pub chunk_id: Option<i64>,
    pub weight: f64,
    pub valid_from: Option<TimestampWithTimeZone>,
    pub valid_to: Option<TimestampWithTimeZone>,
    pub exclusive: bool,
    pub source_type: Option<&'a str>,
    pub target_type: Option<&'a str>,
}

/// Insert or reinforce a (temporal) relation and record its evidence.
/// Returns `None` when both endpoints resolve to the same entity.
pub fn upsert_relation_internal(gid: i32, r: &RelationInput) -> Option<i64> {
    let src = upsert_entity_internal(
        gid,
        &EntityInput {
            name: r.source,
            entity_type: r.source_type,
            chunk_id: r.chunk_id,
            ..Default::default()
        },
    );
    let tgt = upsert_entity_internal(
        gid,
        &EntityInput {
            name: r.target,
            entity_type: r.target_type,
            chunk_id: r.chunk_id,
            ..Default::default()
        },
    );
    if src == tgt {
        notice!(
            "graphrag: skipping self-relation \"{}\" -> \"{}\"",
            r.source,
            r.target
        );
        return None;
    }
    let rel_type = normalize_rel_type(r.rel_type);
    let weight = if r.weight.is_finite() && r.weight > 0.0 {
        r.weight
    } else {
        1.0
    };

    // Exclusive (functional) relations, e.g. "ceo_of" or "located_in": a new
    // fact closes the currently valid facts with the same source and type,
    // the way temporal knowledge graphs for agent memory (Zep / Graphiti)
    // invalidate superseded edges instead of deleting them. A fact is only
    // closed when everyone who can see it may also read the new evidence
    // (graphrag._may_supersede); otherwise a restricted document would change
    // what other roles see, and reveal when the change happened.
    let mut valid_from = r.valid_from;
    if r.exclusive {
        if valid_from.is_none() {
            valid_from = one::<TimestampWithTimeZone>("SELECT now()", &[]);
        }
        let acl: Option<Vec<String>> = r.chunk_id.and_then(|cid| {
            one::<Vec<String>>(
                "SELECT acl FROM graphrag.chunk WHERE id = $1",
                &[cid.into()],
            )
        });
        let (closed, kept_open) = rows_mut(
            "WITH cand AS (
                SELECT id, graphrag._may_supersede(id, $6) AS ok FROM graphrag.relation
                 WHERE graph_id = $1 AND source_id = $2 AND rel_type = $3 AND target_id <> $4
                   AND valid_during @> $5 AND (lower(valid_during) IS NULL OR lower(valid_during) < $5)),
             closed AS (
                UPDATE graphrag.relation r
                   SET valid_during = tstzrange(lower(r.valid_during), $5, '[)'), invalidated_at = now()
                  FROM cand WHERE r.id = cand.id AND cand.ok
                RETURNING r.id)
             SELECT (SELECT count(*) FROM closed), (SELECT count(*) FROM cand WHERE NOT cand.ok)",
            &[
                gid.into(),
                src.into(),
                rel_type.as_str().into(),
                tgt.into(),
                valid_from.into(),
                acl.into(),
            ],
            |row| {
                Ok((
                    get::<i64>(row, 1)?.unwrap_or(0),
                    get::<i64>(row, 2)?.unwrap_or(0),
                ))
            },
        )
        .into_iter()
        .next()
        .unwrap_or((0, 0));
        if closed > 0 {
            debug1!("graphrag: closed {closed} superseded relation(s)");
        }
        if kept_open > 0 {
            notice!(
                "graphrag: {kept_open} current \"{rel_type}\" relation(s) of \"{}\" left open: \
                 roles that can see them may not read the new evidence",
                r.source
            );
        }
    }

    let existing = one::<i64>(
        "SELECT id FROM graphrag.relation
          WHERE graph_id = $1 AND source_id = $2 AND target_id = $3 AND rel_type = $4
            AND valid_during && tstzrange($5, $6, '[)')
          ORDER BY lower(valid_during) IS NOT DISTINCT FROM $5 DESC, id
          LIMIT 1",
        &[
            gid.into(),
            src.into(),
            tgt.into(),
            rel_type.as_str().into(),
            valid_from.into(),
            r.valid_to.into(),
        ],
    );
    let rel_id = match existing {
        Some(id) => {
            let new_evidence = match r.chunk_id {
                Some(cid) => one_mut::<i64>(
                    "INSERT INTO graphrag.relation_evidence (relation_id, chunk_id, description) VALUES ($1, $2, $3)
                     ON CONFLICT DO NOTHING RETURNING relation_id",
                    &[id.into(), cid.into(), r.description.into()],
                )
                .is_some(),
                None => true,
            };
            if new_evidence {
                exec(
                    "UPDATE graphrag.relation SET weight = weight + $2 WHERE id = $1",
                    &[id.into(), weight.into()],
                );
            } else if let (Some(cid), Some(d)) = (r.chunk_id, r.description) {
                exec(
                    "UPDATE graphrag.relation_evidence SET description = CASE
                        WHEN description IS NULL THEN $3
                        WHEN position($3 IN description) > 0 THEN description
                        ELSE description || E'\\n' || $3 END
                      WHERE relation_id = $1 AND chunk_id = $2",
                    &[id.into(), cid.into(), d.into()],
                );
            }
            if r.chunk_id.is_none() {
                exec(
                    "UPDATE graphrag.relation SET is_curated = true WHERE id = $1",
                    &[id.into()],
                );
            }
            id
        }
        None => {
            let id = one_mut::<i64>(
                "INSERT INTO graphrag.relation (graph_id, source_id, target_id, rel_type, weight, valid_during, is_curated)
                 VALUES ($1, $2, $3, $4, $5, tstzrange($6, $7, '[)'), $8) RETURNING id",
                &[
                    gid.into(),
                    src.into(),
                    tgt.into(),
                    rel_type.as_str().into(),
                    weight.into(),
                    valid_from.into(),
                    r.valid_to.into(),
                    r.chunk_id.is_none().into(),
                ],
            )
            .expect("relation id");
            if let Some(cid) = r.chunk_id {
                exec(
                    "INSERT INTO graphrag.relation_evidence (relation_id, chunk_id, description) VALUES ($1, $2, $3)",
                    &[id.into(), cid.into(), r.description.into()],
                );
            }
            id
        }
    };
    Some(rel_id)
}

/// Insert or merge an entity. With `chunk_id` the call records provenance
/// (a mention and the chunk's description of the entity); without it the
/// entity is *curated* and visible to every role.
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn upsert_entity(
    graph: &str,
    name: &str,
    entity_type: default!(Option<&str>, "NULL"),
    description: default!(Option<&str>, "NULL"),
    chunk_id: default!(Option<i64>, "NULL"),
    embedding: default!(Option<Vec<f32>>, "NULL"),
    aliases: default!(Option<Vec<String>>, "NULL"),
    metadata: default!(Option<JsonB>, "NULL"),
) -> i64 {
    let gid = graph_id(graph);
    upsert_entity_internal(
        gid,
        &EntityInput {
            name,
            entity_type,
            description,
            chunk_id,
            embedding,
            aliases: aliases.unwrap_or_default(),
            metadata: metadata.map(|m| m.0),
        },
    )
}

/// Insert or reinforce a relation `source -[rel_type]-> target` valid during
/// `[valid_from, valid_to)`. Missing endpoints are created. With
/// `exclusive => true`, currently valid relations of the same type from the
/// same source to *other* targets are closed at `valid_from` (default now()).
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn upsert_relation(
    graph: &str,
    source: &str,
    target: &str,
    rel_type: default!(&str, "'related_to'"),
    description: default!(Option<&str>, "NULL"),
    chunk_id: default!(Option<i64>, "NULL"),
    weight: default!(f64, 1.0),
    valid_from: default!(Option<TimestampWithTimeZone>, "NULL"),
    valid_to: default!(Option<TimestampWithTimeZone>, "NULL"),
    exclusive: default!(bool, false),
    source_type: default!(Option<&str>, "NULL"),
    target_type: default!(Option<&str>, "NULL"),
) -> Option<i64> {
    let gid = graph_id(graph);
    upsert_relation_internal(
        gid,
        &RelationInput {
            source,
            target,
            rel_type,
            description,
            chunk_id,
            weight,
            valid_from,
            valid_to,
            exclusive,
            source_type,
            target_type,
        },
    )
}

fn str_field<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(|x| x.as_str()))
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn parse_ts(v: &Value, keys: &[&str]) -> Option<TimestampWithTimeZone> {
    let s = str_field(v, keys)?;
    Spi::get_one_with_args::<TimestampWithTimeZone>("SELECT $1::timestamptz", &[s.into()])
        .unwrap_or_else(|e| error!("graphrag: invalid timestamp \"{s}\": {e}"))
}

/// Store the output of an LLM extraction step for one chunk in one call.
///
/// Accepts the common shapes produced by GraphRAG / LightRAG style prompts:
/// `{"entities": [{"name"|"entity_name", "type"|"entity_type", "description", "aliases"}],
///   "relations"|"relationships": [{"source"|"src_id", "target"|"tgt_id",
///   "type"|"relation"|"rel_type", "description", "weight"|"strength",
///   "valid_from", "valid_to", "exclusive"}]}`.
#[pg_extern(schema = "graphrag")]
fn ingest_extraction(chunk_id: i64, extraction: JsonB) -> JsonB {
    let gid = chunk_graph(chunk_id);
    let doc = extraction.0;
    let mut n_entities = 0;
    let mut n_relations = 0;
    let mut skipped = 0;
    if let Some(entities) = doc.get("entities").and_then(|v| v.as_array()) {
        for e in entities {
            let Some(name) = str_field(e, &["name", "entity_name", "title"]) else {
                skipped += 1;
                continue;
            };
            let aliases = e
                .get("aliases")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            upsert_entity_internal(
                gid,
                &EntityInput {
                    name,
                    entity_type: str_field(e, &["type", "entity_type", "label"]),
                    description: str_field(e, &["description", "entity_description", "desc"]),
                    chunk_id: Some(chunk_id),
                    aliases,
                    ..Default::default()
                },
            );
            n_entities += 1;
        }
    }
    let relations = doc
        .get("relations")
        .or_else(|| doc.get("relationships"))
        .and_then(|v| v.as_array());
    if let Some(relations) = relations {
        for r in relations {
            let (Some(source), Some(target)) = (
                str_field(r, &["source", "src_id", "source_entity", "head"]),
                str_field(r, &["target", "tgt_id", "target_entity", "tail"]),
            ) else {
                skipped += 1;
                continue;
            };
            let weight = ["weight", "strength", "relationship_strength"]
                .iter()
                .find_map(|k| {
                    r.get(*k).and_then(|x| {
                        x.as_f64()
                            .or_else(|| x.as_str().and_then(|s| s.parse().ok()))
                    })
                })
                .unwrap_or(1.0);
            let res = upsert_relation_internal(
                gid,
                &RelationInput {
                    source,
                    target,
                    rel_type: str_field(
                        r,
                        &["type", "relation", "rel_type", "relationship", "predicate"],
                    )
                    .unwrap_or("related_to"),
                    description: str_field(r, &["description", "relationship_description", "desc"]),
                    chunk_id: Some(chunk_id),
                    weight,
                    valid_from: parse_ts(r, &["valid_from", "valid_at", "start"]),
                    valid_to: parse_ts(r, &["valid_to", "invalid_at", "end"]),
                    exclusive: r
                        .get("exclusive")
                        .and_then(|x| x.as_bool())
                        .unwrap_or(false),
                    source_type: str_field(r, &["source_type"]),
                    target_type: str_field(r, &["target_type"]),
                },
            );
            if res.is_some() {
                n_relations += 1;
            } else {
                skipped += 1;
            }
        }
    }
    JsonB(
        json!({"chunk_id": chunk_id, "entities": n_entities, "relations": n_relations, "skipped": skipped}),
    )
}

/// LLM-free indexing: extract keyphrases from chunks that have no mentions
/// yet, register them as `CONCEPT` entities and connect phrases that occur
/// in the same chunk with `co_occurs` relations. Costs zero LLM tokens
/// (cf. LazyGraphRAG, LinearRAG); the resulting graph supports PPR,
/// communities and path search like an LLM-extracted one.
#[pg_extern(schema = "graphrag")]
fn lazy_index(
    graph: &str,
    document_id: default!(Option<i64>, "NULL"),
    max_phrases: default!(i32, 12),
) -> JsonB {
    let gid = graph_id(graph);
    let chunks: Vec<(i64, String)> = rows(
        "SELECT c.id, c.content FROM graphrag.chunk c
          WHERE c.graph_id = $1 AND ($2::bigint IS NULL OR c.document_id = $2)
            AND NOT EXISTS (SELECT 1 FROM graphrag.mention m WHERE m.chunk_id = c.id)
          ORDER BY c.id",
        &[gid.into(), document_id.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    );
    let mut n_mentions = 0usize;
    let mut n_pairs = 0usize;
    for (cid, content) in &chunks {
        let phrases = text::extract_keyphrases(content, max_phrases.max(1) as usize);
        let mut ids: Vec<(i64, String)> = Vec::new();
        for (p, _) in &phrases {
            let id = upsert_entity_internal(
                gid,
                &EntityInput {
                    name: p,
                    entity_type: Some("CONCEPT"),
                    chunk_id: Some(*cid),
                    ..Default::default()
                },
            );
            if !ids.iter().any(|(x, _)| *x == id) {
                ids.push((id, p.clone()));
            }
            n_mentions += 1;
        }
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                // Store each undirected pair once, ordered by entity id.
                let (a, b) = if ids[i].0 < ids[j].0 {
                    (&ids[i], &ids[j])
                } else {
                    (&ids[j], &ids[i])
                };
                let existing = one::<i64>(
                    "SELECT id FROM graphrag.relation WHERE graph_id = $1 AND source_id = $2 AND target_id = $3
                       AND rel_type = 'co_occurs' AND upper_inf(valid_during) AND lower_inf(valid_during)",
                    &[gid.into(), a.0.into(), b.0.into()],
                );
                let rel_id = match existing {
                    Some(id) => {
                        let added = one_mut::<i64>(
                            "INSERT INTO graphrag.relation_evidence (relation_id, chunk_id) VALUES ($1, $2)
                             ON CONFLICT DO NOTHING RETURNING relation_id",
                            &[id.into(), (*cid).into()],
                        );
                        if added.is_some() {
                            exec(
                                "UPDATE graphrag.relation SET weight = weight + 1 WHERE id = $1",
                                &[id.into()],
                            );
                        }
                        id
                    }
                    None => {
                        let id = one_mut::<i64>(
                            "INSERT INTO graphrag.relation (graph_id, source_id, target_id, rel_type, weight)
                             VALUES ($1, $2, $3, 'co_occurs', 1) RETURNING id",
                            &[gid.into(), a.0.into(), b.0.into()],
                        )
                        .expect("relation id");
                        exec(
                            "INSERT INTO graphrag.relation_evidence (relation_id, chunk_id) VALUES ($1, $2)",
                            &[id.into(), (*cid).into()],
                        );
                        id
                    }
                };
                let _ = rel_id;
                n_pairs += 1;
            }
        }
    }
    JsonB(
        json!({"chunks": chunks.len(), "mentions": n_mentions, "cooccurrences": n_pairs, "llm_tokens": 0}),
    )
}
