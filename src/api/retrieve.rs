//! Retrieval.
//!
//! * `local_search`  — HippoRAG-style: seed entities from the question
//!   (LLM-free query NER, explicit names, embeddings), spread relevance with
//!   local-push Personalized PageRank over *visible, currently valid* edges,
//!   score chunks by the PPR mass of the entities they mention, and fuse with
//!   dense and BM25 (Japanese-aware) rankings via reciprocal rank fusion.
//! * `global_search` — Microsoft GraphRAG-style community reports, with
//!   redaction when the caller may not see all evidence behind a report.
//! * `relational_paths` — PathRAG-style explainable multi-hop paths.
//! * `build_context` — assembles all of the above into a token-budgeted,
//!   citation-annotated prompt context.

use std::collections::{HashMap, HashSet};

use pgrx::datum::TimestampWithTimeZone;
use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::{json, Value};

use super::common::{filter_exposed, get, graph_id, one, rows, Opts, Vis};
use super::resolve::{match_names, match_query_text};
use crate::guc;
use crate::paths::{self, Edge, PathAdjacency, PathParams};
use crate::ppr::{local_push_ppr, Adjacency, PprParams};
use crate::text::{self, Bm25};
use crate::vector;

pub struct Ctx {
    pub gid: i32,
    pub vis: Vis,
    pub as_of: Option<TimestampWithTimeZone>,
    pub opts: Opts,
}

impl Ctx {
    pub fn new(graph: &str, as_of: Option<TimestampWithTimeZone>, options: Option<JsonB>) -> Ctx {
        let gid = graph_id(graph);
        Ctx {
            gid,
            vis: Vis::for_graph(gid),
            as_of,
            opts: Opts::new(options),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Seed {
    pub entity_id: i64,
    pub name: String,
    pub weight: f64,
    pub source: &'static str,
}

/// Combine all seeding strategies; one entry per entity (max weight), with
/// HippoRAG-style node specificity (rare entities count more).
pub fn collect_seeds(
    ctx: &Ctx,
    query: Option<&str>,
    query_embedding: Option<&[f32]>,
    seed_entities: &[String],
) -> Vec<Seed> {
    let mut seeds: HashMap<i64, Seed> = HashMap::new();
    let mut add = |id: i64, name: String, w: f64, source: &'static str| {
        let e = seeds.entry(id).or_insert(Seed {
            entity_id: id,
            name: name.clone(),
            weight: 0.0,
            source,
        });
        if w > e.weight {
            e.weight = w;
            e.source = source;
        }
    };
    if !seed_entities.is_empty() {
        for m in match_names(
            ctx.gid,
            &ctx.vis,
            seed_entities,
            ctx.opts.f64("min_match", 0.6),
            2,
        ) {
            add(m.entity_id, m.name, m.score, "name");
        }
    }
    if let Some(q) = query.filter(|q| !q.trim().is_empty()) {
        for (id, name, score) in
            match_query_text(ctx.gid, &ctx.vis, q, ctx.opts.usize("query_seed_k", 12))
        {
            add(id, name, score, "query");
        }
    }
    if let Some(qe) = query_embedding.filter(|v| !v.is_empty()) {
        let k = ctx.opts.usize("embedding_seed_k", 5);
        let min = ctx.opts.f64("embedding_min_similarity", 0.3);
        let sql = format!(
            "SELECT id, name, embedding FROM (
                SELECT e.id, e.name, graphrag.entity_embedding(e.id) AS embedding FROM graphrag.entity e
                 WHERE e.graph_id = $1 AND {}) s
              WHERE cardinality(embedding) = $2",
            ctx.vis.entity("e")
        );
        let scored: Vec<((i64, String), f64)> =
            rows(&sql, &[ctx.gid.into(), (qe.len() as i32).into()], |row| {
                let id = get::<i64>(row, 1)?.unwrap();
                let name = get::<String>(row, 2)?.unwrap_or_default();
                let emb = get::<Vec<f32>>(row, 3)?.unwrap_or_default();
                Ok(((id, name), vector::cosine(qe, &emb).unwrap_or(0.0)))
            });
        let mut scored = scored;
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for ((id, name), s) in scored.into_iter().take(k).filter(|(_, s)| *s >= min) {
            add(id, name, s, "embedding");
        }
    }
    let explicit = ctx.opts.ids("seed_ids");
    if !explicit.is_empty() {
        let sql = format!("SELECT e.id, e.name FROM graphrag.entity e WHERE e.id = ANY($1) AND e.graph_id = $2 AND {}", ctx.vis.entity("e"));
        for (id, name) in rows(&sql, &[explicit.into(), ctx.gid.into()], |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        }) {
            add(id, name, 1.0, "id");
        }
    }
    let mut seeds: Vec<Seed> = seeds.into_values().collect();
    if seeds.is_empty() {
        return seeds;
    }
    // Node specificity: down-weight entities mentioned in many chunks.
    // Counted over visible chunks only: the weight is returned to the caller.
    let ids: Vec<i64> = seeds.iter().map(|s| s.entity_id).collect();
    let df_sql = format!(
        "SELECT m.entity_id, count(*) FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
          WHERE m.entity_id = ANY($1) AND {} GROUP BY m.entity_id",
        ctx.vis.chunk("c")
    );
    let df: HashMap<i64, i64> = rows(&df_sql, &[ids.into()], |row| {
        Ok((
            get::<i64>(row, 1)?.unwrap(),
            get::<i64>(row, 2)?.unwrap_or(0),
        ))
    })
    .into_iter()
    .collect();
    for s in seeds.iter_mut() {
        let d = *df.get(&s.entity_id).unwrap_or(&0) as f64;
        s.weight /= 1.0 + d.ln_1p();
    }
    seeds.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap()
            .then(a.entity_id.cmp(&b.entity_id))
    });
    seeds.truncate(ctx.opts.usize("max_seeds", 20));
    seeds
}

fn temporal_clause(param: usize) -> String {
    format!("r.valid_during @> coalesce(${param}::timestamptz, now())")
}

/// Personalized PageRank from the seeds over visible, currently valid edges.
pub fn run_ppr(ctx: &Ctx, seeds: &[Seed]) -> (HashMap<i64, f64>, Value) {
    if seeds.is_empty() {
        return (HashMap::new(), json!({"seeds": 0}));
    }
    let params = PprParams {
        damping: ctx.opts.f64("damping", guc::PPR_DAMPING.get()),
        epsilon: ctx.opts.f64("epsilon", guc::PPR_EPSILON.get()),
        max_nodes: ctx
            .opts
            .usize("max_nodes", guc::MAX_EXPANSION.get().max(10) as usize),
        ..Default::default()
    };
    let sql = format!(
        "SELECT r.source_id, r.target_id, r.weight FROM graphrag.relation r
          WHERE r.graph_id = $1 AND (r.source_id = ANY($2) OR r.target_id = ANY($2)) AND {} AND {}",
        temporal_clause(3),
        ctx.vis.edge("r")
    );
    let seed_pairs: Vec<(i64, f64)> = seeds.iter().map(|s| (s.entity_id, s.weight)).collect();
    let res = local_push_ppr(&seed_pairs, &params, |ids| {
        let wanted: HashSet<i64> = ids.iter().copied().collect();
        let mut adj: Adjacency = HashMap::new();
        for (s, t, w) in rows(
            &sql,
            &[ctx.gid.into(), ids.to_vec().into(), ctx.as_of.into()],
            |row| {
                Ok((
                    get::<i64>(row, 1)?.unwrap(),
                    get::<i64>(row, 2)?.unwrap(),
                    get::<f64>(row, 3)?.unwrap_or(1.0),
                ))
            },
        ) {
            if wanted.contains(&s) {
                adj.entry(s).or_default().push((t, w));
            }
            if wanted.contains(&t) {
                adj.entry(t).or_default().push((s, w));
            }
        }
        adj
    });
    let meta = json!({
        "seeds": seeds.len(),
        "loaded_nodes": res.loaded_nodes,
        "pushes": res.pushes,
        "truncated": res.truncated,
        "damping": params.damping,
        "epsilon": params.epsilon,
    });
    (res.scores, meta)
}

fn sorted_scores(m: &HashMap<i64, f64>) -> Vec<(i64, f64)> {
    let mut v: Vec<(i64, f64)> = m.iter().map(|(k, s)| (*k, *s)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

/// Chunks mentioning high-PPR entities, scored by the PPR mass they contain.
fn graph_chunks(ctx: &Ctx, ppr: &HashMap<i64, f64>) -> Vec<(i64, f64)> {
    let top: Vec<(i64, f64)> = sorted_scores(ppr)
        .into_iter()
        .take(ctx.opts.usize("ppr_entities", 30))
        .collect();
    if top.is_empty() {
        return Vec::new();
    }
    let ids: Vec<i64> = top.iter().map(|(e, _)| *e).collect();
    let sql = format!(
        "SELECT m.chunk_id, m.entity_id FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
          WHERE m.entity_id = ANY($1) AND {}",
        ctx.vis.chunk("c")
    );
    let mut score: HashMap<i64, f64> = HashMap::new();
    for (cid, eid) in rows(&sql, &[ids.into()], |row| {
        Ok((get::<i64>(row, 1)?.unwrap(), get::<i64>(row, 2)?.unwrap()))
    }) {
        *score.entry(cid).or_insert(0.0) += ppr.get(&eid).copied().unwrap_or(0.0);
    }
    sorted_scores(&score)
}

fn dense_chunks(ctx: &Ctx, qe: &[f32], k: usize) -> Vec<(i64, f64)> {
    let sql = format!(
        "SELECT c.id, c.embedding FROM graphrag.chunk c
          WHERE c.graph_id = $1 AND c.embedding IS NOT NULL AND cardinality(c.embedding) = $2 AND {}",
        ctx.vis.chunk("c")
    );
    let scored = rows(&sql, &[ctx.gid.into(), (qe.len() as i32).into()], |row| {
        let id = get::<i64>(row, 1)?.unwrap();
        let emb = get::<Vec<f32>>(row, 2)?.unwrap_or_default();
        Ok((id, vector::cosine(qe, &emb).unwrap_or(0.0)))
    });
    vector::top_k(scored, k)
}

fn tsquery_literal(token: &str) -> String {
    format!("'{}'", token.replace('\\', "\\\\").replace('\'', "''"))
}

/// BM25 over CJK-bigram / word lexemes, using the GIN index for candidates.
pub fn lexical_chunks(gid: i32, vis: &Vis, query: &str, k: usize) -> Vec<(i64, f64)> {
    let mut terms = text::lexical_tokens(query);
    terms.sort();
    terms.dedup();
    if terms.is_empty() {
        return Vec::new();
    }
    // Corpus statistics over the chunks the caller may see: counting hidden
    // chunks would let scores reveal which terms restricted documents contain.
    let n_docs = one::<i64>(
        &format!(
            "SELECT count(*) FROM graphrag.chunk c WHERE c.graph_id = $1 AND {}",
            vis.chunk("c")
        ),
        &[gid.into()],
    )
    .unwrap_or(0) as usize;
    if n_docs == 0 {
        return Vec::new();
    }
    let df_sql = format!(
        "SELECT count(*) FROM graphrag.chunk c WHERE c.graph_id = $1 AND c.lexemes @@ $2::tsquery AND {}",
        vis.chunk("c")
    );
    let mut df: HashMap<String, usize> = HashMap::new();
    for t in &terms {
        let n = one::<i64>(&df_sql, &[gid.into(), tsquery_literal(t).into()]).unwrap_or(0) as usize;
        if n > 0 {
            df.insert(t.clone(), n);
        }
    }
    if df.is_empty() {
        return Vec::new();
    }
    let mut selective: Vec<&String> = df
        .iter()
        .filter(|(_, n)| **n * 2 <= n_docs)
        .map(|(t, _)| t)
        .collect();
    if selective.is_empty() {
        selective = df.keys().collect();
    }
    selective.sort();
    let tsq = selective
        .iter()
        .map(|t| tsquery_literal(t))
        .collect::<Vec<_>>()
        .join(" | ");
    let sql = format!(
        "SELECT c.id, c.content FROM graphrag.chunk c
          WHERE c.graph_id = $1 AND c.lexemes @@ $2::tsquery AND {}
          LIMIT 2000",
        vis.chunk("c")
    );
    let docs: Vec<(i64, Vec<String>)> = rows(&sql, &[gid.into(), tsq.into()], |row| {
        Ok((
            get::<i64>(row, 1)?.unwrap(),
            text::lexical_tokens(&get::<String>(row, 2)?.unwrap_or_default()),
        ))
    });
    if docs.is_empty() {
        return Vec::new();
    }
    let avg = docs.iter().map(|(_, t)| t.len()).sum::<usize>() as f64 / docs.len() as f64;
    let bm = Bm25::default();
    let scored: Vec<(i64, f64)> = docs
        .iter()
        .map(|(id, toks)| (*id, bm.score(&terms, toks, avg, &df, n_docs)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    vector::top_k(scored, k)
}

/// Weighted reciprocal rank fusion.
fn rrf(lists: &[(&[(i64, f64)], f64)], k: f64) -> Vec<(i64, f64)> {
    let mut fused: HashMap<i64, f64> = HashMap::new();
    for (list, weight) in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *fused.entry(*id).or_insert(0.0) += weight / (k + rank as f64 + 1.0);
        }
    }
    sorted_scores(&fused)
}

#[derive(Clone, Debug)]
pub struct ChunkHit {
    pub chunk_id: i64,
    pub document_id: i64,
    pub external_id: Option<String>,
    pub title: Option<String>,
    pub content: String,
    pub score: f64,
    pub graph: Option<f64>,
    pub dense: Option<f64>,
    pub lexical: Option<f64>,
    /// All visible entities mentioned in the chunk, most relevant first.
    pub entities: Vec<(i64, String)>,
}

pub struct Retrieval {
    pub seeds: Vec<Seed>,
    pub ppr: HashMap<i64, f64>,
    pub ppr_meta: Value,
    pub chunks: Vec<ChunkHit>,
}

pub fn retrieve(
    ctx: &Ctx,
    query: Option<&str>,
    query_embedding: Option<&[f32]>,
    seed_entities: &[String],
    top_k: usize,
) -> Retrieval {
    let seeds = collect_seeds(ctx, query, query_embedding, seed_entities);
    let (ppr, ppr_meta) = run_ppr(ctx, &seeds);
    let pool = ctx.opts.usize("candidate_k", (top_k * 4).max(20));
    let graph_list: Vec<(i64, f64)> = graph_chunks(ctx, &ppr).into_iter().take(pool).collect();
    let dense_list: Vec<(i64, f64)> = match query_embedding {
        Some(qe) if !qe.is_empty() => dense_chunks(ctx, qe, pool),
        _ => Vec::new(),
    };
    let lexical_list: Vec<(i64, f64)> = match query {
        Some(q) if ctx.opts.bool("lexical", true) => lexical_chunks(ctx.gid, &ctx.vis, q, pool),
        _ => Vec::new(),
    };
    let fused = rrf(
        &[
            (&graph_list, ctx.opts.f64("graph_weight", 1.0)),
            (&dense_list, ctx.opts.f64("dense_weight", 1.0)),
            (&lexical_list, ctx.opts.f64("lexical_weight", 1.0)),
        ],
        ctx.opts.f64("rrf_k", 60.0),
    );
    let picked: Vec<(i64, f64)> = fused.into_iter().take(top_k).collect();
    let chunks = load_chunks(ctx, &picked, &ppr, &graph_list, &dense_list, &lexical_list);
    Retrieval {
        seeds,
        ppr,
        ppr_meta,
        chunks,
    }
}

fn load_chunks(
    ctx: &Ctx,
    picked: &[(i64, f64)],
    ppr: &HashMap<i64, f64>,
    graph_list: &[(i64, f64)],
    dense_list: &[(i64, f64)],
    lexical_list: &[(i64, f64)],
) -> Vec<ChunkHit> {
    if picked.is_empty() {
        return Vec::new();
    }
    let ids: Vec<i64> = picked.iter().map(|(c, _)| *c).collect();
    let sql = format!(
        "SELECT c.id, c.document_id, c.content, d.external_id, d.title,
                ARRAY(SELECT e.id FROM graphrag.mention m JOIN graphrag.entity e ON e.id = m.entity_id
                       WHERE m.chunk_id = c.id AND {}) AS ents
           FROM graphrag.chunk c JOIN graphrag.document d ON d.id = c.document_id
          WHERE c.id = ANY($1) AND {}",
        ctx.vis.entity("e"),
        ctx.vis.chunk("c")
    );
    let raw = rows(&sql, &[ids.into()], |row| {
        Ok((
            get::<i64>(row, 1)?.unwrap(),
            get::<i64>(row, 2)?.unwrap(),
            get::<String>(row, 3)?.unwrap_or_default(),
            get::<String>(row, 4)?,
            get::<String>(row, 5)?,
            get::<Vec<i64>>(row, 6)?.unwrap_or_default(),
        ))
    });
    let all_entities: Vec<i64> = raw
        .iter()
        .flat_map(|r| r.5.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let names: HashMap<i64, String> = rows(
        "SELECT id, name FROM graphrag.entity WHERE id = ANY($1)",
        &[all_entities.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    )
    .into_iter()
    .collect();
    let lookup = |l: &[(i64, f64)], id: i64| l.iter().find(|(c, _)| *c == id).map(|(_, s)| *s);
    let by_id: HashMap<i64, _> = raw.into_iter().map(|r| (r.0, r)).collect();
    let mut hits = Vec::new();
    for (cid, score) in picked {
        let Some((_, doc, content, ext, title, ents)) = by_id.get(cid) else {
            continue;
        };
        let mut ents: Vec<(i64, String)> = ents
            .iter()
            .map(|e| (*e, names.get(e).cloned().unwrap_or_default()))
            .collect();
        ents.sort_by(|a, b| {
            let pa = ppr.get(&a.0).copied().unwrap_or(0.0);
            let pb = ppr.get(&b.0).copied().unwrap_or(0.0);
            pb.partial_cmp(&pa).unwrap().then(a.1.cmp(&b.1))
        });
        hits.push(ChunkHit {
            chunk_id: *cid,
            document_id: *doc,
            external_id: ext.clone(),
            title: title.clone(),
            content: content.clone(),
            score: *score,
            graph: lookup(graph_list, *cid),
            dense: lookup(dense_list, *cid),
            lexical: lookup(lexical_list, *cid),
            entities: ents,
        });
    }
    hits
}

/// Hybrid graph retrieval of chunks (PPR + dense + BM25, fused by RRF).
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn local_search(
    graph: &str,
    query: default!(Option<&str>, "NULL"),
    query_embedding: default!(Option<Vec<f32>>, "NULL"),
    seed_entities: default!(Option<Vec<String>>, "NULL"),
    top_k: default!(i32, 10),
    as_of: default!(Option<TimestampWithTimeZone>, "NULL"),
    options: default!(Option<JsonB>, "NULL"),
) -> TableIterator<
    'static,
    (
        name!(rank, i32),
        name!(chunk_id, i64),
        name!(document_id, i64),
        name!(score, f64),
        name!(graph_score, Option<f64>),
        name!(dense_score, Option<f64>),
        name!(lexical_score, Option<f64>),
        name!(content, String),
        name!(entities, Vec<String>),
    ),
> {
    let ctx = Ctx::new(graph, as_of, options);
    let r = retrieve(
        &ctx,
        query,
        query_embedding.as_deref(),
        &seed_entities.unwrap_or_default(),
        top_k.max(1) as usize,
    );
    let hits = filter_exposed(
        r.chunks,
        |h| h.entities.iter().map(|(e, _)| *e).collect(),
        ctx.gid,
    );
    let max_names = ctx.opts.usize("entities_per_chunk", 8);
    TableIterator::new(hits.into_iter().enumerate().map(move |(i, h)| {
        (
            i as i32 + 1,
            h.chunk_id,
            h.document_id,
            h.score,
            h.graph,
            h.dense,
            h.lexical,
            h.content,
            h.entities
                .into_iter()
                .take(max_names)
                .map(|(_, n)| n)
                .collect(),
        )
    }))
}

/// Entities ranked by Personalized PageRank from the query's seeds.
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn rank_entities(
    graph: &str,
    query: default!(Option<&str>, "NULL"),
    query_embedding: default!(Option<Vec<f32>>, "NULL"),
    seed_entities: default!(Option<Vec<String>>, "NULL"),
    top_k: default!(i32, 20),
    as_of: default!(Option<TimestampWithTimeZone>, "NULL"),
    options: default!(Option<JsonB>, "NULL"),
) -> TableIterator<
    'static,
    (
        name!(entity_id, i64),
        name!(name, String),
        name!(entity_type, Option<String>),
        name!(score, f64),
        name!(seed, bool),
    ),
> {
    let ctx = Ctx::new(graph, as_of, options);
    let seeds = collect_seeds(
        &ctx,
        query,
        query_embedding.as_deref(),
        &seed_entities.unwrap_or_default(),
    );
    let (ppr, _) = run_ppr(&ctx, &seeds);
    let top: Vec<(i64, f64)> = sorted_scores(&ppr)
        .into_iter()
        .take(top_k.max(1) as usize)
        .collect();
    let ids: Vec<i64> = top.iter().map(|(e, _)| *e).collect();
    let info: HashMap<i64, (String, Option<String>)> = rows(
        "SELECT id, name, entity_type FROM graphrag.entity WHERE id = ANY($1)",
        &[ids.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                (
                    get::<String>(row, 2)?.unwrap_or_default(),
                    get::<String>(row, 3)?,
                ),
            ))
        },
    )
    .into_iter()
    .collect();
    let seed_ids: HashSet<i64> = seeds.iter().map(|s| s.entity_id).collect();
    let out: Vec<(i64, String, Option<String>, f64, bool)> = top
        .into_iter()
        .filter_map(|(id, s)| {
            info.get(&id)
                .map(|(n, t)| (id, n.clone(), t.clone(), s, seed_ids.contains(&id)))
        })
        .collect();
    let out = filter_exposed(out, |r| vec![r.0], ctx.gid);
    TableIterator::new(out)
}

/// BM25 search over chunks (CJK bigrams + words).
#[pg_extern(schema = "graphrag")]
fn lexical_search(
    graph: &str,
    query: &str,
    top_k: default!(i32, 10),
) -> TableIterator<
    'static,
    (
        name!(chunk_id, i64),
        name!(score, f64),
        name!(content, String),
    ),
> {
    let gid = graph_id(graph);
    let vis = Vis::for_graph(gid);
    let hits = lexical_chunks(gid, &vis, query, top_k.max(1) as usize);
    let ids: Vec<i64> = hits.iter().map(|(c, _)| *c).collect();
    let sql = format!(
        "SELECT c.id, c.content,
                ARRAY(SELECT e.id FROM graphrag.mention m JOIN graphrag.entity e ON e.id = m.entity_id
                       WHERE m.chunk_id = c.id AND {})
           FROM graphrag.chunk c WHERE c.id = ANY($1)",
        vis.entity("e")
    );
    let content: HashMap<i64, (String, Vec<i64>)> = rows(&sql, &[ids.into()], |row| {
        Ok((
            get::<i64>(row, 1)?.unwrap(),
            (
                get::<String>(row, 2)?.unwrap_or_default(),
                get::<Vec<i64>>(row, 3)?.unwrap_or_default(),
            ),
        ))
    })
    .into_iter()
    .collect();
    let out: Vec<(i64, f64, String, Vec<i64>)> = hits
        .into_iter()
        .filter_map(|(c, s)| {
            content
                .get(&c)
                .map(|(t, ents)| (c, s, t.clone(), ents.clone()))
        })
        .collect();
    let out = filter_exposed(out, |h| h.3.clone(), gid);
    TableIterator::new(out.into_iter().map(|(c, s, t, _)| (c, s, t)))
}

// ---------------------------------------------------------------------------
// Global search over community reports
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CommunityHit {
    pub id: i64,
    pub level: i32,
    pub parent: Option<i64>,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub score: f64,
    pub visibility: &'static str,
    pub stale: bool,
    pub member_ids: Vec<i64>,
    pub members: Vec<String>,
}

fn token_overlap(query_tokens: &HashSet<String>, text: &str) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let toks: HashSet<String> = text::lexical_tokens(text).into_iter().collect();
    query_tokens.intersection(&toks).count() as f64 / query_tokens.len() as f64
}

pub fn search_communities(
    ctx: &Ctx,
    query: Option<&str>,
    query_embedding: Option<&[f32]>,
    level: Option<i32>,
    restrict_to: Option<&[i64]>,
) -> Vec<CommunityHit> {
    let sql = format!(
        "SELECT c.id, c.level, c.parent_id, c.rank, rep.id IS NOT NULL AS has_report,
                rep.title, rep.content, rep.embedding, coalesce(rep.stale, false),
                CASE WHEN rep.id IS NULL THEN NULL ELSE graphrag.acls_satisfied(rep.required_acls) END AS visible,
                ARRAY(SELECT e.id FROM graphrag.community_member cm JOIN graphrag.entity e ON e.id = cm.entity_id
                       WHERE cm.community_id = c.id AND {}
                       ORDER BY (SELECT count(*) FROM graphrag.relation r WHERE r.source_id = e.id OR r.target_id = e.id) DESC, e.id
                       LIMIT 12) AS member_ids
           FROM graphrag.community c
           LEFT JOIN graphrag.report rep ON rep.community_id = c.id
          WHERE c.graph_id = $1 AND ($2::int IS NULL OR c.level = $2) AND ($3::bigint[] IS NULL OR c.id = ANY($3))",
        ctx.vis.entity("e")
    );
    let raw = rows(
        &sql,
        &[
            ctx.gid.into(),
            level.into(),
            restrict_to.map(|r| r.to_vec()).into(),
        ],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<i32>(row, 2)?.unwrap(),
                get::<i64>(row, 3)?,
                get::<f64>(row, 4)?.unwrap_or(0.0),
                get::<bool>(row, 5)?.unwrap_or(false),
                get::<String>(row, 6)?,
                get::<String>(row, 7)?,
                get::<Vec<f32>>(row, 8)?,
                get::<bool>(row, 9)?.unwrap_or(false),
                get::<bool>(row, 10)?,
                get::<Vec<i64>>(row, 11)?.unwrap_or_default(),
            ))
        },
    );
    let max_rank = raw.iter().map(|r| r.3).fold(0.0_f64, f64::max).max(1e-9);
    let all_members: Vec<i64> = raw
        .iter()
        .flat_map(|r| r.10.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let names: HashMap<i64, String> = rows(
        "SELECT id, name FROM graphrag.entity WHERE id = ANY($1)",
        &[all_members.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    )
    .into_iter()
    .collect();
    let qtokens: HashSet<String> = query
        .map(|q| text::lexical_tokens(q).into_iter().collect())
        .unwrap_or_default();
    let mut hits: Vec<CommunityHit> = Vec::new();
    for (id, lvl, parent, rank, has_report, title, content, emb, stale, visible, member_ids) in raw
    {
        let members: Vec<String> = member_ids
            .iter()
            .filter_map(|m| names.get(m).cloned())
            .collect();
        if members.is_empty() && !matches!(visible, Some(true)) {
            continue; // nothing the caller may see
        }
        let visibility = match (has_report, visible) {
            (true, Some(true)) => "full",
            (true, _) => "redacted",
            (false, _) => "no_report",
        };
        let (title, summary) = if visibility == "full" {
            (title, content)
        } else {
            (None, None)
        };
        let semantic = match (query_embedding, &emb, visibility) {
            (Some(q), Some(e), "full") => vector::cosine(q, e),
            _ => None,
        };
        let lexical = match &summary {
            Some(s) => token_overlap(
                &qtokens,
                &format!("{} {}", title.clone().unwrap_or_default(), s),
            ),
            None => token_overlap(&qtokens, &members.join(" ")),
        };
        let score = semantic.unwrap_or(lexical) + 0.05 * rank / max_rank;
        hits.push(CommunityHit {
            id,
            level: lvl,
            parent,
            title,
            summary,
            score,
            visibility,
            stale,
            member_ids,
            members,
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap()
            .then(a.level.cmp(&b.level))
            .then(a.id.cmp(&b.id))
    });
    hits
}

/// Pick the best communities across levels without returning both a
/// community and its ancestor / descendant (dynamic level selection).
fn select_across_levels(hits: Vec<CommunityHit>, k: usize) -> Vec<CommunityHit> {
    let parent: HashMap<i64, Option<i64>> = hits.iter().map(|h| (h.id, h.parent)).collect();
    let ancestors = |mut id: i64| {
        let mut v = Vec::new();
        while let Some(Some(p)) = parent.get(&id) {
            v.push(*p);
            id = *p;
        }
        v
    };
    let mut chosen: Vec<CommunityHit> = Vec::new();
    let mut blocked: HashSet<i64> = HashSet::new();
    for h in hits {
        if chosen.len() >= k {
            break;
        }
        let anc = ancestors(h.id);
        if blocked.contains(&h.id) || anc.iter().any(|a| chosen.iter().any(|c| c.id == *a)) {
            continue;
        }
        blocked.extend(anc);
        blocked.insert(h.id);
        chosen.push(h);
    }
    chosen
}

/// Community reports relevant to a (global, "sensemaking") question.
/// Reports are returned only when the caller may see every piece of evidence
/// they were generated from; otherwise the row is `redacted` and lists the
/// visible member entities so the application can summarize on the fly.
#[pg_extern(schema = "graphrag")]
fn global_search(
    graph: &str,
    query: default!(Option<&str>, "NULL"),
    query_embedding: default!(Option<Vec<f32>>, "NULL"),
    level: default!(Option<i32>, "NULL"),
    top_k: default!(i32, 10),
) -> TableIterator<
    'static,
    (
        name!(community_id, i64),
        name!(level, i32),
        name!(title, Option<String>),
        name!(summary, Option<String>),
        name!(score, f64),
        name!(visibility, String),
        name!(stale, bool),
        name!(members, Vec<String>),
    ),
> {
    let ctx = Ctx::new(graph, None, None);
    let hits = search_communities(&ctx, query, query_embedding.as_deref(), level, None);
    let hits = if level.is_none() {
        select_across_levels(hits, top_k.max(1) as usize)
    } else {
        hits.into_iter().take(top_k.max(1) as usize).collect()
    };
    let hits = filter_exposed(hits, |h| h.member_ids.clone(), ctx.gid);
    TableIterator::new(hits.into_iter().map(|h| {
        (
            h.id,
            h.level,
            h.title,
            h.summary,
            h.score,
            h.visibility.to_string(),
            h.stale,
            h.members,
        )
    }))
}

// ---------------------------------------------------------------------------
// Relational paths (PathRAG)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PathHit {
    pub reliability: f64,
    pub text: String,
    pub entity_ids: Vec<i64>,
    pub relation_ids: Vec<i64>,
    pub evidence: Vec<i64>,
}

pub fn find_relational_paths(ctx: &Ctx, endpoints: &[i64], params: &PathParams) -> Vec<PathHit> {
    if endpoints.len() < 2 {
        return Vec::new();
    }
    let sql = format!(
        "SELECT r.id, r.source_id, r.target_id FROM graphrag.relation r
          WHERE r.graph_id = $1 AND (r.source_id = ANY($2) OR r.target_id = ANY($2)) AND {} AND {}",
        temporal_clause(3),
        ctx.vis.edge("r")
    );
    let max_nodes = ctx
        .opts
        .usize("max_nodes", guc::MAX_EXPANSION.get().max(10) as usize);
    let mut adj: PathAdjacency = HashMap::new();
    let load = |adj: &mut PathAdjacency, ids: Vec<i64>| {
        let wanted: HashSet<i64> = ids.iter().copied().collect();
        for id in &ids {
            adj.entry(*id).or_default();
        }
        for (rid, s, t) in rows(
            &sql,
            &[ctx.gid.into(), ids.into(), ctx.as_of.into()],
            |row| {
                Ok((
                    get::<i64>(row, 1)?.unwrap(),
                    get::<i64>(row, 2)?.unwrap(),
                    get::<i64>(row, 3)?.unwrap(),
                ))
            },
        ) {
            if wanted.contains(&s) {
                adj.get_mut(&s).unwrap().push(Edge {
                    to: t,
                    relation_id: rid,
                    forward: true,
                });
            }
            if wanted.contains(&t) {
                adj.get_mut(&t).unwrap().push(Edge {
                    to: s,
                    relation_id: rid,
                    forward: false,
                });
            }
        }
    };
    load(&mut adj, endpoints.to_vec());
    let mut frontier: HashMap<i64, f64> = endpoints.iter().map(|e| (*e, 1.0)).collect();
    for _ in 1..params.max_hops {
        let mut next = paths::next_frontier(&adj, &frontier, params);
        if next.is_empty() || adj.len() >= max_nodes {
            break;
        }
        if adj.len() + next.len() > max_nodes {
            let mut v: Vec<(i64, f64)> = next.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v.truncate(max_nodes - adj.len());
            next = v.into_iter().collect();
        }
        load(&mut adj, next.keys().copied().collect());
        frontier = next;
    }
    let found = paths::find_paths(&adj, endpoints, params);
    if found.is_empty() {
        return Vec::new();
    }
    let ent_ids: Vec<i64> = found
        .iter()
        .flat_map(|p| p.nodes.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let rel_ids: Vec<i64> = found
        .iter()
        .flat_map(|p| p.edges.iter().map(|(r, _)| *r))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let names: HashMap<i64, String> = rows(
        "SELECT id, name FROM graphrag.entity WHERE id = ANY($1)",
        &[ent_ids.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    )
    .into_iter()
    .collect();
    let rel_info: HashMap<i64, String> = rows(
        "SELECT id, rel_type FROM graphrag.relation WHERE id = ANY($1)",
        &[rel_ids.clone().into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    )
    .into_iter()
    .collect();
    let ev_sql = format!(
        "SELECT ev.relation_id, ev.chunk_id FROM graphrag.relation_evidence ev JOIN graphrag.chunk c ON c.id = ev.chunk_id
          WHERE ev.relation_id = ANY($1) AND {} ORDER BY ev.chunk_id",
        ctx.vis.chunk("c")
    );
    let mut evidence: HashMap<i64, Vec<i64>> = HashMap::new();
    for (r, c) in rows(&ev_sql, &[rel_ids.into()], |row| {
        Ok((get::<i64>(row, 1)?.unwrap(), get::<i64>(row, 2)?.unwrap()))
    }) {
        evidence.entry(r).or_default().push(c);
    }
    found
        .into_iter()
        .map(|p| {
            let mut s = names.get(&p.nodes[0]).cloned().unwrap_or_default();
            let mut ev: Vec<i64> = Vec::new();
            for (i, (rid, forward)) in p.edges.iter().enumerate() {
                let t = rel_info.get(rid).cloned().unwrap_or_default();
                let next = names.get(&p.nodes[i + 1]).cloned().unwrap_or_default();
                if *forward {
                    s.push_str(&format!(" -[{t}]-> {next}"));
                } else {
                    s.push_str(&format!(" <-[{t}]- {next}"));
                }
                for c in evidence.get(rid).into_iter().flatten() {
                    if !ev.contains(c) {
                        ev.push(*c);
                    }
                }
            }
            PathHit {
                reliability: p.reliability,
                text: s,
                entity_ids: p.nodes,
                relation_ids: p.edges.iter().map(|(r, _)| *r).collect(),
                evidence: ev,
            }
        })
        .collect()
}

fn path_params(opts: &Opts, max_hops: usize, top_k: usize) -> PathParams {
    PathParams {
        max_hops: max_hops.clamp(1, 6),
        decay: opts.f64("decay", 0.8),
        threshold: opts.f64("threshold", 0.05),
        top_k,
        ..Default::default()
    }
}

/// Explainable multi-hop paths between entities (PathRAG flow pruning).
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn relational_paths(
    graph: &str,
    entities: Vec<String>,
    max_hops: default!(i32, 3),
    top_k: default!(i32, 10),
    as_of: default!(Option<TimestampWithTimeZone>, "NULL"),
    options: default!(Option<JsonB>, "NULL"),
) -> TableIterator<
    'static,
    (
        name!(rank, i32),
        name!(reliability, f64),
        name!(path, String),
        name!(entity_ids, Vec<i64>),
        name!(relation_ids, Vec<i64>),
        name!(evidence_chunk_ids, Vec<i64>),
    ),
> {
    let ctx = Ctx::new(graph, as_of, options);
    let mut endpoints: Vec<i64> = Vec::new();
    for m in match_names(
        ctx.gid,
        &ctx.vis,
        &entities,
        ctx.opts.f64("min_match", 0.6),
        1,
    ) {
        if !endpoints.contains(&m.entity_id) {
            endpoints.push(m.entity_id);
        }
    }
    let params = path_params(&ctx.opts, max_hops.max(1) as usize, top_k.max(1) as usize);
    let hits = find_relational_paths(&ctx, &endpoints, &params);
    let hits = filter_exposed(hits, |h| h.entity_ids.clone(), ctx.gid);
    TableIterator::new(hits.into_iter().enumerate().map(|(i, h)| {
        (
            i as i32 + 1,
            h.reliability,
            h.text,
            h.entity_ids,
            h.relation_ids,
            h.evidence,
        )
    }))
}

/// k-hop neighbourhood as JSON (`{"nodes": [...], "edges": [...]}`) for
/// visualisation and debugging.
#[pg_extern(schema = "graphrag")]
fn subgraph(
    graph: &str,
    entities: Vec<String>,
    hops: default!(i32, 1),
    max_nodes: default!(i32, 50),
    as_of: default!(Option<TimestampWithTimeZone>, "NULL"),
) -> JsonB {
    let ctx = Ctx::new(graph, as_of, None);
    let mut nodes: Vec<i64> = Vec::new();
    for m in match_names(ctx.gid, &ctx.vis, &entities, 0.6, 1) {
        if !nodes.contains(&m.entity_id) {
            nodes.push(m.entity_id);
        }
    }
    let sql = format!(
        "SELECT r.id, r.source_id, r.target_id, r.rel_type, r.weight,
                lower(r.valid_during)::text, upper(r.valid_during)::text
           FROM graphrag.relation r
          WHERE r.graph_id = $1 AND (r.source_id = ANY($2) OR r.target_id = ANY($2)) AND {} AND {}
          ORDER BY r.weight DESC, r.id",
        temporal_clause(3),
        ctx.vis.edge("r")
    );
    let limit = max_nodes.max(1) as usize;
    let mut edges: HashMap<i64, Value> = HashMap::new();
    let mut frontier = nodes.clone();
    for _ in 0..hops.max(0) {
        if frontier.is_empty() || nodes.len() >= limit {
            break;
        }
        let mut next = Vec::new();
        for (rid, s, t, rt, w, vf, vt) in rows(
            &sql,
            &[ctx.gid.into(), frontier.clone().into(), ctx.as_of.into()],
            |row| {
                Ok((
                    get::<i64>(row, 1)?.unwrap(),
                    get::<i64>(row, 2)?.unwrap(),
                    get::<i64>(row, 3)?.unwrap(),
                    get::<String>(row, 4)?.unwrap_or_default(),
                    get::<f64>(row, 5)?.unwrap_or(1.0),
                    get::<String>(row, 6)?,
                    get::<String>(row, 7)?,
                ))
            },
        ) {
            for n in [s, t] {
                if !nodes.contains(&n) && nodes.len() < limit {
                    nodes.push(n);
                    next.push(n);
                }
            }
            if nodes.contains(&s) && nodes.contains(&t) {
                edges.insert(rid, json!({"id": rid, "source": s, "target": t, "type": rt, "weight": w, "valid_from": vf, "valid_to": vt}));
            }
        }
        frontier = next;
    }
    let admitted = filter_exposed(nodes, |n| vec![*n], ctx.gid);
    let node_json: Vec<Value> = rows(
        "SELECT id, name, entity_type FROM graphrag.entity WHERE id = ANY($1) ORDER BY id",
        &[admitted.clone().into()],
        |row| {
            Ok(
                json!({"id": get::<i64>(row, 1)?, "name": get::<String>(row, 2)?, "type": get::<String>(row, 3)?}),
            )
        },
    );
    let keep: HashSet<i64> = admitted.into_iter().collect();
    let mut edge_json: Vec<Value> = edges
        .into_values()
        .filter(|e| {
            keep.contains(&e["source"].as_i64().unwrap())
                && keep.contains(&e["target"].as_i64().unwrap())
        })
        .collect();
    edge_json.sort_by_key(|e| e["id"].as_i64());
    JsonB(json!({"nodes": node_json, "edges": edge_json}))
}

// ---------------------------------------------------------------------------
// Context builder
// ---------------------------------------------------------------------------

/// Visible description of entities: a report (if every piece of its
/// evidence is visible to the caller) or the per-chunk descriptions from
/// visible chunks, or the curated description.
fn entity_descriptions(ctx: &Ctx, ids: &[i64]) -> HashMap<i64, (String, Option<String>, String)> {
    let sql = format!(
        "SELECT e.id, e.name, e.entity_type,
                (SELECT rep.content FROM graphrag.report rep
                  WHERE rep.entity_id = e.id AND graphrag.acls_satisfied(rep.required_acls)),
                (SELECT string_agg(d, ' / ') FROM (
                    SELECT DISTINCT m.description AS d FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
                     WHERE m.entity_id = e.id AND m.description IS NOT NULL AND {}
                     ORDER BY 1 LIMIT 3) s),
                e.metadata->>'description'
           FROM graphrag.entity e WHERE e.id = ANY($1)",
        ctx.vis.chunk("c")
    );
    rows(&sql, &[ids.to_vec().into()], |row| {
        let id = get::<i64>(row, 1)?.unwrap();
        let name = get::<String>(row, 2)?.unwrap_or_default();
        let ty = get::<String>(row, 3)?;
        let desc = get::<String>(row, 4)?
            .or(get::<String>(row, 5)?)
            .or(get::<String>(row, 6)?)
            .unwrap_or_default();
        Ok((id, (name, ty, desc)))
    })
    .into_iter()
    .collect()
}

fn fmt_range(from: &Option<String>, to: &Option<String>) -> String {
    match (from, to) {
        (None, None) => String::new(),
        (f, t) => format!(
            " (valid {} – {})",
            f.as_deref().map(|s| &s[..s.len().min(10)]).unwrap_or("…"),
            t.as_deref().map(|s| &s[..s.len().min(10)]).unwrap_or("now")
        ),
    }
}

/// Build a token-budgeted, citation-annotated context for an LLM prompt.
///
/// Sections (each with its own share of the budget, unused budget flows on):
/// entities, relations (with validity), reasoning paths, community reports
/// and source chunks. Every fact carries `[#chunk_id]` citations so answers
/// can be traced back to documents. Returns the text plus structured data.
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn build_context(
    graph: &str,
    query: default!(Option<&str>, "NULL"),
    query_embedding: default!(Option<Vec<f32>>, "NULL"),
    seed_entities: default!(Option<Vec<String>>, "NULL"),
    max_tokens: default!(i32, 3000),
    as_of: default!(Option<TimestampWithTimeZone>, "NULL"),
    options: default!(Option<JsonB>, "NULL"),
) -> JsonB {
    let ctx = Ctx::new(graph, as_of, options);
    let budget = max_tokens.max(100) as usize;
    let r = retrieve(
        &ctx,
        query,
        query_embedding.as_deref(),
        &seed_entities.unwrap_or_default(),
        ctx.opts.usize("chunks", 8),
    );

    // --- candidates -------------------------------------------------------
    let top_entities: Vec<(i64, f64)> = sorted_scores(&r.ppr)
        .into_iter()
        .take(ctx.opts.usize("entities", 12))
        .collect();
    let ent_ids: Vec<i64> = top_entities.iter().map(|(e, _)| *e).collect();
    let rel_sql = format!(
        "SELECT r.id, r.source_id, r.target_id, r.rel_type, r.weight,
                lower(r.valid_during)::text, upper(r.valid_during)::text,
                (SELECT string_agg(d, ' / ') FROM (
                    SELECT DISTINCT ev.description AS d FROM graphrag.relation_evidence ev JOIN graphrag.chunk c ON c.id = ev.chunk_id
                     WHERE ev.relation_id = r.id AND ev.description IS NOT NULL AND {} ORDER BY 1 LIMIT 2) s),
                ARRAY(SELECT ev.chunk_id FROM graphrag.relation_evidence ev JOIN graphrag.chunk c ON c.id = ev.chunk_id
                       WHERE ev.relation_id = r.id AND {} ORDER BY ev.chunk_id LIMIT 3)
           FROM graphrag.relation r
          WHERE r.graph_id = $1 AND r.source_id = ANY($2) AND r.target_id = ANY($2) AND {} AND {}",
        ctx.vis.chunk("c"),
        ctx.vis.chunk("c"),
        temporal_clause(3),
        ctx.vis.edge("r")
    );
    let mut relations = rows(
        &rel_sql,
        &[ctx.gid.into(), ent_ids.clone().into(), ctx.as_of.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<i64>(row, 2)?.unwrap(),
                get::<i64>(row, 3)?.unwrap(),
                get::<String>(row, 4)?.unwrap_or_default(),
                get::<f64>(row, 5)?.unwrap_or(1.0),
                get::<String>(row, 6)?,
                get::<String>(row, 7)?,
                get::<String>(row, 8)?,
                get::<Vec<i64>>(row, 9)?.unwrap_or_default(),
            ))
        },
    );
    relations.sort_by(|a, b| {
        let sa = (r.ppr.get(&a.1).unwrap_or(&0.0) + r.ppr.get(&a.2).unwrap_or(&0.0)) * a.4.ln_1p();
        let sb = (r.ppr.get(&b.1).unwrap_or(&0.0) + r.ppr.get(&b.2).unwrap_or(&0.0)) * b.4.ln_1p();
        sb.partial_cmp(&sa).unwrap().then(a.0.cmp(&b.0))
    });
    let path_hits = if ctx.opts.bool("paths", true) {
        let endpoints: Vec<i64> = r.seeds.iter().take(4).map(|s| s.entity_id).collect();
        find_relational_paths(
            &ctx,
            &endpoints,
            &path_params(&ctx.opts, ctx.opts.usize("max_hops", 3), 5),
        )
    } else {
        Vec::new()
    };
    let community_ids: Vec<i64> = if ent_ids.is_empty() {
        Vec::new()
    } else {
        rows(
            "SELECT cm.community_id FROM graphrag.community_member cm JOIN graphrag.community c ON c.id = cm.community_id
              WHERE cm.entity_id = ANY($1) GROUP BY cm.community_id, c.level
              ORDER BY count(*) DESC, c.level DESC, cm.community_id LIMIT 10",
            &[ent_ids.clone().into()],
            |row| get::<i64>(row, 1).map(|v| v.unwrap()),
        )
    };
    let communities: Vec<CommunityHit> = if community_ids.is_empty() {
        Vec::new()
    } else {
        search_communities(
            &ctx,
            query,
            query_embedding.as_deref(),
            None,
            Some(&community_ids),
        )
        .into_iter()
        .filter(|c| c.visibility == "full")
        .take(3)
        .collect()
    };

    // --- exposure accounting ----------------------------------------------
    let mut exposed_order: Vec<i64> = ent_ids.clone();
    for c in &r.chunks {
        exposed_order.extend(c.entities.iter().map(|(e, _)| *e));
    }
    for p in &path_hits {
        exposed_order.extend(p.entity_ids.iter().copied());
    }
    // Community reports name their members, and seeds are returned by name.
    for c in &communities {
        exposed_order.extend(c.member_ids.iter().copied());
    }
    exposed_order.extend(r.seeds.iter().map(|s| s.entity_id));
    let admitted: Option<HashSet<i64>> = {
        let mut seen = HashSet::new();
        let order: Vec<i64> = exposed_order
            .into_iter()
            .filter(|e| seen.insert(*e))
            .collect();
        super::common::admit_exposure(ctx.gid, &order)
    };
    let ok = |e: &i64| admitted.as_ref().map(|a| a.contains(e)).unwrap_or(true);

    // --- assemble ---------------------------------------------------------
    let descs = entity_descriptions(&ctx, &ent_ids);
    let names: HashMap<i64, String> = descs.iter().map(|(k, v)| (*k, v.0.clone())).collect();
    let share = |key: &str, default: f64| ((ctx.opts.f64(key, default)) * budget as f64) as usize;
    let mut used = 0usize;
    let mut out = String::new();
    let mut sections = serde_json::Map::new();
    let add_section = |title: &str,
                       lines: Vec<(String, Value)>,
                       cap: usize,
                       used: &mut usize,
                       out: &mut String|
     -> Vec<Value> {
        let mut kept = Vec::new();
        let header = format!("## {title}\n");
        let mut section_tokens = text::estimate_tokens(&header);
        let mut body = String::new();
        for (line, v) in lines {
            let t = text::estimate_tokens(&line) + 1;
            if section_tokens + t > cap || *used + section_tokens + t > budget {
                break;
            }
            section_tokens += t;
            body.push_str(&line);
            body.push('\n');
            kept.push(v);
        }
        if !kept.is_empty() {
            out.push_str(&header);
            out.push_str(&body);
            out.push('\n');
            *used += section_tokens;
        }
        kept
    };

    let entity_lines: Vec<(String, Value)> = top_entities
        .iter()
        .filter(|(e, _)| ok(e))
        .filter_map(|(e, s)| {
            descs.get(e).map(|(n, t, d)| {
                let ty = t.as_deref().map(|t| format!(" ({t})")).unwrap_or_default();
                let d = if d.is_empty() {
                    String::new()
                } else {
                    format!(": {d}")
                };
                (
                    format!("- {n}{ty}{d}"),
                    json!({"id": e, "name": n, "type": t, "score": s}),
                )
            })
        })
        .collect();
    let kept = add_section(
        "Entities",
        entity_lines,
        share("entity_share", 0.2),
        &mut used,
        &mut out,
    );
    sections.insert("entities".into(), Value::Array(kept));

    let relation_lines: Vec<(String, Value)> = relations
        .iter()
        .filter(|r| ok(&r.1) && ok(&r.2))
        .map(|(id, s, t, rt, w, vf, vt, d, ev)| {
            let cite = ev.iter().map(|c| format!("[#{c}]")).collect::<Vec<_>>().join("");
            let d = d.as_deref().map(|d| format!(": {d}")).unwrap_or_default();
            let sn = names.get(s).cloned().unwrap_or_else(|| s.to_string());
            let tn = names.get(t).cloned().unwrap_or_else(|| t.to_string());
            (
                format!("- {sn} -[{rt}]-> {tn}{}{d} {cite}", fmt_range(vf, vt)),
                json!({"id": id, "source": sn, "type": rt, "target": tn, "weight": w, "valid_from": vf, "valid_to": vt, "evidence": ev}),
            )
        })
        .collect();
    let kept = add_section(
        "Relationships",
        relation_lines,
        share("relation_share", 0.2),
        &mut used,
        &mut out,
    );
    sections.insert("relations".into(), Value::Array(kept));

    // PathRAG presents the most reliable paths last (closest to the question).
    let mut path_lines: Vec<(String, Value)> = path_hits
        .iter()
        .filter(|p| p.entity_ids.iter().all(&ok))
        .map(|p| {
            let cite = p
                .evidence
                .iter()
                .take(4)
                .map(|c| format!("[#{c}]"))
                .collect::<Vec<_>>()
                .join("");
            (
                format!("- {} (reliability {:.2}) {cite}", p.text, p.reliability),
                json!({"path": p.text, "reliability": p.reliability, "evidence": p.evidence}),
            )
        })
        .collect();
    path_lines.reverse();
    let kept = add_section(
        "Reasoning paths",
        path_lines,
        share("path_share", 0.1),
        &mut used,
        &mut out,
    );
    sections.insert("paths".into(), Value::Array(kept));

    let community_lines: Vec<(String, Value)> = communities
        .iter()
        .filter(|c| c.member_ids.iter().all(&ok))
        .map(|c| {
            let title = c
                .title
                .clone()
                .unwrap_or_else(|| format!("Community {}", c.id));
            let summary = c.summary.clone().unwrap_or_default();
            (
                format!(
                    "- {title}{}: {summary}",
                    if c.stale { " (may be outdated)" } else { "" }
                ),
                json!({"id": c.id, "level": c.level, "title": title, "stale": c.stale}),
            )
        })
        .collect();
    let kept = add_section(
        "Community reports",
        community_lines,
        share("community_share", 0.1),
        &mut used,
        &mut out,
    );
    sections.insert("communities".into(), Value::Array(kept));

    let source_lines: Vec<(String, Value)> = r
        .chunks
        .iter()
        .filter(|c| c.entities.iter().all(|(e, _)| ok(e)))
        .map(|c| {
            let src = c.title.clone().or(c.external_id.clone()).unwrap_or_else(|| format!("document {}", c.document_id));
            (
                format!("[#{}] ({src}) {}", c.chunk_id, c.content.replace('\n', " ")),
                json!({"chunk_id": c.chunk_id, "document_id": c.document_id, "external_id": c.external_id, "title": c.title, "score": c.score}),
            )
        })
        .collect();
    let remaining = budget.saturating_sub(used);
    let kept = add_section("Sources", source_lines, remaining, &mut used, &mut out);
    sections.insert("sources".into(), Value::Array(kept));

    let seeds: Vec<Value> = r
        .seeds
        .iter()
        .filter(|s| ok(&s.entity_id))
        .map(|s| json!({"id": s.entity_id, "name": s.name, "weight": s.weight, "via": s.source}))
        .collect();
    JsonB(json!({
        "context": out.trim_end(),
        "token_estimate": used,
        "max_tokens": budget,
        "seeds": seeds,
        "ppr": r.ppr_meta,
        "sections": Value::Object(sections),
        "as_of": one::<String>("SELECT coalesce($1::timestamptz, now())::text", &[ctx.as_of.into()]),
    }))
}
