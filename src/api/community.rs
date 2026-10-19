//! Communities and LLM reports.
//!
//! `build_communities` runs hierarchical Leiden inside the database and
//! *re-uses* the existing report of every community whose member set did not
//! change, so after incremental ingestion only new or changed communities
//! need (paid) LLM summarization — `graphrag.pending_report` lists them.

use std::collections::HashMap;

use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::json;

use super::common::{exec, get, graph_id, one, one_mut, rows, rows_mut};
use crate::leiden::{hierarchical_leiden, leiden, modularity, Graph, Rng};
use crate::text;

fn member_hash(members: &[i64]) -> String {
    // FNV-1a over the sorted ids: stable across processes and versions.
    let mut sorted = members.to_vec();
    sorted.sort_unstable();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in sorted {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}-{}", members.len())
}

/// Detect hierarchical communities with Leiden over currently valid
/// relations and store them. Reports of communities whose membership is
/// unchanged are carried over.
#[pg_extern(schema = "graphrag")]
fn build_communities(
    graph: &str,
    resolution: default!(f64, 1.0),
    max_cluster_size: default!(i32, 10),
    max_levels: default!(i32, 3),
    seed: default!(i64, 42),
    min_size: default!(i32, 2),
) -> JsonB {
    let gid = graph_id(graph);
    let edges = rows(
        "SELECT least(source_id, target_id), greatest(source_id, target_id), sum(weight)
           FROM graphrag.relation
          WHERE graph_id = $1 AND valid_during @> now()
          GROUP BY 1, 2",
        &[gid.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<i64>(row, 2)?.unwrap(),
                get::<f64>(row, 3)?.unwrap_or(1.0),
            ))
        },
    );
    let mut index: HashMap<i64, usize> = HashMap::new();
    let mut ids: Vec<i64> = Vec::new();
    for (s, t, _) in &edges {
        for n in [*s, *t] {
            index.entry(n).or_insert_with(|| {
                ids.push(n);
                ids.len() - 1
            });
        }
    }
    let local: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|(s, t, w)| (index[s], index[t], *w))
        .collect();
    let g = Graph::from_edges(ids.len(), &local);
    let clusters = hierarchical_leiden(
        &g,
        resolution,
        max_cluster_size.max(1) as usize,
        max_levels.max(1) as usize,
        seed as u64,
    );
    let top_modularity = {
        let flat = leiden(&g, resolution, &mut Rng::new(seed as u64));
        modularity(&g, &flat, resolution)
    };

    // Existing communities and their reports, keyed by member set.
    let old: Vec<(i64, String)> = rows(
        "SELECT id, member_hash FROM graphrag.community WHERE graph_id = $1",
        &[gid.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?.unwrap_or_default(),
            ))
        },
    );
    let mut old_by_hash: HashMap<String, i64> = HashMap::new();
    for (id, h) in old {
        old_by_hash.entry(h).or_insert(id);
    }

    let mut db_id: Vec<Option<i64>> = vec![None; clusters.len()];
    let mut per_level: HashMap<usize, usize> = HashMap::new();
    let mut reused = 0;
    for (i, c) in clusters.iter().enumerate() {
        if c.members.len() < min_size.max(1) as usize {
            continue;
        }
        let members: Vec<i64> = c.members.iter().map(|m| ids[*m]).collect();
        let rank: f64 = c.members.iter().map(|m| g.degree[*m]).sum();
        let parent = c.parent.and_then(|p| db_id[p]);
        let hash = member_hash(&members);
        let id = one_mut::<i64>(
            "INSERT INTO graphrag.community (graph_id, level, parent_id, member_hash, size, rank)
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
            &[
                gid.into(),
                (c.level as i32).into(),
                parent.into(),
                hash.as_str().into(),
                (members.len() as i32).into(),
                rank.into(),
            ],
        )
        .expect("community id");
        exec(
            "INSERT INTO graphrag.community_member (community_id, entity_id) SELECT $1, unnest($2::bigint[])",
            &[id.into(), members.into()],
        );
        if let Some(old_id) = old_by_hash.remove(&hash) {
            let moved = rows_mut(
                "UPDATE graphrag.report SET community_id = $2 WHERE community_id = $1 RETURNING id",
                &[old_id.into(), id.into()],
                |row| get::<i64>(row, 1),
            );
            reused += moved.len();
        }
        db_id[i] = Some(id);
        *per_level.entry(c.level).or_insert(0) += 1;
    }
    // Drop the previous generation (reports that were not carried over go with it).
    let removed = rows_mut(
        "DELETE FROM graphrag.community WHERE graph_id = $1 AND id <> ALL ($2) RETURNING id",
        &[
            gid.into(),
            db_id.iter().flatten().copied().collect::<Vec<i64>>().into(),
        ],
        |row| get::<i64>(row, 1),
    );
    let mut levels: Vec<(usize, usize)> = per_level.into_iter().collect();
    levels.sort();
    let pending = one::<i64>(
        "SELECT count(*) FROM graphrag.pending_report WHERE graph_id = $1 AND kind = 'community'",
        &[gid.into()],
    )
    .unwrap_or(0);
    JsonB(json!({
        "nodes": ids.len(),
        "edges": edges.len(),
        "communities": db_id.iter().flatten().count(),
        "levels": levels.into_iter().map(|(l, n)| json!({"level": l, "communities": n})).collect::<Vec<_>>(),
        "modularity": top_modularity,
        "reports_reused": reused,
        "previous_removed": removed.len(),
        "reports_pending": pending,
    }))
}

/// Text to send to an LLM to write a community report: sub-community
/// reports (if any), member entities with their visible descriptions and the
/// internal relations, highest-degree first, within a token budget.
#[pg_extern(schema = "graphrag")]
fn community_context(community_id: i64, max_tokens: default!(i32, 6000)) -> String {
    let (gid, level) = {
        let r = rows(
            "SELECT graph_id, level FROM graphrag.community WHERE id = $1",
            &[community_id.into()],
            |row| Ok((get::<i32>(row, 1)?.unwrap(), get::<i32>(row, 2)?.unwrap())),
        );
        *r.first()
            .unwrap_or_else(|| error!("graphrag: community {community_id} does not exist"))
    };
    let vis = super::common::Vis::for_graph(gid);
    let budget = max_tokens.max(200) as usize;
    let mut out = format!("# Community {community_id} (level {level})\n\n");
    let mut used = text::estimate_tokens(&out);
    let push = |line: String, used: &mut usize, out: &mut String| -> bool {
        let t = text::estimate_tokens(&line) + 1;
        if *used + t > budget {
            return false;
        }
        *used += t;
        out.push_str(&line);
        out.push('\n');
        true
    };

    let children = rows(
        "SELECT c.id, rep.title, rep.content FROM graphrag.community c
           JOIN graphrag.report rep ON rep.community_id = c.id
          WHERE c.parent_id = $1 AND graphrag.acls_satisfied(rep.required_acls)
          ORDER BY c.rank DESC, c.id",
        &[community_id.into()],
        |row| {
            Ok((
                get::<i64>(row, 1)?.unwrap(),
                get::<String>(row, 2)?,
                get::<String>(row, 3)?.unwrap_or_default(),
            ))
        },
    );
    if !children.is_empty() {
        push("## Sub-community reports".into(), &mut used, &mut out);
        for (id, title, content) in children {
            if !push(
                format!("- [{id}] {}: {content}", title.unwrap_or_default()),
                &mut used,
                &mut out,
            ) {
                break;
            }
        }
        out.push('\n');
    }

    let ent_sql = format!(
        "SELECT e.id, e.name, e.entity_type,
                (SELECT count(*) FROM graphrag.relation r WHERE (r.source_id = e.id OR r.target_id = e.id) AND r.valid_during @> now()) AS degree,
                coalesce(
                    (SELECT rep.content FROM graphrag.report rep WHERE rep.entity_id = e.id AND graphrag.acls_satisfied(rep.required_acls)),
                    (SELECT string_agg(d, ' / ') FROM (
                        SELECT DISTINCT m.description AS d FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
                         WHERE m.entity_id = e.id AND m.description IS NOT NULL AND {} ORDER BY 1 LIMIT 5) s),
                    e.metadata->>'description', '')
           FROM graphrag.community_member cm JOIN graphrag.entity e ON e.id = cm.entity_id
          WHERE cm.community_id = $1 AND {}
          ORDER BY degree DESC, e.id",
        vis.chunk("c"),
        vis.entity("e")
    );
    let entities = rows(&ent_sql, &[community_id.into()], |row| {
        Ok((
            get::<i64>(row, 1)?.unwrap(),
            get::<String>(row, 2)?.unwrap_or_default(),
            get::<String>(row, 3)?,
            get::<i64>(row, 4)?.unwrap_or(0),
            get::<String>(row, 5)?.unwrap_or_default(),
        ))
    });
    push(
        "## Entities (id | name | type | degree | description)".into(),
        &mut used,
        &mut out,
    );
    for (id, name, ty, deg, desc) in &entities {
        if !push(
            format!(
                "{id} | {name} | {} | {deg} | {desc}",
                ty.clone().unwrap_or_default()
            ),
            &mut used,
            &mut out,
        ) {
            break;
        }
    }
    out.push('\n');
    let rel_sql = format!(
        "SELECT s.name, r.rel_type, t.name, r.weight,
                coalesce((SELECT string_agg(d, ' / ') FROM (
                    SELECT DISTINCT ev.description AS d FROM graphrag.relation_evidence ev JOIN graphrag.chunk c ON c.id = ev.chunk_id
                     WHERE ev.relation_id = r.id AND ev.description IS NOT NULL AND {} ORDER BY 1 LIMIT 3) x), ''),
                lower(r.valid_during)::text, upper(r.valid_during)::text
           FROM graphrag.relation r
           JOIN graphrag.entity s ON s.id = r.source_id
           JOIN graphrag.entity t ON t.id = r.target_id
          WHERE r.source_id IN (SELECT entity_id FROM graphrag.community_member WHERE community_id = $1)
            AND r.target_id IN (SELECT entity_id FROM graphrag.community_member WHERE community_id = $1)
            AND r.valid_during @> now() AND {}
          ORDER BY r.weight DESC, r.id",
        vis.chunk("c"),
        vis.edge("r")
    );
    let relations = rows(&rel_sql, &[community_id.into()], |row| {
        Ok((
            get::<String>(row, 1)?.unwrap_or_default(),
            get::<String>(row, 2)?.unwrap_or_default(),
            get::<String>(row, 3)?.unwrap_or_default(),
            get::<f64>(row, 4)?.unwrap_or(1.0),
            get::<String>(row, 5)?.unwrap_or_default(),
            get::<String>(row, 6)?,
            get::<String>(row, 7)?,
        ))
    });
    push(
        "## Relationships (source | type | target | weight | description | valid)".into(),
        &mut used,
        &mut out,
    );
    for (s, ty, t, w, d, vf, vt) in relations {
        let valid = match (vf, vt) {
            (None, None) => String::new(),
            (f, t) => format!(
                "{} – {}",
                f.unwrap_or_default(),
                t.unwrap_or_else(|| "now".into())
            ),
        };
        if !push(
            format!("{s} | {ty} | {t} | {w} | {d} | {valid}"),
            &mut used,
            &mut out,
        ) {
            break;
        }
    }
    out.trim_end().to_string()
}

/// Store an LLM-generated report for an entity, relation or community.
/// `required_acls` is computed from the ACLs of *all* evidence behind the
/// object, so the report is only shown to roles that could read every source.
#[allow(clippy::too_many_arguments)]
#[pg_extern(schema = "graphrag")]
fn set_report(
    kind: &str,
    ref_id: i64,
    content: &str,
    title: default!(Option<&str>, "NULL"),
    embedding: default!(Option<Vec<f32>>, "NULL"),
    generated_by: default!(Option<&str>, "NULL"),
) -> i64 {
    let (column, graph_sql, acl_sql) = match kind {
        "entity" => (
            "entity_id",
            "SELECT graph_id FROM graphrag.entity WHERE id = $1",
            "SELECT graphrag._evidence_acls(ARRAY[$1::bigint], '{}')",
        ),
        "relation" => (
            "relation_id",
            "SELECT graph_id FROM graphrag.relation WHERE id = $1",
            "SELECT graphrag._evidence_acls('{}', ARRAY[$1::bigint])",
        ),
        "community" => (
            "community_id",
            "SELECT graph_id FROM graphrag.community WHERE id = $1",
            "SELECT graphrag._evidence_acls(
                ARRAY(SELECT entity_id FROM graphrag.community_member WHERE community_id = $1),
                ARRAY(SELECT r.id FROM graphrag.relation r
                       WHERE r.source_id IN (SELECT entity_id FROM graphrag.community_member WHERE community_id = $1)
                         AND r.target_id IN (SELECT entity_id FROM graphrag.community_member WHERE community_id = $1)))",
        ),
        other => error!("graphrag: unknown report kind \"{other}\" (expected entity, relation or community)"),
    };
    let gid = one::<i32>(graph_sql, &[ref_id.into()])
        .unwrap_or_else(|| error!("graphrag: {kind} {ref_id} does not exist"));
    let acls = one_mut::<JsonB>(acl_sql, &[ref_id.into()]).unwrap_or(JsonB(json!([])));
    let sql = format!(
        "INSERT INTO graphrag.report (graph_id, {column}, title, content, embedding, required_acls, stale, generated_by)
         VALUES ($1, $2, $3, $4, $5, $6, false, $7)
         ON CONFLICT ({column}) DO UPDATE SET
            title = EXCLUDED.title, content = EXCLUDED.content, embedding = EXCLUDED.embedding,
            required_acls = EXCLUDED.required_acls, stale = false, generated_by = EXCLUDED.generated_by,
            updated_at = now()
         RETURNING id"
    );
    one_mut::<i64>(
        &sql,
        &[
            gid.into(),
            ref_id.into(),
            title.into(),
            content.into(),
            embedding.into(),
            acls.into(),
            generated_by.into(),
        ],
    )
    .expect("report id")
}
