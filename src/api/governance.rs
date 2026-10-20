//! Governance: erasure, temporal invalidation, exposure accounting, stats.
//!
//! Regulations such as the GDPR (Art. 17, right to erasure) and Japan's
//! APPI require that personal data can be deleted on request. In a GraphRAG
//! system the data does not only live in the source document: it has been
//! copied into entity / relation descriptions, LLM summaries, community
//! reports and embeddings. Because every derived object carries provenance,
//! `forget_document` can remove exactly what was derived from a document and
//! leave an auditable record.

use pgrx::datum::TimestampWithTimeZone;
use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::json;

use super::common::{get, graph_id, one, one_mut, rows, rows_mut};

fn forget(document_id: i64, clear_embeddings: bool) -> JsonB {
    // The work happens in graphrag._erase_document (SECURITY DEFINER), so
    // that derived objects hidden from the caller by row security are erased
    // too; it checks that the caller may delete the document itself.
    one_mut::<JsonB>(
        "SELECT graphrag._erase_document($1, $2, current_user)",
        &[document_id.into(), clear_embeddings.into()],
    )
    .unwrap_or_else(|| error!("graphrag: document {document_id} does not exist"))
}

/// Erase a document and everything derived from it (right to erasure).
/// Shared entities / relations survive with their remaining evidence; LLM
/// reports that may contain the erased content are deleted (they show up in
/// `graphrag.pending_report` for regeneration); an entry is written to
/// `graphrag.erasure_log`.
#[pg_extern(schema = "graphrag")]
fn forget_document(document_id: i64, clear_embeddings: default!(bool, true)) -> JsonB {
    forget(document_id, clear_embeddings)
}

/// Erase a document identified by its external id.
#[pg_extern(schema = "graphrag", name = "forget_document")]
fn forget_document_by_external_id(
    graph: &str,
    external_id: &str,
    clear_embeddings: default!(bool, true),
) -> JsonB {
    let gid = graph_id(graph);
    let id = one::<i64>(
        "SELECT id FROM graphrag.document WHERE graph_id = $1 AND external_id = $2",
        &[gid.into(), external_id.into()],
    )
    .unwrap_or_else(|| {
        error!("graphrag: document \"{external_id}\" not found in graph \"{graph}\"")
    });
    forget(id, clear_embeddings)
}

/// End the validity of a fact at `at` (default now()) without deleting it.
#[pg_extern(schema = "graphrag")]
fn invalidate_relation(
    relation_id: i64,
    at: default!(Option<TimestampWithTimeZone>, "NULL"),
) -> bool {
    let ok = one_mut::<i64>(
        "UPDATE graphrag.relation
            SET valid_during = tstzrange(lower(valid_during), coalesce($2, now()), '[)'), invalidated_at = now()
          WHERE id = $1 AND valid_during @> coalesce($2, now())
            AND (lower(valid_during) IS NULL OR lower(valid_during) < coalesce($2, now()))
          RETURNING id",
        &[relation_id.into(), at.into()],
    );
    ok.is_some()
}

/// Per-role disclosure statistics (see `graphrag.exposure_tracking`).
#[pg_extern(schema = "graphrag")]
fn exposure_report(
    graph: &str,
) -> TableIterator<
    'static,
    (
        name!(principal, String),
        name!(exposed_entities, i64),
        name!(total_entities, i64),
        name!(ratio, f64),
        name!(last_seen, Option<TimestampWithTimeZone>),
    ),
> {
    let gid = graph_id(graph);
    let total = one::<i64>(
        "SELECT count(*) FROM graphrag.entity WHERE graph_id = $1",
        &[gid.into()],
    )
    .unwrap_or(0);
    let out = rows(
        "SELECT principal::text, count(*), max(last_seen) FROM graphrag.exposure WHERE graph_id = $1 GROUP BY principal ORDER BY 2 DESC, 1",
        &[gid.into()],
        |row| {
            let n = get::<i64>(row, 2)?.unwrap_or(0);
            Ok((
                get::<String>(row, 1)?.unwrap_or_default(),
                n,
                total,
                if total > 0 { n as f64 / total as f64 } else { 0.0 },
                get::<TimestampWithTimeZone>(row, 3)?,
            ))
        },
    );
    TableIterator::new(out)
}

/// Forget exposure records for a graph (optionally for one role only).
#[pg_extern(schema = "graphrag")]
fn reset_exposure(graph: &str, principal: default!(Option<&str>, "NULL")) -> i64 {
    let gid = graph_id(graph);
    rows_mut(
        "DELETE FROM graphrag.exposure WHERE graph_id = $1 AND ($2::text IS NULL OR principal = $2::name) RETURNING 1",
        &[gid.into(), principal.into()],
        |row| get::<i32>(row, 1),
    )
    .len() as i64
}

/// Size and health of a graph.
#[pg_extern(schema = "graphrag")]
fn stats(graph: &str) -> JsonB {
    let gid = graph_id(graph);
    let v = one::<JsonB>(
        "SELECT jsonb_build_object(
            'documents', (SELECT count(*) FROM graphrag.document WHERE graph_id = $1),
            'chunks', (SELECT count(*) FROM graphrag.chunk WHERE graph_id = $1),
            'restricted_chunks', (SELECT count(*) FROM graphrag.chunk WHERE graph_id = $1 AND cardinality(acl) > 0),
            'chunks_with_embedding', (SELECT count(*) FROM graphrag.chunk WHERE graph_id = $1 AND embedding IS NOT NULL),
            'entities', (SELECT count(*) FROM graphrag.entity WHERE graph_id = $1),
            'entities_with_embedding', (SELECT count(*) FROM graphrag.entity WHERE graph_id = $1 AND embedding IS NOT NULL),
            'relations', (SELECT count(*) FROM graphrag.relation WHERE graph_id = $1),
            'current_relations', (SELECT count(*) FROM graphrag.relation WHERE graph_id = $1 AND valid_during @> now()),
            'communities', (SELECT count(*) FROM graphrag.community WHERE graph_id = $1),
            'reports', (SELECT count(*) FROM graphrag.report WHERE graph_id = $1),
            'stale_reports', (SELECT count(*) FROM graphrag.report WHERE graph_id = $1 AND stale),
            'pending_reports', (SELECT count(*) FROM graphrag.pending_report WHERE graph_id = $1),
            'erasures', (SELECT count(*) FROM graphrag.erasure_log WHERE graph_id = $1))",
        &[gid.into()],
    );
    v.unwrap_or(JsonB(json!({})))
}
