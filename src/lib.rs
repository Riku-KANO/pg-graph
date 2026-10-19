//! pg_graphrag — a GraphRAG engine that lives inside PostgreSQL 18.
//!
//! The extension stores a provenance-tracked knowledge graph (documents,
//! chunks, entities, relations, communities, LLM reports) and runs the graph
//! side of GraphRAG next to the data: local-push Personalized PageRank
//! (HippoRAG-style multi-hop retrieval), hierarchical Leiden communities
//! (Microsoft GraphRAG-style global search), flow-pruned relational paths
//! (PathRAG), Japanese-aware lexical search and entity resolution. LLM calls
//! (extraction, summarization, answering) stay in the application; the
//! database returns ready-to-use, access-controlled, cited context.

// pgrx needs `TableIterator<(name!(..), ..)>` spelled out in the signature
// (type aliases are not accepted by the macro), so these types are long.
#![allow(clippy::type_complexity)]

use pgrx::prelude::*;

::pgrx::pg_module_magic!(name, version);

pub mod leiden;
pub mod paths;
pub mod ppr;
pub mod text;
pub mod vector;

mod api;
mod guc;

extension_sql!(
    r#"
CREATE SCHEMA IF NOT EXISTS graphrag;
COMMENT ON SCHEMA graphrag IS 'pg_graphrag: knowledge graph, retrieval and governance for GraphRAG';
"#,
    name = "graphrag_schema",
    bootstrap
);

extension_sql_file!("sql/schema.sql", name = "graphrag_tables", finalize);

/// Declares the `graphrag` schema to pgrx; the functions themselves use
/// `#[pg_extern(schema = "graphrag")]` across the `api` modules.
#[pg_schema]
mod graphrag {}

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    guc::init();
}

#[cfg(any(test, feature = "pg_test"))]
mod pgtests;

/// Required by `cargo pgrx test`.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
