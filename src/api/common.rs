//! Shared helpers for the SQL API: SPI wrappers, access-control SQL
//! fragments, option parsing and exposure accounting.

use std::collections::HashSet;

use pgrx::datum::DatumWithOid;
use pgrx::prelude::*;
use pgrx::spi::{SpiHeapTupleData, SpiResult};

use crate::guc;

/// Run a read query and map every row.
pub fn rows<T>(
    sql: &str,
    args: &[DatumWithOid<'_>],
    f: impl Fn(&SpiHeapTupleData<'_>) -> SpiResult<T>,
) -> Vec<T> {
    Spi::connect(|client| {
        let table = client.select(sql, None, args)?;
        let mut out = Vec::with_capacity(table.len());
        for row in table {
            out.push(f(&row)?);
        }
        SpiResult::Ok(out)
    })
    .unwrap_or_else(|e| error!("graphrag: query failed: {e}\n{sql}"))
}

/// Run a data-modifying statement and map every returned row.
pub fn rows_mut<T>(
    sql: &str,
    args: &[DatumWithOid<'_>],
    f: impl Fn(&SpiHeapTupleData<'_>) -> SpiResult<T>,
) -> Vec<T> {
    Spi::connect_mut(|client| {
        let table = client.update(sql, None, args)?;
        let mut out = Vec::with_capacity(table.len());
        for row in table {
            out.push(f(&row)?);
        }
        SpiResult::Ok(out)
    })
    .unwrap_or_else(|e| error!("graphrag: statement failed: {e}\n{sql}"))
}

/// Run a statement for its side effects.
pub fn exec(sql: &str, args: &[DatumWithOid<'_>]) {
    Spi::run_with_args(sql, args)
        .unwrap_or_else(|e| error!("graphrag: statement failed: {e}\n{sql}"))
}

/// First column of the first row of a read query.
pub fn one<T: FromDatum + IntoDatum>(sql: &str, args: &[DatumWithOid<'_>]) -> Option<T> {
    Spi::connect(|client| {
        let table = client.select(sql, Some(1), args)?;
        if table.is_empty() {
            return SpiResult::Ok(None);
        }
        table.first().get_one::<T>()
    })
    .unwrap_or_else(|e| error!("graphrag: query failed: {e}\n{sql}"))
}

/// First column of the first row of a data-modifying statement.
pub fn one_mut<T: FromDatum + IntoDatum>(sql: &str, args: &[DatumWithOid<'_>]) -> Option<T> {
    Spi::connect_mut(|client| {
        let table = client.update(sql, None, args)?;
        if table.is_empty() {
            return SpiResult::Ok(None);
        }
        table.first().get_one::<T>()
    })
    .unwrap_or_else(|e| error!("graphrag: statement failed: {e}\n{sql}"))
}

pub fn get<T: FromDatum + IntoDatum>(row: &SpiHeapTupleData<'_>, i: usize) -> SpiResult<Option<T>> {
    row.get::<T>(i)
}

/// Resolve a graph name to its id, raising an error if it does not exist.
pub fn graph_id(name: &str) -> i32 {
    one::<i32>(
        "SELECT id FROM graphrag.graph WHERE name = $1",
        &[name.into()],
    )
    .unwrap_or_else(|| {
        error!("graphrag: graph \"{name}\" does not exist (create it with graphrag.create_graph)")
    })
}

pub fn current_role() -> String {
    one::<String>("SELECT current_user::text", &[]).unwrap_or_default()
}

/// SQL fragments that restrict rows to what the calling role may see.
///
/// Visibility is evidence based: a chunk is visible when its ACL is empty or
/// names a role the caller belongs to; an entity / relation is visible when
/// it is curated or has at least one visible piece of evidence. When the
/// caller is a superuser, or the graph has no restricted chunks, the checks
/// collapse to `true` so the common case pays nothing.
#[derive(Clone, Copy, Debug)]
pub struct Vis {
    pub restricted: bool,
}

impl Vis {
    pub fn for_graph(gid: i32) -> Vis {
        if unsafe { pg_sys::superuser() } {
            return Vis { restricted: false };
        }
        let restricted = one::<bool>(
            "SELECT EXISTS (SELECT 1 FROM graphrag.chunk WHERE graph_id = $1 AND cardinality(acl) > 0)",
            &[gid.into()],
        )
        .unwrap_or(true);
        Vis { restricted }
    }

    pub fn chunk(&self, alias: &str) -> String {
        if self.restricted {
            format!("graphrag.acl_visible({alias}.acl)")
        } else {
            "true".into()
        }
    }

    pub fn entity(&self, alias: &str) -> String {
        if self.restricted {
            format!(
                "({alias}.is_curated OR EXISTS (SELECT 1 FROM graphrag.mention vm JOIN graphrag.chunk vc ON vc.id = vm.chunk_id \
                 WHERE vm.entity_id = {alias}.id AND graphrag.acl_visible(vc.acl)))"
            )
        } else {
            "true".into()
        }
    }

    pub fn entity_id(&self, column: &str) -> String {
        if self.restricted {
            format!(
                "EXISTS (SELECT 1 FROM graphrag.entity ve WHERE ve.id = {column} AND {})",
                self.entity("ve")
            )
        } else {
            "true".into()
        }
    }

    pub fn relation(&self, alias: &str) -> String {
        if self.restricted {
            format!(
                "({alias}.is_curated OR EXISTS (SELECT 1 FROM graphrag.relation_evidence vev JOIN graphrag.chunk vc2 ON vc2.id = vev.chunk_id \
                 WHERE vev.relation_id = {alias}.id AND graphrag.acl_visible(vc2.acl)))"
            )
        } else {
            "true".into()
        }
    }

    /// Relation visible, both endpoints visible.
    pub fn edge(&self, alias: &str) -> String {
        if self.restricted {
            format!(
                "{} AND {} AND {}",
                self.relation(alias),
                self.entity_id(&format!("{alias}.source_id")),
                self.entity_id(&format!("{alias}.target_id"))
            )
        } else {
            "true".into()
        }
    }
}

/// Per-call options passed as `jsonb`.
pub struct Opts(pub serde_json::Value);

impl Opts {
    pub fn new(v: Option<pgrx::JsonB>) -> Opts {
        Opts(v.map(|j| j.0).unwrap_or(serde_json::Value::Null))
    }

    pub fn f64(&self, key: &str, default: f64) -> f64 {
        self.0.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    pub fn usize(&self, key: &str, default: usize) -> usize {
        self.0
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    pub fn bool(&self, key: &str, default: bool) -> bool {
        self.0.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    pub fn ids(&self, key: &str) -> Vec<i64> {
        self.0
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
            .unwrap_or_default()
    }
}

/// Exposure accounting (see `graphrag.exposure_tracking`). Entities must be
/// given best-first; returns `None` when tracking is off (everything
/// admitted), otherwise the admitted subset.
pub fn admit_exposure(gid: i32, entities: &[i64]) -> Option<HashSet<i64>> {
    if !guc::EXPOSURE_TRACKING.get() || entities.is_empty() {
        return None;
    }
    let principal = current_role();
    let admitted: Vec<i64> = one_mut::<Vec<i64>>(
        "SELECT graphrag._exposure_admit($1, $2::name, $3, $4)",
        &[
            gid.into(),
            principal.into(),
            entities.to_vec().into(),
            guc::EXPOSURE_BUDGET.get().into(),
        ],
    )
    .unwrap_or_default();
    Some(admitted.into_iter().collect())
}

/// Keep only items whose entities were all admitted.
pub fn filter_exposed<T>(items: Vec<T>, entities_of: impl Fn(&T) -> Vec<i64>, gid: i32) -> Vec<T> {
    let mut order: Vec<i64> = Vec::new();
    let mut seen = HashSet::new();
    for it in &items {
        for e in entities_of(it) {
            if seen.insert(e) {
                order.push(e);
            }
        }
    }
    match admit_exposure(gid, &order) {
        None => items,
        Some(admitted) => items
            .into_iter()
            .filter(|it| entities_of(it).iter().all(|e| admitted.contains(e)))
            .collect(),
    }
}

/// Normalize relation type labels coming from different extraction prompts
/// (`works for`, `WORKS_FOR`, `Works-For` → `works_for`).
pub fn normalize_rel_type(t: &str) -> String {
    let n = crate::text::normalize(t);
    let mut out = String::new();
    let mut last_us = false;
    for c in n.trim().chars() {
        if c.is_whitespace() || c == '-' || c == '_' {
            if !last_us && !out.is_empty() {
                out.push('_');
                last_us = true;
            }
        } else {
            out.push(c);
            last_us = false;
        }
    }
    let out = out.trim_end_matches('_').to_string();
    if out.is_empty() {
        "related_to".into()
    } else {
        out
    }
}
