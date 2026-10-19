//! Configuration parameters (`SET graphrag.xxx = ...`).

use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

pub static PPR_DAMPING: GucSetting<f64> = GucSetting::<f64>::new(0.5);
pub static PPR_EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1e-4);
pub static MAX_EXPANSION: GucSetting<i32> = GucSetting::<i32>::new(20_000);
pub static EXPOSURE_TRACKING: GucSetting<bool> = GucSetting::<bool>::new(false);
pub static EXPOSURE_BUDGET: GucSetting<f64> = GucSetting::<f64>::new(0.0);

pub fn init() {
    GucRegistry::define_float_guc(
        c"graphrag.ppr_damping",
        c"Probability of following an edge in Personalized PageRank.",
        c"1 - damping is the teleport (restart) probability. HippoRAG uses 0.5.",
        &PPR_DAMPING,
        0.01,
        0.99,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"graphrag.ppr_epsilon",
        c"Residual threshold of the local-push PPR approximation.",
        c"Smaller values are more accurate and touch more of the graph.",
        &PPR_EPSILON,
        1e-9,
        0.1,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"graphrag.max_expansion",
        c"Maximum number of graph nodes loaded by one retrieval call.",
        c"Bounds latency and memory of PPR and path search on large graphs.",
        &MAX_EXPANSION,
        10,
        10_000_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"graphrag.exposure_tracking",
        c"Record which entities each role has been shown.",
        c"Defence against graph-extraction attacks: see graphrag.exposure_report(). Superuser only.",
        &EXPOSURE_TRACKING,
        GucContext::Suset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"graphrag.exposure_budget",
        c"Maximum fraction of a graph's entities a role may be shown (0 = unlimited).",
        c"Requires graphrag.exposure_tracking. Results that would exceed the budget are withheld. Superuser only.",
        &EXPOSURE_BUDGET,
        0.0,
        1.0,
        GucContext::Suset,
        GucFlags::default(),
    );
}
