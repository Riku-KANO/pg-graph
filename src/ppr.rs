//! Personalized PageRank.
//!
//! HippoRAG (NeurIPS 2024) and HippoRAG 2 (2025) showed that running PPR
//! from the entities mentioned in a question is an effective, LLM-free way to
//! do multi-hop retrieval over a knowledge graph. Computing PPR over the whole
//! graph for every query does not scale inside a database, so this module
//! implements the *local push* algorithm of Andersen, Chung and Lang (FOCS
//! 2006): only nodes that receive enough probability mass are ever touched,
//! and their adjacency lists are fetched lazily, in batches, through a loader
//! callback (in practice an SPI query that also enforces access control and
//! temporal validity).

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug)]
pub struct PprParams {
    /// Probability of following an edge (1 - teleport probability).
    pub damping: f64,
    /// Push threshold: residual mass per unit degree below which a node is not pushed.
    pub epsilon: f64,
    /// Upper bound on nodes whose adjacency is loaded.
    pub max_nodes: usize,
    /// Upper bound on push operations.
    pub max_pushes: usize,
}

impl Default for PprParams {
    fn default() -> Self {
        PprParams {
            damping: 0.5,
            epsilon: 1e-4,
            max_nodes: 20_000,
            max_pushes: 2_000_000,
        }
    }
}

#[derive(Debug, Default)]
pub struct PprResult {
    pub scores: HashMap<i64, f64>,
    pub loaded_nodes: usize,
    pub pushes: usize,
    /// True when `max_nodes` or `max_pushes` cut the computation short.
    pub truncated: bool,
}

pub type Adjacency = HashMap<i64, Vec<(i64, f64)>>;

fn normalize_seeds(seeds: &[(i64, f64)]) -> Vec<(i64, f64)> {
    let mut merged: HashMap<i64, f64> = HashMap::new();
    for &(id, w) in seeds {
        if w.is_finite() && w > 0.0 {
            *merged.entry(id).or_insert(0.0) += w;
        }
    }
    let total: f64 = merged.values().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut v: Vec<(i64, f64)> = merged.into_iter().map(|(k, w)| (k, w / total)).collect();
    v.sort_by_key(|(k, _)| *k);
    v
}

/// Approximate PPR by local push. `load` receives node ids whose adjacency is
/// needed and returns `(neighbor, weight)` lists; nodes missing from the
/// returned map are treated as having no (visible) edges.
pub fn local_push_ppr<F>(seeds: &[(i64, f64)], params: &PprParams, mut load: F) -> PprResult
where
    F: FnMut(&[i64]) -> Adjacency,
{
    let seeds = normalize_seeds(seeds);
    let mut result = PprResult::default();
    if seeds.is_empty() {
        return result;
    }
    let alpha = (1.0 - params.damping).clamp(1e-6, 1.0);
    let eps = params.epsilon.max(1e-12);

    let mut residual: HashMap<i64, f64> = seeds.iter().copied().collect();
    let mut adj: Adjacency = HashMap::new();
    let mut weight_sum: HashMap<i64, f64> = HashMap::new();
    let degree = |adj: &Adjacency, u: i64| -> f64 {
        adj.get(&u).map(|e| e.len().max(1)).unwrap_or(1) as f64
    };

    'outer: loop {
        // Load adjacency for every unloaded node that holds enough residual mass.
        let mut need: Vec<(i64, f64)> = residual
            .iter()
            .filter(|(u, r)| !adj.contains_key(u) && **r > eps)
            .map(|(u, r)| (*u, *r))
            .collect();
        if !need.is_empty() {
            let room = params.max_nodes.saturating_sub(adj.len());
            if room < need.len() {
                result.truncated = true;
                need.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                need.truncate(room);
            }
            if !need.is_empty() {
                let ids: Vec<i64> = need.iter().map(|(u, _)| *u).collect();
                let mut loaded = load(&ids);
                for u in ids {
                    let mut edges = loaded.remove(&u).unwrap_or_default();
                    edges.retain(|(v, w)| *v != u && w.is_finite() && *w > 0.0);
                    weight_sum.insert(u, edges.iter().map(|(_, w)| *w).sum());
                    adj.insert(u, edges);
                }
            }
        }

        let mut queue: VecDeque<i64> = residual
            .iter()
            .filter(|(u, r)| adj.contains_key(u) && **r >= eps * degree(&adj, **u))
            .map(|(u, _)| *u)
            .collect();
        if queue.is_empty() {
            break;
        }
        let mut in_queue: HashSet<i64> = queue.iter().copied().collect();

        while let Some(u) = queue.pop_front() {
            in_queue.remove(&u);
            let ru = *residual.get(&u).unwrap_or(&0.0);
            if ru < eps * degree(&adj, u) {
                continue;
            }
            result.pushes += 1;
            if result.pushes > params.max_pushes {
                result.truncated = true;
                break 'outer;
            }
            *result.scores.entry(u).or_insert(0.0) += alpha * ru;
            residual.insert(u, 0.0);
            let spread = (1.0 - alpha) * ru;
            let edges = &adj[&u];
            let ws = weight_sum[&u];
            let targets: Vec<(i64, f64)> = if edges.is_empty() || ws <= 0.0 {
                // Dangling node: teleport back to the seed distribution.
                seeds.iter().map(|(s, w)| (*s, spread * w)).collect()
            } else {
                edges.iter().map(|(v, w)| (*v, spread * w / ws)).collect()
            };
            for (v, mass) in targets {
                let rv = residual.entry(v).or_insert(0.0);
                *rv += mass;
                if adj.contains_key(&v) && *rv >= eps * degree(&adj, v) && !in_queue.contains(&v) {
                    in_queue.insert(v);
                    queue.push_back(v);
                }
            }
        }
    }
    result.loaded_nodes = adj.len();
    result
}

/// Exact PPR by power iteration over a fully materialized adjacency. Used for
/// verification and for small graphs.
pub fn power_iteration_ppr(
    adj: &Adjacency,
    seeds: &[(i64, f64)],
    damping: f64,
    iterations: usize,
) -> HashMap<i64, f64> {
    let seeds = normalize_seeds(seeds);
    let mut nodes: HashSet<i64> = adj.keys().copied().collect();
    for edges in adj.values() {
        nodes.extend(edges.iter().map(|(v, _)| *v));
    }
    nodes.extend(seeds.iter().map(|(s, _)| *s));
    let alpha = 1.0 - damping;
    let mut pi: HashMap<i64, f64> = seeds.iter().copied().collect();
    for _ in 0..iterations {
        let mut next: HashMap<i64, f64> = HashMap::new();
        for (s, w) in &seeds {
            *next.entry(*s).or_insert(0.0) += alpha * w;
        }
        for (&u, &pu) in &pi {
            let edges = adj.get(&u).map(|e| e.as_slice()).unwrap_or(&[]);
            let ws: f64 = edges.iter().filter(|(v, _)| *v != u).map(|(_, w)| *w).sum();
            if ws <= 0.0 {
                for (s, w) in &seeds {
                    *next.entry(*s).or_insert(0.0) += damping * pu * w;
                }
            } else {
                for (v, w) in edges.iter().filter(|(v, _)| *v != u) {
                    *next.entry(*v).or_insert(0.0) += damping * pu * w / ws;
                }
            }
        }
        pi = next;
    }
    pi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn undirected(edges: &[(i64, i64, f64)]) -> Adjacency {
        let mut adj: Adjacency = HashMap::new();
        for &(a, b, w) in edges {
            adj.entry(a).or_default().push((b, w));
            adj.entry(b).or_default().push((a, w));
        }
        adj
    }

    #[test]
    fn push_matches_power_iteration() {
        let adj = undirected(&[
            (1, 2, 1.0),
            (2, 3, 2.0),
            (3, 4, 1.0),
            (4, 1, 0.5),
            (3, 5, 1.0),
            (5, 6, 3.0),
            (6, 7, 1.0),
            (7, 5, 1.0),
        ]);
        let seeds = vec![(1, 1.0), (6, 0.5)];
        let exact = power_iteration_ppr(&adj, &seeds, 0.5, 200);
        let params = PprParams {
            epsilon: 1e-7,
            ..Default::default()
        };
        let mut calls = 0;
        let approx = local_push_ppr(&seeds, &params, |ids| {
            calls += 1;
            ids.iter()
                .filter_map(|id| adj.get(id).map(|e| (*id, e.clone())))
                .collect()
        });
        assert!(calls >= 1);
        for (node, p) in &exact {
            let q = approx.scores.get(node).copied().unwrap_or(0.0);
            assert!((p - q).abs() < 1e-4, "node {node}: exact {p} approx {q}");
        }
        let total: f64 = approx.scores.values().sum();
        assert!(total > 0.99 && total <= 1.0 + 1e-9, "{total}");
    }

    #[test]
    fn push_is_local() {
        // A long path: with a coarse epsilon only the neighbourhood of the seed is loaded.
        let edges: Vec<(i64, i64, f64)> = (0..10_000).map(|i| (i, i + 1, 1.0)).collect();
        let adj = undirected(&edges);
        let params = PprParams {
            epsilon: 1e-3,
            ..Default::default()
        };
        let res = local_push_ppr(&[(5_000, 1.0)], &params, |ids| {
            ids.iter()
                .filter_map(|id| adj.get(id).map(|e| (*id, e.clone())))
                .collect()
        });
        assert!(res.loaded_nodes < 50, "loaded {}", res.loaded_nodes);
        let best = res
            .scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(*best.0, 5_000);
    }

    #[test]
    fn respects_max_nodes() {
        let edges: Vec<(i64, i64, f64)> = (0..1000).map(|i| (0, i + 1, 1.0)).collect();
        let adj = undirected(&edges);
        let params = PprParams {
            epsilon: 1e-9,
            max_nodes: 10,
            ..Default::default()
        };
        let res = local_push_ppr(&[(0, 1.0)], &params, |ids| {
            ids.iter()
                .filter_map(|id| adj.get(id).map(|e| (*id, e.clone())))
                .collect()
        });
        assert!(res.truncated);
        assert!(res.loaded_nodes <= 10);
    }

    #[test]
    fn isolated_seed_keeps_all_mass() {
        let res = local_push_ppr(&[(42, 1.0)], &PprParams::default(), |_| HashMap::new());
        let p = res.scores[&42];
        assert!((p - 1.0).abs() < 1e-3, "{p}");
    }
}
