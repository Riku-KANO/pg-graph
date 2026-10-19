//! Relational path retrieval with flow-based pruning, after PathRAG
//! (Chen et al., 2025, "PathRAG: Pruning Graph-based Retrieval Augmented
//! Generation with Relational Paths").
//!
//! Starting from a source entity with one unit of "resource", each hop
//! passes `decay * resource / degree` to every neighbour. Branches whose
//! resource drops below `threshold` are pruned, which keeps the search local
//! and penalises paths that run through hub nodes. A path's reliability is the
//! mean resource over its hops. Returning explicit paths (rather than a bag of
//! triples) gives the LLM — and the human auditing the answer — a readable
//! chain of evidence.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub to: i64,
    pub relation_id: i64,
    /// True when the stored relation points from the current node to `to`.
    pub forward: bool,
}

pub type PathAdjacency = HashMap<i64, Vec<Edge>>;

#[derive(Clone, Debug)]
pub struct PathParams {
    pub max_hops: usize,
    pub decay: f64,
    pub threshold: f64,
    pub top_k: usize,
    pub max_expansions: usize,
}

impl Default for PathParams {
    fn default() -> Self {
        PathParams {
            max_hops: 3,
            decay: 0.8,
            threshold: 0.05,
            top_k: 10,
            max_expansions: 200_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Path {
    pub nodes: Vec<i64>,
    pub edges: Vec<(i64, bool)>,
    pub reliability: f64,
}

/// Find the most reliable simple paths between every pair of `endpoints`.
/// Pairs are unordered; for each pair the search starts from the endpoint
/// that appears first in `endpoints`.
pub fn find_paths(adj: &PathAdjacency, endpoints: &[i64], params: &PathParams) -> Vec<Path> {
    let targets: Vec<i64> = {
        let mut seen = HashSet::new();
        endpoints
            .iter()
            .copied()
            .filter(|e| seen.insert(*e))
            .collect()
    };
    let mut best_per_pair: HashMap<(i64, i64), Vec<Path>> = HashMap::new();
    let mut expansions = 0usize;

    for (i, &source) in targets.iter().enumerate() {
        let wanted: HashSet<i64> = targets[i + 1..].iter().copied().collect();
        if wanted.is_empty() {
            break;
        }
        let mut nodes = vec![source];
        let mut edges: Vec<(i64, bool)> = Vec::new();
        let mut resources: Vec<f64> = Vec::new();
        let mut on_path: HashSet<i64> = HashSet::from([source]);
        dfs(
            adj,
            params,
            &wanted,
            1.0,
            &mut nodes,
            &mut edges,
            &mut resources,
            &mut on_path,
            &mut expansions,
            &mut |path: Path| {
                let a = path.nodes[0];
                let b = *path.nodes.last().unwrap();
                best_per_pair.entry((a, b)).or_default().push(path);
            },
        );
    }

    let mut all: Vec<Path> = Vec::new();
    for (_, mut paths) in best_per_pair {
        paths.sort_by(|a, b| {
            b.reliability
                .partial_cmp(&a.reliability)
                .unwrap()
                .then(a.nodes.len().cmp(&b.nodes.len()))
        });
        // Keep a few alternatives per pair so one pair cannot crowd out the rest.
        paths.truncate(3);
        all.extend(paths);
    }
    all.sort_by(|a, b| {
        b.reliability
            .partial_cmp(&a.reliability)
            .unwrap()
            .then(a.nodes.len().cmp(&b.nodes.len()))
            .then(a.nodes.cmp(&b.nodes))
    });
    all.truncate(params.top_k);
    all
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    adj: &PathAdjacency,
    params: &PathParams,
    wanted: &HashSet<i64>,
    resource: f64,
    nodes: &mut Vec<i64>,
    edges: &mut Vec<(i64, bool)>,
    resources: &mut Vec<f64>,
    on_path: &mut HashSet<i64>,
    expansions: &mut usize,
    emit: &mut dyn FnMut(Path),
) {
    if edges.len() >= params.max_hops || *expansions >= params.max_expansions {
        return;
    }
    let u = *nodes.last().unwrap();
    let Some(neighbors) = adj.get(&u) else { return };
    if neighbors.is_empty() {
        return;
    }
    let distinct: HashSet<i64> = neighbors.iter().map(|e| e.to).collect();
    let next_resource = params.decay * resource / distinct.len() as f64;
    if next_resource < params.threshold {
        return;
    }
    for e in neighbors {
        if on_path.contains(&e.to) {
            continue;
        }
        *expansions += 1;
        nodes.push(e.to);
        edges.push((e.relation_id, e.forward));
        resources.push(next_resource);
        if wanted.contains(&e.to) {
            let reliability = resources.iter().sum::<f64>() / resources.len() as f64;
            emit(Path {
                nodes: nodes.clone(),
                edges: edges.clone(),
                reliability,
            });
        } else {
            on_path.insert(e.to);
            dfs(
                adj,
                params,
                wanted,
                next_resource,
                nodes,
                edges,
                resources,
                on_path,
                expansions,
                emit,
            );
            on_path.remove(&e.to);
        }
        nodes.pop();
        edges.pop();
        resources.pop();
    }
}

/// Nodes worth loading for path search: breadth-first from the endpoints,
/// following only branches whose resource can stay above the threshold.
/// Returns, per BFS layer, the nodes whose adjacency must be fetched next.
pub fn next_frontier(
    adj: &PathAdjacency,
    frontier: &HashMap<i64, f64>,
    params: &PathParams,
) -> HashMap<i64, f64> {
    let mut next: HashMap<i64, f64> = HashMap::new();
    for (&u, &res) in frontier {
        let Some(neighbors) = adj.get(&u) else {
            continue;
        };
        let distinct: HashSet<i64> = neighbors.iter().map(|e| e.to).collect();
        if distinct.is_empty() {
            continue;
        }
        let r = params.decay * res / distinct.len() as f64;
        if r < params.threshold {
            continue;
        }
        for v in distinct {
            if !adj.contains_key(&v) {
                let slot = next.entry(v).or_insert(0.0);
                if r > *slot {
                    *slot = r;
                }
            }
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(edges: &[(i64, i64, i64)]) -> PathAdjacency {
        let mut adj: PathAdjacency = HashMap::new();
        for &(rid, a, b) in edges {
            adj.entry(a).or_default().push(Edge {
                to: b,
                relation_id: rid,
                forward: true,
            });
            adj.entry(b).or_default().push(Edge {
                to: a,
                relation_id: rid,
                forward: false,
            });
        }
        adj
    }

    #[test]
    fn finds_multi_hop_path() {
        // 1 -> 2 -> 3, plus a hub 4 connected to many leaves and to 3.
        let mut edges = vec![(10, 1, 2), (11, 2, 3), (12, 1, 4), (13, 4, 3)];
        for leaf in 100..120 {
            edges.push((leaf, 4, leaf));
        }
        let adj = build(&edges);
        let paths = find_paths(&adj, &[1, 3], &PathParams::default());
        assert!(!paths.is_empty());
        // The path through the hub is pruned or ranked lower.
        assert_eq!(paths[0].nodes, vec![1, 2, 3]);
        assert_eq!(paths[0].edges, vec![(10, true), (11, true)]);
        assert!(
            paths.iter().all(|p| p.nodes != vec![1, 4, 3])
                || paths[0].reliability > paths[1].reliability
        );
    }

    #[test]
    fn respects_hop_limit_and_direction() {
        let adj = build(&[(1, 1, 2), (2, 3, 2), (3, 3, 4)]);
        let p = find_paths(
            &adj,
            &[1, 4],
            &PathParams {
                max_hops: 2,
                ..Default::default()
            },
        );
        assert!(p.is_empty());
        let p = find_paths(
            &adj,
            &[1, 4],
            &PathParams {
                max_hops: 3,
                ..Default::default()
            },
        );
        assert_eq!(p[0].nodes, vec![1, 2, 3, 4]);
        assert_eq!(p[0].edges, vec![(1, true), (2, false), (3, true)]);
    }

    #[test]
    fn frontier_prunes_hubs() {
        let mut edges = vec![(1, 1, 2)];
        for leaf in 100..200 {
            edges.push((leaf, 2, leaf));
        }
        let mut adj: PathAdjacency = HashMap::new();
        let full = build(&edges);
        adj.insert(1, full[&1].clone());
        let f1 = next_frontier(&adj, &HashMap::from([(1, 1.0)]), &PathParams::default());
        assert_eq!(f1.keys().copied().collect::<Vec<_>>(), vec![2]);
        adj.insert(2, full[&2].clone());
        let f2 = next_frontier(&adj, &f1, &PathParams::default());
        // 0.8 * 0.8 / 101 < 0.05: hub neighbours are not expanded
        assert!(f2.is_empty());
    }
}
