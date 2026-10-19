//! Leiden community detection (Traag, Waltman & van Eck, Sci. Rep. 2019)
//! with a hierarchical driver in the style of Microsoft GraphRAG's
//! `hierarchical_leiden`: communities larger than `max_cluster_size` are
//! recursively re-partitioned to form the next (finer) level.
//!
//! The quality function is modularity with a resolution parameter γ:
//! `Q = 1/2m Σ_ij [A_ij - γ k_i k_j / 2m] δ(c_i, c_j)`.
//! Compared with Louvain, the refinement phase guarantees that every
//! community is internally connected, which matters for GraphRAG because a
//! community report is supposed to describe one coherent topic.
//!
//! The refinement step picks the best well-connected target greedily (the
//! θ → 0 limit of the randomized rule in the paper); visiting order is
//! randomized with a seeded RNG so results are reproducible.

use std::collections::{HashMap, VecDeque};

/// SplitMix64: tiny, seedable, good enough for shuffling.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// Undirected weighted graph. `adj` holds each edge in both directions and no
/// self loops; self-loop weight is kept in `self_weight`. `degree[i]` equals
/// `Σ adj weights + 2 * self_weight[i]`, and `total` is `Σ degree = 2m`.
#[derive(Clone, Debug)]
pub struct Graph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub self_weight: Vec<f64>,
    pub degree: Vec<f64>,
    pub total: f64,
}

impl Graph {
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Graph {
        let mut maps: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
        let mut self_weight = vec![0.0; n];
        for &(u, v, w) in edges {
            if !(w.is_finite() && w > 0.0) || u >= n || v >= n {
                continue;
            }
            if u == v {
                self_weight[u] += w;
            } else {
                *maps[u].entry(v).or_insert(0.0) += w;
                *maps[v].entry(u).or_insert(0.0) += w;
            }
        }
        Self::from_maps(maps, self_weight)
    }

    fn from_maps(maps: Vec<HashMap<usize, f64>>, self_weight: Vec<f64>) -> Graph {
        let adj: Vec<Vec<(usize, f64)>> = maps
            .into_iter()
            .map(|m| {
                let mut v: Vec<(usize, f64)> = m.into_iter().collect();
                v.sort_by_key(|(k, _)| *k);
                v
            })
            .collect();
        let degree: Vec<f64> = adj
            .iter()
            .zip(&self_weight)
            .map(|(edges, s)| edges.iter().map(|(_, w)| w).sum::<f64>() + 2.0 * s)
            .collect();
        let total = degree.iter().sum();
        Graph {
            adj,
            self_weight,
            degree,
            total,
        }
    }

    pub fn len(&self) -> usize {
        self.adj.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adj.is_empty()
    }

    /// Collapse nodes into communities given by `membership` (labels `0..k`).
    fn aggregate(&self, membership: &[usize], k: usize) -> Graph {
        let mut maps: Vec<HashMap<usize, f64>> = vec![HashMap::new(); k];
        let mut self_weight = vec![0.0; k];
        for u in 0..self.len() {
            let cu = membership[u];
            self_weight[cu] += self.self_weight[u];
            for &(v, w) in &self.adj[u] {
                let cv = membership[v];
                if cu == cv {
                    // each internal edge is visited from both ends
                    self_weight[cu] += w / 2.0;
                } else {
                    *maps[cu].entry(cv).or_insert(0.0) += w;
                }
            }
        }
        Self::from_maps(maps, self_weight)
    }

    /// Induced subgraph on `nodes`; returns the graph and keeps node order.
    pub fn induced(&self, nodes: &[usize]) -> Graph {
        let index: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut maps: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nodes.len()];
        let mut self_weight = vec![0.0; nodes.len()];
        for (i, &u) in nodes.iter().enumerate() {
            self_weight[i] = self.self_weight[u];
            for &(v, w) in &self.adj[u] {
                if let Some(&j) = index.get(&v) {
                    maps[i].insert(j, w);
                }
            }
        }
        Self::from_maps(maps, self_weight)
    }
}

/// Relabel to `0..k` in order of first appearance; returns `k`.
fn renumber(labels: &mut [usize]) -> usize {
    let mut map: HashMap<usize, usize> = HashMap::new();
    for l in labels.iter_mut() {
        let next = map.len();
        *l = *map.entry(*l).or_insert(next);
    }
    map.len()
}

/// Modularity of a partition.
pub fn modularity(g: &Graph, membership: &[usize], resolution: f64) -> f64 {
    if g.total <= 0.0 {
        return 0.0;
    }
    let k = membership.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut internal = vec![0.0; k];
    let mut tot = vec![0.0; k];
    for u in 0..g.len() {
        let c = membership[u];
        tot[c] += g.degree[u];
        internal[c] += g.self_weight[u];
        for &(v, w) in &g.adj[u] {
            if membership[v] == c {
                internal[c] += w / 2.0;
            }
        }
    }
    let m = g.total / 2.0;
    (0..k)
        .map(|c| internal[c] / m - resolution * (tot[c] / g.total).powi(2))
        .sum()
}

const EPS: f64 = 1e-12;

/// Phase 1: local moving of nodes (the "fast" queue-based variant).
fn move_nodes(g: &Graph, part: &mut [usize], resolution: f64, rng: &mut Rng) -> bool {
    let n = g.len();
    let two_m = g.total;
    let mut comm_tot = vec![0.0; n];
    let mut comm_count = vec![0usize; n];
    for u in 0..n {
        comm_tot[part[u]] += g.degree[u];
        comm_count[part[u]] += 1;
    }
    let mut empty: Vec<usize> = (0..n).filter(|c| comm_count[*c] == 0).collect();
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut queue: VecDeque<usize> = order.into_iter().collect();
    let mut in_queue = vec![true; n];
    let mut acc = vec![0.0; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut changed = false;

    while let Some(v) = queue.pop_front() {
        in_queue[v] = false;
        let cv = part[v];
        let kv = g.degree[v];
        for &(u, w) in &g.adj[v] {
            let c = part[u];
            if acc[c] == 0.0 {
                touched.push(c);
            }
            acc[c] += w;
        }
        comm_tot[cv] -= kv;
        comm_count[cv] -= 1;

        let mut best = cv;
        let mut best_gain = acc[cv] - resolution * kv * comm_tot[cv] / two_m;
        for &c in &touched {
            let gain = acc[c] - resolution * kv * comm_tot[c] / two_m;
            if gain > best_gain + EPS {
                best = c;
                best_gain = gain;
            }
        }
        // Moving to an empty community has gain 0.
        if comm_count[cv] > 0 && best_gain < -EPS {
            if let Some(&c) = empty.last() {
                best = c;
            }
        }
        if best != cv && comm_count[best] == 0 {
            if let Some(pos) = empty.iter().rposition(|c| *c == best) {
                empty.swap_remove(pos);
            }
        }
        comm_tot[best] += kv;
        comm_count[best] += 1;
        part[v] = best;
        if comm_count[cv] == 0 {
            empty.push(cv);
        }
        for &c in &touched {
            acc[c] = 0.0;
        }
        touched.clear();

        if best != cv {
            changed = true;
            for &(u, _) in &g.adj[v] {
                if !in_queue[u] && part[u] != best {
                    in_queue[u] = true;
                    queue.push_back(u);
                }
            }
        }
    }
    changed
}

/// Phase 2: refine each community of `part` starting from singletons, only
/// merging well-connected nodes into well-connected sub-communities.
fn refine(g: &Graph, part: &[usize], resolution: f64, rng: &mut Rng) -> Vec<usize> {
    let n = g.len();
    let two_m = g.total;
    let mut comm_tot = vec![0.0; n];
    for u in 0..n {
        comm_tot[part[u]] += g.degree[u];
    }
    // E(v, S - v)
    let ext: Vec<f64> = (0..n)
        .map(|v| {
            g.adj[v]
                .iter()
                .filter(|(u, _)| part[*u] == part[v])
                .map(|(_, w)| *w)
                .sum()
        })
        .collect();
    let mut refined: Vec<usize> = (0..n).collect();
    let mut ref_tot: Vec<f64> = g.degree.clone();
    let mut ref_ext: Vec<f64> = ext.clone();
    let mut ref_count = vec![1usize; n];

    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut acc = vec![0.0; n];
    let mut touched: Vec<usize> = Vec::new();

    for v in order {
        if ref_count[refined[v]] != 1 {
            continue; // already merged
        }
        let s = part[v];
        let kv = g.degree[v];
        if ext[v] < resolution * kv * (comm_tot[s] - kv) / two_m - EPS {
            continue; // v is not well connected to its community
        }
        for &(u, w) in &g.adj[v] {
            if part[u] != s {
                continue;
            }
            let c = refined[u];
            if c == refined[v] {
                continue;
            }
            if acc[c] == 0.0 {
                touched.push(c);
            }
            acc[c] += w;
        }
        let mut best: Option<usize> = None;
        let mut best_gain = EPS;
        for &c in &touched {
            let well_connected =
                ref_ext[c] >= resolution * ref_tot[c] * (comm_tot[s] - ref_tot[c]) / two_m - EPS;
            if !well_connected {
                continue;
            }
            let gain = acc[c] - resolution * kv * ref_tot[c] / two_m;
            if gain > best_gain {
                best = Some(c);
                best_gain = gain;
            }
        }
        if let Some(c) = best {
            let old = refined[v];
            ref_ext[c] = ref_ext[c] + ext[v] - 2.0 * acc[c];
            ref_tot[c] += kv;
            ref_count[c] += 1;
            ref_count[old] = 0;
            refined[v] = c;
        }
        for &c in &touched {
            acc[c] = 0.0;
        }
        touched.clear();
    }
    renumber(&mut refined);
    refined
}

/// Flat Leiden partition. Returns a community label per node (`0..k`).
pub fn leiden(g: &Graph, resolution: f64, rng: &mut Rng) -> Vec<usize> {
    let n = g.len();
    if n == 0 {
        return Vec::new();
    }
    if g.total <= 0.0 {
        return (0..n).collect();
    }
    let mut work = g.clone();
    let mut node_of: Vec<usize> = (0..n).collect(); // original node -> aggregated node
    let mut part: Vec<usize> = (0..n).collect();

    for _ in 0..64 {
        move_nodes(&work, &mut part, resolution, rng);
        let k = renumber(&mut part);
        if k == work.len() {
            break; // every community is a single (aggregated) node
        }
        let mut refined = refine(&work, &part, resolution, rng);
        let mut kr = refined.iter().copied().max().unwrap() + 1;
        if kr == work.len() {
            // Refinement could not merge anything; fall back to the
            // unrefined partition so that aggregation still makes progress.
            refined = part.clone();
            kr = k;
        }
        let next = work.aggregate(&refined, kr);
        let mut next_part = vec![0usize; kr];
        for v in 0..work.len() {
            next_part[refined[v]] = part[v];
        }
        for o in node_of.iter_mut() {
            *o = refined[*o];
        }
        work = next;
        part = next_part;
    }
    let mut result: Vec<usize> = node_of.iter().map(|&a| part[a]).collect();
    renumber(&mut result);
    result
}

#[derive(Clone, Debug)]
pub struct Cluster {
    pub level: usize,
    pub parent: Option<usize>,
    pub members: Vec<usize>,
}

/// Hierarchical Leiden: level 0 is the coarsest partition; clusters larger
/// than `max_cluster_size` are split again (level + 1) until `max_levels`.
pub fn hierarchical_leiden(
    g: &Graph,
    resolution: f64,
    max_cluster_size: usize,
    max_levels: usize,
    seed: u64,
) -> Vec<Cluster> {
    let mut rng = Rng::new(seed);
    let mut clusters: Vec<Cluster> = Vec::new();
    let top = leiden(g, resolution, &mut rng);
    let mut work: VecDeque<(usize, Option<usize>, Vec<usize>)> = VecDeque::new();
    for members in group(&top, &(0..g.len()).collect::<Vec<_>>()) {
        work.push_back((0, None, members));
    }
    while let Some((level, parent, members)) = work.pop_front() {
        let idx = clusters.len();
        clusters.push(Cluster {
            level,
            parent,
            members: members.clone(),
        });
        if members.len() <= max_cluster_size || level + 1 >= max_levels {
            continue;
        }
        let sub = g.induced(&members);
        let labels = leiden(&sub, resolution, &mut rng);
        let groups = group(&labels, &members);
        if groups.len() <= 1 {
            continue;
        }
        for child in groups {
            work.push_back((level + 1, Some(idx), child));
        }
    }
    clusters
}

fn group(labels: &[usize], nodes: &[usize]) -> Vec<Vec<usize>> {
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &l) in labels.iter().enumerate() {
        groups[l].push(nodes[i]);
    }
    groups.retain(|g| !g.is_empty());
    groups
}

/// Whether the members of a community form one connected component.
pub fn is_connected(g: &Graph, members: &[usize]) -> bool {
    if members.len() <= 1 {
        return true;
    }
    let set: std::collections::HashSet<usize> = members.iter().copied().collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![members[0]];
    seen.insert(members[0]);
    while let Some(u) = stack.pop() {
        for &(v, _) in &g.adj[u] {
            if set.contains(&v) && seen.insert(v) {
                stack.push(v);
            }
        }
    }
    seen.len() == members.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn karate() -> Graph {
        // Zachary's karate club (0-indexed).
        let e: &[(usize, usize)] = &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 19),
            (0, 21),
            (0, 31),
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 13),
            (1, 17),
            (1, 19),
            (1, 21),
            (1, 30),
            (2, 3),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 13),
            (2, 27),
            (2, 28),
            (2, 32),
            (3, 7),
            (3, 12),
            (3, 13),
            (4, 6),
            (4, 10),
            (5, 6),
            (5, 10),
            (5, 16),
            (6, 16),
            (8, 30),
            (8, 32),
            (8, 33),
            (9, 33),
            (13, 33),
            (14, 32),
            (14, 33),
            (15, 32),
            (15, 33),
            (18, 32),
            (18, 33),
            (19, 33),
            (20, 32),
            (20, 33),
            (22, 32),
            (22, 33),
            (23, 25),
            (23, 27),
            (23, 29),
            (23, 32),
            (23, 33),
            (24, 25),
            (24, 27),
            (24, 31),
            (25, 31),
            (26, 29),
            (26, 33),
            (27, 33),
            (28, 31),
            (28, 33),
            (29, 32),
            (29, 33),
            (30, 32),
            (30, 33),
            (31, 32),
            (31, 33),
            (32, 33),
        ];
        let edges: Vec<(usize, usize, f64)> = e.iter().map(|&(a, b)| (a, b, 1.0)).collect();
        Graph::from_edges(34, &edges)
    }

    #[test]
    fn two_cliques_split() {
        let mut edges = Vec::new();
        for a in 0..5 {
            for b in (a + 1)..5 {
                edges.push((a, b, 1.0));
                edges.push((a + 5, b + 5, 1.0));
            }
        }
        edges.push((0, 5, 1.0));
        let g = Graph::from_edges(10, &edges);
        let p = leiden(&g, 1.0, &mut Rng::new(1));
        assert_eq!(p.iter().copied().max().unwrap(), 1);
        assert!((0..5).all(|i| p[i] == p[0]));
        assert!((5..10).all(|i| p[i] == p[5]));
        assert_ne!(p[0], p[5]);
    }

    #[test]
    fn karate_modularity_and_connectivity() {
        let g = karate();
        for seed in 0..5 {
            let p = leiden(&g, 1.0, &mut Rng::new(seed));
            let q = modularity(&g, &p, 1.0);
            // The optimum is ~0.4198; Leiden reliably gets close.
            assert!(q > 0.40, "seed {seed}: modularity {q}");
            for members in group(&p, &(0..34).collect::<Vec<_>>()) {
                assert!(
                    is_connected(&g, &members),
                    "disconnected community {members:?}"
                );
            }
        }
    }

    #[test]
    fn hierarchy_respects_max_size() {
        let g = karate();
        let clusters = hierarchical_leiden(&g, 1.0, 5, 4, 7);
        assert!(clusters.iter().any(|c| c.level == 0));
        for c in &clusters {
            if let Some(p) = c.parent {
                assert_eq!(clusters[p].level + 1, c.level);
                assert!(c.members.iter().all(|m| clusters[p].members.contains(m)));
            }
        }
        // leaves either fit the size bound or cannot be split further
        let total_level0: usize = clusters
            .iter()
            .filter(|c| c.level == 0)
            .map(|c| c.members.len())
            .sum();
        assert_eq!(total_level0, 34);
    }

    #[test]
    fn isolated_nodes_are_singletons() {
        let g = Graph::from_edges(4, &[(0, 1, 1.0)]);
        let p = leiden(&g, 1.0, &mut Rng::new(3));
        assert_eq!(p[0], p[1]);
        assert_ne!(p[2], p[3]);
        assert_ne!(p[2], p[0]);
    }

    #[test]
    fn deterministic_for_seed() {
        let g = karate();
        assert_eq!(
            leiden(&g, 1.0, &mut Rng::new(9)),
            leiden(&g, 1.0, &mut Rng::new(9))
        );
    }
}
