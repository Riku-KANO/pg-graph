//! Dense-vector helpers. Embeddings are stored as `real[]` so that the
//! extension has no hard dependency on pgvector; for large corpora an
//! approximate index (pgvector HNSW on an expression such as
//! `(embedding::vector(1024))`) can produce seeds that are passed in through
//! the `seed_entities` / `options` parameters instead.

pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

pub fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// Cosine similarity; `None` when dimensions differ or a vector is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot(a, b) / (na * nb))
}

/// Keep the `k` best `(id, score)` pairs, highest score first.
pub fn top_k<T: Copy>(mut scored: Vec<(T, f64)>, k: usize) -> Vec<(T, f64)> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-9);
        assert!(cosine(&[1.0], &[1.0, 2.0]).is_none());
        assert!(cosine(&[0.0, 0.0], &[1.0, 2.0]).is_none());
        let t = top_k(vec![(1, 0.1), (2, 0.9), (3, 0.5)], 2);
        assert_eq!(t, vec![(2, 0.9), (3, 0.5)]);
    }
}
