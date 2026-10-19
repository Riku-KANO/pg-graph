//! Text utilities exposed to SQL.

use pgrx::prelude::*;

use crate::text;
use crate::vector;

/// Canonical entity key: NFKC, lower case, separators and legal-form
/// designators (株式会社, (株), Inc., Ltd., ...) removed.
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn canonical_key(name: &str) -> String {
    text::canonical_key(name)
}

/// Name similarity in [0, 1] (Jaro–Winkler + bigram Dice over canonical keys).
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn name_similarity(a: &str, b: &str) -> f64 {
    text::name_similarity(a, b)
}

/// Tokens for the lexical index: CJK character bigrams + lower-cased words.
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn lexical_tokens(content: &str) -> Vec<String> {
    text::lexical_tokens(content)
}

/// Approximate LLM token count (CJK ≈ 1 token / char, others ≈ 4 chars / token).
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn estimate_tokens(content: &str) -> i32 {
    text::estimate_tokens(content).min(i32::MAX as usize) as i32
}

/// Bigrams of a canonical key and of its aliases (maintained by trigger).
#[pg_extern(immutable, parallel_safe, schema = "graphrag")]
fn entity_grams(key: Option<&str>, aliases: Option<Vec<Option<String>>>) -> Vec<String> {
    let aliases: Vec<String> = aliases.unwrap_or_default().into_iter().flatten().collect();
    text::entity_grams(key.unwrap_or(""), &aliases)
}

/// LLM-free keyphrase extraction for Japanese and English text.
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn extract_keyphrases(
    content: &str,
    max_phrases: default!(i32, 20),
) -> TableIterator<'static, (name!(phrase, String), name!(score, f64))> {
    TableIterator::new(text::extract_keyphrases(
        content,
        max_phrases.max(0) as usize,
    ))
}

/// Sentence-aware chunking with token budget and overlap.
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn chunk_text(
    content: &str,
    max_tokens: default!(i32, 400),
    overlap_tokens: default!(i32, 40),
) -> TableIterator<'static, (name!(ord, i32), name!(content, String), name!(tokens, i32))> {
    let chunks = text::chunk_text(
        content,
        max_tokens.max(1) as usize,
        overlap_tokens.max(0) as usize,
    );
    TableIterator::new(chunks.into_iter().enumerate().map(|(i, c)| {
        let t = text::estimate_tokens(&c) as i32;
        (i as i32, c, t)
    }))
}

/// Cosine similarity of two `real[]` vectors (NULL when dimensions differ).
#[pg_extern(immutable, parallel_safe, strict, schema = "graphrag")]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> Option<f64> {
    vector::cosine(&a, &b)
}
