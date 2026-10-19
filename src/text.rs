//! Text processing that is aware of Japanese (and other CJK) scripts.
//!
//! Everything here is pure Rust so it can be unit-tested without a running
//! PostgreSQL. The SQL-facing wrappers live in `api::text`.
//!
//! Design notes
//! * Entity names coming out of LLM extraction are noisy: full-width vs.
//!   half-width forms (`ＡＢＣ` / `ABC`, `ｶﾀｶﾅ` / `カタカナ`), middle dots
//!   (`ジョン・スミス` / `ジョンスミス`) and corporate designators
//!   (`トヨタ自動車株式会社` / `トヨタ自動車(株)`). [`canonical_key`] folds
//!   these into one key so that the knowledge graph does not fragment.
//! * PostgreSQL's built-in full text search does not segment Japanese.
//!   [`lexical_tokens`] emits character bigrams for CJK runs (the same idea as
//!   pg_bigm) and lower-cased words for alphabetic scripts, which can be
//!   stored as a `tsvector` and searched through a GIN index.

use std::collections::{HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CharClass {
    Han,
    Katakana,
    Hiragana,
    /// Letters and digits of space-delimited scripts (Latin, Cyrillic, Hangul, ...).
    Word,
    Space,
    Other,
}

pub fn classify(c: char) -> CharClass {
    let u = c as u32;
    match u {
        0x3005
        | 0x3006
        | 0x3007
        | 0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0xF900..=0xFAFF
        | 0x20000..=0x2FFFF => CharClass::Han,
        0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => CharClass::Katakana,
        0x3040..=0x309F => CharClass::Hiragana,
        _ if c.is_whitespace() => CharClass::Space,
        _ if c.is_alphanumeric() => CharClass::Word,
        _ => CharClass::Other,
    }
}

fn is_cjk(class: CharClass) -> bool {
    matches!(
        class,
        CharClass::Han | CharClass::Katakana | CharClass::Hiragana
    )
}

/// NFKC + lower case. Full-width ASCII becomes ASCII, half-width katakana
/// becomes full-width katakana, compatibility ideographs are unified.
pub fn normalize(s: &str) -> String {
    s.nfkc().collect::<String>().to_lowercase()
}

fn is_key_separator(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '・' | '･'
                | '·'
                | '-'
                | '‐'
                | '‑'
                | '‒'
                | '–'
                | '—'
                | '―'
                | '_'
                | '.'
                | ','
                | '、'
                | '。'
                | '\''
                | '"'
                | '`'
                | '’'
                | '‘'
                | '“'
                | '”'
                | '「'
                | '」'
                | '『'
                | '』'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '【'
                | '】'
                | '〈'
                | '〉'
                | '《'
                | '》'
                | '/'
                | '\\'
                | ':'
                | ';'
                | '!'
                | '?'
                | '~'
                | '〜'
        )
}

const JA_LEGAL_FORMS: &[&str] = &[
    "株式会社",
    "有限会社",
    "合同会社",
    "合資会社",
    "合名会社",
    "一般社団法人",
    "一般財団法人",
    "公益社団法人",
    "公益財団法人",
    "(株)",
    "(有)",
    "(同)",
];

const EN_LEGAL_FORMS: &[&str] = &[
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "ltd",
    "limited",
    "llc",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "kk",
];

/// Remove legal-form designators such as `株式会社` or `Inc.` so that
/// `トヨタ自動車株式会社`, `トヨタ自動車(株)` and `トヨタ自動車` collapse to the
/// same key. Operates on already-normalized (NFKC, lower-cased) text.
fn strip_legal_forms(normalized: &str) -> String {
    let mut s = normalized.trim().to_string();
    for form in JA_LEGAL_FORMS {
        if let Some(rest) = s.strip_prefix(form) {
            if !rest.trim().is_empty() {
                s = rest.trim().to_string();
            }
        }
        if let Some(rest) = s.strip_suffix(form) {
            if !rest.trim().is_empty() {
                s = rest.trim().to_string();
            }
        }
    }
    // English designators only count as separate trailing words.
    loop {
        let trimmed = s.trim_end_matches(|c: char| c == '.' || c == ',' || c.is_whitespace());
        let Some(pos) = trimmed.rfind(|c: char| c.is_whitespace() || c == ',') else {
            break;
        };
        let last = &trimmed[pos + 1..];
        if EN_LEGAL_FORMS.contains(&last) && !trimmed[..pos].trim().is_empty() {
            s = trimmed[..pos].to_string();
        } else {
            break;
        }
    }
    s
}

/// Canonical key used to de-duplicate entities.
pub fn canonical_key(name: &str) -> String {
    let normalized = normalize(name);
    let stripped = strip_legal_forms(&normalized);
    let key: String = stripped.chars().filter(|c| !is_key_separator(*c)).collect();
    if key.is_empty() {
        // Names made only of separators: keep the normalized text so that the
        // key is never empty.
        normalized.trim().to_string()
    } else {
        key
    }
}

/// Character n-grams of a string (by Unicode scalar values).
pub fn char_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() || n == 0 {
        return Vec::new();
    }
    if chars.len() <= n {
        return vec![chars.iter().collect()];
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

/// Bigram set of a canonical key plus the keys of all aliases. Stored in
/// `graphrag.entity.key_grams` and indexed with GIN for fuzzy lookup.
pub fn entity_grams(key: &str, aliases: &[String]) -> Vec<String> {
    let mut set: HashSet<String> = char_ngrams(key, 2).into_iter().collect();
    for a in aliases {
        set.extend(char_ngrams(&canonical_key(a), 2));
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

/// Sørensen–Dice coefficient over character bigrams.
pub fn bigram_dice(a: &str, b: &str) -> f64 {
    let ga: HashSet<String> = char_ngrams(a, 2).into_iter().collect();
    let gb: HashSet<String> = char_ngrams(b, 2).into_iter().collect();
    if ga.is_empty() || gb.is_empty() {
        return 0.0;
    }
    let inter = ga.intersection(&gb).count() as f64;
    2.0 * inter / (ga.len() + gb.len()) as f64
}

/// Jaro–Winkler similarity over Unicode scalar values.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let window = (a.len().max(b.len()) / 2).saturating_sub(1);
    let mut a_match = vec![false; a.len()];
    let mut b_match = vec![false; b.len()];
    let mut matches = 0usize;
    for i in 0..a.len() {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(b.len());
        for j in lo..hi {
            if !b_match[j] && a[i] == b[j] {
                a_match[i] = true;
                b_match[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..a.len() {
        if a_match[i] {
            while !b_match[k] {
                k += 1;
            }
            if a[i] != b[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }
    let m = matches as f64;
    let jaro =
        (m / a.len() as f64 + m / b.len() as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0;
    let prefix = a
        .iter()
        .zip(b.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count() as f64;
    jaro + prefix * 0.1 * (1.0 - jaro)
}

/// Similarity of two entity names in [0, 1]. Both names are canonicalized
/// first, so `ＩＢＭ` and `IBM Corp.` score 1.0. The score averages
/// Jaro–Winkler (good for typos and prefixes) and bigram Dice (robust for CJK,
/// where Jaro–Winkler over-rewards short shared prefixes such as 東京/東北).
pub fn name_similarity(a: &str, b: &str) -> f64 {
    let ka = canonical_key(a);
    let kb = canonical_key(b);
    if ka == kb {
        return 1.0;
    }
    0.5 * jaro_winkler(&ka, &kb) + 0.5 * bigram_dice(&ka, &kb)
}

const EN_STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "as", "from",
    "but", "not", "no", "do", "does", "did", "has", "have", "had", "which", "who", "whom", "what",
    "when", "where", "why", "how", "than", "then", "there", "their", "they", "them", "he", "she",
    "his", "her", "we", "our", "you", "your", "i", "me", "my", "so", "if", "into", "about", "can",
    "could", "would", "should", "will", "may", "might", "also", "such",
];

/// Tokens for the lexical (BM25) index: CJK runs become overlapping character
/// bigrams, other scripts become lower-cased words without stopwords.
pub fn lexical_tokens(text: &str) -> Vec<String> {
    let norm = normalize(text);
    let mut out = Vec::new();
    let mut run: Vec<char> = Vec::new();
    let mut run_is_cjk = false;

    let flush = |run: &mut Vec<char>, run_is_cjk: bool, out: &mut Vec<String>| {
        if run.is_empty() {
            return;
        }
        if run_is_cjk {
            if run.len() == 1 {
                out.push(run.iter().collect());
            } else {
                for w in run.windows(2) {
                    out.push(w.iter().collect());
                }
            }
        } else {
            let word: String = run.iter().collect();
            if !EN_STOPWORDS.contains(&word.as_str()) {
                out.push(word);
            }
        }
        run.clear();
    };

    for c in norm.chars() {
        let class = classify(c);
        match class {
            CharClass::Han | CharClass::Katakana | CharClass::Hiragana => {
                if !run_is_cjk {
                    flush(&mut run, run_is_cjk, &mut out);
                    run_is_cjk = true;
                }
                run.push(c);
            }
            CharClass::Word => {
                if run_is_cjk {
                    flush(&mut run, run_is_cjk, &mut out);
                    run_is_cjk = false;
                }
                run.push(c);
            }
            CharClass::Space | CharClass::Other => {
                flush(&mut run, run_is_cjk, &mut out);
            }
        }
    }
    flush(&mut run, run_is_cjk, &mut out);
    out
}

/// Rough LLM token estimate. CJK characters cost roughly one token each with
/// modern BPE vocabularies; other text averages about four characters per
/// token. The estimate is deliberately conservative so that context budgets
/// are not exceeded.
pub fn estimate_tokens(text: &str) -> usize {
    let mut cjk = 0usize;
    let mut other = 0usize;
    for c in text.chars() {
        let class = classify(c);
        if is_cjk(class) {
            cjk += 1;
        } else if class != CharClass::Space {
            other += 1;
        }
    }
    cjk + other.div_ceil(4)
}

const JA_GENERIC_NOUNS: &[&str] = &[
    "場合", "今回", "以下", "以上", "方法", "一方", "可能", "必要", "問題", "結果", "内容", "対象",
    "状況", "部分", "全体", "程度", "前回", "次回", "現在", "今後", "最近", "同様", "関係", "意味",
    "目的", "大幅", "多数", "一部", "様々", "各種", "全て", "以前", "以後", "今年", "昨年", "来年",
    "理由", "影響", "情報", "提供", "利用", "使用", "実施", "対応", "確認", "予定",
];

const NUMERIC_SUFFIXES: &[char] = &[
    '年', '月', '日', '時', '分', '秒', '人', '個', '件', '回', '円', '万', '億', '兆', '千', '百',
    '第', '号', '歳', '週', '度', '期', '台', '社', '名', '割', '倍',
];

const EN_SENTENCE_STARTERS: &[&str] = &[
    "The", "A", "An", "This", "That", "These", "Those", "It", "In", "On", "At", "For", "By",
    "With", "As", "But", "And", "Or", "If", "When", "While", "After", "Before", "However",
    "Although", "Since", "There", "Here", "We", "They", "He", "She", "I", "You", "Our", "Its",
    "Their", "His", "Her", "What", "Who", "Which", "Where", "Why", "How", "Is", "Are", "Was",
    "Were", "Do", "Does", "Did",
];

#[derive(Debug)]
struct Candidate {
    surface: String,
    count: usize,
    first_pos: usize,
}

/// LLM-free keyphrase extraction (used for "lazy" GraphRAG indexing and for
/// seeding queries without an NER call).
///
/// * Japanese: maximal runs of kanji / katakana (optionally fused with
///   adjacent Latin letters such as `AI倫理`), split at hiragana and
///   punctuation, which approximates noun compounds.
/// * Latin script: sequences of capitalised words, acronyms and tokens with
///   digits (`PostgreSQL`, `GPT-4`, `Microsoft Research`).
///
/// Scores are `count * (1 + ln(len))` and the result is de-duplicated by
/// canonical key, best first.
pub fn extract_keyphrases(text: &str, max: usize) -> Vec<(String, f64)> {
    let norm: String = text.nfkc().collect();
    let chars: Vec<char> = norm.chars().collect();
    let mut found: HashMap<String, Candidate> = HashMap::new();
    let mut push = |surface: String, pos: usize| {
        let key = canonical_key(&surface);
        if key.chars().count() < 2 {
            return;
        }
        let e = found.entry(key).or_insert(Candidate {
            surface: surface.clone(),
            count: 0,
            first_pos: pos,
        });
        e.count += 1;
    };

    // Pass 1: CJK compounds — maximal runs of Han / Katakana fused with directly
    // adjacent ASCII letters or digits, split at hiragana, spaces and punctuation.
    let in_compound = |idx: usize| -> bool {
        let c = chars[idx];
        match classify(c) {
            CharClass::Han | CharClass::Katakana => true,
            CharClass::Word => c.is_ascii_alphanumeric(),
            _ => {
                c == '・'
                    && idx + 1 < chars.len()
                    && classify(chars[idx + 1]) == CharClass::Katakana
            }
        }
    };
    let mut i = 0;
    while i < chars.len() {
        if !in_compound(i) {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && in_compound(i) {
            i += 1;
        }
        let run = &chars[start..i];
        let cjk_count = run
            .iter()
            .filter(|c| matches!(classify(**c), CharClass::Han | CharClass::Katakana))
            .count();
        if cjk_count < 2 || run.len() > 24 {
            continue;
        }
        // Dates, counts and amounts such as 2024年4月 or 300万円 are not topics.
        let numeric_expression = run.iter().any(|c| c.is_ascii_digit())
            && run
                .iter()
                .all(|c| c.is_ascii_digit() || NUMERIC_SUFFIXES.contains(c));
        let surface: String = run.iter().collect();
        if !numeric_expression && !JA_GENERIC_NOUNS.contains(&surface.as_str()) {
            push(surface, start);
        }
    }

    // Pass 2: Latin-script proper nouns / acronyms.
    let words: Vec<(usize, String)> = {
        let mut v = Vec::new();
        let mut cur = String::new();
        let mut cur_start = 0;
        for (idx, &c) in chars.iter().enumerate() {
            let class = classify(c);
            let is_word_char = (class == CharClass::Word && !is_cjk(class))
                || (!cur.is_empty() && matches!(c, '-' | '+' | '#' | '.' | '&'));
            if is_word_char && class != CharClass::Han {
                if cur.is_empty() {
                    cur_start = idx;
                }
                cur.push(c);
            } else {
                if !cur.is_empty() {
                    v.push((cur_start, cur.trim_end_matches(['.', '-']).to_string()));
                    cur.clear();
                }
                // Hard boundary between phrases unless it is a single space.
                if c != ' ' {
                    v.push((idx, String::new()));
                }
            }
        }
        if !cur.is_empty() {
            v.push((cur_start, cur.trim_end_matches(['.', '-']).to_string()));
        }
        v
    };
    let is_proper = |w: &str| -> bool {
        let first = w.chars().next().unwrap_or(' ');
        let has_inner_upper = w.chars().skip(1).any(|c| c.is_uppercase());
        let has_digit = w.chars().any(|c| c.is_ascii_digit());
        let all_digits = w.chars().all(|c| c.is_ascii_digit());
        if all_digits || w.chars().any(|c| is_cjk(classify(c))) {
            return false;
        }
        (first.is_uppercase() && !EN_SENTENCE_STARTERS.contains(&w))
            || has_inner_upper
            || (has_digit && w.len() > 1)
    };
    let mut k = 0;
    while k < words.len() {
        let (pos, ref w) = words[k];
        if w.is_empty() || !is_proper(w) {
            k += 1;
            continue;
        }
        let mut phrase = vec![w.clone()];
        let mut j = k + 1;
        while j < words.len() && phrase.len() < 5 {
            let (_, ref next) = words[j];
            if !next.is_empty() && is_proper(next) {
                phrase.push(next.clone());
                j += 1;
            } else {
                break;
            }
        }
        let surface = phrase.join(" ");
        if surface.chars().count() >= 2 {
            push(surface, pos);
        }
        k = j;
    }

    let mut scored: Vec<(String, f64, usize)> = found
        .into_values()
        .map(|c| {
            let len = c.surface.chars().count().max(1) as f64;
            (c.surface, c.count as f64 * (1.0 + len.ln()), c.first_pos)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.2.cmp(&b.2)));
    scored.truncate(max);
    scored.into_iter().map(|(s, score, _)| (s, score)).collect()
}

fn is_sentence_end(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '!' | '?' | '\n') || c == '.'
}

/// Split text into chunks of at most `max_tokens` (estimated) tokens,
/// preferring sentence boundaries (`。`, `.`, `!`, `?`, newlines) and carrying
/// over roughly `overlap_tokens` of trailing context into the next chunk.
pub fn chunk_text(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    let max_tokens = max_tokens.max(16);
    let overlap_tokens = overlap_tokens.min(max_tokens / 2);
    // 1. sentences
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let boundary = if c == '.' {
            // "3.14" or "e.g." should not split; require whitespace after a period.
            chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true)
        } else {
            is_sentence_end(c)
        };
        if boundary {
            if !cur.trim().is_empty() {
                sentences.push(cur.trim().to_string());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }
    // 2. split over-long sentences by characters
    let mut pieces: Vec<String> = Vec::new();
    for s in sentences {
        if estimate_tokens(&s) <= max_tokens {
            pieces.push(s);
            continue;
        }
        let mut buf = String::new();
        for c in s.chars() {
            buf.push(c);
            if estimate_tokens(&buf) >= max_tokens {
                pieces.push(std::mem::take(&mut buf));
            }
        }
        if !buf.is_empty() {
            pieces.push(buf);
        }
    }
    // 3. pack sentences into chunks with overlap
    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_tokens = 0usize;
    let joiner = |parts: &[String]| -> String {
        let mut out = String::new();
        for p in parts {
            if !out.is_empty() {
                let last = out.chars().last().unwrap();
                let first = p.chars().next().unwrap_or(' ');
                if !(is_cjk(classify(last)) || is_cjk(classify(first)) || last == '。') {
                    out.push(' ');
                }
            }
            out.push_str(p);
        }
        out
    };
    for p in pieces {
        let t = estimate_tokens(&p);
        if current_tokens + t > max_tokens && !current.is_empty() {
            chunks.push(joiner(&current));
            // carry over trailing sentences as overlap
            let mut carried: Vec<String> = Vec::new();
            let mut carried_tokens = 0;
            for s in current.iter().rev() {
                let st = estimate_tokens(s);
                if carried_tokens + st > overlap_tokens {
                    break;
                }
                carried.insert(0, s.clone());
                carried_tokens += st;
            }
            current = carried;
            current_tokens = carried_tokens;
        }
        current.push(p);
        current_tokens += t;
    }
    if !current.is_empty() {
        let last = joiner(&current);
        if chunks.last().map(|c| c != &last).unwrap_or(true) {
            chunks.push(last);
        }
    }
    chunks
}

/// Okapi BM25 over pre-tokenized documents.
pub struct Bm25 {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25 {
    fn default() -> Self {
        Bm25 { k1: 1.2, b: 0.75 }
    }
}

impl Bm25 {
    /// `df` maps query terms to document frequency, `n_docs` is the corpus size.
    pub fn score(
        &self,
        query_terms: &[String],
        doc_tokens: &[String],
        avg_len: f64,
        df: &HashMap<String, usize>,
        n_docs: usize,
    ) -> f64 {
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in doc_tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        let len = doc_tokens.len() as f64;
        let avg = if avg_len > 0.0 { avg_len } else { 1.0 };
        let mut score = 0.0;
        for q in query_terms {
            let Some(&f) = tf.get(q.as_str()) else {
                continue;
            };
            let d = *df.get(q).unwrap_or(&0) as f64;
            let n = n_docs.max(1) as f64;
            let idf = ((n - d + 0.5) / (d + 0.5) + 1.0).ln();
            let f = f as f64;
            score +=
                idf * f * (self.k1 + 1.0) / (f + self.k1 * (1.0 - self.b + self.b * len / avg));
        }
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_key_folds_width_and_legal_forms() {
        assert_eq!(
            canonical_key("トヨタ自動車株式会社"),
            canonical_key("トヨタ自動車(株)")
        );
        assert_eq!(canonical_key("トヨタ自動車㈱"), "トヨタ自動車");
        assert_eq!(
            canonical_key("株式会社サイバーエージェント"),
            "サイバーエージェント"
        );
        assert_eq!(canonical_key("ＩＢＭ"), "ibm");
        assert_eq!(canonical_key("IBM Corp."), "ibm");
        assert_eq!(canonical_key("ｶﾀｶﾅ"), "カタカナ");
        assert_eq!(
            canonical_key("ジョン・スミス"),
            canonical_key("ジョン スミス")
        );
        assert_eq!(canonical_key("New York"), "newyork");
        // Symbols that carry meaning are kept.
        assert_ne!(canonical_key("C++"), canonical_key("C"));
        assert_ne!(canonical_key("C#"), canonical_key("C"));
        // A designator alone is not stripped to nothing.
        assert_eq!(canonical_key("株式会社"), "株式会社");
        assert_eq!(canonical_key("Inc."), "inc");
    }

    #[test]
    fn similarity_is_sane() {
        assert_eq!(name_similarity("ＩＢＭ", "IBM Corp."), 1.0);
        assert!(name_similarity("Microsoft", "Microsft") > 0.85);
        assert!(name_similarity("東京大学", "東京工業大学") < 0.85);
        assert!(name_similarity("東京", "東北") < 0.7);
        assert!(jaro_winkler("martha", "marhta") > 0.95);
    }

    #[test]
    fn lexical_tokens_bigram_cjk() {
        let t = lexical_tokens("東京タワーは港区にある。The Tower of Tokyo");
        assert!(t.contains(&"東京".to_string()));
        assert!(t.contains(&"タワ".to_string()));
        assert!(t.contains(&"港区".to_string()));
        assert!(t.contains(&"tower".to_string()));
        assert!(t.contains(&"tokyo".to_string()));
        assert!(!t.contains(&"the".to_string()));
        assert!(!t.contains(&"of".to_string()));
        // full-width digits normalize
        assert!(lexical_tokens("２０２５年").contains(&"2025".to_string()));
    }

    #[test]
    fn keyphrases_ja_en() {
        let kp = extract_keyphrases(
            "トヨタ自動車は愛知県豊田市に本社を置く。トヨタ自動車の社長は佐藤恒治である。Microsoft Research released GraphRAG in 2024.",
            20,
        );
        let phrases: Vec<&str> = kp.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(phrases[0], "トヨタ自動車", "{phrases:?}");
        assert!(phrases.contains(&"愛知県豊田市"), "{phrases:?}");
        assert!(phrases.contains(&"佐藤恒治"), "{phrases:?}");
        assert!(phrases.contains(&"Microsoft Research"), "{phrases:?}");
        assert!(phrases.contains(&"GraphRAG"), "{phrases:?}");
        assert!(!phrases.contains(&"2024"), "{phrases:?}");
    }

    #[test]
    fn keyphrases_fuse_latin_and_kanji() {
        let kp = extract_keyphrases("AI倫理の議論が進む。PostgreSQL拡張を使う。", 10);
        let phrases: Vec<&str> = kp.iter().map(|(s, _)| s.as_str()).collect();
        assert!(phrases.contains(&"AI倫理"), "{phrases:?}");
        assert!(phrases.contains(&"PostgreSQL拡張"), "{phrases:?}");
    }

    #[test]
    fn token_estimate() {
        assert_eq!(estimate_tokens("東京タワー"), 5);
        assert_eq!(estimate_tokens("hello world"), 3);
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn chunking_respects_budget_and_sentences() {
        let text = "これは最初の文です。これは二番目の文です。これは三番目の文です。This is English. Another one here.";
        let chunks = chunk_text(text, 20, 0);
        assert!(chunks.len() >= 2, "{chunks:?}");
        for c in &chunks {
            assert!(estimate_tokens(c) <= 20, "{c}");
        }
        assert!(chunks[0].starts_with("これは最初の文です。"));
        let joined: String = chunks.concat();
        assert!(joined.contains("三番目"));
        // overlap carries the previous sentence
        let with_overlap = chunk_text(text, 20, 10);
        assert!(with_overlap.len() >= chunks.len());
        // decimals do not split
        assert_eq!(chunk_text("Pi is 3.14 roughly.", 100, 0).len(), 1);
    }

    #[test]
    fn bm25_prefers_matching_doc() {
        let bm = Bm25::default();
        let q = lexical_tokens("東京タワー");
        let d1 = lexical_tokens("東京タワーは電波塔です");
        let d2 = lexical_tokens("大阪城は城です");
        let mut df = HashMap::new();
        for t in &q {
            df.insert(t.clone(), 1);
        }
        let s1 = bm.score(&q, &d1, 8.0, &df, 2);
        let s2 = bm.score(&q, &d2, 8.0, &df, 2);
        assert!(s1 > 0.0 && s2 == 0.0);
    }
}
