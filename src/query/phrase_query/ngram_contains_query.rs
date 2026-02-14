//! N-gram based contains query for fast substring search.
//!
//! Uses a `._ngram` field (trigrams) to quickly identify candidate documents,
//! then verifies matches by reading stored text. Replaces the expensive FST walk
//! (levels 3–4 of the cascade) with O(k lookups + candidates) via trigram index.
//!
//! Cascade:
//!   1. **Exact**: term dict lookup on raw field — O(1)
//!   2. **Ngram**: trigram lookup on ngram field + verification — O(k + candidates)

use std::cmp::min;
use std::sync::Arc;

use super::scoring_utils::{
    edit_distance, generate_trigrams, intersect_sorted_vecs, ngram_threshold, token_match_distance,
    tokenize_raw, HighlightSink,
};
use crate::fieldnorm::FieldNormReader;
use crate::query::bm25::Bm25Weight;
use crate::query::{EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::document::Value;
use crate::schema::{Field, IndexRecordOption, Term};
use crate::{DocId, DocSet, InvertedIndexReader, Score, SegmentReader, TantivyDocument, TERMINATED};

// ─── Candidate Collection ──────────────────────────────────────────────────

/// Collect doc_ids from a single term's posting list.
fn collect_posting_docs(
    inverted_index: &InvertedIndexReader,
    term: &Term,
) -> crate::Result<Vec<DocId>> {
    let term_info = match inverted_index.get_term_info(term)? {
        Some(ti) => ti,
        None => return Ok(Vec::new()),
    };
    let mut docs = Vec::new();
    let mut block_postings =
        inverted_index.read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
    loop {
        let block = block_postings.docs();
        if block.is_empty() {
            break;
        }
        docs.extend_from_slice(block);
        block_postings.advance();
    }
    Ok(docs)
}

/// Get candidate doc_ids for a query token via threshold-based trigram intersection.
fn ngram_candidates_for_token(
    token: &str,
    ngram_field: Field,
    ngram_inverted: &InvertedIndexReader,
    fuzzy_distance: u8,
) -> crate::Result<Vec<DocId>> {
    let trigrams = generate_trigrams(token);
    let threshold = ngram_threshold(trigrams.len(), fuzzy_distance);

    if trigrams.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all doc_ids from all trigram posting lists.
    let mut all_docs: Vec<DocId> = Vec::new();
    for trigram in &trigrams {
        let term = Term::from_field_text(ngram_field, trigram);
        let docs = collect_posting_docs(ngram_inverted, &term)?;
        all_docs.extend(docs);
    }

    // Sort and count: keep doc_ids appearing >= threshold times.
    all_docs.sort_unstable();

    let mut candidates = Vec::new();
    let mut i = 0;
    while i < all_docs.len() {
        let doc = all_docs[i];
        let mut count = 0usize;
        while i < all_docs.len() && all_docs[i] == doc {
            count += 1;
            i += 1;
        }
        if count >= threshold {
            candidates.push(doc);
        }
    }

    Ok(candidates)
}

// ─── Query ─────────────────────────────────────────────────────────────────

/// N-gram based contains query: trigram lookup + stored text verification.
///
/// Uses a trigram field to quickly narrow candidates, then verifies matches
/// by reading stored text. This avoids expensive FST walks for substring matching.
#[derive(Clone, Debug)]
pub struct NgramContainsQuery {
    raw_field: Field,
    ngram_field: Field,
    stored_field: Option<Field>,
    tokens: Vec<String>,
    separators: Vec<String>,
    prefix: String,
    suffix: String,
    fuzzy_distance: u8,
    distance_budget: u32,
    strict_separators: bool,
    highlight_sink: Option<Arc<HighlightSink>>,
}

impl NgramContainsQuery {
    /// Creates a new `NgramContainsQuery`.
    ///
    /// * `raw_field` - Lowercase raw field for exact term lookups.
    /// * `ngram_field` - Trigram field for candidate collection.
    /// * `stored_field` - Field to load stored text from (if different from `raw_field`).
    /// * `tokens` - Query tokens (lowercased).
    /// * `separators` - Separators between consecutive tokens in the query string.
    /// * `prefix` / `suffix` - Characters before/after the token span.
    /// * `fuzzy_distance` - Max Levenshtein distance per token.
    /// * `distance_budget` - Max cumulative edit distance.
    /// * `strict_separators` - Whether separators must match exactly (edit distance) vs relaxed.
    pub fn new(
        raw_field: Field,
        ngram_field: Field,
        stored_field: Option<Field>,
        tokens: Vec<String>,
        separators: Vec<String>,
        prefix: String,
        suffix: String,
        fuzzy_distance: u8,
        distance_budget: u32,
        strict_separators: bool,
    ) -> Self {
        NgramContainsQuery {
            raw_field,
            ngram_field,
            stored_field,
            tokens,
            separators,
            prefix,
            suffix,
            fuzzy_distance,
            distance_budget,
            strict_separators,
            highlight_sink: None,
        }
    }

    /// Attach a highlight sink to capture byte offsets during scoring.
    pub fn with_highlight_sink(mut self, sink: Arc<HighlightSink>) -> Self {
        self.highlight_sink = Some(sink);
        self
    }
}

impl Query for NgramContainsQuery {
    fn weight(&self, enable_scoring: EnableScoring) -> crate::Result<Box<dyn Weight>> {
        let bm25_weight = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => {
                let terms: Vec<Term> = self
                    .tokens
                    .iter()
                    .map(|t| Term::from_field_text(self.raw_field, t))
                    .collect();
                if terms.is_empty() {
                    Bm25Weight::for_one_term(0, 1, 1.0)
                } else {
                    Bm25Weight::for_terms(statistics_provider, &terms)?
                }
            }
            EnableScoring::Disabled { .. } => Bm25Weight::for_one_term(0, 1, 1.0),
        };

        Ok(Box::new(NgramContainsWeight {
            raw_field: self.raw_field,
            ngram_field: self.ngram_field,
            stored_field: self.stored_field,
            tokens: self.tokens.clone(),
            separators: self.separators.clone(),
            prefix: self.prefix.clone(),
            suffix: self.suffix.clone(),
            fuzzy_distance: self.fuzzy_distance,
            distance_budget: self.distance_budget,
            strict_separators: self.strict_separators,
            highlight_sink: self.highlight_sink.clone(),
            bm25_weight,
        }))
    }
}

// ─── Weight ────────────────────────────────────────────────────────────────

struct NgramContainsWeight {
    raw_field: Field,
    ngram_field: Field,
    stored_field: Option<Field>,
    tokens: Vec<String>,
    separators: Vec<String>,
    prefix: String,
    suffix: String,
    fuzzy_distance: u8,
    distance_budget: u32,
    strict_separators: bool,
    highlight_sink: Option<Arc<HighlightSink>>,
    bm25_weight: Bm25Weight,
}

impl Weight for NgramContainsWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let segment_ord = self
            .highlight_sink
            .as_ref()
            .map(|s| s.next_segment())
            .unwrap_or(0);
        let raw_inverted = reader.inverted_index(self.raw_field)?;
        let ngram_inverted = reader.inverted_index(self.ngram_field)?;

        // Collect candidate doc_ids for each query token.
        let mut per_token_candidates: Vec<Vec<DocId>> = Vec::new();

        for token in &self.tokens {
            // Level 1: Exact lookup in raw field.
            let term = Term::from_field_text(self.raw_field, token);
            let exact_docs = collect_posting_docs(&raw_inverted, &term)?;
            if !exact_docs.is_empty() {
                per_token_candidates.push(exact_docs);
                continue;
            }

            // Level 2: Ngram lookup with threshold-based intersection.
            let candidates = ngram_candidates_for_token(
                token,
                self.ngram_field,
                &ngram_inverted,
                self.fuzzy_distance,
            )?;
            per_token_candidates.push(candidates);
        }

        // Intersect across all tokens.
        let final_candidates = intersect_sorted_vecs(per_token_candidates);

        if final_candidates.is_empty() {
            return Ok(Box::new(EmptyScorer));
        }

        // Create scorer that verifies each candidate via stored text.
        let store_reader = reader
            .get_store_reader(50)
            .map_err(crate::TantivyError::from)?;
        let text_field = self.stored_field.unwrap_or(self.raw_field);

        let fieldnorm_reader = if let Some(fnr) = reader
            .fieldnorms_readers()
            .get_field(self.raw_field)?
        {
            fnr
        } else {
            FieldNormReader::constant(reader.max_doc(), 1)
        };

        Ok(Box::new(NgramContainsScorer::new(
            final_candidates,
            store_reader,
            text_field,
            self.tokens.clone(),
            self.separators.clone(),
            self.prefix.clone(),
            self.suffix.clone(),
            self.fuzzy_distance,
            self.distance_budget,
            self.strict_separators,
            self.bm25_weight.boost_by(boost),
            fieldnorm_reader,
            self.highlight_sink.clone(),
            segment_ord,
        )))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(crate::TantivyError::InvalidArgument(format!(
                "Document {doc} does not match"
            )));
        }
        Ok(Explanation::new("NgramContainsScorer", scorer.score()))
    }
}

// ─── Scorer ────────────────────────────────────────────────────────────────

struct NgramContainsScorer {
    candidates: Vec<DocId>,
    cursor: usize,
    store_reader: crate::store::StoreReader,
    text_field: Field,
    tokens: Vec<String>,
    separators: Vec<String>,
    prefix: String,
    suffix: String,
    fuzzy_distance: u8,
    distance_budget: u32,
    strict_separators: bool,
    bm25_weight: Bm25Weight,
    fieldnorm_reader: FieldNormReader,
    last_tf: u32,
    highlight_sink: Option<Arc<HighlightSink>>,
    segment_ord: u32,
}

impl NgramContainsScorer {
    fn new(
        candidates: Vec<DocId>,
        store_reader: crate::store::StoreReader,
        text_field: Field,
        tokens: Vec<String>,
        separators: Vec<String>,
        prefix: String,
        suffix: String,
        fuzzy_distance: u8,
        distance_budget: u32,
        strict_separators: bool,
        bm25_weight: Bm25Weight,
        fieldnorm_reader: FieldNormReader,
        highlight_sink: Option<Arc<HighlightSink>>,
        segment_ord: u32,
    ) -> Self {
        let mut scorer = NgramContainsScorer {
            candidates,
            cursor: 0,
            store_reader,
            text_field,
            tokens,
            separators,
            prefix,
            suffix,
            fuzzy_distance,
            distance_budget,
            strict_separators,
            bm25_weight,
            fieldnorm_reader,
            last_tf: 0,
            highlight_sink,
            segment_ord,
        };
        // Advance to first valid doc.
        if scorer.doc() != TERMINATED && !scorer.verify() {
            scorer.advance();
        }
        scorer
    }

    /// Verify the current candidate doc by reading stored text.
    /// Counts all matches (term frequency) and stores in `last_tf`.
    fn verify(&mut self) -> bool {
        self.last_tf = 0;
        let doc_id = self.doc();
        if doc_id == TERMINATED {
            return false;
        }

        let doc: TantivyDocument = match self.store_reader.get(doc_id) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let stored_text = match doc.get_first(self.text_field).and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return false,
        };

        let doc_tokens = tokenize_raw(stored_text);

        let tf = if self.tokens.len() == 1 {
            self.count_single_token(stored_text, &doc_tokens)
        } else {
            self.count_multi_token(stored_text, &doc_tokens)
        };
        self.last_tf = tf;
        tf > 0
    }

    /// Count all matching positions for a single-token query.
    fn count_single_token(&self, stored_text: &str, doc_tokens: &[(usize, usize)]) -> u32 {
        let query_token = &self.tokens[0];
        let mut count = 0u32;

        for &(start, end) in doc_tokens {
            let doc_token = stored_text[start..end].to_lowercase();

            let distance = match token_match_distance(&doc_token, query_token, self.fuzzy_distance)
            {
                Some(d) => d,
                None => continue,
            };

            let mut total_distance = distance;

            // Validate prefix.
            if !self.prefix.is_empty() {
                if self.strict_separators {
                    let prefix_len = self.prefix.len();
                    let doc_prefix_start = start.saturating_sub(prefix_len);
                    let doc_prefix = &stored_text[doc_prefix_start..start];
                    total_distance += edit_distance(&self.prefix, doc_prefix);
                    if total_distance > self.distance_budget {
                        continue;
                    }
                } else {
                    if start == 0 {
                        continue;
                    }
                    if stored_text.as_bytes()[start - 1].is_ascii_alphanumeric() {
                        continue;
                    }
                }
            }

            // Validate suffix.
            if !self.suffix.is_empty() {
                if self.strict_separators {
                    let suffix_len = self.suffix.len();
                    let doc_suffix_end = min(end + suffix_len, stored_text.len());
                    let doc_suffix = &stored_text[end..doc_suffix_end];
                    total_distance += edit_distance(&self.suffix, doc_suffix);
                    if total_distance > self.distance_budget {
                        continue;
                    }
                } else {
                    if end >= stored_text.len() {
                        continue;
                    }
                    if stored_text.as_bytes()[end].is_ascii_alphanumeric() {
                        continue;
                    }
                }
            }

            count += 1;
            if let Some(ref sink) = self.highlight_sink {
                sink.insert(self.segment_ord, self.doc(), vec![[start, end]]);
            }
        }
        count
    }

    /// Count all matching positions for a multi-token query.
    fn count_multi_token(&self, stored_text: &str, doc_tokens: &[(usize, usize)]) -> u32 {
        let num_query = self.tokens.len();
        if doc_tokens.len() < num_query {
            return 0;
        }

        let mut count = 0u32;
        for start_idx in 0..=(doc_tokens.len() - num_query) {
            if self.check_at_position(stored_text, doc_tokens, start_idx) {
                count += 1;
            }
        }
        count
    }

    fn check_at_position(
        &self,
        stored_text: &str,
        doc_tokens: &[(usize, usize)],
        start_idx: usize,
    ) -> bool {
        let mut total_distance = 0u32;

        // Check each query token matches the corresponding doc token.
        for (q_idx, query_token) in self.tokens.iter().enumerate() {
            let (start, end) = doc_tokens[start_idx + q_idx];
            let doc_token = stored_text[start..end].to_lowercase();

            match token_match_distance(&doc_token, query_token, self.fuzzy_distance) {
                Some(d) => total_distance += d,
                None => return false,
            }

            if total_distance > self.distance_budget {
                return false;
            }
        }

        // Validate separators between consecutive tokens.
        for (sep_idx, query_sep) in self.separators.iter().enumerate() {
            let (_, end_i) = doc_tokens[start_idx + sep_idx];
            let (start_next, _) = doc_tokens[start_idx + sep_idx + 1];
            if end_i > stored_text.len() || start_next > stored_text.len() || end_i > start_next {
                return false;
            }
            let doc_sep = &stored_text[end_i..start_next];
            if self.strict_separators {
                total_distance += edit_distance(query_sep, doc_sep);
                if total_distance > self.distance_budget {
                    return false;
                }
            } else if doc_sep.is_empty() || !doc_sep.bytes().any(|b| !b.is_ascii_alphanumeric()) {
                return false;
            }
        }

        // Validate prefix.
        if !self.prefix.is_empty() {
            let (first_start, _) = doc_tokens[start_idx];
            if self.strict_separators {
                let prefix_len = self.prefix.len();
                let doc_prefix_start = first_start.saturating_sub(prefix_len);
                let doc_prefix = &stored_text[doc_prefix_start..first_start];
                total_distance += edit_distance(&self.prefix, doc_prefix);
                if total_distance > self.distance_budget {
                    return false;
                }
            } else {
                if first_start == 0 {
                    return false;
                }
                let before = &stored_text[..first_start];
                if before
                    .as_bytes()
                    .last()
                    .is_none_or(|b| b.is_ascii_alphanumeric())
                {
                    return false;
                }
            }
        }

        // Validate suffix.
        if !self.suffix.is_empty() {
            let num_query = self.tokens.len();
            let (_, last_end) = doc_tokens[start_idx + num_query - 1];
            if self.strict_separators {
                let suffix_len = self.suffix.len();
                let doc_suffix_end = min(last_end + suffix_len, stored_text.len());
                let doc_suffix = &stored_text[last_end..doc_suffix_end];
                total_distance += edit_distance(&self.suffix, doc_suffix);
                if total_distance > self.distance_budget {
                    return false;
                }
            } else {
                if last_end >= stored_text.len() {
                    return false;
                }
                if stored_text.as_bytes()[last_end].is_ascii_alphanumeric() {
                    return false;
                }
            }
        }

        if let Some(ref sink) = self.highlight_sink {
            let offsets: Vec<[usize; 2]> = (0..self.tokens.len())
                .map(|i| {
                    let (s, e) = doc_tokens[start_idx + i];
                    [s, e]
                })
                .collect();
            sink.insert(self.segment_ord, self.doc(), offsets);
        }
        true
    }
}

impl DocSet for NgramContainsScorer {
    fn advance(&mut self) -> DocId {
        loop {
            self.cursor += 1;
            let doc = self.doc();
            if doc == TERMINATED || self.verify() {
                return doc;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        while self.cursor < self.candidates.len() && self.candidates[self.cursor] < target {
            self.cursor += 1;
        }
        if self.doc() == TERMINATED || self.verify() {
            return self.doc();
        }
        self.advance()
    }

    fn doc(&self) -> DocId {
        if self.cursor < self.candidates.len() {
            self.candidates[self.cursor]
        } else {
            TERMINATED
        }
    }

    fn size_hint(&self) -> u32 {
        self.candidates.len().saturating_sub(self.cursor) as u32
    }
}

impl Scorer for NgramContainsScorer {
    fn score(&mut self) -> Score {
        let doc = self.doc();
        let fieldnorm_id = self.fieldnorm_reader.fieldnorm_id(doc);
        self.bm25_weight.score(fieldnorm_id, self.last_tf)
    }
}
