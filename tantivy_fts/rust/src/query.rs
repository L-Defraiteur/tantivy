//! Query and schema JSON parsing.
//!
//! Query routing with dual-field layout:
//!   - phrase, parse            → stemmed field (recall: "run" matches "running")
//!   - term, fuzzy, regex       → raw field (precision: exact word forms, lowercase only)
//! The user always references the base field name; routing is transparent.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use ld_tantivy::collector::{FilterCollector, TopDocs};
use ld_tantivy::query::{
    AutomatonPhraseQuery, BooleanQuery, FuzzyTermQuery, HighlightSink, NgramContainsQuery, Occur,
    PhraseQuery, Query, QueryParser, RegexQuery, TermQuery,
};
use ld_tantivy::schema::{Field, FieldType, IndexRecordOption, Schema, Term};
use ld_tantivy::{Document, Index, Searcher, TantivyDocument};

// ─── Schema Config ──────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize)]
pub struct SchemaConfig {
    pub fields: Vec<FieldDef>,
    pub tokenizer: Option<String>,
    pub stemmer: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct FieldDef {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub stored: Option<bool>,
    pub indexed: Option<bool>,
    pub fast: Option<bool>,
}

// ─── Query Config ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct QueryConfig {
    #[serde(rename = "type")]
    pub query_type: String,
    pub field: Option<String>,
    pub fields: Option<Vec<String>>,
    pub value: Option<String>,
    pub terms: Option<Vec<String>>,
    pub pattern: Option<String>,
    pub distance: Option<u8>,
    pub strict_separators: Option<bool>,
    pub highlight: Option<bool>,
    // Boolean query sub-clauses
    pub must: Option<Vec<QueryConfig>>,
    pub should: Option<Vec<QueryConfig>>,
    pub must_not: Option<Vec<QueryConfig>>,
}

// ─── Search Result ──────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
pub struct SearchResult {
    pub score: f32,
    pub doc: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlights: Option<std::collections::HashMap<String, Vec<[usize; 2]>>>,
}

// ─── Tokenization Helper ────────────────────────────────────────────────────

/// Tokenize text through the tokenizer configured for a field.
/// Returns the list of tokens (e.g. ["lazi", "dog"] for "lazy dog" with English stemmer).
fn tokenize_for_field(index: &Index, field: Field, schema: &Schema, text: &str) -> Vec<String> {
    let tokenizer_name = match schema.get_field_entry(field).field_type() {
        FieldType::Str(opts) => opts
            .get_indexing_options()
            .map(|o| o.tokenizer())
            .unwrap_or("default"),
        _ => "default",
    };

    if let Some(mut tokenizer) = index.tokenizers().get(tokenizer_name) {
        let mut stream = tokenizer.token_stream(text);
        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text.clone());
        }
        tokens
    } else {
        // Fallback: just lowercase
        vec![text.to_lowercase()]
    }
}

/// Token with byte offsets, used to extract separators from the query string.
struct TokenWithOffsets {
    text: String,
    offset_from: usize,
    offset_to: usize,
}

/// Tokenize text through the tokenizer configured for a field, preserving byte offsets.
fn tokenize_with_offsets(
    index: &Index,
    field: Field,
    schema: &Schema,
    text: &str,
) -> Vec<TokenWithOffsets> {
    let tokenizer_name = match schema.get_field_entry(field).field_type() {
        FieldType::Str(opts) => opts
            .get_indexing_options()
            .map(|o| o.tokenizer())
            .unwrap_or("default"),
        _ => "default",
    };

    if let Some(mut tokenizer) = index.tokenizers().get(tokenizer_name) {
        let mut stream = tokenizer.token_stream(text);
        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(TokenWithOffsets {
                text: token.text.clone(),
                offset_from: token.offset_from,
                offset_to: token.offset_to,
            });
        }
        tokens
    } else {
        vec![TokenWithOffsets {
            text: text.to_lowercase(),
            offset_from: 0,
            offset_to: text.len(),
        }]
    }
}

// ─── Field Resolution ───────────────────────────────────────────────────────

/// Resolve a field by name from the query config.
/// If `use_raw` is true and a `._raw` counterpart exists, use that instead.
fn resolve_field(
    config: &QueryConfig,
    schema: &Schema,
    raw_pairs: &[(String, String)],
    use_raw: bool,
) -> Result<Field, String> {
    let name = config
        .field
        .as_deref()
        .ok_or("query requires 'field'")?;

    let actual_name = if use_raw {
        raw_pairs
            .iter()
            .find(|(user, _)| user == name)
            .map(|(_, raw)| raw.as_str())
            .unwrap_or(name) // no raw counterpart → use base field
    } else {
        name
    };

    schema
        .get_field(actual_name)
        .map_err(|_| format!("unknown field: {actual_name}"))
}

// ─── Query Building ─────────────────────────────────────────────────────────

pub fn build_query(
    config: &QueryConfig,
    schema: &Schema,
    index: &Index,
    raw_pairs: &[(String, String)],
    ngram_pairs: &[(String, String)],
    highlight_sink: Option<Arc<HighlightSink>>,
) -> Result<Box<dyn Query>, String> {
    match config.query_type.as_str() {
        "term" => build_term_query(config, schema, index, raw_pairs, highlight_sink),
        "fuzzy" => build_fuzzy_query(config, schema, index, raw_pairs, highlight_sink),
        "phrase" => build_phrase_query(config, schema, index, raw_pairs, highlight_sink),
        "regex" => build_regex_query(config, schema, raw_pairs, highlight_sink),
        "contains" => {
            build_contains_query(config, schema, index, raw_pairs, ngram_pairs, highlight_sink)
        }
        "boolean" => build_boolean_query(config, schema, index, raw_pairs, ngram_pairs),
        "parse" => build_parsed_query(config, schema, index),
        other => Err(format!("unknown query type: {other}")),
    }
}

/// Term query: exact token match on raw field (lowercased only, no stemming).
/// Use `parse` query for stemmed/analyzed search.
fn build_term_query(
    config: &QueryConfig,
    schema: &Schema,
    _index: &Index,
    raw_pairs: &[(String, String)],
    highlight_sink: Option<Arc<HighlightSink>>,
) -> Result<Box<dyn Query>, String> {
    let field = resolve_field(config, schema, raw_pairs, true)?;
    let value = config.value.as_deref().ok_or("term query requires 'value'")?;

    // Direct lowercase — no tokenizer pipeline, just case-fold for exact token lookup.
    let term = Term::from_field_text(field, &value.to_lowercase());
    let mut query = TermQuery::new(term, IndexRecordOption::WithFreqs);
    if let Some(sink) = highlight_sink {
        query = query.with_highlight_sink(sink);
    }
    Ok(Box::new(query))
}

/// Fuzzy query: Levenshtein match on raw field (lowercased only, no stemming).
fn build_fuzzy_query(
    config: &QueryConfig,
    schema: &Schema,
    _index: &Index,
    raw_pairs: &[(String, String)],
    highlight_sink: Option<Arc<HighlightSink>>,
) -> Result<Box<dyn Query>, String> {
    let field = resolve_field(config, schema, raw_pairs, true)?;
    let value = config.value.as_deref().ok_or("fuzzy query requires 'value'")?;
    let distance = config.distance.unwrap_or(1);

    // Direct lowercase — Levenshtein automaton runs on the raw token form.
    let term = Term::from_field_text(field, &value.to_lowercase());
    let mut query = FuzzyTermQuery::new(term, distance, true);
    if let Some(sink) = highlight_sink {
        query = query.with_highlight_sink(sink);
    }
    Ok(Box::new(query))
}

/// Phrase query: tokenize each term through stemmed field, search stemmed index.
fn build_phrase_query(
    config: &QueryConfig,
    schema: &Schema,
    index: &Index,
    raw_pairs: &[(String, String)],
    highlight_sink: Option<Arc<HighlightSink>>,
) -> Result<Box<dyn Query>, String> {
    let field = resolve_field(config, schema, raw_pairs, false)?;
    let terms_str = config
        .terms
        .as_ref()
        .ok_or("phrase query requires 'terms'")?;

    let terms: Vec<Term> = terms_str
        .iter()
        .map(|t| {
            let tokens = tokenize_for_field(index, field, schema, t);
            let stemmed = tokens.first().map(|s| s.as_str()).unwrap_or(t);
            Term::from_field_text(field, stemmed)
        })
        .collect();

    let mut query = PhraseQuery::new(terms);
    if let Some(sink) = highlight_sink {
        query = query.with_highlight_sink(sink);
    }
    Ok(Box::new(query))
}

/// Contains query: auto-cascade per position (exact → fuzzy → substring).
///   - Uses AutomatonPhraseQuery which handles both single-token and multi-token.
///   - Fuzzy distance defaults to 1 (configurable via `distance` field).
///   - Extracts separators, prefix, and suffix from the query string for validation.
/// Resolve the ngram field for a user field name, if ngram pairs are configured.
fn resolve_ngram_field(
    config: &QueryConfig,
    schema: &Schema,
    ngram_pairs: &[(String, String)],
) -> Option<Field> {
    let name = config.field.as_deref()?;
    let ngram_name = ngram_pairs
        .iter()
        .find(|(user, _)| user == name)
        .map(|(_, ngram)| ngram.as_str())?;
    schema.get_field(ngram_name).ok()
}

fn build_contains_query(
    config: &QueryConfig,
    schema: &Schema,
    index: &Index,
    raw_pairs: &[(String, String)],
    ngram_pairs: &[(String, String)],
    highlight_sink: Option<Arc<HighlightSink>>,
) -> Result<Box<dyn Query>, String> {
    let field = resolve_field(config, schema, raw_pairs, true)?;
    // Resolve the base (stored) field for loading text during separator validation.
    // The raw field may not be stored, but the base field is.
    let stored_field = resolve_field(config, schema, raw_pairs, false).ok();
    let value = config.value.as_deref().ok_or("contains query requires 'value'")?;
    let fuzzy_distance = config.distance.unwrap_or(1);
    let strict_separators = config.strict_separators.unwrap_or(true);

    // Tokenize through the raw field's tokenizer, preserving byte offsets.
    let tokens = tokenize_with_offsets(index, field, schema, value);

    if tokens.is_empty() {
        // Empty input: regex substring on the raw value as fallback.
        let escaped = regex_escape(&value.to_lowercase());
        let pattern = format!(".*{escaped}.*");
        return RegexQuery::from_pattern(&pattern, field)
            .map(|q| Box::new(q) as Box<dyn Query>)
            .map_err(|e| format!("invalid contains pattern: {e}"));
    }

    // Extract separators between consecutive tokens from the original query string.
    let separators: Vec<String> = tokens
        .windows(2)
        .map(|w| value[w[0].offset_to..w[1].offset_from].to_string())
        .collect();

    // Extract prefix (chars before first token) and suffix (chars after last token).
    let prefix = value[..tokens.first().map(|t| t.offset_from).unwrap_or(0)].to_string();
    let suffix = value[tokens.last().map(|t| t.offset_to).unwrap_or(value.len())..].to_string();

    let token_texts: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();

    // distance_budget defaults to fuzzy_distance (enough for one fuzzy token match).
    let distance_budget = fuzzy_distance as u32;

    // If ngram field is available, use NgramContainsQuery (fast trigram lookup + verification).
    if let Some(ngram_field) = resolve_ngram_field(config, schema, ngram_pairs) {
        let mut query = NgramContainsQuery::new(
            field,
            ngram_field,
            stored_field,
            token_texts,
            separators,
            prefix,
            suffix,
            fuzzy_distance,
            distance_budget,
            strict_separators,
        );
        if let Some(sink) = highlight_sink {
            query = query.with_highlight_sink(sink);
        }
        return Ok(Box::new(query));
    }

    // Fallback: AutomatonPhraseQuery (FST walk cascade).
    let phrase_terms: Vec<(usize, String)> = token_texts
        .into_iter()
        .enumerate()
        .collect();

    if !separators.is_empty() || !prefix.is_empty() || !suffix.is_empty() {
        let mut query = AutomatonPhraseQuery::new_with_separators(
            field,
            stored_field,
            phrase_terms,
            1000,
            fuzzy_distance,
            separators,
            prefix,
            suffix,
            distance_budget,
            strict_separators,
        );
        if let Some(sink) = highlight_sink {
            query = query.with_highlight_sink(sink);
        }
        Ok(Box::new(query))
    } else {
        let mut query = AutomatonPhraseQuery::new(
            field,
            phrase_terms,
            1000,
            fuzzy_distance,
        );
        if let Some(sink) = highlight_sink {
            query = query.with_highlight_sink(sink);
        }
        Ok(Box::new(query))
    }
}

/// Escape regex special characters in a string.
fn regex_escape(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\.*+?()[]{}|^$".contains(c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

/// Regex query: pattern applies to raw field terms (lowercased, not stemmed).
fn build_regex_query(
    config: &QueryConfig,
    schema: &Schema,
    raw_pairs: &[(String, String)],
    highlight_sink: Option<Arc<HighlightSink>>,
) -> Result<Box<dyn Query>, String> {
    let field = resolve_field(config, schema, raw_pairs, true)?;
    let pattern = config
        .pattern
        .as_deref()
        .ok_or("regex query requires 'pattern'")?;
    let mut query = RegexQuery::from_pattern(pattern, field)
        .map_err(|e| format!("invalid regex: {e}"))?;
    if let Some(sink) = highlight_sink {
        query = query.with_highlight_sink(sink);
    }
    Ok(Box::new(query) as Box<dyn Query>)
}

fn build_boolean_query(
    config: &QueryConfig,
    schema: &Schema,
    index: &Index,
    raw_pairs: &[(String, String)],
    ngram_pairs: &[(String, String)],
) -> Result<Box<dyn Query>, String> {
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    if let Some(ref must) = config.must {
        for sub in must {
            clauses.push((Occur::Must, build_query(sub, schema, index, raw_pairs, ngram_pairs, None)?));
        }
    }
    if let Some(ref should) = config.should {
        for sub in should {
            clauses.push((Occur::Should, build_query(sub, schema, index, raw_pairs, ngram_pairs, None)?));
        }
    }
    if let Some(ref must_not) = config.must_not {
        for sub in must_not {
            clauses.push((Occur::MustNot, build_query(sub, schema, index, raw_pairs, ngram_pairs, None)?));
        }
    }

    if clauses.is_empty() {
        return Err("boolean query has no clauses".to_string());
    }

    Ok(Box::new(BooleanQuery::new(clauses)))
}

/// Parse query: already uses the field's configured tokenizer (stemmed pipeline).
fn build_parsed_query(
    config: &QueryConfig,
    schema: &Schema,
    index: &Index,
) -> Result<Box<dyn Query>, String> {
    let value = config
        .value
        .as_deref()
        .ok_or("parse query requires 'value'")?;

    let fields: Vec<Field> = if let Some(ref field_names) = config.fields {
        field_names
            .iter()
            .map(|n| {
                schema
                    .get_field(n)
                    .map_err(|_| format!("unknown field: {n}"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some(ref field_name) = config.field {
        vec![schema
            .get_field(field_name)
            .map_err(|_| format!("unknown field: {field_name}"))?]
    } else {
        return Err("parse query requires 'field' or 'fields'".to_string());
    };

    let parser = QueryParser::for_index(index, fields);
    parser
        .parse_query(value)
        .map_err(|e| format!("query parse error: {e}"))
}

// ─── Search Execution ───────────────────────────────────────────────────────

pub fn execute_search(
    searcher: &Searcher,
    query: &dyn Query,
    limit: u32,
    schema: &Schema,
    highlight_sink: Option<&HighlightSink>,
    highlight_field: Option<&str>,
) -> Result<Vec<SearchResult>, String> {
    let collector = TopDocs::with_limit(limit as usize).order_by_score();
    let top_docs = searcher
        .search(query, &collector)
        .map_err(|e| format!("search error: {e}"))?;

    collect_results(searcher, &top_docs, schema, highlight_sink, highlight_field)
}

/// Search with pre-filtering by allowed node IDs.
pub fn execute_search_filtered(
    searcher: &Searcher,
    query: &dyn Query,
    limit: u32,
    schema: &Schema,
    allowed_ids: HashSet<u64>,
    highlight_sink: Option<&HighlightSink>,
    highlight_field: Option<&str>,
) -> Result<Vec<SearchResult>, String> {
    let inner = TopDocs::with_limit(limit as usize).order_by_score();
    let collector = FilterCollector::new(
        crate::handle::NODE_ID_FIELD.to_string(),
        move |value: u64| allowed_ids.contains(&value),
        inner,
    );

    let top_docs = searcher
        .search(query, &collector)
        .map_err(|e| format!("filtered search error: {e}"))?;

    collect_results(searcher, &top_docs, schema, highlight_sink, highlight_field)
}

fn collect_results(
    searcher: &Searcher,
    top_docs: &[(f32, ld_tantivy::DocAddress)],
    schema: &Schema,
    highlight_sink: Option<&HighlightSink>,
    highlight_field: Option<&str>,
) -> Result<Vec<SearchResult>, String> {
    let mut results = Vec::with_capacity(top_docs.len());
    for &(score, doc_address) in top_docs {
        let doc: TantivyDocument = searcher
            .doc(doc_address)
            .map_err(|e| format!("doc retrieval error: {e}"))?;

        let doc_json = doc.to_named_doc(schema);
        let doc_value = serde_json::to_value(&doc_json)
            .map_err(|e| format!("json serialization error: {e}"))?;

        let highlights = highlight_sink.and_then(|sink| {
            let offsets = sink.get(doc_address.segment_ord, doc_address.doc_id)?;
            let field_name = highlight_field?;
            let mut map = HashMap::new();
            map.insert(field_name.to_string(), offsets);
            Some(map)
        });

        results.push(SearchResult {
            score,
            doc: doc_value,
            highlights,
        });
    }

    Ok(results)
}
