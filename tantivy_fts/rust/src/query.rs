//! Query and schema JSON parsing.
//!
//! Query routing with dual-field layout:
//!   - phrase, parse            → stemmed field (recall: "run" matches "running")
//!   - term, fuzzy, regex       → raw field (precision: exact word forms, lowercase only)
//! The user always references the base field name; routing is transparent.

use std::ops::Bound;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use ld_tantivy::query::{
    AutomatonPhraseQuery, BooleanQuery, FuzzyTermQuery, HighlightSink, NgramContainsQuery, Occur,
    PhraseQuery, Query, QueryParser, RangeQuery, RegexQuery, TermQuery,
};
use ld_tantivy::schema::{Field, FieldType, IndexRecordOption, Schema, Term};
use ld_tantivy::Index;

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
pub struct FilterClause {
    pub field: String,
    pub op: String, // "eq", "ne", "lt", "lte", "gt", "gte", "in"
    pub value: serde_json::Value,
}

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
    // Boolean query sub-clauses
    pub must: Option<Vec<QueryConfig>>,
    pub should: Option<Vec<QueryConfig>>,
    pub must_not: Option<Vec<QueryConfig>>,
    // Filter clauses on non-text fields
    pub filters: Option<Vec<FilterClause>>,
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
    let text_query = match config.query_type.as_str() {
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
    }?;

    // Wrap with filter clauses if present.
    if let Some(ref filters) = config.filters {
        if !filters.is_empty() {
            let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
            clauses.push((Occur::Must, text_query));
            for filter in filters {
                clauses.push((Occur::Must, build_filter_clause(filter, schema)?));
            }
            return Ok(Box::new(BooleanQuery::new(clauses)));
        }
    }

    Ok(text_query)
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

// ─── Filter Clause Building ────────────────────────────────────────────────

/// Helper: extract a JSON value as the appropriate Term for a given field type.
fn json_to_term(field: Field, field_type: &FieldType, value: &serde_json::Value) -> Result<Term, String> {
    match field_type {
        FieldType::U64(_) => {
            let v = value.as_u64().ok_or_else(|| format!("expected u64 value, got {value}"))?;
            Ok(Term::from_field_u64(field, v))
        }
        FieldType::I64(_) => {
            let v = value.as_i64().ok_or_else(|| format!("expected i64 value, got {value}"))?;
            Ok(Term::from_field_i64(field, v))
        }
        FieldType::F64(_) => {
            let v = value.as_f64().ok_or_else(|| format!("expected f64 value, got {value}"))?;
            Ok(Term::from_field_f64(field, v))
        }
        FieldType::Str(_) => {
            let v = value.as_str().ok_or_else(|| format!("expected string value, got {value}"))?;
            Ok(Term::from_field_text(field, v))
        }
        _ => Err(format!("unsupported field type for filter")),
    }
}

fn build_filter_clause(filter: &FilterClause, schema: &Schema) -> Result<Box<dyn Query>, String> {
    let field = schema
        .get_field(&filter.field)
        .map_err(|_| format!("unknown filter field: {}", filter.field))?;
    let field_type = schema.get_field_entry(field).field_type().clone();

    match filter.op.as_str() {
        "eq" => {
            let term = json_to_term(field, &field_type, &filter.value)?;
            Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
        }
        "ne" => {
            let term = json_to_term(field, &field_type, &filter.value)?;
            let eq_query = TermQuery::new(term, IndexRecordOption::Basic);
            Ok(Box::new(BooleanQuery::new(vec![
                (Occur::MustNot, Box::new(eq_query) as Box<dyn Query>),
            ])))
        }
        "lt" => {
            let term = json_to_term(field, &field_type, &filter.value)?;
            Ok(Box::new(RangeQuery::new(Bound::Unbounded, Bound::Excluded(term))))
        }
        "lte" => {
            let term = json_to_term(field, &field_type, &filter.value)?;
            Ok(Box::new(RangeQuery::new(Bound::Unbounded, Bound::Included(term))))
        }
        "gt" => {
            let term = json_to_term(field, &field_type, &filter.value)?;
            Ok(Box::new(RangeQuery::new(Bound::Excluded(term), Bound::Unbounded)))
        }
        "gte" => {
            let term = json_to_term(field, &field_type, &filter.value)?;
            Ok(Box::new(RangeQuery::new(Bound::Included(term), Bound::Unbounded)))
        }
        "in" => {
            let values = filter.value.as_array().ok_or("'in' operator requires an array value")?;
            let clauses: Vec<(Occur, Box<dyn Query>)> = values
                .iter()
                .map(|v| {
                    let term = json_to_term(field, &field_type, v)?;
                    Ok((Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>))
                })
                .collect::<Result<Vec<_>, String>>()?;
            if clauses.is_empty() {
                return Err("'in' filter requires at least one value".to_string());
            }
            Ok(Box::new(BooleanQuery::new(clauses)))
        }
        other => Err(format!("unknown filter operator: {other}")),
    }
}

