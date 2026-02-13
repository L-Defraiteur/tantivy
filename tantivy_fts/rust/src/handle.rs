//! Index handle management.
//!
//! Each TantivyHandle holds an Index, an IndexWriter, and an IndexReader.
//! Handles are identified by opaque pointers passed through the C FFI.
//!
//! When a stemmer is configured, every "text" field gets a dual-field layout:
//!   - `{name}` : tokenized + stemmed (for phrase/parse queries — recall)
//!   - `{name}._raw` : lowercased only (for term/fuzzy/regex queries — precision)
//! The routing is transparent — users always reference the base field name.

use std::path::Path;
use std::sync::Mutex;

use ld_tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED, TEXT,
};
use ld_tantivy::{Index, IndexReader, IndexSettings, IndexWriter, ReloadPolicy};

use crate::directory::StdFsDirectory;
use crate::query::SchemaConfig;

/// Reserved field name for Rag3db node IDs, used for filtered search.
pub const NODE_ID_FIELD: &str = "_node_id";

/// Suffix appended to text fields for the non-stemmed counterpart.
pub const RAW_SUFFIX: &str = "._raw";

/// Suffix appended to text fields for the n-gram (trigram) counterpart.
pub const NGRAM_SUFFIX: &str = "._ngram";

/// Tokenizer name for stemmed fields.
const STEMMED_TOKENIZER: &str = "stemmed";

/// Tokenizer name for n-gram (trigram) fields.
const NGRAM_TOKENIZER: &str = "ngram";

/// Opaque handle exposed through the C FFI.
pub struct TantivyHandle {
    pub index: Index,
    pub writer: Mutex<IndexWriter>,
    pub reader: IndexReader,
    pub schema: Schema,
    /// Maps field names (including internal `._raw` names) to Field objects.
    pub field_map: Vec<(String, Field)>,
    /// Maps user field names to their `._raw` counterpart names.
    /// Only populated when a stemmer is active and only for "text" fields.
    pub raw_field_pairs: Vec<(String, String)>,
    /// Maps user field names to their `._ngram` counterpart names.
    /// Only populated when a stemmer is active and only for "text" fields.
    pub ngram_field_pairs: Vec<(String, String)>,
}

/// Default writer heap size (50MB).
const WRITER_HEAP_SIZE: usize = 50_000_000;

/// Config file stored alongside the index for reopening.
const CONFIG_FILE: &str = "_config.json";

impl TantivyHandle {
    /// Create a new index at the given path.
    pub fn create(path: &str, config: &SchemaConfig) -> Result<Self, String> {
        let (schema, field_map, raw_field_pairs, ngram_field_pairs) = build_schema(config)?;
        let directory =
            StdFsDirectory::open(path).map_err(|e| format!("cannot open directory: {e}"))?;
        let index = Index::create(directory, schema.clone(), IndexSettings::default())
            .map_err(|e| format!("cannot create index: {e}"))?;

        configure_tokenizers(&index, config);

        // Persist config so open() can re-register tokenizers and rebuild raw_field_pairs.
        let config_json =
            serde_json::to_string(config).map_err(|e| format!("cannot serialize config: {e}"))?;
        std::fs::write(Path::new(path).join(CONFIG_FILE), config_json)
            .map_err(|e| format!("cannot write config: {e}"))?;

        let writer = index
            .writer(WRITER_HEAP_SIZE)
            .map_err(|e| format!("cannot create writer: {e}"))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| format!("cannot create reader: {e}"))?;

        Ok(Self {
            index,
            writer: Mutex::new(writer),
            reader,
            schema,
            field_map,
            raw_field_pairs,
            ngram_field_pairs,
        })
    }

    /// Open an existing index at the given path.
    pub fn open(path: &str) -> Result<Self, String> {
        let directory =
            StdFsDirectory::open(path).map_err(|e| format!("cannot open directory: {e}"))?;
        let index = Index::open(directory).map_err(|e| format!("cannot open index: {e}"))?;

        // Load config to re-register tokenizers and rebuild raw_field_pairs.
        let config_path = Path::new(path).join(CONFIG_FILE);
        let (raw_field_pairs, ngram_field_pairs) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| format!("cannot read config: {e}"))?;
            let config: SchemaConfig = serde_json::from_str(&config_str)
                .map_err(|e| format!("cannot parse config: {e}"))?;
            configure_tokenizers(&index, &config);
            if config.stemmer.is_some() {
                let text_fields: Vec<_> = config
                    .fields
                    .iter()
                    .filter(|f| f.field_type == "text")
                    .collect();
                let raw: Vec<_> = text_fields
                    .iter()
                    .map(|f| (f.name.clone(), format!("{}{RAW_SUFFIX}", f.name)))
                    .collect();
                let ngram: Vec<_> = text_fields
                    .iter()
                    .map(|f| (f.name.clone(), format!("{}{NGRAM_SUFFIX}", f.name)))
                    .collect();
                (raw, ngram)
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        };

        let schema = index.schema();
        let field_map = schema
            .fields()
            .map(|(field, entry)| (entry.name().to_string(), field))
            .collect();

        let writer = index
            .writer(WRITER_HEAP_SIZE)
            .map_err(|e| format!("cannot create writer: {e}"))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| format!("cannot create reader: {e}"))?;

        Ok(Self {
            index,
            writer: Mutex::new(writer),
            reader,
            schema,
            field_map,
            raw_field_pairs,
            ngram_field_pairs,
        })
    }

    /// Get a field by name.
    pub fn field(&self, name: &str) -> Option<Field> {
        self.field_map
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| *f)
    }

}

fn build_schema(
    config: &SchemaConfig,
) -> Result<(Schema, Vec<(String, Field)>, Vec<(String, String)>, Vec<(String, String)>), String> {
    let mut builder = Schema::builder();
    let mut field_map = Vec::new();
    let mut raw_field_pairs = Vec::new();
    let mut ngram_field_pairs = Vec::new();
    let has_stemmer = config.stemmer.is_some();

    // Auto-add _node_id as u64 FAST + INDEXED field for filtered search.
    let node_id_field = builder.add_u64_field(NODE_ID_FIELD, FAST | INDEXED);
    field_map.push((NODE_ID_FIELD.to_string(), node_id_field));

    for field_def in &config.fields {
        match field_def.field_type.as_str() {
            "text" => {
                if has_stemmer {
                    // Stemmed field: uses "stemmed" tokenizer.
                    // Uses WithFreqsAndPositionsAndOffsets so PhraseScorer can capture
                    // byte offsets for highlighting (stemmer preserves original offsets).
                    let indexing = TextFieldIndexing::default()
                        .set_tokenizer(STEMMED_TOKENIZER)
                        .set_index_option(IndexRecordOption::WithFreqsAndPositionsAndOffsets);
                    let mut opts = TextOptions::default().set_indexing_options(indexing);
                    if field_def.stored.unwrap_or(true) {
                        opts = opts.set_stored();
                    }
                    let field = builder.add_text_field(&field_def.name, opts);
                    field_map.push((field_def.name.clone(), field));

                    // Raw counterpart: "default" tokenizer (lowercase only), NOT stored.
                    // Uses WithFreqsAndPositionsAndOffsets so ContainsScorer can read
                    // byte offsets directly from the index (no re-tokenization needed).
                    let raw_indexing = TextFieldIndexing::default()
                        .set_tokenizer("default")
                        .set_index_option(IndexRecordOption::WithFreqsAndPositionsAndOffsets);
                    let raw_opts = TextOptions::default().set_indexing_options(raw_indexing);
                    let raw_name = format!("{}{RAW_SUFFIX}", field_def.name);
                    let raw_field = builder.add_text_field(&raw_name, raw_opts);
                    field_map.push((raw_name.clone(), raw_field));
                    raw_field_pairs.push((field_def.name.clone(), raw_name));

                    // N-gram counterpart: trigrams for fast substring candidate generation.
                    // Uses IndexRecordOption::Basic (doc IDs only — no positions/offsets needed).
                    let ngram_indexing = TextFieldIndexing::default()
                        .set_tokenizer(NGRAM_TOKENIZER)
                        .set_index_option(IndexRecordOption::Basic);
                    let ngram_opts = TextOptions::default().set_indexing_options(ngram_indexing);
                    let ngram_name = format!("{}{NGRAM_SUFFIX}", field_def.name);
                    let ngram_field = builder.add_text_field(&ngram_name, ngram_opts);
                    field_map.push((ngram_name.clone(), ngram_field));
                    ngram_field_pairs.push((field_def.name.clone(), ngram_name));
                } else {
                    // No stemmer: single field with default tokenizer.
                    let opts = if field_def.stored.unwrap_or(true) {
                        TEXT | STORED
                    } else {
                        TEXT
                    };
                    let field = builder.add_text_field(&field_def.name, opts);
                    field_map.push((field_def.name.clone(), field));
                }
            }
            "u64" => {
                use ld_tantivy::schema::{NumericOptions, FAST, INDEXED};
                let mut opts = NumericOptions::default();
                if field_def.stored.unwrap_or(true) {
                    opts = opts | STORED;
                }
                if field_def.indexed.unwrap_or(false) {
                    opts = opts | INDEXED;
                }
                if field_def.fast.unwrap_or(false) {
                    opts = opts | FAST;
                }
                let field = builder.add_u64_field(&field_def.name, opts);
                field_map.push((field_def.name.clone(), field));
            }
            "i64" => {
                use ld_tantivy::schema::{NumericOptions, FAST, INDEXED};
                let mut opts = NumericOptions::default();
                if field_def.stored.unwrap_or(true) {
                    opts = opts | STORED;
                }
                if field_def.indexed.unwrap_or(false) {
                    opts = opts | INDEXED;
                }
                if field_def.fast.unwrap_or(false) {
                    opts = opts | FAST;
                }
                let field = builder.add_i64_field(&field_def.name, opts);
                field_map.push((field_def.name.clone(), field));
            }
            "string" => {
                use ld_tantivy::schema::STRING;
                let opts = if field_def.stored.unwrap_or(true) {
                    STRING | STORED
                } else {
                    STRING
                };
                let field = builder.add_text_field(&field_def.name, opts);
                field_map.push((field_def.name.clone(), field));
            }
            other => return Err(format!("unknown field type: {other}")),
        }
    }

    Ok((builder.build(), field_map, raw_field_pairs, ngram_field_pairs))
}

fn configure_tokenizers(index: &Index, config: &SchemaConfig) {
    if let Some(ref stemmer_lang) = config.stemmer {
        use ld_tantivy::tokenizer::{LowerCaser, SimpleTokenizer, Stemmer, TextAnalyzer};

        use crate::tokenizer::NgramFilter;

        let lang = match stemmer_lang.as_str() {
            "english" => ld_tantivy::tokenizer::Language::English,
            "french" => ld_tantivy::tokenizer::Language::French,
            "german" => ld_tantivy::tokenizer::Language::German,
            "spanish" => ld_tantivy::tokenizer::Language::Spanish,
            "italian" => ld_tantivy::tokenizer::Language::Italian,
            "portuguese" => ld_tantivy::tokenizer::Language::Portuguese,
            "dutch" => ld_tantivy::tokenizer::Language::Dutch,
            "russian" => ld_tantivy::tokenizer::Language::Russian,
            _ => return,
        };

        let tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(Stemmer::new(lang))
            .build();

        // Register as "stemmed" — the "default" tokenizer stays untouched (lowercase only).
        index.tokenizers().register(STEMMED_TOKENIZER, tokenizer);

        // N-gram tokenizer: SimpleTokenizer → lowercase → trigrams.
        let ngram_tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(NgramFilter)
            .build();
        index.tokenizers().register(NGRAM_TOKENIZER, ngram_tokenizer);
    }
}
