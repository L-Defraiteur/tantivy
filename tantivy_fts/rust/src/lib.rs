//! tantivy-fts: C FFI bindings for Tantivy full-text search.
//!
//! This crate provides a C API for creating, managing, and querying
//! Tantivy indexes. It is designed to be compiled as a static library
//! and linked into the rag3db C++ extension.
//!
//! The API is platform-agnostic: on native targets, indexes are stored
//! on the real filesystem; on Emscripten (WASM), they go through the
//! Emscripten VFS (MEMFS/IDBFS).

mod bridge;
mod directory;
mod handle;
mod query;
mod tokenizer;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use std::sync::Arc;

use handle::TantivyHandle;
use ld_tantivy::query::HighlightSink;
use ld_tantivy::schema::Term;

/// Opaque handle type for the C API.
pub type TantivyHandlePtr = *mut TantivyHandle;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Convert a C string to a Rust &str. Returns None if null or invalid UTF-8.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Convert a Rust String to a C string (caller must free with tantivy_free_string).
fn string_to_cstr(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Return an error as a JSON string: {"error": "message"}.
fn error_json(msg: &str) -> *mut c_char {
    let json = format!(r#"{{"error":"{}"}}"#, msg.replace('"', r#"\""#));
    string_to_cstr(json)
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────

/// Create a new Tantivy index at the given path.
///
/// `path`: filesystem path for the index directory.
/// `schema_json`: JSON string defining the schema (see query::SchemaConfig).
///
/// Returns an opaque handle, or null on error.
#[no_mangle]
pub unsafe extern "C" fn tantivy_create_index(
    path: *const c_char,
    schema_json: *const c_char,
) -> TantivyHandlePtr {
    let path = match cstr_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let schema_str = match cstr_to_str(schema_json) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    let config: query::SchemaConfig = match serde_json::from_str(schema_str) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("tantivy_create_index: invalid schema JSON: {e}");
            return std::ptr::null_mut();
        }
    };

    match TantivyHandle::create(path, &config) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            eprintln!("tantivy_create_index: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Open an existing Tantivy index at the given path.
///
/// Returns an opaque handle, or null on error.
#[no_mangle]
pub unsafe extern "C" fn tantivy_open_index(path: *const c_char) -> TantivyHandlePtr {
    let path = match cstr_to_str(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match TantivyHandle::open(path) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            eprintln!("tantivy_open_index: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Close an index and free its resources.
#[no_mangle]
pub unsafe extern "C" fn tantivy_close_index(handle: TantivyHandlePtr) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

// ─── Write Operations ───────────────────────────────────────────────────────

/// Add a document to the index. The document is a JSON object whose keys
/// correspond to field names in the schema.
///
/// Returns the opstamp (monotonic operation id), or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn tantivy_add_document(
    handle: TantivyHandlePtr,
    doc_json: *const c_char,
) -> i64 {
    if handle.is_null() {
        return -1;
    }
    let h = &*handle;

    let doc_str = match cstr_to_str(doc_json) {
        Some(s) => s,
        None => return -1,
    };

    // Parse JSON and duplicate text values into ._raw fields (transparent to caller).
    let mut json_value: serde_json::Value = match serde_json::from_str(doc_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tantivy_add_document: invalid JSON: {e}");
            return -1;
        }
    };
    if let serde_json::Value::Object(ref mut obj) = json_value {
        for (user_field, raw_field) in &h.raw_field_pairs {
            if let Some(value) = obj.get(user_field).cloned() {
                obj.insert(raw_field.clone(), value);
            }
        }
        for (user_field, ngram_field) in &h.ngram_field_pairs {
            if let Some(value) = obj.get(user_field).cloned() {
                obj.insert(ngram_field.clone(), value);
            }
        }
    }

    // Convert to TantivyDocument
    let doc = match ld_tantivy::TantivyDocument::parse_json(&h.schema, &json_value.to_string()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("tantivy_add_document: cannot parse document: {e}");
            return -1;
        }
    };

    let writer = match h.writer.lock() {
        Ok(w) => w,
        Err(_) => return -1,
    };

    match writer.add_document(doc) {
        Ok(opstamp) => opstamp as i64,
        Err(e) => {
            eprintln!("tantivy_add_document: {e}");
            -1
        }
    }
}

/// Delete documents matching a term (field=value).
///
/// Deletions take effect on the next commit.
/// Returns the opstamp, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn tantivy_delete_by_term(
    handle: TantivyHandlePtr,
    field_name: *const c_char,
    value: *const c_char,
) -> i64 {
    if handle.is_null() {
        return -1;
    }
    let h = &*handle;

    let field_name = match cstr_to_str(field_name) {
        Some(s) => s,
        None => return -1,
    };
    let value = match cstr_to_str(value) {
        Some(s) => s,
        None => return -1,
    };

    let field = match h.field(field_name) {
        Some(f) => f,
        None => {
            eprintln!("tantivy_delete_by_term: unknown field: {field_name}");
            return -1;
        }
    };

    let term = Term::from_field_text(field, value);
    let writer = match h.writer.lock() {
        Ok(w) => w,
        Err(_) => return -1,
    };

    writer.delete_term(term) as i64
}

/// Commit pending operations (adds and deletes).
///
/// Creates a new segment for added documents. Returns the opstamp, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn tantivy_commit(handle: TantivyHandlePtr) -> i64 {
    if handle.is_null() {
        return -1;
    }
    let h = &*handle;

    let mut writer = match h.writer.lock() {
        Ok(w) => w,
        Err(_) => return -1,
    };

    match writer.commit() {
        Ok(opstamp) => opstamp as i64,
        Err(e) => {
            eprintln!("tantivy_commit: {e}");
            -1
        }
    }
}

/// Rollback pending operations.
#[no_mangle]
pub unsafe extern "C" fn tantivy_rollback(handle: TantivyHandlePtr) {
    if handle.is_null() {
        return;
    }
    let h = &*handle;

    let mut writer = match h.writer.lock() {
        Ok(w) => w,
        Err(_) => return,
    };

    if let Err(e) = writer.rollback() {
        eprintln!("tantivy_rollback: {e}");
    }
}

// ─── Read Operations ────────────────────────────────────────────────────────

/// Search the index.
///
/// `query_json`: JSON query (see query::QueryConfig for format).
/// `limit`: maximum number of results.
///
/// Returns a JSON string: [{"score": 1.23, "doc": {...}}, ...].
/// Caller must free the returned string with tantivy_free_string.
/// Returns an error JSON on failure.
#[no_mangle]
pub unsafe extern "C" fn tantivy_search(
    handle: TantivyHandlePtr,
    query_json: *const c_char,
    limit: u32,
) -> *mut c_char {
    if handle.is_null() {
        return error_json("null handle");
    }
    let h = &*handle;

    let query_str = match cstr_to_str(query_json) {
        Some(s) => s,
        None => return error_json("invalid query string"),
    };

    let config: query::QueryConfig = match serde_json::from_str(query_str) {
        Ok(c) => c,
        Err(e) => return error_json(&format!("invalid query JSON: {e}")),
    };

    let highlight = config.highlight.unwrap_or(false);
    let highlight_sink = if highlight {
        Some(Arc::new(HighlightSink::new()))
    } else {
        None
    };
    let highlight_field = config.field.clone();

    let query = match query::build_query(&config, &h.schema, &h.index, &h.raw_field_pairs, &h.ngram_field_pairs, highlight_sink.clone()) {
        Ok(q) => q,
        Err(e) => return error_json(&e),
    };

    let searcher = h.reader.searcher();
    let results = match query::execute_search(
        &searcher,
        query.as_ref(),
        limit,
        &h.schema,
        highlight_sink.as_deref(),
        highlight_field.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => return error_json(&e),
    };

    match serde_json::to_string(&results) {
        Ok(json) => string_to_cstr(json),
        Err(e) => error_json(&format!("serialization error: {e}")),
    }
}

/// Search the index with pre-filtering by allowed node IDs.
///
/// Only documents whose `_node_id` value is in the `allowed_ids` array are considered.
/// This is the key function for Rag3db graph-filtered FTS:
///   1. C++ extension runs Cypher WHERE → gets matching node IDs
///   2. Passes them here → Tantivy only scores those documents
///
/// `allowed_ids`: pointer to a u64 array of allowed node IDs.
/// `num_ids`: number of elements in the array.
///
/// Returns a JSON string like tantivy_search. Caller must free with tantivy_free_string.
#[no_mangle]
pub unsafe extern "C" fn tantivy_search_filtered(
    handle: TantivyHandlePtr,
    query_json: *const c_char,
    limit: u32,
    allowed_ids: *const u64,
    num_ids: u32,
) -> *mut c_char {
    if handle.is_null() {
        return error_json("null handle");
    }
    let h = &*handle;

    let query_str = match cstr_to_str(query_json) {
        Some(s) => s,
        None => return error_json("invalid query string"),
    };

    let config: query::QueryConfig = match serde_json::from_str(query_str) {
        Ok(c) => c,
        Err(e) => return error_json(&format!("invalid query JSON: {e}")),
    };

    let highlight = config.highlight.unwrap_or(false);
    let highlight_sink = if highlight {
        Some(Arc::new(HighlightSink::new()))
    } else {
        None
    };
    let highlight_field = config.field.clone();

    let query = match query::build_query(&config, &h.schema, &h.index, &h.raw_field_pairs, &h.ngram_field_pairs, highlight_sink.clone()) {
        Ok(q) => q,
        Err(e) => return error_json(&e),
    };

    // Build the allowed set from the C array
    let id_set: std::collections::HashSet<u64> = if allowed_ids.is_null() || num_ids == 0 {
        std::collections::HashSet::new()
    } else {
        let slice = std::slice::from_raw_parts(allowed_ids, num_ids as usize);
        slice.iter().copied().collect()
    };

    let searcher = h.reader.searcher();
    let results = match query::execute_search_filtered(
        &searcher,
        query.as_ref(),
        limit,
        &h.schema,
        id_set,
        highlight_sink.as_deref(),
        highlight_field.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => return error_json(&e),
    };

    match serde_json::to_string(&results) {
        Ok(json) => string_to_cstr(json),
        Err(e) => error_json(&format!("serialization error: {e}")),
    }
}

/// Reload the reader to see the latest committed segments.
///
/// Call this after tantivy_commit to make new documents visible to search.
#[no_mangle]
pub unsafe extern "C" fn tantivy_reload_reader(handle: TantivyHandlePtr) {
    if handle.is_null() {
        return;
    }
    let h = &*handle;
    if let Err(e) = h.reader.reload() {
        eprintln!("tantivy_reload_reader: {e}");
    }
}

// ─── Info ───────────────────────────────────────────────────────────────────

/// Get the index schema as a JSON string.
///
/// Caller must free the returned string with tantivy_free_string.
#[no_mangle]
pub unsafe extern "C" fn tantivy_get_schema(handle: TantivyHandlePtr) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    let h = &*handle;

    let schema_json = serde_json::to_string(&h.schema);
    match schema_json {
        Ok(json) => string_to_cstr(json),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get the number of documents in the index (including deleted but not yet merged).
#[no_mangle]
pub unsafe extern "C" fn tantivy_num_docs(handle: TantivyHandlePtr) -> u64 {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    let searcher = h.reader.searcher();
    searcher.num_docs()
}

// ─── Memory Management ─────────────────────────────────────────────────────

/// Free a string returned by any tantivy_* function.
#[no_mangle]
pub unsafe extern "C" fn tantivy_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
