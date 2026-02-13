//! tantivy-fts: typed Rust ↔ C++ bridge for Tantivy full-text search.
//!
//! This crate provides a cxx bridge for creating, managing, and querying
//! Tantivy indexes. It is designed to be compiled as a static library
//! and linked into the rag3db C++ extension.

mod bridge;
mod directory;
mod handle;
mod query;
mod tokenizer;
