//! lucivy-fts: typed Rust ↔ C++ bridge for Lucivy full-text search.
//!
//! This crate provides a cxx bridge for creating, managing, and querying
//! Lucivy indexes. It is designed to be compiled as a static library
//! and linked into the rag3db C++ extension.

mod bridge;
pub mod directory;
pub mod handle;
pub mod query;
pub mod tokenizer;
