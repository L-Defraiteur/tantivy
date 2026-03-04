# lucivy

Fork of [tantivy](https://github.com/quickwit-oss/tantivy) v0.26.0 — a fast full-text search engine in Rust — with **fuzzy substring search**, **contains_split multi-word matching**, **trigram-accelerated regex**, and **BM25 scoring**. Designed for code search, technical docs, and mixed-language content where exact matching fails.

```
quickwit-oss/tantivy v0.22
  -> izihawa/tantivy v0.26.0
    -> L-Defraiteur/lucivy (this fork)
```

## Python package

lucivy ships with Python bindings via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs/). Use it as a standalone BM25 index alongside vector databases like Qdrant, Milvus, etc.

```bash
cd lucivy && maturin develop --release
```

```python
import lucivy

# Create an index with text + filter fields
index = lucivy.Index.create("./my_index", fields=[
    {"name": "title", "type": "text"},
    {"name": "body", "type": "text"},
    {"name": "category", "type": "string"},
    {"name": "price", "type": "f64"},
], stemmer="english")

# Add documents
index.add(1, title="Rust programming guide", body="Learn systems programming with Rust")
index.add(2, title="Python for data science", body="Data analysis with pandas and numpy")
index.add(3, title="C++ template metaprogramming", body="Advanced C++ techniques")
index.commit()

# Search — default mode is contains_split: multi-word queries are split
# into individual substring matches combined with boolean OR
results = index.search("rust program", limit=10)
# Matches doc 1: "rust" matches title, "program" matches title + body

# With highlights — get byte offsets of matches per field
results = index.search("rust", limit=10, highlights=True)
for r in results:
    print(r.doc_id, r.score, r.highlights)
    # highlights = {"title": [(0, 4)], "body": [(42, 46)]}

# Pre-filtered by document IDs (for hybrid search with vector DBs)
results = index.search("programming", limit=10, allowed_ids=[1, 3])

# Dict queries for full control
results = index.search({
    "type": "contains",
    "field": "body",
    "value": "programing",  # typo — fuzzy distance=1 catches it
    "distance": 1,
}, limit=10)

# Delete, update, context manager
index.delete(2)
index.update(1, title="Updated title")
index.commit()
```

### Python API

| Method | Description |
|--------|-------------|
| `Index.create(path, fields, stemmer?)` | Create a new index |
| `Index.open(path)` | Open an existing index |
| `index.add(doc_id, **fields)` | Add a document |
| `index.add_many([{doc_id, ...}])` | Batch add |
| `index.delete(doc_id)` | Delete a document |
| `index.update(doc_id, **fields)` | Delete + re-add |
| `index.commit()` / `index.rollback()` | Flush / discard |
| `index.search(query, limit, highlights?, allowed_ids?)` | Search |
| `index.num_docs` / `index.schema` / `index.path` | Properties |

## Key features

### contains_split — multi-word substring search

The default search mode for string queries. Each word is searched independently as a **substring match** (trigram-accelerated) and results are combined with boolean OR. This means `"rust async"` finds documents containing "rust" OR "async" anywhere in any text field.

```python
# "hello world" → contains("hello") OR contains("world") across all text fields
results = index.search("hello world", limit=10)
```

In dict query mode:
```python
results = index.search({
    "type": "contains_split",
    "value": "rust async",
    "field": "body",
}, limit=10)
```

### NgramContainsQuery — trigram-accelerated substring search

Fast substring search using a **trigram index** for candidate lookup + **stored text verification** + **BM25 scoring**. Three verification modes:

- **Fuzzy** (default) — token-by-token Levenshtein matching with separator validation
- **Regex** — compiled regex on stored text, with trigram-accelerated candidate collection
- **Hybrid** — regex OR fuzzy, returns `max(tf_regex, tf_fuzzy)`

### How it works

Every text field automatically gets 3 sub-fields (triple-field layout):

| Sub-field | Tokenizer | Purpose |
|-----------|-----------|---------|
| `{name}` | stemmed or lowercase | Phrase/parse queries (recall) |
| `{name}._raw` | lowercase only | Term/fuzzy/regex/contains (precision) |
| `{name}._ngram` | trigrams | Fast candidate generation for contains |

When a contains query runs:

1. **Candidate collection** — depends on mode:
   - *Fuzzy*: exact lookup on `._raw` + trigram intersection on `._ngram`
   - *Regex*: trigram union on `._ngram` from extracted regex literals
   - *Short literals*: full segment scan when literals < 3 chars
2. **Verification** — read stored text, dispatch to fuzzy or regex
3. **BM25 scoring** — `idf * (1 + k1) * tf / (tf + k1 * (1 - b + b * dl / avgdl))`

### What it matches

**Fuzzy mode** (default):

| Query | Document | Match? | Why |
|-------|----------|--------|-----|
| `programming` | `"Rust programming is fun"` | yes | exact token match |
| `programing` (typo) | `"Rust programming is fun"` | yes | fuzzy distance=1 |
| `program` | `"Rust programming is fun"` | yes | substring via trigram |
| `c++` | `"c++ and c# are popular"` | yes | separator validation |
| `std::collections` | `"use std::collections::HashMap"` | yes | multi-token + `::` separator |

**Regex mode** (`regex: true`):

| Pattern | Document | Match? | Why |
|---------|----------|--------|-----|
| `program[a-z]+` | `"Rust programming is fun"` | yes | regex on stored text |
| `v[0-9]+` | `"version v2.0 released"` | yes | full-scan fallback (literal < 3 chars) |

### HighlightSink — byte offset capture

Thread-safe side-channel for capturing match byte offsets during scoring. Zero extra cost — offsets are captured as a free byproduct of existing verification work.

Supported by: contains, ngram contains, term, fuzzy, regex, phrase.

### ContainsQuery (FST-based fallback)

When no `._ngram` field is available, falls back to a 4-level cascade on the term dictionary:

1. **Exact** — term dict lookup
2. **Fuzzy** — Levenshtein automaton
3. **Substring** — regex `.*token.*`
4. **Fuzzy substring** — combined Levenshtein + substring automaton

### WithFreqsAndPositionsAndOffsets

New `IndexRecordOption` variant that stores `(offset_from, offset_to)` per token in the postings, like Lucene's `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`.

## Building

```bash
# Rust library tests (1064 tests)
cargo test --lib

# Python bindings
cd lucivy && maturin develop --release
```

## Usage as a Rust dependency

```toml
[dependencies]
ld-lucivy = { path = "../ld-lucivy", features = ["stopwords", "lz4-compression", "stemmer"] }
```

Also used by [lucivy_fts](../../lucivy_fts/), a cxx FFI crate that exposes full-text search for [rag3db](https://github.com/L-Defraiteur/rag3db).

## Lineage

- [quickwit-oss/tantivy](https://github.com/quickwit-oss/tantivy) — original full-text search engine in Rust
- [izihawa/tantivy](https://github.com/izihawa/tantivy) — v0.26.0 fork with regex phrase queries, FST improvements
- **L-Defraiteur/lucivy** — this fork: NgramContainsQuery, contains_split, fuzzy + regex + hybrid modes, BM25, HighlightSink, byte offsets, Python bindings

## License

MIT — same license as upstream tantivy.
