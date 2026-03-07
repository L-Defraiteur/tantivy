# lucivy

BM25 search engine with cross-token fuzzy matching — it finds substrings, handles typos, and matches across word boundaries. Built for code search, technical docs, and as a BM25 complement to vector databases.

## Install

Official bindings are **MIT-licensed** — use them freely, no revenue restrictions.

| Language | Install | License |
|----------|---------|---------|
| Python | `pip install lucivy` (or build from source) | MIT |
| Node.js | `npm install lucivy` (or build from source) | MIT |
| Rust | Direct dependency on `ld-lucivy` | LRSL |

The core engine (`ld-lucivy`) is under LRSL v1.3 (source-available). See [License](#license) for details.

## Node.js

```bash
cd bindings/nodejs && npm run build
node test.mjs
```

```javascript
const { Index } = require('lucivy');

// Create an index with text + filter fields
const index = Index.create('./my_index', [
    { name: 'title', type: 'text' },
    { name: 'body', type: 'text' },
    { name: 'year', type: 'i64', indexed: true, fast: true },
], 'english');

// Add documents
index.add(1, { title: 'Rust programming guide', body: 'Learn systems programming with Rust', year: 2024 });
index.add(2, { title: 'Python for data science', body: 'Data analysis with pandas and numpy', year: 2023 });
index.add(3, { title: 'C++ template metaprogramming', body: 'Advanced C++ techniques', year: 2022 });
index.commit();

// Search — string queries use contains_split: each word is a fuzzy
// substring match, combined with boolean OR, across all text fields
let results = index.search('rust program');

// With highlights — get byte offsets of matches per field
results = index.search('rust', { highlights: true });
for (const r of results) {
    console.log(r.docId, r.score, r.highlights);
    // highlights = { title: [[0, 4]], body: [[42, 46]] }
}

// Contains with fuzzy tolerance (catches typos)
results = index.search({ type: 'contains', field: 'body', value: 'programing', distance: 1 });

// Contains with regex (cross-token pattern matching)
results = index.search({ type: 'contains', field: 'body', value: 'program.*language', regex: true });

// Boolean: must + must_not
results = index.search({
    type: 'boolean',
    must: [{ type: 'contains', field: 'body', value: 'programming' }],
    must_not: [{ type: 'contains', field: 'body', value: 'python' }],
});

// Filters on non-text fields
results = index.search({
    type: 'contains',
    field: 'body',
    value: 'programming',
    filters: [{ field: 'year', op: 'gte', value: 2023 }],
});

// Pre-filtered by document IDs (for hybrid search with vector DBs)
results = index.search('programming', { allowedIds: [1, 3] });

// Delete, update, persistence
index.delete(2);
index.update(1, { title: 'Updated title', body: 'Updated body', year: 2025 });
index.commit();

// Reopen from disk
const index2 = Index.open('./my_index');
```

### Node.js API

| Method | Description |
|--------|-------------|
| `Index.create(path, fields, stemmer?)` | Create a new index |
| `Index.open(path)` | Open an existing index |
| `index.add(docId, fields)` | Add a document |
| `index.addMany([{docId, ...}])` | Batch add |
| `index.delete(docId)` | Delete a document |
| `index.update(docId, fields)` | Delete + re-add |
| `index.commit()` / `index.rollback()` | Flush / discard |
| `index.search(query, options?)` | Search |
| `index.numDocs` / `index.schema` / `index.path` | Properties |

## Python

```bash
cd bindings/python && pip install maturin && maturin develop --release
pytest tests/  # 64 tests
```

```python
import lucivy

# Create an index with text + filter fields
index = lucivy.Index.create("./my_index", fields=[
    {"name": "title", "type": "text"},
    {"name": "body", "type": "text"},
    {"name": "category", "type": "string"},
    {"name": "year", "type": "i64", "indexed": True, "fast": True},
], stemmer="english")

# Add documents
index.add(1, title="Rust programming guide", body="Learn systems programming with Rust", year=2024)
index.add(2, title="Python for data science", body="Data analysis with pandas and numpy", year=2023)
index.add(3, title="C++ template metaprogramming", body="Advanced C++ techniques", year=2022)
index.commit()

# Search — string queries use contains_split: each word is a fuzzy
# substring match, combined with boolean OR, across all text fields
results = index.search("rust program", limit=10)

# With highlights — get byte offsets of matches per field
results = index.search("rust", limit=10, highlights=True)
for r in results:
    print(r.doc_id, r.score, r.highlights)
    # highlights = {"title": [(0, 4)], "body": [(42, 46)]}

# Pre-filtered by document IDs (for hybrid search with vector DBs)
results = index.search("programming", limit=10, allowed_ids=[1, 3])

# Delete, update, persistence
index.delete(2)
index.update(1, title="Updated title", body="Updated body", year=2025)
index.commit()  # required to persist changes

# Reopen from disk
index2 = lucivy.Index.open("./my_index")
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

## Query types

lucivy queries operate on **stored text** (cross-token). They handle multi-word phrases, substrings, separators, and special characters naturally.

### `contains` — the workhorse query

Fuzzy substring match with separator awareness.

```python
# Exact substring
index.search({"type": "contains", "field": "body", "value": "programming language"})

# Substring within a token: "program" matches "programming"
index.search({"type": "contains", "field": "body", "value": "program"})

# Fuzzy tolerance (default distance=1, catches typos)
index.search({"type": "contains", "field": "body", "value": "programing languag", "distance": 1})

# Strict exact: distance=0 disables fuzzy
index.search({"type": "contains", "field": "body", "value": "programming", "distance": 0})
```

### `contains` + `regex`

Regex on stored text (cross-token).

```python
# Matches "programming language" — the .* spans the space between tokens
index.search({"type": "contains", "field": "body", "value": "program.*language", "regex": True})

# Alternation
index.search({"type": "contains", "field": "body", "value": "python|rust", "regex": True})
```

### `contains_split`

Splits query into words, each word is a `contains`, combined with OR.

```python
# String query (auto contains_split across all text fields)
index.search("rust async programming")

# Explicit dict query on a specific field
index.search({"type": "contains_split", "field": "body", "value": "memory safety"})
```

### `boolean`

Combine sub-queries with must (AND), should (OR), must_not (NOT).

```python
index.search({
    "type": "boolean",
    "must": [
        {"type": "contains", "field": "body", "value": "rust"},
        {"type": "contains", "field": "body", "value": "programming"},
    ],
    "must_not": [{"type": "contains", "field": "body", "value": "javascript"}],
})
```

### Filters on non-text fields

Non-text fields (`i64`, `f64`, `u64`, `string`) can be filtered via the `filters` key. Fields must be created with `indexed: true, fast: true`.

```python
index.search({
    "type": "contains",
    "field": "body",
    "value": "programming",
    "filters": [
        {"field": "year", "op": "gte", "value": 2023},
    ],
})
# Supported ops: eq, ne, lt, lte, gt, gte, in, not_in, between, starts_with, contains
```

### Highlights

All query types support byte-offset highlights. Internal fields (`._raw`, `._ngram`) are automatically filtered out.

```python
results = index.search("rust programming", highlights=True)
for r in results:
    if r.highlights:
        for field, offsets in r.highlights.items():
            print(f"  {field}: {offsets}")  # e.g. "body": [(5, 9), (20, 31)]
```

## Internals

### Triple-field layout

Every text field automatically gets 3 sub-fields:

| Sub-field | Tokenizer | Used by |
|-----------|-----------|---------|
| `{name}` | stemmed or lowercase | `phrase`, `parse` queries (recall) |
| `{name}._raw` | lowercase only | `contains` verification (precision) |
| `{name}._ngram` | character trigrams | `contains` candidate generation |

This is transparent to the user — you always reference the base field name.

### NgramContainsQuery — how `contains` works

The `contains` query type uses trigram-accelerated substring search on stored text:

1. **Candidate collection** — depends on mode:
   - *Fuzzy*: term dictionary lookup on `._raw` (O(1) via FST), falling back to trigram intersection on `._ngram` if the exact term isn't found
   - *Regex*: trigram union on `._ngram` from extracted regex literals
   - *Short literals*: full segment scan when literals < 3 chars
2. **Verification** — read stored text, dispatch to fuzzy or regex verifier
3. **BM25 scoring** — standard `idf * (1 + k1) * tf / (tf + k1 * (1 - b + b * dl / avgdl))`

### What `contains` matches

**Fuzzy mode** (default):

| Query | Document | Match? | Why |
|-------|----------|--------|-----|
| `programming` | `"Rust programming is fun"` | yes | exact token match |
| `programing` (typo) | `"Rust programming is fun"` | yes | fuzzy distance=1 |
| `program` | `"Rust programming is fun"` | yes | substring of token |
| `programming language` | `"...programming language used..."` | yes | cross-token with separator |
| `c++` | `"c++ and c# are popular"` | yes | separator-aware |
| `std::collections` | `"use std::collections::HashMap"` | yes | multi-token + `::` separator |

**Regex mode** (`regex: true`):

| Pattern | Document | Match? | Why |
|---------|----------|--------|-----|
| `program.*language` | `"...programming language used..."` | yes | cross-token regex on stored text |
| `python\|rust` | `"Python is versatile"` | yes | alternation |
| `v[0-9]+` | `"version v2.0 released"` | yes | full-scan fallback (literal < 3 chars) |

## Building

```bash
# Rust library tests
cargo test --lib

# Python bindings
cd bindings/python
maturin develop --release
pytest tests/ -v  # 64 tests

# Node.js bindings
cargo build -p lucivy-napi --release
cp target/release/liblucivy_napi.so bindings/nodejs/lucivy.linux-x64-gnu.node
node bindings/nodejs/test.mjs
```

## Usage as a Rust dependency

```toml
[dependencies]
ld-lucivy = { path = "../ld-lucivy", features = ["stopwords", "lz4-compression", "stemmer"] }
```

Also used by [lucivy_fts](../../lucivy_fts/), a cxx FFI crate that exposes full-text search for [rag3db](https://github.com/L-Defraiteur/rag3db).

## Lineage

Fork of [tantivy](https://github.com/quickwit-oss/tantivy) v0.26.0 (via [izihawa/tantivy](https://github.com/izihawa/tantivy)).

```
quickwit-oss/tantivy v0.22
  -> izihawa/tantivy v0.26.0 (regex phrase queries, FST improvements)
    -> L-Defraiteur/lucivy (NgramContainsQuery, contains_split, fuzzy/regex/hybrid modes, HighlightSink, Python/Node.js bindings)
```

## License

The core engine (`ld-lucivy`) is licensed under **LRSL v1.3** (Luciform Research Source License) — source-available. Free for research, personal, academic use and businesses under 100k EUR annual revenue. Commercial use above threshold requires a separate agreement. See [LICENSE](LICENSE).

**Official bindings (Python, Node.js) are MIT-licensed.** Per LRSL Section 4.3, you can use them freely regardless of your revenue — no commercial agreement needed. Each binding has its own MIT [LICENSE](bindings/nodejs/LICENSE) file.

The original Tantivy code is MIT-licensed (see [NOTICE](NOTICE)).
