# ld-tantivy

Fork of [tantivy](https://github.com/quickwit-oss/tantivy) v0.26.0 for **fuzzy substring search with BM25 scoring** and **trigram-accelerated regex**. Designed for code search, technical docs, and mixed-language content where exact matching fails.

```
quickwit-oss/tantivy v0.22
  -> izihawa/tantivy v0.26.0
    -> L-Defraiteur/tantivy (this fork)
```

## Key feature: NgramContainsQuery

Fast substring search using a **trigram index** for candidate lookup + **stored text verification** + **BM25 scoring**. Supports three verification modes:

- **Fuzzy** (default) — token-by-token Levenshtein matching with separator validation
- **Regex** — compiled regex on stored text, with trigram-accelerated candidate collection from extracted literals
- **Hybrid** — regex OR fuzzy on extracted literals (`distance > 0`), returns `max(tf_regex, tf_fuzzy)`

### How it works

Every text field automatically gets 3 sub-fields (triple-field layout):

| Sub-field | Tokenizer | Purpose |
|-----------|-----------|---------|
| `{name}` | stemmed or lowercase | Phrase/parse queries (recall) |
| `{name}._raw` | lowercase only | Term/fuzzy/regex/contains (precision) |
| `{name}._ngram` | trigrams | Fast candidate generation for contains |

When a contains query runs:

1. **Candidate collection** — depends on mode:
   - *Fuzzy*: exact lookup on `._raw` + trigram intersection on `._ngram` (all tokens must match)
   - *Regex*: trigram union on `._ngram` from extracted regex literals (each literal is an alternative)
   - *Regex (short literals)*: full segment scan when literals < 3 chars (no usable trigrams)
2. **Verification** — read stored text, dispatch to fuzzy or regex verification
3. **BM25 scoring** — score based on term frequency and document length (k1=1.2, b=0.75)

Works **with or without** a stemmer configured.

### What it matches

**Fuzzy mode** (default):

| Query | Document | Match? | Why |
|-------|----------|--------|-----|
| `programming` | `"Rust programming is fun"` | yes | exact token match |
| `programing` (typo) | `"Rust programming is fun"` | yes | fuzzy distance=1 |
| `program` | `"Rust programming is fun"` | yes | substring via trigram |
| `c++` | `"c++ and c# are popular"` | yes | separator validation (`++`) |
| `c++` | `"the cat sat"` | no | no non-alnum separator after "c" |
| `std::collections` | `"use std::collections::HashMap"` | yes | multi-token + `::` separator |
| `os.path.join` | `"import os.path.join for files"` | yes | `.` separators exact |

**Regex mode** (`regex: true`):

| Pattern | Document | Match? | Why |
|---------|----------|--------|-----|
| `program[a-z]+` | `"Rust programming is fun"` | yes | regex on stored text |
| `program[a-z]+` | `"A programmer's guide"` | yes | trigram "program" → candidate |
| `programing[a-z]+` (typo, distance=1) | `"Rust programming is fun"` | yes | hybrid: fuzzy on literal "programing" |
| `v[0-9]+` | `"version v2.0 released"` | yes | full-scan fallback (literal "v" < 3 chars) |
| `python[a-z]+` | `"Rust programming is fun"` | no | regex doesn't match |

### BM25 scoring

Documents with more occurrences of the query term score higher. Shorter documents score higher than longer ones with the same frequency.

```
score = idf * (1 + k1) * tf / (tf + k1 * (1 - b + b * dl / avgdl))
```

- `tf` = number of matches counted during verification (not just boolean match)
- `dl` = document field length from fieldnorm reader
- `avgdl` = average field length across the index
- `idf` = inverse document frequency from the `._raw` field statistics

### Separator validation

For queries containing non-alphanumeric characters (`c++`, `std::collections`, `option<result<(i32`), two modes:

- **`strict_separators: true`** (default) — separators must match within edit distance budget
- **`strict_separators: false`** — only checks that a non-alnum character exists between tokens

Cumulative distance budget: the sum of fuzzy distances from tokens + separator distances must stay within budget.

### Regex acceleration

When `regex: true` is set, NgramContainsQuery uses `regex-syntax` to parse the pattern's HIR and extract obligatory literals via `Extractor`. These literals serve as trigram sources for candidate collection:

- **Literals >= 3 chars** → trigram-based candidate collection (fast path, union across alternatives)
- **Literals < 3 chars** → full segment scan (all docs are candidates, regex verification filters)
- **No ngram field** → falls back to standard `RegexQuery` (FST walk, no BM25)

The compiled regex (`regex` crate, case-insensitive) runs on stored text during verification. When `distance > 0`, hybrid mode tries both regex exact matching and fuzzy matching on extracted literals, returning `max(tf_regex, tf_fuzzy)`.

**Files:** `ngram_contains_query.rs` (VerificationMode::Regex, verify_regex), `tantivy_fts/rust/src/query.rs` (build_contains_regex)

## Other features

### ContainsQuery (AutomatonPhraseQuery) — FST-based fallback

When no `._ngram` field is available, falls back to a 4-level cascade on the term dictionary:

1. **Exact** — term dict lookup
2. **Fuzzy** — Levenshtein automaton
3. **Substring** — regex `.*token.*`
4. **Fuzzy substring** — combined Levenshtein + substring automaton

Slower than NgramContainsQuery but works on any indexed field.

**Files:** `automaton_phrase_query.rs`, `automaton_phrase_weight.rs`, `contains_scorer.rs`, `fuzzy_substring_automaton.rs`

### HighlightSink — byte offset capture for all query types

Thread-safe side-channel for capturing match byte offsets during scoring. Zero extra cost — offsets are captured as a free byproduct of existing verification/scoring work.

Supported by: contains, ngram contains, term, fuzzy, regex, phrase.

**File:** `scoring_utils.rs` (HighlightSink), integrated in all scorer types.

### WithFreqsAndPositionsAndOffsets — byte offsets in postings

New `IndexRecordOption` variant that stores `(offset_from, offset_to)` per token in the postings, like Lucene's `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`. Separate `.offsets` file, bitpacked blocks of 128, interleaved delta-encoding.

**21 files modified** across `src/schema/`, `src/postings/`, `src/index/`, `src/termdict/`, `src/query/`.

## Building

```bash
cargo test --lib    # 1025 tests (7 ignored — format compat v6/v7)
```

## Usage with tantivy_fts

This fork is used as a dependency of [tantivy_fts](../../tantivy_fts/), a cxx FFI crate that exposes full-text search for [rag3db](https://github.com/L-Defraiteur/rag3db).

```toml
[dependencies]
ld-tantivy = { path = "../ld-tantivy", features = ["stopwords", "lz4-compression", "stemmer"] }
```

### Rebuild after modifying ld-tantivy sources

CMake does not detect changes in Rust source files. After modifying files in `src/`, you must manually rebuild:

```bash
# 1. Rebuild Rust (from this directory)
cargo build --release -p ld-tantivy -p tantivy-fts

# 2. Re-link the extension shared lib (from rag3db/build/release/)
cmake --build . --target rag3db_tantivy_fts_extension -j$(nproc)
```

See [`../../tantivy_fts/BUILD.md`](../../tantivy_fts/BUILD.md) for the full build guide.

## Lineage

- [quickwit-oss/tantivy](https://github.com/quickwit-oss/tantivy) — original full-text search engine in Rust
- [izihawa/tantivy](https://github.com/izihawa/tantivy) — v0.26.0 fork with regex phrase queries, FST improvements
- **L-Defraiteur/tantivy** — this fork: NgramContainsQuery (fuzzy + regex + hybrid), BM25 scoring, ContainsQuery, byte offsets, separator validation, HighlightSink

## License

MIT — same license as upstream tantivy.
