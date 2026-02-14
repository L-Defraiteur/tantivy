# ld-tantivy

Fork of [tantivy](https://github.com/quickwit-oss/tantivy) (via [izihawa/tantivy](https://github.com/izihawa/tantivy) v0.26.0) with extensions for content search: **ContainsQuery**, **NgramContainsQuery**, **byte offsets in postings**, **separator validation**, and **HighlightSink**.

```
quickwit-oss/tantivy v0.22
  -> izihawa/tantivy v0.26.0 (regex phrase queries, FST improvements)
    -> L-Defraiteur/tantivy (this fork)
```

## Changes from upstream

**48 files changed** compared to `izihawa/tantivy@main`.

### 1. ContainsQuery — multi-strategy search with auto-cascade

A new query type that searches for substrings within indexed terms, with automatic fallback per position:

1. **Exact** — direct term dictionary lookup
2. **Fuzzy** — Levenshtein automaton (configurable distance, default 1)
3. **Substring** — regex `.*token.*` on the term dictionary
4. **Fuzzy substring** — combined `.*{levenshtein(token,d)}.*` automaton on the term dictionary

Early termination: as soon as one level finds matches for a position, lower levels are skipped.

For multi-token queries (`"std::collections"`, `"os.path.join"`), the `PhraseScorer` then verifies that positions are consecutive in the document.

**Files:**
- `src/query/phrase_query/automaton_phrase_query.rs` — Query struct, `new()` and `new_with_separators()` constructors
- `src/query/phrase_query/automaton_phrase_weight.rs` — Weight with 4-level cascade, `CascadeLevel` enum, 6 unit tests
- `src/query/fuzzy_substring_automaton.rs` — `FuzzySubstringAutomaton` for level 4

### 2. ContainsScorer — separator validation and cumulative distance

A custom scorer that validates non-alphanumeric characters (separators) between query tokens match those in the document. This is what allows `c++` to match only documents containing `c++` and not every occurrence of the word "c".

**How it works (multi-token):**
- Reads byte offsets directly from the postings index via `append_positions_and_offsets()` — no re-tokenization needed
- Extracts actual separators from stored text (text between `offset_to[token_i]` and `offset_from[token_i+1]`)
- Compares with query separators via edit distance (Levenshtein)
- Global cumulative distance budget: the sum of fuzzy distances from tokens + separator distances must stay within budget
- Falls back to re-tokenization if the field was indexed without offsets (`WithFreqsAndPositions`)

**Two validation modes:**
- `strict_separators: true` (default) — separators must match exactly (within edit distance budget). `c++` does not match `c--` (distance 2 > budget 1)
- `strict_separators: false` — only checks that a non-alphanumeric character exists between tokens. `c--` matches `c++`, `std collections` matches `std::collections`

**Edge constraints:**
- First token: no constraint on what precedes it in the document (unless the query has characters before the first token)
- Last token: same for what follows

**Match examples:**

| Query | Document | Result |
|-------|----------|--------|
| `c++` | `"c++ and c# are popular"` | match (`++` separators valid) |
| `c++` | `"the cat sat"` | reject (no non-alnum separator after "c") |
| `std::collections` | `"use std::collections::HashMap"` | match (`::` separator exact) |
| `os.path.join` | `"import os.path.join for files"` | match (`.` separators exact) |
| `option<result<(i32` | `"Vec<Option<Result<(i32,&str)>>"` | match (`<`, `<(` separators valid) |

**Files:**
- `src/query/phrase_query/contains_scorer.rs` — `ContainsScorer` (multi-token) + `ContainsSingleScorer` (single-token)
- `src/query/phrase_query/scoring_utils.rs` — shared `edit_distance()`, `tokenize_raw()`, `token_match_distance()`, `HighlightSink`

### 3. NgramContainsQuery — trigram-accelerated contains search

A fast alternative to AutomatonPhraseQuery that uses a trigram (ngram) index for candidate lookup, then verifies matches against stored text. Best for fields with a dedicated `._ngram` subfield.

**How it works:**
1. Extract trigrams from query tokens → look up in the ngram inverted index → get candidate doc IDs
2. For each candidate, read stored text and re-tokenize with `tokenize_raw()`
3. Verify token matches (exact, fuzzy, or substring) with separator validation and cumulative distance budget
4. Same two validation modes as ContainsScorer (`strict_separators` / relaxed)

**File:** `src/query/phrase_query/ngram_contains_query.rs` — Query, Weight, Scorer with trigram candidate filtering

### 4. HighlightSink — side-channel byte offset capture for all query types

A thread-safe sink for capturing byte offsets during scoring, used by the FFI layer to return highlight ranges without re-tokenization.

```rust
pub struct HighlightSink {
    data: Mutex<HashMap<(u32, DocId), Vec<[usize; 2]>>>,
    segment_counter: AtomicU32,
}
```

**How it works:**
- Shared via `Arc<HighlightSink>` between the caller and scorers
- Each `Weight::scorer()` call increments `segment_counter` to track segment ordinals (since `SegmentReader` doesn't expose `segment_ordinal()`)
- Scorers call `sink.insert(segment_ord, doc_id, offsets)` as a free byproduct of scoring — no extra work
- After search, the caller reads offsets via `sink.get(segment_ord, doc_id)`
- **Important:** `next_segment()` must be called for every segment, even when returning an empty scorer (e.g. term not found), to keep the counter in sync with the real segment ordinals used by `TopDocs`

**Supported query types:**

| Query type | Scorer | Offset source |
|------------|--------|---------------|
| **contains** | `ContainsScorer`, `ContainsSingleScorer` | Stored text verification (free byproduct) |
| **ngram contains** | `NgramContainsScorer` | Stored text verification (free byproduct) |
| **term** | `TermScorer` | Postings byte offsets via `append_offsets()` |
| **fuzzy** | `AutomatonWeight` | Postings byte offsets (captured during scorer construction) |
| **regex** | `AutomatonWeight` | Postings byte offsets (captured during scorer construction) |
| **phrase** | `PhraseScorer` | Postings byte offsets via `drain_or_capture_offsets()` |

All scorers accept the sink via `with_highlight_sink()`.

**Files:**
- `src/query/phrase_query/scoring_utils.rs` — `HighlightSink` struct + `HighlightKey` type alias
- `src/query/term_query/term_scorer.rs` — `capture_offsets()` in `advance()` / `seek()`
- `src/query/term_query/term_weight.rs` — forces `WithFreqsAndPositionsAndOffsets`, segment_ord sync
- `src/query/automaton_weight.rs` — highlight path reads full postings with offsets per matching term
- `src/query/phrase_query/phrase_scorer.rs` — `drain_or_capture_offsets()` keeps position/offset readers in sync
- `src/query/phrase_query/phrase_weight.rs` — forces `WithFreqsAndPositionsAndOffsets`, segment_ord sync
- `src/query/fuzzy_query.rs` — sink propagation + custom `Debug` (excludes sink)
- `src/query/regex_query.rs` — sink propagation
- `src/query/phrase_query/phrase_query.rs` — sink propagation
- `src/query/term_query/term_query.rs` — sink propagation

### 5. WithFreqsAndPositionsAndOffsets — byte offsets in postings

New `IndexRecordOption` variant that stores byte offsets (`offset_from`, `offset_to`) of each token occurrence directly in the postings, like Lucene's `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`.

```rust
pub enum IndexRecordOption {
    Basic,                              // doc IDs
    WithFreqs,                          // + term frequencies
    WithFreqsAndPositions,              // + token positions
    WithFreqsAndPositionsAndOffsets,    // + byte offsets (NEW)
}
```

**Storage format:**
- Separate `.offsets` file (new `SegmentComponent::Offsets`)
- CompositeFile per field (same architecture as `.pos`)
- Interleaved delta-encoding: `(from_delta_0, to_delta_0, from_delta_1, to_delta_1, ...)` — 2 values per token
- Bitpacked in blocks of 128 (reuses `PositionSerializer`)
- Extended `TermInfo`: added `offsets_range: Range<usize>` (40 bytes instead of 28)

**Write pipeline:**
```
Token (offset_from, offset_to, position)
  -> PostingsWriter::subscribe_with_offsets()
  -> TfPositionAndOffsetRecorder::record_position_with_offsets()
  -> FieldSerializer::write_doc_with_offsets()
  -> PositionSerializer (.offsets) — bitpacked blocks
```

**Read pipeline:**
```
InvertedIndexReader::read_postings_from_terminfo()
  -> PositionReader (.offsets)
  -> SegmentPostings::offsets() -> Vec<(u32, u32)>
```

**Propagation through unions:**
- `SegmentPostings` — reads from PositionReader
- `LoadedPostings` — from offsets loaded in memory
- `SimpleUnion` / `BitSetPostingUnion` — merge + sort + dedup from all docsets
- `PostingsWithOffset::positions_and_offsets()` — delegates with position offset, byte offsets stay absolute

**Joint method** `append_positions_and_offsets(offset, output)` on the `Postings` trait returns `(position, byte_from, byte_to)` tuples, keeping positions and byte offsets correlated through unions (unlike separate `append_positions_with_offset` + `append_offsets` which sort/dedup independently).

**21 files modified** across `src/schema/`, `src/postings/`, `src/index/`, `src/termdict/`, `src/query/`.

## Building

```bash
cargo test --lib    # 1015 tests (7 ignored — format compat v6/v7)
```

## Usage with tantivy_fts

This fork is used as a dependency of tantivy_fts, a C FFI crate that exposes full-text search for [rag3db](https://github.com/L-Defraiteur/rag3db).

```toml
[dependencies]
ld-tantivy = { path = "../ld-tantivy", features = ["stopwords", "lz4-compression", "stemmer"] }
```

### Rebuild after modifying ld-tantivy sources

CMake does not detect changes in Rust source files. After modifying files in `src/`, you must manually rebuild the Rust static library and re-link the extension:

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
- **L-Defraiteur/tantivy** — this fork: ContainsQuery, NgramContainsQuery, byte offsets, separator validation, HighlightSink for all query types

## License

MIT — same license as upstream tantivy.
