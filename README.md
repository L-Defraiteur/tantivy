# ld-tantivy

Fork of [tantivy](https://github.com/quickwit-oss/tantivy) (via [izihawa/tantivy](https://github.com/izihawa/tantivy) v0.26.0) with three major extensions for content search: **ContainsQuery**, **byte offsets in postings**, and **separator validation**.

```
quickwit-oss/tantivy v0.22
  -> izihawa/tantivy v0.26.0 (regex phrase queries, FST improvements)
    -> L-Defraiteur/tantivy (this fork)
```

## Changes from upstream

**39 files changed, +1873 lines, -59 lines** compared to `izihawa/tantivy@main`.

### 1. ContainsQuery — multi-strategy search with auto-cascade

A new query type that searches for substrings within indexed terms, with automatic fallback per position:

1. **Exact** — direct term dictionary lookup
2. **Fuzzy** — Levenshtein automaton (configurable distance, default 1)
3. **Substring** — regex `.*token.*` on the term dictionary

Early termination: as soon as one level finds matches for a position, lower levels are skipped.

For multi-token queries (`"std::collections"`, `"os.path.join"`), the `PhraseScorer` then verifies that positions are consecutive in the document.

**Files:**
- `src/query/phrase_query/automaton_phrase_query.rs` (162 lines) — Query struct, `new()` and `new_with_separators()` constructors
- `src/query/phrase_query/automaton_phrase_weight.rs` (415 lines) — Weight with cascade, `CascadeLevel` enum, 6 unit tests

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

**File:** `src/query/phrase_query/contains_scorer.rs` (605 lines) — `ContainsScorer` (multi-token) + `ContainsSingleScorer` (single-token) + `edit_distance()` + `tokenize_raw()`

### 3. WithFreqsAndPositionsAndOffsets — byte offsets in postings

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
cargo test --lib    # 997 tests (7 ignored — format compat v6/v7)
```

## Usage with tantivy_fts

This fork is used as a dependency of tantivy_fts, a C FFI crate that exposes full-text search for [rag3db](https://github.com/L-Defraiteur/rag3db).

```toml
[dependencies]
ld-tantivy = { path = "../ld-tantivy", features = ["stopwords", "lz4-compression", "stemmer"] }
```

## Lineage

- [quickwit-oss/tantivy](https://github.com/quickwit-oss/tantivy) — original full-text search engine in Rust
- [izihawa/tantivy](https://github.com/izihawa/tantivy) — v0.26.0 fork with regex phrase queries, FST improvements
- **L-Defraiteur/tantivy** — this fork: ContainsQuery, byte offsets, separator validation, cumulative distance

## License

MIT — same license as upstream tantivy.
