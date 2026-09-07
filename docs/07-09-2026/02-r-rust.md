lucivy: one index that answers substring, fuzzy-across-tokens and regex queries — and every answer is checked against a scan of the files (Rust, MIT)

**What it is.** lucivy is a full-text search library in Rust, with Python, Node.js, C++ and WASM bindings, built on a suffix FST instead of a token index. One default index answers exact substrings, matches across separators (`spin_lock` finds `spin lock`, `spin-lock` and `spinlock`), typos across token boundaries, regular expressions, two-character needles and boolean queries — with BM25 and the exact bytes of every match, and nothing to configure per question. It runs in your process, inside your transaction if you plug your own storage (a `BlobStore` trait: load, save, delete, list), and the same engine runs in the browser through emscripten with threads.

**The part I care about most: every answer is checked.** The ground-truth harness indexes the Linux kernel (93 983 files, 857 MB of text), runs a panel of queries, and compares every count *and every byte span* to a byte-by-byte scan of the files. It fails on any disagreement. Zero mismatches in 4.0.

**Against Elasticsearch and tantivy, same corpus, each configured at its best for substring search** — Elasticsearch with a trigram analyzer plus a `wildcard` field, tantivy (upstream, not our fork) with its `NgramTokenizer` — the "truth" column being that scan:

| asked | truth (scan of the files) | lucivy 4.0 | Elasticsearch 8.19 | tantivy 0.25 |
|---|---|---|---|---|
| `spin_lock`, separators relaxed (`spin lock`, `spin-lock`, `spinlock`) | 9 552 | **9 552**, 23 ms | 6 577 | 6 601 |
| `spinlokc`, two edits, across the token boundary | 10 034 | **10 034**, 148 ms | 3 549 | 6 557 |
| `spin_lock_[a-z]+`, a regex | 5 510 | **5 510**, 219 ms | 5 440, 480 ms | 0 |
| `de`, two characters | 93 009 | **93 009**, 561 ms | 0, silently | 0, silently |
| `retur -ENOMEM`, a fuzzy phrase | 14 449 | **14 449**, 30 ms | 14 446, 24 ms | — |
| `mutex_lock`: where it matched, in 5 145 documents | 20 797 spans | **all 20 797, 15 ms** | top 200 only: 179 ms | 96 ms |

Where they win, because they do: tantivy indexes the corpus in 1-5 s against 107 s here, and its index is 7× smaller; Elasticsearch does the fuzzy phrase as well as we do. The report has the sizes, the exact configurations and the lines where each engine's own documentation stops.

**The price.** The index is 5.8× the text (3.9× with the `derived_in_ram` option, which rebuilds three sidecars at open instead of storing them), against 3.6× for Elasticsearch's trigram setup and 0.8× for tantivy's n-grams. Indexing costs ×1.5 with the default shared dictionary. Queries stay in the tens of milliseconds for substrings, under a quarter of a second for fuzzy and regex; the one query above half a second returns 7.7 million positions.

**A few Rust things.** Forked from tantivy 0.22 for the segment layer; the suffix engine, the sharded handle, the snapshot/delta formats and the actor/DAG scheduler (`luciole`, WASM-safe, no `thread::spawn`) are ours. Five crates at the same version. The 4.0 format opens 3.0.x indexes and converts them on the first commit; that contract is a test against a fixture the published 3.0.8 wheel built.

![lucivy demo: lucivy's own source indexed in the browser in 3 s, then PostgreSQL's 5 199 files in 14 s, every search timed live](https://raw.githubusercontent.com/L-Defraiteur/lucivy/main/docs/07-09-2026/images/demo.gif)

**The demo** above is the real thing: the page clones lucivy's own source from GitHub and indexes 1 272 files in your tab in 3 s, then PostgreSQL's 5 199 files in 14 s, and every search you see is timed live — `--strict`, `--fuzzy 1 "vaccum"`, `--regex "ExecInit[A-Z][a-zA-Z]+\("`, an emoji, a boolean. You can type your own.

- [**Try it in your browser**](https://l-defraiteur.github.io/lucivy/) — the playground
- [Source on GitHub](https://github.com/L-Defraiteur/lucivy) (MIT)
- [The comparison with Elasticsearch and tantivy](https://github.com/L-Defraiteur/lucivy/blob/main/docs/compare-engines-2026-09-05.md), reproducible with one script
- [PyPI](https://pypi.org/project/lucivy/) · [npm](https://www.npmjs.com/package/lucivy) · [crates.io](https://crates.io/crates/lucivy-core) — 4.0.2

I'd take criticism on the comparison first: if you know a configuration of either engine that gets closer on a row, I'll add it to the report, with your name on the line.
