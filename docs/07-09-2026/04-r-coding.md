Search code the way you grep it: a full-text engine with exact byte spans, verified answers, running in your process or in the browser

**Full-text search that doesn't lie.** Every answer is compared to a byte-by-byte scan of the files — the whole Linux kernel, 93 983 files — and the build fails on any disagreement. The index answers substrings, typos across token boundaries (`spinlokc` finds `spin_lock`), regex and two-character queries, with the exact position of every match and BM25 ranking. Nothing to configure per query.

**Compared with Elasticsearch and tantivy** on that corpus, each configured for substring search: they agree on the plain substring; on relaxed separators, fuzzy across the boundary, regex and two characters, they return partial counts or 0 — silently. Where they win: indexing speed (1-5 s against 107 s) and index size (7× smaller for tantivy). The report is one script, reproducible, with their wins written in.

**The price** is disk: the index is 5.8× the text (3.9× with an option).

![lucivy demo: lucivy's own source indexed in the browser in 3 s, then PostgreSQL's 5 199 files in 14 s, every search timed live](https://raw.githubusercontent.com/L-Defraiteur/lucivy/main/docs/07-09-2026/images/demo.gif)

**Where it runs.** A Rust library with Python, Node.js, C++ and browser bindings, MIT. In your process, in your transaction if you plug your own storage, and in the browser: the demo above clones its own source from GitHub and indexes it in your tab in 3 s, then PostgreSQL's 5 199 files in 14 s, every search timed live.

- [**Try it in your browser**](https://l-defraiteur.github.io/lucivy/) — the playground
- [Source on GitHub](https://github.com/L-Defraiteur/lucivy) (MIT)
- [The comparison with Elasticsearch and tantivy](https://github.com/L-Defraiteur/lucivy/blob/main/docs/compare-engines-2026-09-05.md), reproducible with one script
- [PyPI](https://pypi.org/project/lucivy/) · [npm](https://www.npmjs.com/package/lucivy) · [crates.io](https://crates.io/crates/lucivy-core) — 4.0.2
