<!-- Copie markdown de playground/blog/every-engine-lies-a-little.html (la page publiée, canonique). Pour dev.to : coller tel quel, canonical_url = la page. -->

# Every full-text engine lies a little

*Lucie Defraiteur · 7 September 2026 · what Elasticsearch, tantivy and my own engine really return on 93 983 Linux kernel files, judged by a byte-by-byte scan of the same files*

Here is a thing I did not expect to find: ask a search engine for the two letters `de` over the source of the Linux kernel, and two of the three engines I tested answer **0 documents**. Not an error, not a warning. Zero. The truth is 93 009 of 93 983 files.

I write a search engine, so I am not neutral, and I will say where it loses. But the point of this post is not my engine. It is the method: **never trust a count you have not checked against the files.** Once you have that habit, every engine turns out to lie a little, mine included until the test caught it.

## The setup

One corpus: a Linux kernel tree (7.2), 93 983 text files after filtering, 857 MB of text. Three engines, each configured the way its own documentation says to do substring search, not left at defaults that would make it a straw man:

- **Elasticsearch 8.19**, a trigram analyzer (min and max gram 3) on the content field, plus a `wildcard` field for regular expressions. The standard analyzer is measured too, for size.
- **tantivy 0.25**, upstream, with its `NgramTokenizer` (3 grams); its default tokenizer measured too. Its n-gram phrase query returns 0 because all n-gram positions are 0, so I run a boolean AND of the trigrams and re-read the stored text of each candidate to verify. That is the fairest thing I could do for it.
- **lucivy 4.0**, mine, a suffix FST over every token, defaults.

And one judge: a scan of the files. For every query, a small program reads each file and finds the matches with plain string operations or a regex, then counts the documents and, when it can, the byte offsets. An engine's answer is right when its documents *and its spans* match the scan. The scan takes seconds per query; that is fine, it only has to be right.

> Everything below is reproducible with one script in the repository, `benches/compare_engines.sh`, which writes [this report](https://github.com/L-Defraiteur/lucivy/blob/main/docs/compare-engines-2026-09-05.md). The exact index mappings and tokenizer settings are in it.

## Where everyone agrees

On a plain substring, all three agree with the scan to the document: `mutex_lock` is in 5 145 files, `spin_lock` in 6 569, `sched` in 9 289 (5 284 as a whole word: the difference is `sched_clock`, `schedule`, `sched_domain`). Good. That is the case everyone tests, and everyone passes it.

## Where they part

| asked | truth (the scan) | lucivy | Elasticsearch | tantivy |
|---|---|---|---|---|
| `spin_lock` with separators relaxed: also `spin lock`, `spin-lock`, `spinlock` | 9 552 | **9 552**, 23 ms | 6 577 — its trigrams carry the underscore, so `spin lock` and `spinlock` are other strings | 6 601 — relaxed is the only mode it has: the separator never enters its index, so `spin_lock` and `spinlock` are the same thing to it |
| `spinlokc`, two edits, the typo sitting across the token boundary | 10 034 | **10 034**, 148 ms | 3 549 — fuzziness compares whole terms | 6 557 — same |
| `spin_lock_[a-z]+`, a regular expression | 5 510 | **5 510**, 219 ms | 5 440 on the wildcard field, 480 ms — 70 short | 0 — the terms are already cut into grams |
| `de`, two characters | 93 009 | **93 009**, 7.7 million positions, 561 ms | 0, silently | 0, silently |
| `retur -ENOMEM`, a fuzzy phrase | 14 449 | **14 449**, 30 ms | 14 446 with `span_near`, 24 ms — it does this well | — |
| **where it matched**: every position of `mutex_lock` in its 5 145 files | 20 797 spans | **all 20 797**, 15 ms | `highlight` on the top 200 documents only: 179 ms | re-reading 5 145 stored texts: 96 ms |

Read the zeros again. They are the interesting part. An engine that returns 6 577 where the truth is 9 552 is wrong in a way you might notice. An engine that returns **0** for two characters looks like it worked and found nothing, and nobody notices, because who checks a zero?

The typo row is the same story. `spinlokc` is one transposition away from `spin_lock`, but the transposition straddles the underscore. Term-level fuzziness compares `spinlokc` to whole terms, and no single term is within two edits of it, so a third to two thirds of the documents are missing. The count is not zero, so it looks fine.

## Where they win, because they do

This is not a hit piece. Same corpus, same machine, idle:

| engine | index | × the text | indexing |
|---|---|---|---|
| Elasticsearch, standard analyzer | 781 MB | ×0.9 | 28 s |
| Elasticsearch, trigrams + `wildcard` | 3 082 MB | ×3.6 | 123 s |
| tantivy, default tokenizer | 612 MB | ×0.7 | **1 s** |
| tantivy, `NgramTokenizer` | 680 MB | ×0.8 | **5 s** |
| lucivy 4.0 | 4 926 MB | ×5.8 | 107 s |
| lucivy 4.0 with `derived_in_ram` | 3 335 MB | ×3.9 | 111 s |

tantivy indexes this corpus in one to five seconds. Mine takes a hundred. Its index is seven times smaller. On a whole-word query it answers in 0 ms where mine takes 27. Elasticsearch does the fuzzy phrase as well as I do. If your queries are whole words, use them and be happy. And note the difference of kind: Elasticsearch is a service you run next to your application; mine is a library that goes inside it. The index lives in your process, in your transaction if you plug your own storage, on your machine — nothing to deploy beside your service, nothing that leaves it, and the same engine runs in the browser with the data staying in the tab.

What my index buys with its size is the other half of the first table: the questions a trigram index cannot ask, and the exact position of every match delivered with the answer instead of recomputed for the top 200.

## The method is the point

I did not start with the comparison. I started with a test that indexes the kernel, runs a panel of queries, and fails if any count or any span disagrees with the scan. It runs on every release. It caught my own engine lying twice this summer: once dropping documents whose fuzzy match crossed a separator (3.0.2 to 3.0.6, a whole month), once returning a stale dictionary one run in three under a race. Both were invisible in the ordinary tests, which showed "20 hits" because 20 was the result limit.

If you take one thing from this post: put a scan of the files next to your engine, on a real corpus, and diff. It is a slow, dumb program. It is also the only one in the room that cannot be wrong about what is in the files.

```bash
git clone --depth=1 --branch v7.2 https://github.com/torvalds/linux /tmp/linux-bench
V3_CORPUS=/tmp/linux-bench V3_SFX_VERSION=4 cargo test --release -p lucivy-core \
    --test test_sfx_v3_ground_truth v3_ground_truth_demo -- --ignored --nocapture
```

## If you want to try the engine

lucivy is a Rust library, MIT, with Python, Node.js and C++ bindings, and it runs in the browser: [the playground](https://l-defraiteur.github.io/lucivy/) clones its own source from GitHub and indexes it in your tab in about three seconds, then you can type `index postgres` or `index linux` and search a whole tree with `--fuzzy`, `--regex`, `--strict`. The repository is [here](https://github.com/L-Defraiteur/lucivy). And if you know a configuration of Elasticsearch or tantivy that gets closer on any row above, tell me: it goes into the report with your name on the line.
