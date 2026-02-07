# ld-tantivy

Fork de [tantivy](https://github.com/quickwit-oss/tantivy) (via [izihawa/tantivy](https://github.com/izihawa/tantivy) v0.26.0) avec trois extensions majeures pour la recherche par contenu : **ContainsQuery**, **byte offsets dans les postings**, et **validation des separateurs**.

```
quickwit-oss/tantivy v0.22
  -> izihawa/tantivy v0.26.0 (regex phrase queries, FST ameliorations)
    -> L-Defraiteur/tantivy (ce fork)
```

## Changements par rapport a upstream

**39 fichiers modifies, +1873 lignes, -59 lignes** par rapport a `izihawa/tantivy@main`.

### 1. ContainsQuery — recherche multi-strategie avec auto-cascade

Un nouveau type de query qui cherche des sous-chaines dans les termes indexes, avec fallback automatique par position :

1. **Exact** — lookup direct dans le dictionnaire de termes
2. **Fuzzy** — automate Levenshtein (distance configurable, defaut 1)
3. **Substring** — regex `.*token.*` sur le dictionnaire de termes

Early termination : des qu'un niveau trouve des matches pour une position, les niveaux inferieurs sont ignores.

Pour les queries multi-tokens (`"std::collections"`, `"os.path.join"`), le `PhraseScorer` verifie ensuite que les positions sont consecutives dans le document.

**Fichiers :**
- `src/query/phrase_query/automaton_phrase_query.rs` (162 lignes) — struct Query, constructeurs `new()` et `new_with_separators()`
- `src/query/phrase_query/automaton_phrase_weight.rs` (415 lignes) — Weight avec cascade, `CascadeLevel` enum, 6 tests unitaires

### 2. ContainsScorer — validation des separateurs et distance cumulative

Un scorer custom qui valide que les caracteres non-alphanumeriques (separateurs) entre les tokens de la query correspondent a ceux du document. C'est ce qui permet a `c++` de matcher uniquement les documents contenant `c++` et pas chaque occurrence du mot "c".

**Fonctionnement :**
- Charge le texte stocke du document
- Re-tokenise pour obtenir les byte offsets de chaque token
- Extrait les separateurs reels (texte entre `offset_to[token_i]` et `offset_from[token_i+1]`)
- Compare avec les separateurs de la query via distance d'edition (Levenshtein)
- Budget de distance cumulatif global : la somme des distances fuzzy des tokens + distances des separateurs doit rester dans le budget

**Deux modes de validation :**
- `strict_separators: true` (defaut) — les separateurs doivent correspondre exactement (avec budget edit distance). `c++` ne matche pas `c--` (distance 2 > budget 1)
- `strict_separators: false` — verifie seulement qu'un caractere non-alphanumerique existe entre les tokens. `c--` matche `c++`, `std collections` matche `std::collections`

**Contraintes aux bords :**
- Premier token : pas de contrainte sur ce qui precede dans le document (sauf si la query a des caracteres avant le premier token)
- Dernier token : idem pour ce qui suit

**Exemples de matches :**

| Query | Document | Resultat |
|-------|----------|----------|
| `c++` | `"c++ and c# are popular"` | match (separateurs `++` valides) |
| `c++` | `"the cat sat"` | rejet (pas de separateur non-alnum apres "c") |
| `std::collections` | `"use std::collections::HashMap"` | match (separateur `::` exact) |
| `os.path.join` | `"import os.path.join for files"` | match (separateurs `.` exacts) |
| `option<result<(i32` | `"Vec<Option<Result<(i32,&str)>>"` | match (separateurs `<`, `<(` valides) |

**Fichier :** `src/query/phrase_query/contains_scorer.rs` (605 lignes) — `ContainsScorer` (multi-token) + `ContainsSingleScorer` (single-token) + `edit_distance()` + `tokenize_raw()`

### 3. WithFreqsAndPositionsAndOffsets — byte offsets dans les postings

Nouveau variant de `IndexRecordOption` qui stocke les byte offsets (`offset_from`, `offset_to`) de chaque occurrence de token directement dans les postings, comme `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS` de Lucene.

```rust
pub enum IndexRecordOption {
    Basic,                              // doc IDs
    WithFreqs,                          // + term frequencies
    WithFreqsAndPositions,              // + positions de tokens
    WithFreqsAndPositionsAndOffsets,    // + byte offsets (NOUVEAU)
}
```

**Format de stockage :**
- Fichier `.offsets` separe (nouveau `SegmentComponent::Offsets`)
- CompositeFile par champ (meme architecture que `.pos`)
- Delta-encoding interleave : `(from_delta_0, to_delta_0, from_delta_1, to_delta_1, ...)` — 2 valeurs par token
- Bitpacked en blocks de 128 (reutilise `PositionSerializer`)
- `TermInfo` etendu : ajout `offsets_range: Range<usize>` (40 bytes au lieu de 28)

**Pipeline ecriture :**
```
Token (offset_from, offset_to, position)
  -> PostingsWriter::subscribe_with_offsets()
  -> TfPositionAndOffsetRecorder::record_position_with_offsets()
  -> FieldSerializer::write_doc_with_offsets()
  -> PositionSerializer (.offsets) — bitpacked blocks
```

**Pipeline lecture :**
```
InvertedIndexReader::read_postings_from_terminfo()
  -> PositionReader (.offsets)
  -> SegmentPostings::offsets() -> Vec<(u32, u32)>
```

**Propagation a travers les unions :**
- `SegmentPostings::append_offsets()` — lecture depuis le PositionReader
- `LoadedPostings::append_offsets()` — depuis les offsets charges en memoire
- `SimpleUnion::append_offsets()` — merge + sort + dedup des offsets de tous les docsets
- `BitSetPostingUnion::append_offsets()` — idem
- `PostingsWithOffset::append_offsets()` — delegation (byte offsets sont absolus)

**21 fichiers modifies** dans `src/schema/`, `src/postings/`, `src/index/`, `src/termdict/`, `src/query/`.

## Building

```bash
cargo test --lib    # 997 tests (7 ignored — compat format v6/v7)
```

## Utilisation avec tantivy_fts

Ce fork est utilise comme dependance de tantivy_fts, une crate FFI C qui expose la recherche full-text pour [rag3db](https://github.com/L-Defraiteur/rag3db).

```toml
[dependencies]
ld-tantivy = { path = "../ld-tantivy", features = ["stopwords", "lz4-compression", "stemmer"] }
```

## Lineage

- [quickwit-oss/tantivy](https://github.com/quickwit-oss/tantivy) — moteur de recherche full-text original en Rust
- [izihawa/tantivy](https://github.com/izihawa/tantivy) — fork v0.26.0 avec regex phrase queries, ameliorations FST
- **L-Defraiteur/tantivy** — ce fork : ContainsQuery, byte offsets, validation separateurs, distance cumulative

## License

MIT — meme licence que tantivy upstream.
