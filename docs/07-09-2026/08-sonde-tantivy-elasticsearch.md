# Sonde du 7 septembre — où tantivy et Elasticsearch trébuchent vraiment

Question de Lucie : « `ust_lan` de `rust_lang`, il trouvera pas, peu importe la
query ou l'index, et pire en fuzzy, et les emojis même pas la peine ». Vérifié
plutôt que supposé, sur le noyau (93 983 fichiers), avec les index du banc.

## Résultat

- **Faux pour les index en trigrammes.** Elasticsearch (`match_phrase` sur le
  champ trigrammes, `token_chars: []`) et tantivy (`NgramTokenizer` + vérification
  sur le texte stocké) trouvent **tout ce qui fait trois caractères ou plus**,
  quel que soit le contenu : `pin_loc` (l'équivalent noyau de `ust_lan`) est
  6 591 chez les trois, `Müller`, `naïve`, `->next`, `int i`, `#include <linux/`
  pareil. Ça ne tombe qu'avec leur **tokenizer par défaut** (termes entiers),
  ce que le rapport disait déjà.
- **Vrai en dessous de trois caractères** : `©` (1 878 fichiers) → 0, `→` (34)
  → 0, silencieusement, comme `de`. Un emoji seul est un caractère : même trou.
  Le noyau n'a aucun emoji, donc rien mesuré dessus (`grep -rlP '\xF0\x9F'` : 0).
- **Le prix de tantivy** : correct par construction mais il relit les candidats ;
  `#include <linux/` garde 40 741 documents après le AND des trigrammes, tous
  relus, 455 ms (Elasticsearch 55, lucivy 72). Un balayage des textes déjà en
  mémoire et en minuscules fait 24-66 ms — le chemin vérifié est un scan déguisé.
- **lucivy** : 12/12 exacts, positions comprises, 5-72 ms sauf `de` (565 ms,
  7,7 M de positions).

Tableau complet : `docs/compare-engines-2026-09-05.md` §3 bis. L'article a
gagné la ligne `©` et une phrase honnête sur `pin_loc`.

## Commandes

```bash
# tantivy (bâtit /tmp/tv_ngram, ~1 min ; CMP_REUSE=1 pour le réutiliser)
CMP_CORPUS=/tmp/lucivy-cmp-90k CMP_PROBE='de|©|pin_loc' cargo test --release -p lucivy-core \
    --test compare_tantivy probe_substrings -- --ignored --nocapture
# Elasticsearch (conteneur lucivy-es, index cmp_ngram encore présent)
curl -s localhost:9200/cmp_ngram/_search -H 'Content-Type: application/json' \
    -d '{"size":0,"track_total_hits":true,"query":{"match_phrase":{"body":"pin_loc"}}}'
# lucivy (index dictionnaire du noyau dans /tmp/lucivy-compare/dict, 107,9 s à bâtir)
V3_CORPUS=/tmp/lucivy-cmp-90k V3_MAX_DOCS=1000000 V3_COMMIT_EVERY=10000 V3_SFX_VERSION=4 \
    V3_INDEX_DIR=/tmp/lucivy-compare/dict LUCIVY_HIGHLIGHT_SPAN_CAP=0 \
    V3_QUERIES='©:strict,pin_loc:strict,int\si:strict' cargo test --release -p lucivy-core \
    --test test_sfx_v3_ground_truth v3_ground_truth_demo -- --ignored --nocapture
```

## Leçon

Chercher où un concurrent tombe se fait en le sondant, pas en le supposant :
l'intuition « il ne trouvera pas `ust_lan` » était vraie pour l'outil tel qu'il
vient et fausse pour l'outil configuré. Le tableau reste solide parce qu'il ne
contient que des lignes mesurées, et la ligne « `pin_loc` : tout le monde le
trouve » vaut autant que les zéros — elle coupe l'objection avant qu'on la fasse.
