# Chantier 4.1 — index sans positions (`positions: false`), spans par relecture

Branche `v4.1` depuis `main` (8 septembre 2026). Une option de création, jamais
le défaut ; le format reste 4.x (un index existant s'ouvre tel quel, un index
bâti avec l'option ne s'ouvre pas en 4.0.x — le contrat de `derived_in_ram`).

## 1. Ce qu'on vise, mesuré

Index du noyau (5 156 Mo, 857 Mo de texte, `docs/07-09-2026/09`) :

| | aujourd'hui | sans positions |
|---|---|---|
| `.sfxpost` | 771 Mo (167 M d'entrées) | 168 Mo (76 M de paires terme-doc + tf) |
| `.word_sfxpost` | 626 Mo | 128 Mo |
| `.posmap`, `.word_pos_map`, `.sibling_v3` | 1 667 Mo | 0 |
| reste (FST, `.termtexts`, `.gmap`, `store`) | 1 948 Mo | 1 948 Mo |
| **total** | **5 156 Mo, ×6,0** | **≈ 2 250 Mo, ×2,6** |

Le prix : tout ce qui a besoin d'une position (adjacence entre jetons, spans,
fuzzy, regex bornée) se vérifie en relisant le texte stocké des candidats.
`stored: true` devient obligatoire pour le champ.

## 2. Où les positions servent aujourd'hui (cartographie du 8 septembre)

- **Phase FST** (`plan.rs`, `fst_walk.rs`, `falling_walk_v3`, `cross_token_chain_v3`,
  `sibling_table` à la lecture) : **aucune position**. Elle produit des ordinaux et
  des chaînes d'ordinaux. Inchangée.
- **Résolution** (`resolve.rs`) : `resolve_single_v3` émet un `MatchV3` par
  occurrence (position = ancre du span) ; `resolve_chains_impl` vérifie
  l'adjacence stricte par `posmap.ordinal_at(doc, pos + 1)` — seule la première
  liste lit les postings, le reste lit `.posmap`.
- **Composite** : `find_multi_token_v3` (positions consécutives), les chaînes
  de trigrammes du fuzzy (`build_trigram_chains`, distances en positions),
  `rebuild_window_opts` (fenêtre rebâtie depuis `posmap` + `termtexts`, ancrée
  par un `byte_at`), `verify_candidates`.
- **Placement** : `orchestrator::place_spans` — le seul endroit qui produit des
  octets, deux `byte_at` par match. Le sink d'highlights ne voit que des octets.
- **BM25** : tf = nombre de `MatchV3` par document (donc des occurrences
  énumérées par position) ; df = `SfxPostReaderV2::doc_freq`, sans position.
- **Précédent doc-only** : `regex_verified.rs:161-181`, branche « motif non
  borné ou sans littéral » : documents candidats rebâtis entiers et balayés
  par `find_iter`, spans par `back[]`. Exact par construction. Sa source est
  `posmap` + `termtexts` ; elle devient le docstore.
- **Danger** : `derived_in_ram` rebâtit les dérivés *depuis les postings*
  (`derived.rs`). Sans positions il n'y a rien à rebâtir et rien à dériver : les
  deux options sont exclusives (refusées ensemble, comme `shared_dictionary`
  contredit par `sfx_version`).

## 3. Le design

**Layout.** `SFP6` : par ordinal, `(delta doc, tf)` en varint, blocs et points
de contrôle comme `SFP5` ; `WSP6` : idem pour les mots (plus de `first`/`last`,
plus de `tail_off`). Les lecteurs répondent `has_positions()` comme ils
répondent `has_byte_spans()`. Rien de dérivé n'est écrit ni rebâti
(`components_for` sans `posmap`, `word_pos_map`, `sibling_v3`). FST,
`.termtexts`, `.gmap`, docstore inchangés.

**Requête, trois régimes, tous branchés sur `reader.has_positions()`.**

1. *Un seul jeton, séparateurs stricts* (`contains "mutex_lock"`, mot entier,
   préfixe) : l'ensemble des documents = union des listes des ordinaux trouvés
   par la FST — **exact sans relecture**, compte en millisecondes ; tf = somme
   des tf des ordinaux ; spans par relecture des seuls documents qu'on
   affiche (le sink d'highlights sait déjà refaire une passe restreinte aux ids
   du top-k : `LUCIVY_HIGHLIGHT_SPAN_CAP` et sa relance).
2. *Chaînes* (séparateurs relâchés, plusieurs jetons, `find_multi_token_v3`) :
   candidats = intersection des listes des ordinaux de la chaîne — un
   sur-ensemble (comme le AND de trigrammes de tantivy, mais sur des jetons
   exacts, donc étroit) — puis **vérification par relecture** du texte stocké
   avec un apparieur en espace d'octets (le harnais en a un :
   `grep_spans` / séparateurs relâchés), qui rend compte, tf et spans exacts.
3. *Fuzzy et regex* : candidats par la FST comme aujourd'hui, vérification en
   espace d'octets par `fuzzy_spans` (déjà sans position) et `find_iter`
   (précédent ci-dessus), source = docstore.

Les matches sortent **déjà placés en octets** : `place_spans` et
`place_overlap_overflow` sont sautés. Le harnais vérifie les deux layouts avec
le même panel ; le contrat est le même, seul le temps des spans change.

**Coût attendu.** Comptes exacts et rapides pour le régime 1 (la majorité des
requêtes du panel) ; le régime 2 paie la relecture des candidats (ordre de
100 ms pour 5 000 documents, tantivy fait 96 sur 5 145) ; le régime 3 aussi.
Une requête sans highlights ne relit rien en régime 1.

## 4. Les étapes, chacune avec sa mesure

1. **Layout + plomberie** : `positions: bool` dans `SchemaConfig` →
   `IndexSettings` → `components_for` → écrivain (`sfx_dag_v3.rs`) → lecteurs
   (`has_positions`) ; `derived_in_ram` refusé avec ; `list_files_for` ;
   snapshot/sync suivent. Mesure : taille de l'index 10 000 et noyau dans les
   deux layouts, les lecteurs s'ouvrent, le panel refuse proprement (pas encore
   de chemin de requête) avec un message net.
2. **Régime 1** : ensembles de documents depuis les listes, tf depuis les
   postings, spans par relecture du top-k. Mesure : lignes strict/mot
   entier/préfixe du panel 10/10, temps de compte et temps des spans à part.
3. **Régime 2** : intersection + apparieur d'octets (déplacé du harnais dans
   le moteur, testé contre lui). Mesure : lignes relâchées et multi-jetons du
   panel, nombre de candidats relus par requête.
4. **Régime 3** : fuzzy et regex sur le docstore. Mesure : lignes fz1/fz2/rx.
5. **Le noyau entier**, `V3_POSITIONS=0` dans le harnais, A/B temps contre le
   layout par défaut, tableau dans le rapport.
6. Bindings (`positions` dans les quatre), README, CHANGELOG, playground
   (`?nopos`), et la comparaison mise à jour.

Règle du chantier : jamais un chiffre sans le panel vert à côté ; le layout par
défaut ne bouge pas d'un octet (les tests existants le prouvent).
