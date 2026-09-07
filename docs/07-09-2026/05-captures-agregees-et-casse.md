# Captures agrégées et casse exacte — suggestions 4.1 (7 septembre 2026)

Deuxième chantier candidat pour 4.1, à côté de l'import
(`docs/06-09-2026/02-import-tantivy-elasticsearch.md`). Il vient d'un
scénario de Lucie : « je fais un commit pour un service, je ne suis plus sûre
que mon commit inclut de nouvelles variables d'environnement à rajouter dans
mon `.env` ».

## Ce que le moteur fait déjà (vérifié dans le code le 7 septembre)

- **`contains` est insensible à la casse.** La FST et la requête sont en
  minuscules ; la comparaison finale sur `.termtexts` se fait en minuscules
  (`composite.rs`, « termtexts keeps the original case; the FST and the query
  are lowercase »). Il n'existe pas d'option de casse sur `contains`.
- **La regex est sensible à la casse.** `regex_verified.rs` : les littéraux du
  motif sont extraits puis mis en minuscules pour trouver les candidats par
  `contains` ; le motif entier, parsé *case-sensitive* exprès (une HIR
  insensible à la casse ferait 64 variantes par littéral, 1,5 s de CPU mesurés
  sur rag3db), est ensuite vérifié par `find_iter` sur le texte d'origine.
  Les positions rendues sont celles du fichier. Donc `process\.env\.[A-Z_]+`
  ne rend que des majuscules, à l'octet près.
- **Une regex sans littéral** (`[A-Z_]{4,}` seul, `.*`) n'a rien pour
  localiser des candidats : tous les documents sont rebâtis et balayés une
  fois, et `query_warnings` le dit. Exact, au prix d'un scan.
- **Le pré-filtre `allowed_ids`** restreint tout ça aux fichiers d'un commit.

Le scénario tient donc aujourd'hui en quelques lignes de Python : une regex
par idiome (`process.env.`, `os.environ[`, `getenv(`, `env::var(`), chacune
avec son littéral donc rapide, restreinte aux ids des fichiers du commit, le
nom lu dans le span, l'ensemble comparé aux clés du `.env`.

## 1. Captures agrégées (la vraie feature)

**Quoi.** Une requête regex qui rend, au lieu de documents, **les valeurs
distinctes d'un groupe de capture** avec leur compte et leurs documents :

```json
{"type": "regex", "field": "content", "pattern": "process\\.env\\.([A-Z_][A-Z0-9_]+)",
 "capture": 1, "allowed_ids": [12, 40, 41]}
→ [{"value": "DATABASE_URL", "count": 7, "docs": [12, 40]}, {"value": "REDIS_TTL", "count": 1, "docs": [41]}]
```

**Pourquoi c'est peu de travail.** Tout existe : les spans exacts, le texte
des documents, le pré-filtre. C'est une couche au-dessus de `regex_verified`
qui, au lieu d'accumuler des `MatchV3`, applique `re.captures` sur chaque
fenêtre vérifiée et agrège dans un `HashMap<String, (count, Vec<DocId>)>`.
Ordre : par compte décroissant puis valeur. Plafond : `LUCIVY_MAX_MATCHES_PER_
SEGMENT` s'applique déjà, et la réponse dit si elle est tronquée
(`last_search_truncated`).

**Ce que ça sert, au-delà du `.env`** : « quels `#define` ce commit ajoute »,
« quelles routes ce service expose » (`app\.(get|post)\("([^"]+)"`), « quels
crates un workspace importe » (`^use ([a-z_]+)::`), « quelles clés de config
sont lues ». C'est une facette sur le match, ce qu'aucun des deux autres
moteurs ne propose sans post-traitement côté client.

**Interfaces.** `QueryConfig.capture: Option<usize>` ; Python `search_captures`,
Node `searchCaptures`, C++ `lucivy_search_captures`, terminal du playground
`--capture N`. Vérité terrain : la même agrégation faite par un `re.finditer`
sur les fichiers, comparée valeur par valeur et compte par compte.

## 2. Casse exacte sur `contains`

**Quoi.** `QueryConfig.case_sensitive: bool` (défaut `false`, inchangé).
Terminal : `--case`.

**Comment.** Rien ne change dans l'index (la FST reste en minuscules, une
seule clé par texte). La marche FST rend les candidats comme aujourd'hui ;
la comparaison finale sur `.termtexts`, qui garde la casse d'origine, se fait
sans `to_lowercase` quand l'option est posée. Pour les chemins fuzzy et
séparateurs relâchés, même règle au moment de la vérification du span.
Coût : nul à l'indexation, une comparaison de moins à la requête.

**Pourquoi.** `--strict --case "Config"` sans écrire une regex ; et la
cohérence avec la regex, qui est déjà sensible à la casse — aujourd'hui les
deux modes de la même page ne répondent pas pareil à `Sharded`.

## Ordre proposé

Casse d'abord (une journée, pas de format), captures ensuite (deux ou trois
jours avec les quatre bindings et la vérité terrain). Les deux avant l'import,
qui est plus long et dépend d'une décision de forme.
