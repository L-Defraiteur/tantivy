# 01 — Bindings Node.js (napi-rs) + réorganisation bindings

Date : 7 mars 2026, 20h50

## Contexte

Lucivy est actuellement consommable depuis :
- **Rust** directement (crate `ld-lucivy` + `lucivy-fts`)
- **C++** via le bridge CXX dans `lucivy_fts/rust/src/bridge.rs` (extension rag3db)
- **Python** via PyO3 dans `lucivy/` (package pip `lucivy`)

Le binding Python est rangé en vrac à la racine du workspace (`ld-lucivy/lucivy/`).
Il n'y a pas de binding Node.js.

## Objectif

1. **Réorganiser** les bindings dans un dossier `bindings/` dédié.
2. **Créer un binding Node.js natif** via napi-rs pour publier un package npm.
3. Exposer la surface complète de l'API Lucivy (pas le legacy Tantivy).

## Réorganisation des bindings

### Avant

```
ld-lucivy/
├── lucivy_fts/rust/        ← bridge CXX (rag3db) + logique métier (handle, query)
├── lucivy/                 ← binding Python (PyO3) — en vrac
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/lib.rs
```

### Après

```
ld-lucivy/
├── lucivy_fts/rust/        ← inchangé
├── bindings/
│   ├── python/             ← déplacé depuis lucivy/
│   │   ├── Cargo.toml         (name = "lucivy-python")
│   │   ├── pyproject.toml
│   │   └── src/lib.rs
│   └── nodejs/             ← nouveau
│       ├── Cargo.toml         (name = "lucivy-napi")
│       ├── package.json
│       ├── index.js           (re-export JS)
│       ├── index.d.ts         (types TS)
│       └── src/lib.rs
```

Le workspace `Cargo.toml` racine sera mis à jour :
- Remplacer `"lucivy"` par `"bindings/python"` dans `[workspace] members`
- Ajouter `"bindings/nodejs"` dans `[workspace] members`

## Surface d'API Node.js

Le binding réutilise `lucivy_fts::handle::LucivyHandle` et `lucivy_fts::query::build_query`,
identique au binding Python. Pas d'exposition du QueryParser legacy de Tantivy.

### Classe `Index`

```typescript
class Index {
  // Lifecycle
  static create(path: string, fields: FieldDef[], stemmer?: string): Index
  static open(path: string): Index

  // CRUD
  add(docId: number, fields: Record<string, string | number>): void
  addMany(docs: Array<{ docId: number } & Record<string, string | number>>): void
  delete(docId: number): void
  update(docId: number, fields: Record<string, string | number>): void

  // Persistence
  commit(): void
  rollback(): void

  // Search
  search(query: string | QueryConfig, options?: SearchOptions): SearchResult[]

  // Getters
  get numDocs(): number
  get path(): string
  get schema(): FieldDef[]
}
```

### Types de query supportés

| Type | Champs requis | Description |
|------|---------------|-------------|
| `contains` | `field`, `value` | Substring fuzzy (trigram + Levenshtein). Options : `distance` (def 1), `regex: true`, `strictSeparators` |
| `boolean` | `must?`, `should?`, `mustNot?` | Composition de sous-queries |
| `term` | `field`, `value` | Token exact (sur champ raw, lowercase) |
| `fuzzy` | `field`, `value` | Levenshtein sur token. Option : `distance` (def 1) |
| `phrase` | `field`, `terms` | Phrase exacte (tokenisée, stemmée) |
| `regex` | `field`, `pattern` | Regex sur tokens raw |
| `parse` | `field`/`fields`, `value` | Query parser Tantivy (stemmed) |

String query → sucre pour `contains_split` sur tous les champs text (chaque mot = contains, combinés en boolean should).

### Filters (sur champs non-text)

Opérateurs : `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `in`, `notIn`, `between`, `startsWith`, `contains`.
Composites : `must`, `should`, `mustNot` (avec `clauses`).

### SearchResult

```typescript
interface SearchResult {
  docId: number
  score: number
  highlights?: Record<string, Array<[number, number]>>
}
```

### FieldDef

```typescript
interface FieldDef {
  name: string
  type: 'text' | 'string' | 'u64' | 'i64' | 'f64'
  stored?: boolean
  indexed?: boolean
  fast?: boolean
}
```

## Étapes d'implémentation

1. Créer `bindings/nodejs/` avec Cargo.toml, package.json, src/lib.rs
2. Implémenter le binding napi-rs (miroir de la structure Python)
3. Déplacer `lucivy/` → `bindings/python/` et mettre à jour les chemins
4. Mettre à jour le workspace Cargo.toml
5. Vérifier que tout compile (Python + Node.js + lucivy-fts)
6. Ajouter les types TypeScript (index.d.ts)

## Dépendances Rust (bindings/nodejs/Cargo.toml)

```toml
[dependencies]
lucivy-fts = { path = "../../lucivy_fts/rust" }
ld-lucivy = { path = "../..", default-features = false, features = ["stopwords", "lz4-compression", "stemmer"] }
napi = { version = "2", features = ["napi9", "serde-json"] }
napi-derive = "2"
serde_json = "1"
```

## WASM (hors scope pour l'instant)

Le WASM viendra dans un second temps. La question du stockage (IndexedDB/OPFS)
est le point bloquant principal et nécessite un adapter de directory Tantivy dédié.
La structure `bindings/wasm/` est réservée pour plus tard.
