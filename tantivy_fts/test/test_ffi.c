/**
 * Native test for tantivy-fts C FFI.
 *
 * Tests the full cycle: create → add → commit → reload → search → filtered search → delete → close.
 * Also tests: reopen, num_docs, get_schema.
 *
 * Build:
 *   cargo build --release  (in rust/)
 *   cc -o test_ffi test_ffi.c \
 *      -I../include \
 *      -L../rust/target/release -ltantivy_fts \
 *      -lpthread -lm -ldl
 *
 * Run:
 *   ./test_ffi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tantivy_fts.h"

#define TEST_DIR "/tmp/tantivy_fts_test"

/* Colors for terminal output */
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

static void check(int condition, const char *name) {
    if (condition) {
        printf(GREEN "  PASS" RESET " %s\n", name);
        tests_passed++;
    } else {
        printf(RED "  FAIL" RESET " %s\n", name);
        tests_failed++;
    }
}

/* Remove test directory (best effort) */
static void cleanup(void) {
    system("rm -rf " TEST_DIR);
}

int main(void) {
    cleanup();

    printf("\n=== tantivy-fts C FFI test ===\n\n");

    /* ── 1. Create index ──────────────────────────────────────── */
    printf("-- Create index --\n");

    const char *schema = "{"
        "\"fields\": ["
            "{\"name\": \"title\", \"type\": \"text\", \"stored\": true},"
            "{\"name\": \"body\",  \"type\": \"text\", \"stored\": true}"
        "],"
        "\"stemmer\": \"english\""
    "}";

    TantivyHandlePtr h = tantivy_create_index(TEST_DIR, schema);
    check(h != NULL, "tantivy_create_index returns non-null handle");

    /* ── 2. Get schema ────────────────────────────────────────── */
    printf("\n-- Schema --\n");

    char *schema_out = tantivy_get_schema(h);
    check(schema_out != NULL, "tantivy_get_schema returns non-null");
    if (schema_out) {
        check(strstr(schema_out, "_node_id") != NULL, "schema contains _node_id field");
        check(strstr(schema_out, "title") != NULL, "schema contains title field");
        check(strstr(schema_out, "body") != NULL, "schema contains body field");
        /* Dual-field: ._raw counterparts should exist */
        check(strstr(schema_out, "title._raw") != NULL, "schema contains title._raw field");
        check(strstr(schema_out, "body._raw") != NULL, "schema contains body._raw field");
        tantivy_free_string(schema_out);
    }

    /* ── 3. Add documents ─────────────────────────────────────── */
    printf("\n-- Add documents --\n");

    /* Note: _node_id is included in each document JSON for filtered search */
    const char *docs[] = {
        "{\"_node_id\": 100, \"title\": \"The quick brown fox\",  \"body\": \"A fox jumped over the lazy dog\"}",
        "{\"_node_id\": 200, \"title\": \"Rust programming\",     \"body\": \"Rust is a systems programming language\"}",
        "{\"_node_id\": 300, \"title\": \"Graph databases\",      \"body\": \"Rag3db is an embedded graph database\"}",
        "{\"_node_id\": 400, \"title\": \"Search engines\",       \"body\": \"Tantivy is a full-text search engine written in Rust\"}",
        "{\"_node_id\": 500, \"title\": \"The lazy dog sleeps\",  \"body\": \"The dog was too lazy to chase the fox\"}"
    };
    int ndocs = sizeof(docs) / sizeof(docs[0]);

    for (int i = 0; i < ndocs; i++) {
        int64_t opstamp = tantivy_add_document(h, docs[i]);
        check(opstamp >= 0, "tantivy_add_document succeeds");
    }

    /* ── 4. Commit ────────────────────────────────────────────── */
    printf("\n-- Commit --\n");

    int64_t commit_stamp = tantivy_commit(h);
    check(commit_stamp >= 0, "tantivy_commit succeeds");

    /* ── 5. Reload reader ─────────────────────────────────────── */
    tantivy_reload_reader(h);

    /* ── 6. Num docs ──────────────────────────────────────────── */
    printf("\n-- Num docs --\n");

    uint64_t num = tantivy_num_docs(h);
    check(num == 5, "tantivy_num_docs == 5");

    /* ── 7. Term search ───────────────────────────────────────── */
    printf("\n-- Term search --\n");

    char *results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"rust\"}", 10);
    check(results != NULL, "tantivy_search (term) returns non-null");
    if (results) {
        /* "rust" (lowercased by tokenizer) should match docs about Rust */
        check(strstr(results, "programming") != NULL, "term search finds 'rust' in body");
        check(strstr(results, "error") == NULL, "term search has no error");
        tantivy_free_string(results);
    }

    /* ── 8. Fuzzy search ──────────────────────────────────────── */
    printf("\n-- Fuzzy search --\n");

    results = tantivy_search(h,
        "{\"type\": \"fuzzy\", \"field\": \"body\", \"value\": \"rast\", \"distance\": 1}", 10);
    check(results != NULL, "tantivy_search (fuzzy) returns non-null");
    if (results) {
        /* "rast" with distance 1 should match "rust" */
        check(strstr(results, "Rust") != NULL || strstr(results, "rust") != NULL,
              "fuzzy search 'rast' (distance=1) finds 'rust'");
        tantivy_free_string(results);
    }

    /* ── 9. Phrase search ─────────────────────────────────────── */
    printf("\n-- Phrase search --\n");

    /* With dual-field, phrase query auto-tokenizes through the stemmed field.
       So "lazy dog" works directly — no need to pre-stem. */
    results = tantivy_search(h,
        "{\"type\": \"phrase\", \"field\": \"body\", \"terms\": [\"lazy\", \"dog\"]}", 10);
    check(results != NULL, "tantivy_search (phrase) returns non-null");
    if (results) {
        check(strstr(results, "error") == NULL, "phrase search has no error");
        check(strstr(results, "fox") != NULL || strstr(results, "dog") != NULL,
              "phrase search 'lazy dog' finds matching doc (auto-stemmed)");
        tantivy_free_string(results);
    }

    /* ── 10. Regex search ─────────────────────────────────────── */
    printf("\n-- Regex search --\n");

    results = tantivy_search(h,
        "{\"type\": \"regex\", \"field\": \"title\", \"pattern\": \".*program.*\"}", 10);
    check(results != NULL, "tantivy_search (regex) returns non-null");
    if (results) {
        check(strstr(results, "Rust") != NULL || strstr(results, "programming") != NULL,
              "regex search '.*program.*' finds programming doc");
        tantivy_free_string(results);
    }

    /* ── 11. Parse query ──────────────────────────────────────── */
    printf("\n-- Parse query --\n");

    results = tantivy_search(h,
        "{\"type\": \"parse\", \"fields\": [\"title\", \"body\"], \"value\": \"graph database\"}", 10);
    check(results != NULL, "tantivy_search (parse) returns non-null");
    if (results) {
        check(strstr(results, "Rag3db") != NULL || strstr(results, "graph") != NULL,
              "parse query 'graph database' finds relevant doc");
        tantivy_free_string(results);
    }

    /* ── 12. Boolean query ────────────────────────────────────── */
    printf("\n-- Boolean query --\n");

    results = tantivy_search(h,
        "{"
            "\"type\": \"boolean\","
            "\"must\": [{\"type\": \"term\", \"field\": \"body\", \"value\": \"rust\"}],"
            "\"must_not\": [{\"type\": \"term\", \"field\": \"title\", \"value\": \"search\"}]"
        "}", 10);
    check(results != NULL, "tantivy_search (boolean) returns non-null");
    if (results) {
        /* Must have "rust" in body but NOT "search" in title → only "Rust programming" */
        check(strstr(results, "programming") != NULL,
              "boolean query: must=rust, must_not=search → finds Rust programming");
        check(strstr(results, "Tantivy") == NULL,
              "boolean query: excludes 'Search engines' doc");
        tantivy_free_string(results);
    }

    /* ── 13. Dual-field stemming tests ────────────────────────── */
    printf("\n-- Dual-field stemming --\n");

    /* Term query: "running" should match "run" in stemmed index (auto-tokenized) */
    results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"running\"}", 10);
    check(results != NULL, "term 'running' returns non-null");
    if (results) {
        /* "running" → stemmed → "run", which matches "Rust is a systems programming language"
           where "programming" stems to "program", not "run". But doc 0 has "jumped" → "jump".
           Actually none have "run" in body... let's test with "programming" instead. */
        tantivy_free_string(results);
    }

    /* Term query is now exact match on raw field: "programs" ≠ "programming" */
    results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"programs\"}", 10);
    check(results != NULL, "term 'programs' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "term exact: 'programs' does NOT match 'programming' (different words)");
        tantivy_free_string(results);
    }

    /* But term "programming" matches exactly */
    results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"programming\"}", 10);
    check(results != NULL, "term 'programming' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "term exact: 'programming' matches exactly");
        tantivy_free_string(results);
    }

    /* Parse query uses stemmer: "programs" → "program" matches "programming" */
    results = tantivy_search(h,
        "{\"type\": \"parse\", \"fields\": [\"body\"], \"value\": \"programs\"}", 10);
    check(results != NULL, "parse 'programs' returns non-null");
    if (results) {
        check(strstr(results, "programming") != NULL,
              "parse stemmed: 'programs' matches 'programming' (both → program)");
        tantivy_free_string(results);
    }

    /* Fuzzy on raw field: "rast" distance 1 → "rust" (on raw, not stemmed) */
    results = tantivy_search(h,
        "{\"type\": \"fuzzy\", \"field\": \"body\", \"value\": \"programing\", \"distance\": 1}", 10);
    check(results != NULL, "fuzzy 'programing' returns non-null");
    if (results) {
        /* "programing" (typo) distance 1 from "programming" (raw field) → match */
        check(strstr(results, "Rust") != NULL || strstr(results, "programming") != NULL,
              "fuzzy on raw: 'programing' (typo) finds 'programming'");
        tantivy_free_string(results);
    }

    /* Regex on raw field: "program.*" matches full word forms (not stemmed) */
    results = tantivy_search(h,
        "{\"type\": \"regex\", \"field\": \"body\", \"pattern\": \"program.*\"}", 10);
    check(results != NULL, "regex 'program.*' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "regex on raw: 'program.*' matches 'programming'");
        tantivy_free_string(results);
    }

    /* Phrase with stemmed words: "lazy dog" should now work (auto-tokenized) */
    results = tantivy_search(h,
        "{\"type\": \"phrase\", \"field\": \"body\", \"terms\": [\"jumped\", \"over\"]}", 10);
    check(results != NULL, "phrase 'jumped over' returns non-null");
    if (results) {
        /* "jumped" → stemmed → "jump", "over" → "over". Both in doc 0 as consecutive terms. */
        check(strstr(results, "fox") != NULL,
              "phrase: 'jumped over' auto-stemmed finds fox doc");
        tantivy_free_string(results);
    }

    /* Parse query uses stemmer pipeline automatically */
    results = tantivy_search(h,
        "{\"type\": \"parse\", \"fields\": [\"body\"], \"value\": \"lazy dogs\"}", 10);
    check(results != NULL, "parse 'lazy dogs' returns non-null");
    if (results) {
        /* "lazy" → "lazi", "dogs" → "dog" — matches docs with "lazy" and "dog" */
        check(strstr(results, "score") != NULL,
              "parse: 'lazy dogs' finds results via stemming");
        tantivy_free_string(results);
    }

    /* ── 13b. Contains search ────────────────────────────────── */
    printf("\n-- Contains search --\n");

    /* "program" as substring: matches "programming" token */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"program\"}", 10);
    check(results != NULL, "contains 'program' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "contains: 'program' matches 'programming' (substring)");
        tantivy_free_string(results);
    }

    /* "engine" as substring: matches "engine" in doc 400 body */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"engine\"}", 10);
    check(results != NULL, "contains 'engine' returns non-null");
    if (results) {
        check(strstr(results, "Tantivy") != NULL || strstr(results, "search") != NULL,
              "contains: 'engine' matches search engine doc");
        tantivy_free_string(results);
    }

    /* Term "program" should NOT match (different token than "programming") */
    results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"program\"}", 10);
    check(results != NULL, "term 'program' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "term vs contains: 'program' does NOT match 'programming' with term (exact token)");
        tantivy_free_string(results);
    }

    /* Contains with regex metacharacters: "c++" should be escaped properly */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c++\"}", 10);
    check(results != NULL, "contains 'c++' returns non-null (regex escape)");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "contains: 'c++' finds nothing (no c++ in docs) but doesn't crash");
        tantivy_free_string(results);
    }

    /* Stress test: every regex metacharacter in one string */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"f(o)o.*+?[b]a{r}|^z$\\\\\"}", 10);
    check(results != NULL, "contains all-metachar nightmare returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "contains: all-metachar string doesn't crash (properly escaped)");
        tantivy_free_string(results);
    }

    /* Real stress: add a doc with regex metacharacters, then find it via contains */
    tantivy_add_document(h,
        "{\"_node_id\": 600, \"title\": \"Regex hell\","
        " \"body\": \"use std::vec<Option<Result<(i32,&str),Box<dyn Error+Send>>>>\"}" );
    tantivy_add_document(h,
        "{\"_node_id\": 700, \"title\": \"More regex\","
        " \"body\": \"pattern: foo(bar)?[0-9]{2,4}|baz.*qux$end\"}" );
    tantivy_add_document(h,
        "{\"_node_id\": 800, \"title\": \"C++ guide\","
        " \"body\": \"c++ and c# are popular languages\"}" );
    tantivy_add_document(h,
        "{\"_node_id\": 900, \"title\": \"Standard library\","
        " \"body\": \"use std::collections::HashMap in your code\"}" );
    tantivy_add_document(h,
        "{\"_node_id\": 1000, \"title\": \"Python paths\","
        " \"body\": \"import os.path.join for file operations\"}" );
    tantivy_commit(h);
    tantivy_reload_reader(h);

    /* Multi-token contains: tokenizer splits on non-alnum, then phrase query on raw.
       "option<result<(i32" → tokens ["option","result","i32"] → phrase match. */

    /* Nested generics: multi-token phrase on raw */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"option<result<(i32\"}", 10);
    check(results != NULL, "contains nested-generics returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "contains: 'option<result<(i32' finds generics doc (phrase on raw)");
        tantivy_free_string(results);
    }

    /* All regex metacharacters: multi-token phrase */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"(bar)?[0-9]{2,4}|baz.*qux$\"}", 10);
    check(results != NULL, "contains regex-in-a-regex returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "contains: '(bar)?[0-9]{2,4}|baz.*qux$' finds pattern doc (phrase on raw)");
        tantivy_free_string(results);
    }

    /* Plus sign is token separator: "error+send" → tokens ["error","send"] → phrase */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"error+send\"}", 10);
    check(results != NULL, "contains 'error+send' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "contains: 'error+send' finds doc (+ split → phrase [error,send])");
        tantivy_free_string(results);
    }

    /* Single-token contains still does regex substring */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"option\"}", 10);
    check(results != NULL, "contains 'option' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "contains: 'option' matches as substring (single token → regex)");
        tantivy_free_string(results);
    }

    /* ── 13c. Contains stress: separator validation ────────── */
    printf("\n-- Contains stress: separator validation --\n");

    /* "c++" should find doc 800 which has literal c++ */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c++\"}", 10);
    check(results != NULL, "stress: 'c++' with c++ doc returns non-null");
    if (results) {
        check(strstr(results, "popular") != NULL,
              "stress: 'c++' finds doc with actual c++ content");
        tantivy_free_string(results);
    }

    /* "c#" should find doc 800 */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c#\"}", 10);
    check(results != NULL, "stress: 'c#' returns non-null");
    if (results) {
        check(strstr(results, "popular") != NULL,
              "stress: 'c#' finds doc with c# content");
        tantivy_free_string(results);
    }

    /* "c--" should NOT match (suffix "--" vs "++" or "#", dist 2 > budget 1) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c--\"}", 10);
    check(results != NULL, "stress: 'c--' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'c--' rejected (suffix '--' too far from '++')");
        tantivy_free_string(results);
    }

    /* "std::collections" should match doc 900 (sep "::" matches exactly) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std::collections\"}", 10);
    check(results != NULL, "stress: 'std::collections' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "stress: 'std::collections' matches (sep '::' = '::')");
        tantivy_free_string(results);
    }

    /* "std..collections" should NOT match (sep ".." vs "::", dist 2 > budget 1) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std..collections\"}", 10);
    check(results != NULL, "stress: 'std..collections' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'std..collections' rejected (sep '..' != '::', dist 2)");
        tantivy_free_string(results);
    }

    /* "std collections" should NOT match (sep " " vs "::", dist 2 > budget 1) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std collections\"}", 10);
    check(results != NULL, "stress: 'std collections' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'std collections' rejected (sep ' ' != '::', dist 2)");
        tantivy_free_string(results);
    }

    /* "collections::hashmap" should match doc 900 */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"collections::hashmap\"}", 10);
    check(results != NULL, "stress: 'collections::hashmap' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "stress: 'collections::hashmap' matches (sep '::' = '::')");
        tantivy_free_string(results);
    }

    /* "os.path" should match doc 1000 (sep "." matches) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"os.path\"}", 10);
    check(results != NULL, "stress: 'os.path' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "stress: 'os.path' matches (sep '.' = '.')");
        tantivy_free_string(results);
    }

    /* "os::path" should NOT match (sep "::" vs ".", dist 2 > budget 1) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"os::path\"}", 10);
    check(results != NULL, "stress: 'os::path' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'os::path' rejected (sep '::' != '.', dist 2)");
        tantivy_free_string(results);
    }

    /* "path.join" should match doc 1000 */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"path.join\"}", 10);
    check(results != NULL, "stress: 'path.join' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "stress: 'path.join' matches (sep '.' = '.')");
        tantivy_free_string(results);
    }

    /* "os.path.join" 3-token chain with two separators "." */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"os.path.join\"}", 10);
    check(results != NULL, "stress: 'os.path.join' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "stress: 'os.path.join' matches 3-token chain (both seps '.')");
        tantivy_free_string(results);
    }

    /* "os.path::join" mixed seps: first "." OK, second "::" vs "." dist 2 → reject */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"os.path::join\"}", 10);
    check(results != NULL, "stress: 'os.path::join' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'os.path::join' rejected (second sep '::' != '.')");
        tantivy_free_string(results);
    }

    /* "error::send" should NOT match doc 600 (sep "::" vs "+", dist 2 > budget 1) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"error::send\"}", 10);
    check(results != NULL, "stress: 'error::send' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'error::send' rejected (sep '::' != '+', dist 2)");
        tantivy_free_string(results);
    }

    /* Fuzzy + correct separator: "std::collectons" (typo i→o) should match */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std::collectons\"}", 10);
    check(results != NULL, "stress: 'std::collectons' (typo) returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "stress: 'std::collectons' fuzzy match + sep '::' OK (budget 1)");
        tantivy_free_string(results);
    }

    /* Fuzzy + wrong separator: "std..collectons" should NOT match (fuzzy 1 + sep 2 = 3 > 1) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std..collectons\"}", 10);
    check(results != NULL, "stress: 'std..collectons' (typo+wrong sep) returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "stress: 'std..collectons' rejected (fuzzy 1 + sep dist 2 > budget 1)");
        tantivy_free_string(results);
    }

    /* ── 13d. Contains relaxed separators (strict_separators=false) ── */
    printf("\n-- Contains relaxed separators --\n");

    /* "c--" with strict=false → should match doc 800 (non-alnum suffix exists) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c--\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'c--' returns non-null");
    if (results) {
        check(strstr(results, "popular") != NULL,
              "relaxed: 'c--' matches c++ doc (non-alnum suffix exists)");
        tantivy_free_string(results);
    }

    /* "c++" with strict=false → still matches doc 800 */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c++\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'c++' returns non-null");
    if (results) {
        check(strstr(results, "popular") != NULL,
              "relaxed: 'c++' matches c++ doc");
        tantivy_free_string(results);
    }

    /* "std collections" with strict=false → matches doc 900 (sep '::' is non-alnum) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std collections\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'std collections' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "relaxed: 'std collections' matches (non-alnum sep exists in doc)");
        tantivy_free_string(results);
    }

    /* "os::path" with strict=false → matches doc 1000 (sep '.' is non-alnum) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"os::path\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'os::path' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "relaxed: 'os::path' matches (non-alnum sep exists in doc)");
        tantivy_free_string(results);
    }

    /* "os.path::join" with strict=false → matches doc 1000 (both seps are non-alnum) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"os.path::join\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'os.path::join' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "relaxed: 'os.path::join' matches (both seps non-alnum)");
        tantivy_free_string(results);
    }

    /* "error::send" with strict=false → matches doc 600 (sep '+' is non-alnum) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"error::send\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'error::send' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "relaxed: 'error::send' matches (non-alnum sep exists in doc)");
        tantivy_free_string(results);
    }

    /* "std..collectons" (typo+wrong sep) with strict=false → matches (fuzzy 1 OK + sep exists) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"std..collectons\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'std..collectons' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "relaxed: 'std..collectons' matches (fuzzy OK + sep exists)");
        tantivy_free_string(results);
    }

    /* Confirm strict=false still rejects when NO non-alnum separator:
       "program" single token, no suffix → should match as substring (no validation needed) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"program\","
        " \"strict_separators\": false}", 10);
    check(results != NULL, "relaxed: 'program' returns non-null");
    if (results) {
        check(strstr(results, "score") != NULL,
              "relaxed: 'program' still matches as substring");
        tantivy_free_string(results);
    }

    /* Confirm strict=true is still default (no strict_separators field) */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"c--\"}", 10);
    check(results != NULL, "default strict: 'c--' returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "default strict: 'c--' still rejected without strict_separators flag");
        tantivy_free_string(results);
    }

    /* ── 14. Filtered search ──────────────────────────────────── */
    printf("\n-- Filtered search --\n");

    /* Search for "rust" but only in nodes 100, 300 (fox and graph docs → no match) */
    uint64_t filter_miss[] = {100, 300};
    results = tantivy_search_filtered(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"rust\"}",
        10, filter_miss, 2);
    check(results != NULL, "tantivy_search_filtered returns non-null");
    if (results) {
        /* Node 200 (Rust programming) and 400 (search engine) have "rust", but they're excluded */
        check(strcmp(results, "[]") == 0, "filtered search: rust not in nodes {100,300} → empty");
        tantivy_free_string(results);
    }

    /* Search for "rust" but only in nodes 200, 400 (these have rust) */
    uint64_t filter_hit[] = {200, 400};
    results = tantivy_search_filtered(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"rust\"}",
        10, filter_hit, 2);
    check(results != NULL, "tantivy_search_filtered returns non-null (hit)");
    if (results) {
        check(strstr(results, "score") != NULL, "filtered search: rust in nodes {200,400} → results");
        tantivy_free_string(results);
    }

    /* Filtered search with single node */
    uint64_t filter_one[] = {400};
    results = tantivy_search_filtered(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"rust\"}",
        10, filter_one, 1);
    check(results != NULL, "tantivy_search_filtered single node returns non-null");
    if (results) {
        check(strstr(results, "Tantivy") != NULL || strstr(results, "search") != NULL,
              "filtered search: rust in node {400} → finds search engine doc");
        /* Should NOT contain the programming doc (node 200) */
        check(strstr(results, "programming language") == NULL,
              "filtered search: node {400} excludes node 200 doc");
        tantivy_free_string(results);
    }

    /* ── 15. Highlighting ─────────────────────────────────────── */
    printf("\n-- Highlighting --\n");

    /* Single-token contains with highlight: "engine" in doc 400 body
       "Tantivy is a full-text search engine written in Rust"
       "engine" = bytes [30, 36] */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"engine\", \"highlight\": true}",
        10);
    check(results != NULL, "highlight: contains 'engine' returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") != NULL,
              "highlight: response contains 'highlights' key");
        check(strstr(results, "[30,36]") != NULL,
              "highlight: 'engine' → byte range [30,36]");
        tantivy_free_string(results);
    }

    /* Substring match: "program" matches "programming" in doc 200
       "Rust is a systems programming language"
       "programming" = bytes [18, 29] */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"program\", \"highlight\": true}",
        10);
    check(results != NULL, "highlight: contains 'program' returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") != NULL,
              "highlight: 'program' response contains highlights");
        check(strstr(results, "[18,29]") != NULL,
              "highlight: 'program' → byte range [18,29] (from 'programming')");
        tantivy_free_string(results);
    }

    /* Multi-token contains: "search engine" in doc 400
       "search" = [23, 29], "engine" = [30, 36] */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"search engine\", \"highlight\": true}",
        10);
    check(results != NULL, "highlight: contains 'search engine' returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") != NULL,
              "highlight: 'search engine' response contains highlights");
        check(strstr(results, "[23,29]") != NULL,
              "highlight: 'search engine' → first token [23,29]");
        check(strstr(results, "[30,36]") != NULL,
              "highlight: 'search engine' → second token [30,36]");
        tantivy_free_string(results);
    }

    /* highlight:false → no highlights key */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"engine\", \"highlight\": false}",
        10);
    check(results != NULL, "highlight=false: returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") == NULL,
              "highlight=false: no highlights key in response");
        tantivy_free_string(results);
    }

    /* highlight absent → no highlights key */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"engine\"}",
        10);
    check(results != NULL, "no highlight param: returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") == NULL,
              "no highlight param: no highlights key in response");
        tantivy_free_string(results);
    }

    /* Term query with highlight: "rust" in doc 200 body
       "Rust is a systems programming language"
       "rust" = bytes [0, 4] */
    results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"body\", \"value\": \"rust\", \"highlight\": true}",
        10);
    check(results != NULL, "highlight on term query: returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") != NULL,
              "highlight on term query: has highlights");
        check(strstr(results, "[0,4]") != NULL,
              "highlight on term query: rust → [0,4]");
        tantivy_free_string(results);
    }

    /* Fuzzy query with highlight: "rrust" (distance 1) → matches "rust"
       doc 200 body: "Rust is a systems programming language" → [0, 4] */
    results = tantivy_search(h,
        "{\"type\": \"fuzzy\", \"field\": \"body\", \"value\": \"rrust\", \"distance\": 1, \"highlight\": true}",
        10);
    check(results != NULL, "highlight on fuzzy query: returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") != NULL,
              "highlight on fuzzy query: has highlights");
        check(strstr(results, "[0,4]") != NULL,
              "highlight on fuzzy query: rrust~1 → [0,4]");
        tantivy_free_string(results);
    }

    /* Phrase query with highlight: ["search", "engine"] on stemmed field
       doc 400 body: "Tantivy is a full-text search engine written in Rust"
       "search" = [22,28], "engine" = [29,35] (stemmed field offsets) */
    results = tantivy_search(h,
        "{\"type\": \"phrase\", \"field\": \"body\", \"terms\": [\"search\", \"engine\"], \"highlight\": true}",
        10);
    check(results != NULL, "highlight on phrase query: returns non-null");
    if (results) {
        check(strstr(results, "\"highlights\"") != NULL,
              "highlight on phrase query: has highlights");
        tantivy_free_string(results);
    }

    /* No match with highlight → empty array */
    results = tantivy_search(h,
        "{\"type\": \"contains\", \"field\": \"body\", \"value\": \"xyznonexistent\", \"highlight\": true}",
        10);
    check(results != NULL, "highlight no match: returns non-null");
    if (results) {
        check(strcmp(results, "[]") == 0,
              "highlight no match: empty array");
        tantivy_free_string(results);
    }

    /* ── 16. Delete by term ───────────────────────────────────── */
    printf("\n-- Delete --\n");

    int64_t del_stamp = tantivy_delete_by_term(h, "title", "rust");
    check(del_stamp >= 0, "tantivy_delete_by_term succeeds");

    commit_stamp = tantivy_commit(h);
    check(commit_stamp >= 0, "commit after delete succeeds");
    tantivy_reload_reader(h);

    num = tantivy_num_docs(h);
    check(num == 9, "num_docs == 9 after deleting 'Rust programming' (10 - 1)");

    /* Verify deleted doc is gone from search */
    results = tantivy_search(h,
        "{\"type\": \"term\", \"field\": \"title\", \"value\": \"rust\"}", 10);
    if (results) {
        check(strcmp(results, "[]") == 0, "deleted doc no longer found by search");
        tantivy_free_string(results);
    }

    /* ── 17. Close and reopen ─────────────────────────────────── */
    printf("\n-- Close and reopen --\n");

    tantivy_close_index(h);

    h = tantivy_open_index(TEST_DIR);
    check(h != NULL, "tantivy_open_index reopens successfully");

    if (h) {
        num = tantivy_num_docs(h);
        check(num == 9, "reopened index has 9 docs (delete persisted)");

        results = tantivy_search(h,
            "{\"type\": \"term\", \"field\": \"body\", \"value\": \"fox\"}", 10);
        if (results) {
            check(strstr(results, "fox") != NULL, "search after reopen finds 'fox'");
            tantivy_free_string(results);
        }

        tantivy_close_index(h);
    }

    /* ── 18. Error handling ───────────────────────────────────── */
    printf("\n-- Error handling --\n");

    check(tantivy_create_index(NULL, schema) == NULL, "null path → null handle");
    check(tantivy_create_index(TEST_DIR "/sub", NULL) == NULL, "null schema → null handle");
    check(tantivy_add_document(NULL, "{}") == -1, "null handle add → -1");
    check(tantivy_commit(NULL) == -1, "null handle commit → -1");

    char *err = tantivy_search(NULL, "{}", 10);
    if (err) {
        check(strstr(err, "error") != NULL, "null handle search → error json");
        tantivy_free_string(err);
    }

    /* ── Summary ──────────────────────────────────────────────── */
    cleanup();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
