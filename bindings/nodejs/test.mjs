import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { Index } = require('./index.js');

import { tmpdir } from 'os';
import { join } from 'path';
import { mkdirSync, rmSync } from 'fs';

const testDir = join(tmpdir(), 'lucivy_node_test_' + Date.now());
mkdirSync(testDir, { recursive: true });

try {
    // 1. Create index
    const idx = Index.create(testDir, [
        { name: 'title', type: 'text' },
        { name: 'body', type: 'text' },
    ], 'english');

    console.log('Created index at:', idx.path);
    console.log('Schema:', idx.schema);

    // 2. Add documents
    idx.add(0, { title: 'Rust programming', body: 'Rust is a systems programming language' });
    idx.add(1, { title: 'Python scripting', body: 'Python is great for scripting and data science' });
    idx.add(2, { title: 'JavaScript everywhere', body: 'JavaScript runs in browsers and on servers with Node.js' });
    idx.commit();

    console.log('Num docs:', idx.numDocs);

    // 3. String search (contains_split on all text fields)
    console.log('\n--- String search: "rust programming" ---');
    const r1 = idx.search('rust programming');
    console.log(r1);

    // 4. Contains query with highlights
    console.log('\n--- Contains "script" with highlights ---');
    const r2 = idx.search(
        { type: 'contains', field: 'body', value: 'script' },
        { highlights: true }
    );
    console.log(JSON.stringify(r2, null, 2));

    // 5. Boolean query (composed of contains)
    console.log('\n--- Boolean: must contains "programming", must_not contains "python" ---');
    const r3 = idx.search({
        type: 'boolean',
        must: [{ type: 'contains', field: 'body', value: 'programming' }],
        must_not: [{ type: 'contains', field: 'body', value: 'python' }],
    });
    console.log(r3);

    // 6. Contains with fuzzy (distance option) — typo tolerance
    console.log('\n--- Contains fuzzy: "javascrip" (distance 2) ---');
    const r4 = idx.search({ type: 'contains', field: 'title', value: 'javascrip', distance: 2 });
    console.log(r4);

    // 7. Contains with regex option
    console.log('\n--- Contains regex: "program[a-z]+" ---');
    const r5 = idx.search({ type: 'contains', field: 'body', value: 'program[a-z]+', regex: true });
    console.log(r5);

    // 8. Contains multi-word (contains_split via string)
    console.log('\n--- Contains split: "systems language" ---');
    const r6 = idx.search('systems language');
    console.log(r6);

    // 9. Delete + update
    idx.delete(1);
    idx.update(2, { title: 'Node.js rocks', body: 'Node.js is JavaScript on the server side' });
    idx.commit();

    console.log('\nAfter delete+update, num docs:', idx.numDocs);
    const r7 = idx.search('node');
    console.log('Search "node":', r7);

    console.log('\nAll tests passed!');
} finally {
    rmSync(testDir, { recursive: true, force: true });
}
