// slothbox — local-first semantic memory
// slothbox — local-first semantic memory. No cloud, no API keys, no database.
// The model runs on your CPU via ONNX. The index is one JSON file you own.
import { pipeline, env } from '@huggingface/transformers';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join, extname, relative, resolve } from 'node:path';
import { createHash } from 'node:crypto';

env.allowRemoteModels = true; // only used on first run to cache the model
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const DIM = 384;
const EXTS = new Set(['.md', '.markdown', '.txt']);
const STORE_DIR = '.slothbox';

const args = process.argv.slice(2);
const cmd = args[0];
const opt = (name, def) => { const i = args.indexOf('--' + name); return i > -1 ? args[i + 1] : def; };

function storePath(root) { return join(root, STORE_DIR, 'index.json'); }

function loadStore(root) {
  const p = storePath(root);
  if (!existsSync(p)) return { model: MODEL, dim: DIM, files: {}, chunks: [] };
  return JSON.parse(readFileSync(p, 'utf8'));
}

function saveStore(root, store) {
  const dir = join(root, STORE_DIR);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(storePath(root), JSON.stringify(store));
}

function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    if (name === STORE_DIR || name === 'node_modules' || name.startsWith('.')) continue;
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, acc);
    else if (EXTS.has(extname(name).toLowerCase())) acc.push(full);
  }
  return acc;
}

function chunk(text, max = 900) {
  const paras = text.split(/\n\s*\n/).map(s => s.trim()).filter(Boolean);
  const out = [];
  let cur = '';
  for (const p of paras) {
    if ((cur + '\n\n' + p).length > max && cur) { out.push(cur); cur = p; }
    else cur = cur ? cur + '\n\n' + p : p;
  }
  if (cur) out.push(cur);
  return out;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function getEmbedder() {
  return pipeline('feature-extraction', MODEL, { dtype: 'q8' });
}

async function embedBatch(embed, texts) {
  const out = await embed(texts, { pooling: 'mean', normalize: true });
  const [n] = out.dims;
  const rows = [];
  for (let i = 0; i < n; i++) rows.push(Array.from(out.data.slice(i * DIM, (i + 1) * DIM)));
  return rows;
}

async function index(root, pace) {
  const store = loadStore(root);
  const files = walk(root);
  const embed = await getEmbedder();
  let added = 0, skipped = 0, chunksAdded = 0;
  for (const f of files) {
    const rel = relative(root, f).replace(/\\/g, '/');
    const buf = readFileSync(f);
    const hash = createHash('sha256').update(buf).digest('hex').slice(0, 16);
    if (store.files[rel] && store.files[rel].hash === hash) { skipped++; continue; }
    store.chunks = store.chunks.filter(c => c.path !== rel);
    const pieces = chunk(buf.toString('utf8'));
    for (let i = 0; i < pieces.length; i += 16) {
      const batch = pieces.slice(i, i + 16);
      const vecs = await embedBatch(embed, batch);
      batch.forEach((text, j) => store.chunks.push({ path: rel, text, vec: vecs[j].map(v => Math.round(v * 1e5) / 1e5) }));
      chunksAdded += batch.length;
    }
    store.files[rel] = { hash, chunks: pieces.length, indexedAt: new Date().toISOString() };
    saveStore(root, store); // resumable: crash here and nothing is lost
    added++;
    if (pace) await sleep(pace); // slow and steady
    process.stdout.write('.');
  }
  saveStore(root, store);
  console.log('\nindexed ' + added + ' file(s), skipped ' + skipped + ' unchanged, ' + chunksAdded + ' passage(s)');
  console.log('store: ' + storePath(root) + ' (' + store.chunks.length + ' passages total)');
}

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

async function ask(root, query, k) {
  const store = loadStore(root);
  if (!store.chunks.length) { console.error('nothing indexed yet — run: node sloth.mjs index <dir>'); process.exit(1); }
  const embed = await getEmbedder();
  const [q] = await embedBatch(embed, [query]);
  const scored = store.chunks.map(c => ({ ...c, score: cosine(q, c.vec) })).sort((x, y) => y.score - x.score).slice(0, k);
  for (const r of scored) {
    console.log('\n[' + r.score.toFixed(3) + '] ' + r.path);
    console.log(r.text.length > 320 ? r.text.slice(0, 320) + '…' : r.text);
  }
}

function status(root) {
  const store = loadStore(root);
  const files = Object.keys(store.files).length;
  console.log('model    ' + store.model + ' (' + store.dim + 'd, local)');
  console.log('files    ' + files);
  console.log('passages ' + store.chunks.length);
  const last = Object.values(store.files).map(f => f.indexedAt).sort().pop();
  console.log('last run ' + (last || 'never'));
  console.log('store    ' + storePath(root));
}

const root = resolve(opt('root', process.cwd()));
if (cmd === 'index') { const dir = args[1] ? resolve(args[1]) : root; await index(dir, Number(opt('pace', 0))); }
else if (cmd === 'ask') { await ask(root, args[1] || '', Number(opt('k', 5))); }
else if (cmd === 'status') { status(root); }
else { console.log('usage:\n  node sloth.mjs index <dir> [--pace 80]\n  node sloth.mjs ask "question" [--k 5]\n  node sloth.mjs status'); }
