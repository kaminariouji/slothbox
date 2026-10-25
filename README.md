# slothbox

Local-first semantic memory for a folder of notes. No cloud, no API keys, no database.

slothbox embeds your markdown and text files with an open-weight sentence model that runs **on your CPU**, stores the vectors in **one JSON file you own**, and answers questions with citations. After the model is cached once, it works with the network cable unplugged.

Built for the DEV Hacktoberfest Open-Source AI Challenge (Week 3: Slow and Steady).

## Why

Most "ask your notes" tools ship your notes to someone else's server. slothbox does the opposite:

- **Open weights.** `Xenova/all-MiniLM-L6-v2` (Apache-2.0), 384 dimensions, quantised to int8, ~23 MB.
- **Local inference.** ONNX Runtime through `@huggingface/transformers`. No account, no key, no rate limit.
- **One file of state.** `.slothbox/index.json`. Back it up with a copy. Delete it and nothing is lost that you cannot rebuild.
- **Zero marginal cost.** Embedding 1,000 passages costs electricity, not tokens.

## Slow and steady

The indexer is deliberately lazy:

- a file whose SHA-256 has not changed is **skipped** (no re-embedding, ever);
- progress is saved **after every file**, so Ctrl-C loses at most one file of work;
- `--pace 80` sleeps 80 ms between files so the index can crawl along in the background while you use the laptop;
- nothing is uploaded, so there is no quota to burn and no bill waiting at the end of the month.

## Install

```bash
npm install
```

The first run downloads the model into the Hugging Face cache (~23 MB). Every run after that is offline.

## Use

```bash
# build or refresh the index (only changed files are re-embedded)
node sloth.mjs index ./sample

# ask a question in plain language
node sloth.mjs ask "which lemon is sweeter for salad dressing" --root ./sample

# what is in the store?
node sloth.mjs status --root ./sample
```

Output:

```
[0.482] notes/citrus.md
Meyer lemons are sweeter than Eureka lemons and work well in dressings. Yuzu has a floral
aroma; a little goes a long way.
```

## Numbers from a laptop (CPU only)

| Step | Measured |
|---|---|
| Model load, first call | ~3.1 s |
| Embed 5 files / 5 passages | < 2 s |
| Store size, 5 passages | ~60 KB JSON |
| Network calls after first run | 0 |

## Limitations

- Passage-level retrieval only: it finds the right paragraph, it does not summarise a document.
- English-centric model; other languages work but rank worse.
- Cosine similarity over 384 dimensions is cheap and blunt. It will happily return a confident wrong answer when nothing in the corpus is relevant.
- No incremental vector database: the whole index is loaded into memory. Fine for thousands of notes, not for millions.
- No encryption at rest. The index is as private as the folder it sits in.

## License

MIT.
