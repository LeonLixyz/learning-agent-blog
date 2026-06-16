# Language Models Can Think, But Not Learn

An essay on the missing *write path* in large language models: why today's
models reason brilliantly over frozen weights but have no good way to write
what they figure out back into themselves.

**Thesis:** today's LLMs are trained to *think*, not to *learn*. "Observe, then
write it into the weights" is the missing operation, for fresh data and fresh
experience alike. The essay walks the problem, the mirages, how brains solve it,
and a way forward.

## Read it

- **Interactive version:** open [`index.html`](index.html) in a browser.
  Self-contained: rendered math (KaTeX from a CDN), animated and static diagrams,
  dark mode, a collapsible contents menu. This is the canonical reading version.
- **Plain-text version:** [`essay.md`](essay.md). The same essay as a single
  Markdown file (prose, figure callouts, references). Handy for quoting, posting,
  or reading without a browser.

## One source, two outputs

There is a **single source**: `build/assemble.py` (the prose) plus
`build/figs/` (one `<key>.html` + `<key>.css` per diagram). Running the build
generates **both** files:

```bash
python3 build/assemble.py   # writes ./index.html AND ./essay.md
```

`index.html` and `essay.md` are **generated artifacts. Do not hand-edit them** —
they get overwritten on every build. Edit the prose in `assemble.py` (the `BODY`
string) or a diagram in `build/figs/`, then rebuild. No dependencies beyond
Python 3.

## Layout

```
.
├── index.html         ← generated: the interactive site (do not hand-edit)
├── essay.md           ← generated: single-file Markdown (do not hand-edit)
├── README.md
├── build/             ← the source
│   ├── assemble.py        prose + the build (writes both outputs)
│   └── figs/              figure sources (<key>.html + <key>.css)
└── notes/             ← working material (not part of the published essay)
```
