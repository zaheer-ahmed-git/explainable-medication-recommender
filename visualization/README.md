# Visualization Pack

Generators under this folder draw aggregate-only figures for meetings and
architecture reviews. They do not read raw clinical rows.

## Hybrid Transformer / GNN internals

Stacked-block diagrams of the active model modules (Pre-LN Transformer encoder
stack and R-GCN message-passing layers):

```bash
uv run python -m visualization.hybrid_architecture_diagrams
```

Writes `figures/transformer_architecture.png` (and `.pdf`),
`figures/gnn_architecture.png` (and `.pdf`), plus
`hybrid_architecture_diagrams.md`. Generated PNGs/PDFs under `figures/` are
gitignored; regenerate locally when presenting.

## Phase 4–9 meeting pack

Aggregate report charts for milestones 4–9:

```bash
uv run python -m visualization.phase4_to_9
```

Requires the relevant `reports/*.json` manifests. Outputs land under
`figures/` plus `meeting_figure_pack.md` / `meeting_figure_pack.json`
(gitignored).
