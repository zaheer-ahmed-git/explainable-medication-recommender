# Phase 4-9 Visualization Pack

This folder contains an aggregate-only meeting visualization generator for the
current Phase 4 through Phase 9 work.

Run from the project root after the relevant `reports/*.json` manifests exist:

```bash
uv run python -m visualization.phase4_to_9
```

Outputs are written under `visualization/figures/` plus
`visualization/meeting_figure_pack.md` and
`visualization/meeting_figure_pack.json`. Those generated files are gitignored.

The generator reads aggregate report manifests only. It does not inspect raw
clinical rows, note text, patient identifiers, row-level scores, or local model
artifacts.

