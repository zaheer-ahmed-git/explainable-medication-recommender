# Hybrid Architecture Diagrams

Generated: 2026-08-25 14:44 UTC
Schema: `hybrid-architecture-diagrams-v2`

These figures show the **internal layer stacks** of the active Transformer and
GNN modules (stacked-block style), drawn from the implementation rather than
aspirational sketches.

| Diagram | PNG | PDF | Primary modules |
| --- | --- | --- | --- |
| Transformer patient/context branch | `figures/transformer_architecture.png` | `figures/transformer_architecture.pdf` | `pipeline/neural_training/model.py` (`EventSequenceEncoder`, Pre-LN) |
| GNN relation branch + fusion | `figures/gnn_architecture.png` | `figures/gnn_architecture.pdf` | `pipeline/gnn_training/model.py`, `graph_encode.py`, `fusion.py` |

## How to regenerate

```bash
uv run python -m visualization.hybrid_architecture_diagrams
```

## Reading the legend

- **Solid borders**: components present in the active pipeline implementation.
- **Dashed / warm borders**: cross-module handoffs (Transformer diagram) or
  pending protected-training status notes (GNN diagram)—not invented layers.
- **Transformer stack**: PyTorch `TransformerEncoderLayer(..., norm_first=True)`
  — Pre-LN order (`Norm → Attn → Add`, `Norm → FFN → Add`), not the classic
  Post-LN “Add & Norm” paper figure.
- **GNN stack**: two `RelationMessagePassingLayer` blocks with per-relation
  `W_r`, sigmoid gates, and `LayerNorm(x + Dropout(GELU(Δ)))`.
- Late and residual fusion are both implemented in `pipeline.gnn_training.fusion`;
  protected training outcomes remain pending.

## Hybrid coupling (implemented)

1. Transformer trains alone (`pipeline.neural_training`) and exports a frozen
   checkpoint plus `encode_context()` vectors.
2. GNN trains on patient query subgraphs (`pipeline.gnn_training`) with its own
   R-GCN-style encoder.
3. Fusion keeps the Transformer immutable and either (a) late-fuses z-scored
   logits with a constrained α, or (b) adds a zero-initialized residual head
   over `[transformer_context ‖ gnn_candidate_representation]`.

Protected GNN/fusion training success remains pending; do not treat these
diagrams as evidence of clinical performance.
