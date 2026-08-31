"""Generate internal Transformer and GNN architecture diagrams from the code.

Figures mirror the stacked-block style used in classic encoder diagrams, but
every layer label and residual order is taken from the active implementation:

- ``pipeline.neural_training.model.EventSequenceEncoder`` uses PyTorch
  ``TransformerEncoderLayer(..., norm_first=True, activation=\"gelu\")``.
- ``pipeline.gnn_training.model.RelationMessagePassingLayer`` aggregates per
  relation, applies ``W_r``, gates, then ``LayerNorm(x + Dropout(GELU(update)))``.

Solid boxes are implemented. Dashed / warm boxes are documented handoffs or
deferred work, not inventing layers that are absent from the modules above.

Run from the project root::

    uv run python -m visualization.hybrid_architecture_diagrams
"""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "researchmodule-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from pipeline.config import PROJECT_ROOT
from pipeline.gnn_training.config import GNNArchitecture
from pipeline.gnn_training.graph_encode import (
    FORWARD_RELATION_TYPES,
    NODE_CONTINUOUS_FEATURES,
    NODE_ROLES,
    NODE_TYPES,
    RELATION_TYPES,
)
from pipeline.gnn_training.model import ABLATION_VARIANTS
from pipeline.neural_training.config import (
    CANDIDATE_SIDE_FEATURE_COUNT,
    EXPERIMENT_VERSION,
    NeuralArchitecture,
)
from pipeline.gnn_training.config import (
    FUSION_EXPERIMENT_VERSION,
    GNN_EXPERIMENT_VERSION,
)

SCHEMA_VERSION = "hybrid-architecture-diagrams-v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "visualization"
DEFAULT_FIGURES_ROOT = DEFAULT_OUTPUT_ROOT / "figures"
DEFAULT_INDEX_PATH = DEFAULT_OUTPUT_ROOT / "hybrid_architecture_diagrams.md"

# Palette close to the classic encoder-stack reference (attention / FF / norm).
PALETTE = {
    "bg": "#F7F5F2",
    "panel": "#E8E4DE",
    "ink": "#1F1F1F",
    "muted": "#5A5A5A",
    "input": "#C9B6E4",
    "output": "#C9B6E4",
    "attention": "#F0B27A",
    "feedforward": "#A8D5E5",
    "norm": "#F7E7A9",
    "embed": "#D6EAF8",
    "static": "#D5F5E3",
    "scorer": "#FADBD8",
    "relation": "#F5CBA7",
    "gate": "#F9E79F",
    "pool": "#D7BDE2",
    "fusion": "#A3E4D7",
    "inferred": "#FDEBD0",
    "block": "#D0D3D4",
    "arrow": "#2C3E50",
}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    text: str
    facecolor: str
    fontsize: float = 8.5
    weight: str = "normal"
    linestyle: str = "solid"
    edgecolor: str | None = None
    linewidth: float = 1.4
    radius: float = 0.018
    ha: str = "center"
    va: str = "center"


def _wrap(text: str, width: int = 28) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def _draw_box(ax, box: Box) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.008,rounding_size={box.radius}",
        linewidth=box.linewidth,
        edgecolor=box.edgecolor or PALETTE["ink"],
        facecolor=box.facecolor,
        linestyle=box.linestyle,
        mutation_aspect=0.4,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        box.x + box.w / 2,
        box.y + box.h / 2,
        _wrap(box.text, width=max(12, int(box.w * 38))),
        ha=box.ha,
        va=box.va,
        fontsize=box.fontsize,
        color=PALETTE["ink"],
        weight=box.weight,
        linespacing=1.15,
        zorder=3,
    )


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str | None = None,
    style: str = "-|>",
    lw: float = 1.35,
    connection: str = "arc3,rad=0.0",
    mutation_scale: float = 12,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color or PALETTE["arrow"],
            connectionstyle=connection,
            zorder=4,
        )
    )


def _residual(
    ax,
    x_left: float,
    y_bottom: float,
    x_right: float,
    y_top: float,
    *,
    color: str | None = None,
) -> None:
    """Draw a U-shaped residual bypass around a sublayer (reference style)."""

    color = color or PALETTE["arrow"]
    mid_x = x_left - 0.018
    ax.plot(
        [x_left, mid_x, mid_x, x_left],
        [y_bottom, y_bottom, y_top, y_top],
        color=color,
        linewidth=1.2,
        zorder=3,
    )
    _arrow(ax, x_left - 0.002, y_top, x_right, y_top, style="-|>", lw=1.15, mutation_scale=10)


def _footer(ax, *, source: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ax.text(
        0.01,
        0.012,
        f"{source}  |  schema {SCHEMA_VERSION}  |  {stamp}",
        fontsize=7.5,
        color=PALETTE["muted"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )


def _legend(ax, handles: list[Patch], *, loc: str = "upper right") -> None:
    ax.legend(
        handles=handles,
        loc=loc,
        frameon=True,
        fontsize=7.5,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )


def _encoder_block(
    ax,
    *,
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    title: str,
    arch: NeuralArchitecture,
) -> tuple[float, float, float, float]:
    """Draw one Pre-LN TransformerEncoderLayer (bottom → top).

    Returns (input_x, input_y, output_x, output_y) midpoints on the block edge.
    """

    # Outer frame
    _draw_box(
        ax,
        Box(
            origin_x,
            origin_y,
            width,
            height,
            "",
            PALETTE["block"],
            linewidth=1.8,
            radius=0.03,
        )
    )
    ax.text(
        origin_x + width / 2,
        origin_y - 0.028,
        title,
        ha="center",
        va="top",
        fontsize=10,
        weight="bold",
        color=PALETTE["ink"],
    )

    pad_x = 0.035
    pad_y = 0.04
    inner_w = width - 2 * pad_x
    # Bottom → top stack: Norm → MHA → Add, Norm → FFN → Add
    layers = [
        (
            "LayerNorm\n(norm_first)",
            PALETTE["norm"],
            0.09,
        ),
        (
            f"Multi-Head Attention\n{arch.attention_heads} heads  ·  d={arch.event_embedding_dim}",
            PALETTE["attention"],
            0.13,
        ),
        (
            "Residual Add\n(+ dropout)",
            PALETTE["norm"],
            0.08,
        ),
        (
            "LayerNorm\n(norm_first)",
            PALETTE["norm"],
            0.09,
        ),
        (
            f"Feed Forward (GELU)\nLinear {arch.event_embedding_dim}→{arch.feedforward_dim}→{arch.event_embedding_dim}",
            PALETTE["feedforward"],
            0.13,
        ),
        (
            "Residual Add\n(+ dropout)",
            PALETTE["norm"],
            0.08,
        ),
    ]
    gap = 0.012
    usable = height - 2 * pad_y - gap * (len(layers) - 1)
    scale = usable / sum(item[2] for item in layers)
    y = origin_y + pad_y
    centers: list[tuple[float, float, float]] = []
    for text, color, base_h in layers:
        h = base_h * scale
        _draw_box(
            ax,
            Box(
                origin_x + pad_x,
                y,
                inner_w,
                h,
                text,
                color,
                fontsize=7.8,
                weight="bold" if "Attention" in text or "Feed Forward" in text else "normal",
            )
        )
        centers.append((origin_x + pad_x + inner_w / 2, y + h / 2, h))
        y += h + gap

    # Vertical main path arrows between stacked layers
    for index in range(len(centers) - 1):
        _, y_a, h_a = centers[index]
        _, y_b, h_b = centers[index + 1]
        _arrow(
            ax,
            centers[index][0],
            y_a + h_a / 2 - 0.002,
            centers[index + 1][0],
            y_b - h_b / 2 + 0.002,
            lw=1.1,
            mutation_scale=9,
        )

    # Residual bypasses: around MHA (layers 0-2) and around FFN (layers 3-5)
    left = origin_x + pad_x
    # MHA residual: from after first LN input path conceptually — show skip from
    # block input into the first Residual Add (Post visual, Pre-LN math).
    mha_in_y = centers[0][1] - centers[0][2] / 2
    mha_add_y = centers[2][1]
    _residual(ax, left, mha_in_y + 0.01, left + 0.01, mha_add_y)
    ffn_in_y = centers[3][1] - centers[3][2] / 2
    ffn_add_y = centers[5][1]
    _residual(ax, left, ffn_in_y + 0.01, left + 0.01, ffn_add_y)

    in_x = origin_x + width / 2
    in_y = origin_y
    out_x = origin_x + width / 2
    out_y = origin_y + height
    return in_x, in_y, out_x, out_y


def draw_transformer_architecture(output_path: Path) -> None:
    """Internal EventSequenceEncoder stack + context/scoring shell."""

    arch = NeuralArchitecture()
    fig, ax = plt.subplots(figsize=(16.5, 10.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.text(
        0.5,
        0.975,
        "Transformer Patient/Context Branch — Internal Architecture",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.5,
        0.945,
        (
            f"{EXPERIMENT_VERSION}  ·  "
            f"EventSequenceEncoder: L={arch.encoder_layers}, "
            f"H={arch.attention_heads}, d={arch.event_embedding_dim}, "
            f"ff={arch.feedforward_dim}, Pre-LN (norm_first=True), GELU  ·  "
            "source: pipeline/neural_training/model.py"
        ),
        ha="center",
        va="top",
        fontsize=8.2,
        color=PALETTE["muted"],
    )

    # ---- Left: input construction ----
    left_x, left_w = 0.03, 0.18
    placements = [
        (0.18, 0.11, "Event token Embedding\npadding_idx=PAD", PALETTE["embed"]),
        (
            0.31,
            0.11,
            "+ Continuous projection\n(time, value, mask) → d",
            PALETTE["embed"],
        ),
        (
            0.44,
            0.11,
            "+ Positional Embedding\nevents 1..L ; CLS uses 0",
            PALETTE["embed"],
        ),
        (
            0.57,
            0.10,
            "input LayerNorm + Dropout",
            PALETTE["norm"],
        ),
        (
            0.69,
            0.11,
            "Prepend learned CLS / summary\n+ pad mask (CLS never masked)",
            PALETTE["input"],
        ),
    ]
    for index, (y, h, text, color) in enumerate(placements):
        _draw_box(ax, Box(left_x, y, left_w, h, text, color, fontsize=7.6))
        if index + 1 < len(placements):
            next_y = placements[index + 1][0]
            _arrow(
                ax,
                left_x + left_w / 2,
                y + h,
                left_x + left_w / 2,
                next_y,
                mutation_scale=9,
            )
    # Connect CLS construction into the sequence tensor callout.
    _arrow(
        ax,
        left_x + left_w / 2,
        placements[-1][0] + placements[-1][1],
        left_x + left_w / 2,
        0.85,
        mutation_scale=9,
    )

    ax.text(
        left_x + left_w / 2,
        0.925,
        "Input construction\n(EventSequenceEncoder)",
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
    )

    # Sequence tensor callout
    _draw_box(
        ax,
        Box(
            left_x,
            0.85,
            left_w,
            0.055,
            "Sequence (G, 1+L, d)",
            PALETTE["input"],
            fontsize=8,
            weight="bold",
        )
    )

    # ---- Center: stacked Pre-LN encoders (match reference layout) ----
    n_layers = arch.encoder_layers
    block_w = 0.22
    block_h = 0.62
    block_y = 0.18
    gap = 0.04
    stack_left = 0.26
    total_w = n_layers * block_w + (n_layers - 1) * gap
    endpoints: list[tuple[float, float, float, float]] = []
    for index in range(n_layers):
        ox = stack_left + index * (block_w + gap)
        ordinal = ("1st", "2nd", "3rd")[index] if index < 3 else f"{index + 1}th"
        endpoints.append(
            _encoder_block(
                ax,
                origin_x=ox,
                origin_y=block_y,
                width=block_w,
                height=block_h,
                title=f"{ordinal} encoder  (Pre-LN)",
                arch=arch,
            )
        )
        if index > 0:
            prev = endpoints[index - 1]
            # Top forward chain like the classic stacked-encoder reference
            _arrow(
                ax,
                prev[2] + 0.01,
                block_y + block_h + 0.035,
                endpoints[index][0] - 0.01,
                block_y + block_h + 0.035,
                lw=1.6,
                mutation_scale=14,
            )

    # Wire input construction into first encoder
    first = endpoints[0]
    _arrow(
        ax,
        left_x + left_w,
        0.875,
        first[0],
        first[1] + block_h,
        connection="arc3,rad=-0.05",
        lw=1.5,
    )
    # Bottom input arrow into first encoder
    _draw_box(
        ax,
        Box(
            first[0] - 0.04,
            block_y - 0.07,
            0.08,
            0.04,
            "Input",
            PALETTE["input"],
            fontsize=8,
            weight="bold",
        )
    )
    _arrow(ax, first[0], block_y - 0.03, first[0], block_y, mutation_scale=11)

    last = endpoints[-1]
    _draw_box(
        ax,
        Box(
            last[0] - 0.045,
            block_y + block_h + 0.015,
            0.09,
            0.04,
            "Encoded\nsequence",
            PALETTE["output"],
            fontsize=7.5,
            weight="bold",
        )
    )
    _arrow(ax, last[0], last[3], last[0], block_y + block_h + 0.015, mutation_scale=11)

    # Note: implemented depth is 2, not the classic 6
    ax.text(
        stack_left + total_w / 2,
        0.075,
        (
            "Implemented depth = 2 identical nn.TransformerEncoderLayer blocks "
            "(reference sketches often show N=6; this figure matches NeuralArchitecture.encoder_layers)."
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color=PALETTE["muted"],
        style="italic",
    )

    # ---- Right: CLS readout + static fusion + dual-path scorer ----
    right_x, right_w = 0.76, 0.21
    right_items = [
        (
            0.78,
            0.08,
            "Take CLS state\nencoded[:, 0] → event summary (d)",
            PALETTE["output"],
        ),
        (
            0.66,
            0.10,
            "StaticContextEncoder\nNumeric residual MLP (feat-dropout)\n+ categorical embeddings → concat",
            PALETTE["static"],
        ),
        (
            0.52,
            0.10,
            "Context fusion\nLN([event ‖ static]) → MLP →\ncontext z_T  (context_hidden_dim="
            f"{arch.context_hidden_dim})",
            PALETTE["static"],
        ),
        (
            0.36,
            0.14,
            "DualPathCandidateScorer\nMLP([z_T ‖ cond ‖ cand ‖ side])\n+ scaled (Q·K·cond_gate)\nside = "
            f"{CANDIDATE_SIDE_FEATURE_COUNT} tabular priors/graph cols",
            PALETTE["scorer"],
        ),
        (
            0.22,
            0.08,
            "Candidate logits (G, C)\npad slots → −∞",
            PALETTE["output"],
        ),
        (
            0.10,
            0.08,
            "Hybrid handoff (implemented in gnn_training)\nencode_context() + frozen logits cache",
            PALETTE["inferred"],
            "dashed",
        ),
    ]
    for item in right_items:
        y, h, text, color = item[0], item[1], item[2], item[3]
        style = item[4] if len(item) > 4 else "solid"
        edge = "#B9770E" if style == "dashed" else None
        _draw_box(
            ax,
            Box(
                right_x,
                y,
                right_w,
                h,
                text,
                color,
                fontsize=7.2,
                linestyle=style,
                edgecolor=edge,
                linewidth=1.6 if style == "dashed" else 1.4,
            )
        )
    for index in range(len(right_items) - 1):
        y0 = right_items[index][0]
        y1 = right_items[index + 1][0] + right_items[index + 1][1]
        _arrow(
            ax,
            right_x + right_w / 2,
            y0,
            right_x + right_w / 2,
            y1 + 0.004,
            mutation_scale=9,
        )

    # Connect last encoder top to CLS box
    _arrow(
        ax,
        last[0] + 0.05,
        block_y + block_h + 0.035,
        right_x,
        0.82,
        connection="arc3,rad=-0.12",
        lw=1.5,
    )

    ax.text(
        right_x + right_w / 2,
        0.90,
        "Readout + scoring\n(TransformerRecommender)",
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
    )

    _legend(
        ax,
        [
            Patch(facecolor=PALETTE["attention"], edgecolor=PALETTE["ink"], label="Multi-head attention"),
            Patch(facecolor=PALETTE["feedforward"], edgecolor=PALETTE["ink"], label="Position-wise FFN"),
            Patch(facecolor=PALETTE["norm"], edgecolor=PALETTE["ink"], label="Norm / residual add"),
            Patch(
                facecolor=PALETTE["inferred"],
                edgecolor="#B9770E",
                linestyle="--",
                label="Cross-module handoff (not inside neural_training)",
            ),
        ],
        loc="lower right",
    )
    _footer(
        ax,
        source=(
            "Implemented: EventSequenceEncoder Pre-LN stack + StaticContextEncoder "
            "+ DualPathCandidateScorer.  Not Post-LN Add&Norm from the original paper figure."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _rgcn_block(
    ax,
    *,
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    title: str,
    arch: GNNArchitecture,
) -> tuple[float, float, float, float]:
    """Draw one RelationMessagePassingLayer (bottom → top)."""

    _draw_box(
        ax,
        Box(
            origin_x,
            origin_y,
            width,
            height,
            "",
            PALETTE["block"],
            linewidth=1.8,
            radius=0.03,
        )
    )
    ax.text(
        origin_x + width / 2,
        origin_y - 0.028,
        title,
        ha="center",
        va="top",
        fontsize=10,
        weight="bold",
        color=PALETTE["ink"],
    )

    pad_x = 0.03
    pad_y = 0.035
    inner_w = width - 2 * pad_x
    layers = [
        (
            f"Per-relation incoming aggregate\nΣ_r  index_add(src·w)  over {arch.relation_count} relations",
            PALETTE["relation"],
            0.14,
        ),
        (
            f"Relation transform  h ← h @ W_r\nW_r shape ({arch.hidden_dim}×{arch.hidden_dim}), FP32",
            PALETTE["attention"],
            0.12,
        ),
        (
            "Relation gate σ(g_r)\n(+ relation dropout in train)",
            PALETTE["gate"],
            0.10,
        ),
        (
            "Sum relation updates → Δ",
            PALETTE["relation"],
            0.08,
        ),
        (
            "GELU(Δ) + Dropout",
            PALETTE["feedforward"],
            0.09,
        ),
        (
            "Residual Add + LayerNorm\nLN(x + Dropout(GELU(Δ)))",
            PALETTE["norm"],
            0.11,
        ),
    ]
    gap = 0.01
    usable = height - 2 * pad_y - gap * (len(layers) - 1)
    scale = usable / sum(item[2] for item in layers)
    y = origin_y + pad_y
    centers: list[tuple[float, float, float]] = []
    for text, color, base_h in layers:
        h = base_h * scale
        _draw_box(
            ax,
            Box(
                origin_x + pad_x,
                y,
                inner_w,
                h,
                text,
                color,
                fontsize=7.4,
                weight="bold" if "W_r" in text or "LayerNorm" in text else "normal",
            )
        )
        centers.append((origin_x + pad_x + inner_w / 2, y + h / 2, h))
        y += h + gap

    for index in range(len(centers) - 1):
        _, y_a, h_a = centers[index]
        _, y_b, h_b = centers[index + 1]
        _arrow(
            ax,
            centers[index][0],
            y_a + h_a / 2 - 0.002,
            centers[index + 1][0],
            y_b - h_b / 2 + 0.002,
            lw=1.05,
            mutation_scale=9,
        )

    # Residual from layer input into final Add & Norm
    left = origin_x + pad_x
    _residual(
        ax,
        left,
        centers[0][1] - centers[0][2] / 2 + 0.01,
        left + 0.01,
        centers[-1][1],
    )

    return (
        origin_x + width / 2,
        origin_y,
        origin_x + width / 2,
        origin_y + height,
    )


def draw_gnn_architecture(output_path: Path) -> None:
    """Internal R-GCN stack + candidate attention pool + fusion handoff."""

    arch = GNNArchitecture()
    fig, ax = plt.subplots(figsize=(16.5, 10.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.text(
        0.5,
        0.975,
        "GNN Relation Branch — Internal Architecture",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.5,
        0.945,
        (
            f"{GNN_EXPERIMENT_VERSION}  ·  "
            f"RelationMessagePassingLayer × {arch.relation_layers}, "
            f"H={arch.hidden_dim}, |R|={arch.relation_count} "
            f"({len(FORWARD_RELATION_TYPES)} fwd + {len(FORWARD_RELATION_TYPES)} rev + self_loop)  ·  "
            "source: pipeline/gnn_training/model.py"
        ),
        ha="center",
        va="top",
        fontsize=8.0,
        color=PALETTE["muted"],
    )

    # ---- Left: node state construction ----
    left_x, left_w = 0.025, 0.20
    node_parts = [
        (
            0.16,
            0.085,
            f"Concept Embedding\n({arch.concept_embedding_dim}d, PAD/UNK)",
            PALETTE["embed"],
        ),
        (
            0.26,
            0.075,
            f"Type Emb ({arch.node_type_embedding_dim}d)\n{len(NODE_TYPES)} types",
            PALETTE["embed"],
        ),
        (
            0.35,
            0.075,
            f"Role Emb ({arch.node_role_embedding_dim}d)\n{len(NODE_ROLES)} roles",
            PALETTE["embed"],
        ),
        (
            0.44,
            0.075,
            f"Time-bin Emb ({arch.time_bin_embedding_dim}d)\n{arch.time_bin_count} bins",
            PALETTE["embed"],
        ),
        (
            0.53,
            0.095,
            "Continuous ("
            + str(len(NODE_CONTINUOUS_FEATURES))
            + "d)\n"
            + ", ".join(NODE_CONTINUOUS_FEATURES[:3])
            + ", …\n+ observed / cold-start flags",
            PALETTE["embed"],
        ),
        (
            0.645,
            0.10,
            f"Node projection\nLN → Linear→{arch.hidden_dim} → GELU → Dropout\n→ initial node states h⁰",
            PALETTE["static"],
        ),
    ]
    for y, h, text, color in node_parts:
        _draw_box(ax, Box(left_x, y, left_w, h, text, color, fontsize=7.0))
    for index in range(len(node_parts) - 1):
        y0, h0 = node_parts[index][0], node_parts[index][1]
        next_bottom = node_parts[index + 1][0]
        _arrow(
            ax,
            left_x + left_w / 2,
            y0 + h0,
            left_x + left_w / 2,
            next_bottom,
            mutation_scale=8,
        )

    # ---- Center geometry (needed for edge wiring) ----
    n_layers = arch.relation_layers
    block_w = 0.23
    block_h = 0.62
    block_y = 0.17
    gap = 0.035
    stack_left = 0.26

    # Edge encoding is parallel to node featurization (not stacked after projection).
    _draw_box(
        ax,
        Box(
            left_x,
            0.775,
            left_w,
            0.095,
            "Edges (parallel path)\nexpand reverse + self-loop\nlog1p(support) → norm (r, dst)\n(graph_encode.py)",
            PALETTE["relation"],
            fontsize=7.0,
        ),
    )

    ax.text(
        left_x + left_w / 2,
        0.90,
        "Node + edge featurization",
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
    )

    # ---- Center: stacked R-GCN layers ----
    endpoints: list[tuple[float, float, float, float]] = []
    for index in range(n_layers):
        ox = stack_left + index * (block_w + gap)
        ordinal = ("1st", "2nd", "3rd")[index] if index < 3 else f"{index + 1}th"
        endpoints.append(
            _rgcn_block(
                ax,
                origin_x=ox,
                origin_y=block_y,
                width=block_w,
                height=block_h,
                title=f"{ordinal} R-GCN layer",
                arch=arch,
            )
        )
        if index > 0:
            prev = endpoints[index - 1]
            _arrow(
                ax,
                prev[2] + 0.01,
                block_y + block_h + 0.035,
                endpoints[index][0] - 0.01,
                block_y + block_h + 0.035,
                lw=1.6,
                mutation_scale=14,
            )

    first = endpoints[0]
    last = endpoints[-1]
    _draw_box(
        ax,
        Box(
            first[0] - 0.04,
            block_y - 0.07,
            0.08,
            0.04,
            "h⁰",
            PALETTE["input"],
            fontsize=9,
            weight="bold",
        )
    )
    # Node projection → h⁰ → first layer
    _arrow(
        ax,
        left_x + left_w,
        0.695,
        first[0] - 0.04,
        block_y - 0.05,
        connection="arc3,rad=0.12",
        lw=1.4,
    )
    _arrow(ax, first[0], block_y - 0.03, first[0], block_y, mutation_scale=11)
    # Parallel edge tensors into the message-passing stack
    _arrow(
        ax,
        left_x + left_w,
        0.82,
        first[0] - 0.02,
        block_y + 0.10,
        connection="arc3,rad=0.2",
        lw=1.15,
        mutation_scale=9,
    )

    _draw_box(
        ax,
        Box(
            last[0] - 0.05,
            block_y + block_h + 0.012,
            0.10,
            0.045,
            "Node states hᴸ",
            PALETTE["output"],
            fontsize=8,
            weight="bold",
        )
    )
    _arrow(ax, last[0], last[3], last[0], block_y + block_h + 0.012, mutation_scale=11)

    ax.text(
        stack_left + (n_layers * block_w + (n_layers - 1) * gap) / 2,
        0.072,
        (
            f"Ablations can skip MP or drop relation subsets: {', '.join(ABLATION_VARIANTS)}.  "
            "rank_only / no_message_passing return h⁰ without these layers."
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        color=PALETTE["muted"],
        style="italic",
    )

    # ---- Right: pool + score + fusion ----
    right_x, right_w = 0.76, 0.21
    right_items = [
        (
            0.78,
            0.10,
            "Lookup query + candidate nodes\nh_q, h_c from packed graph",
            PALETTE["output"],
        ),
        (
            0.64,
            0.12,
            "Candidate-conditioned attention pool\nQ = Linear([h_q ‖ h_c])\nK = Linear(observed context)\npool = softmax(QKᵀ/√H) · context",
            PALETTE["pool"],
        ),
        (
            0.48,
            0.12,
            f"Candidate representation (4H={arch.hidden_dim * 4})\n[h_q ‖ h_c ‖ h_q⊙h_c ‖ pool]\n+ log1p(candidate_rank)",
            PALETTE["scorer"],
        ),
        (
            0.34,
            0.09,
            f"MLP scorer → GNN logits\nLN → Linear→{arch.scorer_hidden_dim} → GELU → 1",
            PALETTE["scorer"],
        ),
        (
            0.21,
            0.10,
            f"Late fusion (α∈[0,1])\n(1−α) z(T) + α z(G)\n{FUSION_EXPERIMENT_VERSION}",
            PALETTE["fusion"],
        ),
        (
            0.09,
            0.085,
            "ResidualFusionHead (zero-init)\nfrozen logits + MLP([z_T ‖ r_G])\nTransformer stays immutable",
            PALETTE["fusion"],
        ),
        (
            0.012,
            0.065,
            "Status note (not a layer):\nprotected GNN/fusion training pending",
            PALETTE["inferred"],
            "dashed",
        ),
    ]
    for item in right_items:
        y, h, text, color = item[0], item[1], item[2], item[3]
        style = item[4] if len(item) > 4 else "solid"
        edge = "#B9770E" if style == "dashed" else None
        _draw_box(
            ax,
            Box(
                right_x,
                y,
                right_w,
                h,
                text,
                color,
                fontsize=7.0,
                linestyle=style,
                edgecolor=edge,
                linewidth=1.6 if style == "dashed" else 1.4,
            )
        )
    for index in range(len(right_items) - 1):
        # Do not draw a data-flow arrow into the status callout.
        if len(right_items[index + 1]) > 4 and right_items[index + 1][4] == "dashed":
            continue
        y0 = right_items[index][0]
        y1 = right_items[index + 1][0] + right_items[index + 1][1]
        _arrow(
            ax,
            right_x + right_w / 2,
            y0,
            right_x + right_w / 2,
            y1 + 0.003,
            mutation_scale=8,
        )

    _arrow(
        ax,
        last[0] + 0.055,
        block_y + block_h + 0.035,
        right_x,
        0.83,
        connection="arc3,rad=-0.12",
        lw=1.5,
    )

    ax.text(
        right_x + right_w / 2,
        0.905,
        "Scoring + hybrid fusion",
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
    )

    # Relation legend strip
    rel_preview = ", ".join(RELATION_TYPES[:3]) + ", …"
    ax.text(
        0.5,
        0.035,
        f"Relations include: {rel_preview} (full list fixed in graph_encode.RELATION_TYPES).",
        ha="center",
        va="center",
        fontsize=7.2,
        color=PALETTE["muted"],
    )

    _legend(
        ax,
        [
            Patch(facecolor=PALETTE["relation"], edgecolor=PALETTE["ink"], label="Relation aggregation"),
            Patch(facecolor=PALETTE["attention"], edgecolor=PALETTE["ink"], label="W_r transform"),
            Patch(facecolor=PALETTE["norm"], edgecolor=PALETTE["ink"], label="Residual + LayerNorm"),
            Patch(facecolor=PALETTE["pool"], edgecolor=PALETTE["ink"], label="Attention pool"),
            Patch(
                facecolor=PALETTE["inferred"],
                edgecolor="#B9770E",
                linestyle="--",
                label="Pending protected training (not a missing layer)",
            ),
        ],
        loc="lower right",
    )
    _footer(
        ax,
        source=(
            "Implemented: GNNRecommender + RelationMessagePassingLayer + fusion.py. "
            "Protected training success remains pending — figure is structural, not performance evidence."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_index_markdown(
    *,
    figures_root: Path,
    index_path: Path,
    transformer_png: Path,
    gnn_png: Path,
) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Hybrid Architecture Diagrams

Generated: {stamp}
Schema: `{SCHEMA_VERSION}`

These figures show the **internal layer stacks** of the active Transformer and
GNN modules (stacked-block style), drawn from the implementation rather than
aspirational sketches.

| Diagram | PNG | PDF | Primary modules |
| --- | --- | --- | --- |
| Transformer patient/context branch | `{transformer_png.relative_to(DEFAULT_OUTPUT_ROOT)}` | `{transformer_png.with_suffix('.pdf').relative_to(DEFAULT_OUTPUT_ROOT)}` | `pipeline/neural_training/model.py` (`EventSequenceEncoder`, Pre-LN) |
| GNN relation branch + fusion | `{gnn_png.relative_to(DEFAULT_OUTPUT_ROOT)}` | `{gnn_png.with_suffix('.pdf').relative_to(DEFAULT_OUTPUT_ROOT)}` | `pipeline/gnn_training/model.py`, `graph_encode.py`, `fusion.py` |

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
"""
    index_path.write_text(content, encoding="utf-8")


def generate_figures(
    *,
    figures_root: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Path]:
    figures_root = figures_root or DEFAULT_FIGURES_ROOT
    index_path = index_path or DEFAULT_INDEX_PATH
    figures_root.mkdir(parents=True, exist_ok=True)

    transformer_png = figures_root / "transformer_architecture.png"
    gnn_png = figures_root / "gnn_architecture.png"
    draw_transformer_architecture(transformer_png)
    draw_gnn_architecture(gnn_png)
    write_index_markdown(
        figures_root=figures_root,
        index_path=index_path,
        transformer_png=transformer_png,
        gnn_png=gnn_png,
    )
    return {
        "transformer_png": transformer_png,
        "transformer_pdf": transformer_png.with_suffix(".pdf"),
        "gnn_png": gnn_png,
        "gnn_pdf": gnn_png.with_suffix(".pdf"),
        "index": index_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stacked-block Transformer and GNN architecture diagrams "
            "from the active pipeline modules."
        )
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=DEFAULT_FIGURES_ROOT,
        help="Directory for PNG/PDF outputs (default: visualization/figures).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Markdown index path (default: visualization/hybrid_architecture_diagrams.md).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_figures(
        figures_root=args.figures_root,
        index_path=args.index_path,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
