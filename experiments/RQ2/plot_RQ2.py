#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUP_ORDER_SINGLE = ["Seen", "Level 1", "Level 2", "Level 3", "Level 4", "Unseen"]
GROUP_DISPLAY_LABELS = {
    "Seen": "Member",
    "v1": "Level 1",
    "v2": "Level 2",
    "v3": "Level 3",
    "v4": "Level 4",
    "Unseen": "Non-member",
}
FONT_SIZES = {
    "title": 26,
    "axis_label": 28,
    "tick_label": 28,
    "trend_value": 16,
    "legend": 20,
    "no_data": 26,
}
FIGURE_LAYOUT = {
    "fig_width": 18.0,
    "fig_height": 8.0,
    "tight_pad": 0.75,
    "tight_w_pad": 1.0,
    "tight_h_pad": 1.5,
    "subplot_wspace": 0.25,
    "subplot_hspace": 0.80,
}


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _display_groups(group_order: List[str]) -> List[str]:
    return [GROUP_DISPLAY_LABELS.get(g, g) for g in group_order]


def _label_type(label) -> int:
    if isinstance(label, list):
        values = [int(value) for value in label]
        if 2 in values:
            return 2
        if len(values) >= 2:
            if values[0] > 0 and values[1] > 0:
                return 1
            if values[0] == 0 and values[1] > 0:
                return 2
            return 0
        return 1 if values and values[0] > 0 else 0
    value = int(label)
    return 2 if value == 2 else int(value > 0)


def _old_to_new_index_map(question_id: int, num_choices: int) -> Dict[int, int]:
    old_indices = list(range(num_choices))
    random.Random(question_id).shuffle(old_indices)
    return {old_index: new_index for new_index, old_index in enumerate(old_indices)}


def _collect_scores(data: Dict, layer: int, independent_old_indices: List[int]):
    grouped_mem = {group: [] for group in GROUP_ORDER_SINGLE}
    grouped_syn = {group: [] for group in GROUP_ORDER_SINGLE}
    for item in data.get("results", []):
        if int(item.get("layer", 99999)) != layer:
            continue
        labels = item.get("correct_label_list", [])
        mem_scores = item.get("score_mem_X", [])
        syn_scores = item.get("score_syn_Y", [])
        count = min(len(labels), len(mem_scores), len(syn_scores))
        for index in range(count):
            label = _label_type(labels[index])
            group = "Seen" if label == 1 else "Unseen" if label == 0 else None
            if group:
                grouped_mem[group].append(float(mem_scores[index]))
                grouped_syn[group].append(float(syn_scores[index]))
        try:
            index_map = _old_to_new_index_map(int(item.get("question_id")), count)
        except (TypeError, ValueError):
            continue
        for level, old_index in enumerate(independent_old_indices, 1):
            new_index = index_map.get(old_index)
            if new_index is None or new_index >= count:
                continue
            group = f"Level {level}"
            grouped_mem[group].append(float(mem_scores[new_index]))
            grouped_syn[group].append(float(syn_scores[new_index]))
    return grouped_mem, grouped_syn


def _outlier_bounds(values: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    if values.size == 0 or args.outlier_filter == "none":
        return -np.inf, np.inf
    if args.outlier_filter == "quantile":
        return float(np.quantile(values, args.q_low)), float(np.quantile(values, args.q_high))
    if args.outlier_filter == "iqr":
        q1, q3 = np.quantile(values, [0.25, 0.75])
        return float(q1 - args.iqr_k * (q3 - q1)), float(q3 + args.iqr_k * (q3 - q1))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-12:
        return -np.inf, np.inf
    sigma = 1.4826 * mad
    return median - args.mad_k * sigma, median + args.mad_k * sigma


def _apply_outlier_filter_dict(grouped: Dict[str, List[float]], args: argparse.Namespace):
    arrays = {group: np.asarray(grouped[group], dtype=float) for group in GROUP_ORDER_SINGLE}
    if args.outlier_scope == "global":
        non_empty = [values for values in arrays.values() if values.size]
        all_values = np.concatenate(non_empty) if non_empty else np.array([], dtype=float)
        bounds = {group: _outlier_bounds(all_values, args) for group in GROUP_ORDER_SINGLE}
    else:
        bounds = {group: _outlier_bounds(values, args) for group, values in arrays.items()}
    filtered: Dict[str, List[float]] = {}
    for group, values in arrays.items():
        low, high = bounds[group]
        if args.outlier_action == "drop":
            values = values[(values >= low) & (values <= high)]
        else:
            values = np.clip(values, low, high)
        filtered[group] = values.tolist()
    return filtered


def _load_model_scores(
    lat_json: Path,
    independent_old_indices: List[int],
    filter_args: argparse.Namespace,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Optional[int]]:
    if not lat_json.is_file():
        empty = {g: [] for g in GROUP_ORDER_SINGLE}
        return empty, empty, None

    with lat_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    best_layer = data.get("linear_classifier", {}).get("best_layer", data.get("best_layer", -3))
    grouped_mem, grouped_syn = _collect_scores(data, int(best_layer), independent_old_indices)
    filtered_mem = _apply_outlier_filter_dict(grouped_mem, filter_args)
    filtered_syn = _apply_outlier_filter_dict(grouped_syn, filter_args)
    return filtered_mem, filtered_syn, int(best_layer)


def _plot_violin_on_ax(
    ax,
    group_order: List[str],
    group_to_scores: Dict[str, List[float]],
    title: str,
    ylabel: str,
    trend_stat: str,
) -> None:
    raw_data = [np.array(group_to_scores.get(g, []), dtype=float) for g in group_order]
    ax.set_facecolor("#F7FAFC")

    if all(arr.size == 0 for arr in raw_data):
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_SIZES["no_data"],
            fontweight="bold",
        )
        ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold", pad=13)
        ax.set_xticks(np.arange(1, len(group_order) + 1))
        ax.set_xticklabels(
            _display_groups(group_order),
            rotation=18,
            ha="right",
            fontsize=FONT_SIZES["tick_label"],
            fontweight="bold",
        )
        return

    positions = []
    data = []
    for idx, arr in enumerate(raw_data, 1):
        if arr.size > 0:
            positions.append(idx)
            data.append(arr)

    showmeans = trend_stat == "mean"
    showmedians = trend_stat == "median"
    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=showmeans,
        showmedians=showmedians,
        showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#6BAED6")
        pc.set_edgecolor("#6BAED6")
        pc.set_alpha(0.58)

    if trend_stat in {"mean", "median"}:
        x_all = np.arange(1, len(group_order) + 1, dtype=float)
        y_all = np.array(
            [
                float(np.mean(group_to_scores[g]))
                if trend_stat == "mean"
                else float(np.median(group_to_scores[g]))
                if group_to_scores[g]
                else np.nan
                for g in group_order
            ],
            dtype=float,
        )
        valid = np.isfinite(y_all)
        if np.any(valid):
            ax.plot(
                x_all[valid],
                y_all[valid],
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                color="#008B8B",
                label=f"{trend_stat} trend",
                zorder=5,
            )
            for xi, yi in zip(x_all[valid], y_all[valid]):
                ax.text(
                    xi,
                    yi,
                    f"{yi:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["trend_value"],
                    fontweight="bold",
                    color="#008B8B",
                )
            ax.legend(loc="best", framealpha=0.85, fontsize=FONT_SIZES["legend"])

    ax.set_xticks(np.arange(1, len(group_order) + 1))
    ax.set_xticklabels(
        _display_groups(group_order),
        rotation=18,
        ha="right",
        fontsize=FONT_SIZES["tick_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlim(0.5, len(group_order) + 0.5)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold", pad=13)
    ax.grid(axis="y", color="#E6E6E6", alpha=1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot RQ2 multi-model Mem/Syn score violins as a 2x3 grid."
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing results/{model}/{parent-prefix}.",
    )
    p.add_argument(
        "--parent-prefix",
        type=str,
        required=True,
        help="Suffix under each model result directory, e.g. stack2/RQ2.",
    )
    p.add_argument(
        "--single-run-subdir",
        type=str,
        default="single_run",
        help="Subdirectory containing LAT.json under each model result directory.",
    )
    p.add_argument("--models", type=str, required=True, help="Comma-separated model ids.")
    p.add_argument(
        "--model-labels",
        type=str,
        default="",
        help="Comma-separated display labels matching --models.",
    )
    p.add_argument(
        "--independent-old-indices",
        type=str,
        default="2,3,4,5",
        help="Old candidate indices corresponding to v1..v4.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Default: <results-root>/<parent-prefix>/single_run/rq2_score_violin_grid_2x3.png",
    )
    p.add_argument(
        "--violin-trend-stat",
        choices=["mean", "median", "none"],
        default="mean",
        help="Statistic used for trend line on violin plot.",
    )
    p.add_argument(
        "--outlier-filter",
        choices=["none", "quantile", "iqr", "mad"],
        default="quantile",
        help="Outlier filter method.",
    )
    p.add_argument(
        "--outlier-action",
        choices=["clip", "drop"],
        default="clip",
        help="How to handle outliers.",
    )
    p.add_argument(
        "--outlier-scope",
        choices=["global", "per-group"],
        default="global",
        help="Compute outlier bounds globally or per group.",
    )
    p.add_argument("--q-low", type=float, default=0.01, help="Lower quantile for quantile filter.")
    p.add_argument("--q-high", type=float, default=0.99, help="Upper quantile for quantile filter.")
    p.add_argument("--iqr-k", type=float, default=1.5, help="IQR multiplier.")
    p.add_argument("--mad-k", type=float, default=3.5, help="MAD multiplier.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.q_low < args.q_high <= 1.0):
        raise ValueError("Require 0 <= q-low < q-high <= 1.")

    models = _split_csv(args.models)
    labels = _split_csv(args.model_labels)
    if labels and len(labels) != len(models):
        raise ValueError("--model-labels must have the same length as --models.")
    if not labels:
        labels = [Path(m).name for m in models]
    if len(models) != 3:
        raise ValueError("This grid expects exactly 3 models for a 2x3 layout.")

    old_indices = [int(x.strip()) for x in args.independent_old_indices.split(",") if x.strip()]
    if len(old_indices) != 4:
        raise ValueError("--independent-old-indices must have exactly 4 entries for v1..v4.")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_LAYOUT["fig_width"], FIGURE_LAYOUT["fig_height"]),
        facecolor="#FFFFFF",
    )

    for col, (model, label) in enumerate(zip(models, labels)):
        lat_json = args.results_root / model / args.parent_prefix / args.single_run_subdir / "LAT.json"
        grouped_mem, grouped_syn, _best_layer = _load_model_scores(lat_json, old_indices, args)
        _plot_violin_on_ax(
            axes[0, col],
            GROUP_ORDER_SINGLE,
            grouped_mem,
            f"{label} Mem",
            "Mem Score" if col == 0 else "",
            args.violin_trend_stat,
        )
        _plot_violin_on_ax(
            axes[1, col],
            GROUP_ORDER_SINGLE,
            grouped_syn,
            f"{label} Syn",
            "Syn Score" if col == 0 else "",
            args.violin_trend_stat,
        )

    fig.tight_layout(
        pad=FIGURE_LAYOUT["tight_pad"],
        w_pad=FIGURE_LAYOUT["tight_w_pad"],
        h_pad=FIGURE_LAYOUT["tight_h_pad"],
    )
    fig.subplots_adjust(
        wspace=FIGURE_LAYOUT["subplot_wspace"],
        hspace=FIGURE_LAYOUT["subplot_hspace"],
    )

    if args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = (
            args.results_root
            / args.parent_prefix
            / args.single_run_subdir
            / "rq2_score_violin_grid_2x3.png"
        ).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[rq2-violin-grid] saved: {out_path}")


if __name__ == "__main__":
    main()
