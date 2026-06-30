#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RQ2_DIR = Path(__file__).resolve().parents[1] / "RQ2"
if str(RQ2_DIR) not in sys.path:
    sys.path.insert(0, str(RQ2_DIR))

from plot_RQ2 import (  # noqa: E402
    FONT_SIZES as RQ2_FONT_SIZES,
    GROUP_ORDER_SINGLE,
    _load_model_scores,
    _plot_violin_on_ax as _plot_rq2_violin_on_ax,
)


DEFAULT_METHODS = [
    "loss",
    "ppl",
    "zlib",
    "min_k",
    "confidence",
    "entropy",
    "msp",
    "margin",
]
SUPPORTED_METHODS = DEFAULT_METHODS
GROUP_DISPLAY_LABELS = {
    "seen": "Member",
    "v1": "Level 1",
    "v2": "Level 2",
    "v3": "Level 3",
    "v4": "Level 4",
    "unseen": "Non-member",
}
METHOD_DISPLAY_LABELS = {
    "confidence": "Confidence",
    "loss": "Loss",
    "entropy": "Entropy",
    "zlib": "Zlib",
    "margin": "Margin",
    "msp": "MP",
    "min_k": "Min-K",
    "ppl": "Perplexity",
}
FONT_SIZES = {
    "title": 22,
    "axis_label": 24,
    "tick_label": 22,
    "trend_value": 16,
    "legend": 16,
    "no_data": 20,
}
FIGURE_LAYOUT = {
    "fig_width": 27.0,
    "fig_height": 7.6,
    "tight_pad": 0.55,
    "tight_w_pad": 0.55,
    "tight_h_pad": 1.15,
    "subplot_wspace": 0.22,
    "subplot_hspace": 0.8,
}
RQ2_FONT_SIZES.update(FONT_SIZES)


def _parse_methods(value: str) -> List[str]:
    methods: List[str] = []
    for method in [x.strip() for x in value.split(",") if x.strip()]:
        methods.extend(DEFAULT_METHODS if method == "all" else [method])
    return list(dict.fromkeys(methods))


def _resolve_result_files(from_result: Path, methods: List[str]) -> Dict[str, Path]:
    path = from_result.resolve()
    result_files: Dict[str, Path] = {}
    if path.is_file():
        if path.name == "all_methods_trans_code_levels_results.json":
            with path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            return {name: Path(value).resolve() for name, value in summary.get("result_files", {}).items()}
        suffix = "_trans_code_levels_results.json"
        if path.name.endswith(suffix):
            return {path.name.removesuffix(suffix): path}
        raise FileNotFoundError(f"Unsupported RQ1 result file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"RQ1 result path not found: {path}")

    summary_path = path / "all_methods_trans_code_levels_results.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        for name, value in summary.get("result_files", {}).items():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = path / candidate.name
            if candidate.is_file():
                result_files[name] = candidate.resolve()
    for method in methods:
        candidate = path / f"{method}_trans_code_levels_results.json"
        if candidate.is_file():
            result_files[method] = candidate.resolve()
    if not result_files:
        raise FileNotFoundError(f"No RQ1 result JSON found under: {path}")
    return result_files


def _extract_group_scores(result: Dict, score_field: str) -> Dict[str, np.ndarray]:
    groups = result.get("groups", [])
    grouped: Dict[str, List[float]] = {group: [] for group in groups}
    for row in result.get("group_candidate_details", []):
        group = row.get("group")
        score = row.get(score_field)
        if score is None:
            for fallback in ("score", "score_raw", "member_like_score"):
                score = row.get(fallback)
                if score is not None:
                    break
        if group in grouped and score is not None:
            grouped[group].append(float(score))
    return {group: np.asarray(values, dtype=float) for group, values in grouped.items()}


def _compute_outlier_bounds(values: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
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


def _filter_array(values: np.ndarray, low: float, high: float, action: str) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if action == "drop":
        return values[(values >= low) & (values <= high)]
    return np.clip(values, low, high)


def _apply_outlier_filter(
    group_order: List[str],
    grouped: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    if args.outlier_filter == "none":
        return {group: grouped[group].copy() for group in group_order}
    if args.outlier_scope == "global":
        non_empty = [grouped[group] for group in group_order if grouped[group].size]
        all_values = np.concatenate(non_empty) if non_empty else np.array([], dtype=float)
        low, high = _compute_outlier_bounds(all_values, args)
        return {group: _filter_array(grouped[group], low, high, args.outlier_action) for group in group_order}
    filtered: Dict[str, np.ndarray] = {}
    for group in group_order:
        low, high = _compute_outlier_bounds(grouped[group], args)
        filtered[group] = _filter_array(grouped[group], low, high, args.outlier_action)
    return filtered


def _parse_method_q_high_overrides(value: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in [x.strip() for x in value.split(",") if x.strip()]:
        method, separator, raw_value = item.partition(":")
        if not separator or method not in SUPPORTED_METHODS:
            raise ValueError(f"Invalid method q-high override: {item}")
        q_high = float(raw_value)
        if not 0.0 < q_high <= 1.0:
            raise ValueError(f"Invalid q-high for {method}: {q_high}")
        overrides[method] = q_high
    return overrides


def _plot_rq1_violin_on_ax(
    ax,
    group_order: List[str],
    grouped: Dict[str, np.ndarray],
    title: str,
    trend_stat: str,
) -> None:
    raw_data = [grouped[group] for group in group_order]
    ax.set_facecolor("#F7FAFC")
    positions = [index for index, values in enumerate(raw_data, 1) if values.size]
    data = [values for values in raw_data if values.size]
    if data:
        parts = ax.violinplot(
            data,
            positions=positions,
            showmeans=trend_stat == "mean",
            showmedians=trend_stat == "median",
            showextrema=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#6BAED6")
            body.set_edgecolor("#6BAED6")
            body.set_alpha(0.58)
    else:
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

    if trend_stat in {"mean", "median"}:
        statistic = np.mean if trend_stat == "mean" else np.median
        x_values = np.arange(1, len(group_order) + 1, dtype=float)
        y_values = np.asarray(
            [float(statistic(grouped[group])) if grouped[group].size else np.nan for group in group_order],
            dtype=float,
        )
        valid = np.isfinite(y_values)
        if np.any(valid):
            ax.plot(
                x_values[valid],
                y_values[valid],
                marker="o",
                color="#008B8B",
                linewidth=2.0,
                label=f"{trend_stat} trend",
            )
            for x_value, y_value in zip(x_values[valid], y_values[valid]):
                ax.text(
                    x_value,
                    y_value,
                    f"{y_value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["trend_value"],
                    fontweight="bold",
                    color="#008B8B",
                )
            ax.legend(loc="best", framealpha=0.85, fontsize=FONT_SIZES["legend"])

    ax.set_xticks(np.arange(1, len(group_order) + 1))
    ax.set_xticklabels(
        [GROUP_DISPLAY_LABELS.get(group, group) for group in group_order],
        rotation=18,
        ha="right",
        fontsize=FONT_SIZES["tick_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlim(0.5, len(group_order) + 0.5)
    ax.set_ylabel("Value", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold", pad=13)
    ax.grid(axis="y", color="#E6E6E6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final RQ1 2x5 violin figure.")
    parser.add_argument("--from-result", type=Path, required=True)
    parser.add_argument("--rq2-lat-json", type=Path, required=True)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--violin-trend-stat", choices=["mean", "median", "none"], default="mean")
    parser.add_argument(
        "--violin-score-field",
        choices=["score_raw", "member_like_score", "score_before_orientation", "score"],
        default="score_raw",
    )
    parser.add_argument("--outlier-filter", choices=["none", "quantile", "iqr", "mad"], default="quantile")
    parser.add_argument("--outlier-action", choices=["clip", "drop"], default="clip")
    parser.add_argument("--outlier-scope", choices=["global", "per-group"], default="global")
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--mad-k", type=float, default=3.5)
    parser.add_argument("--method-q-high-overrides", default="")
    parser.add_argument("--independent-old-indices", default="2,3,4,5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.q_low < args.q_high <= 1.0:
        raise ValueError("Require 0 <= q-low < q-high <= 1.")
    methods = _parse_methods(args.methods)
    result_files = _resolve_result_files(args.from_result, methods)
    ordered_methods = [method for method in DEFAULT_METHODS if method in methods and method in result_files]
    if len(ordered_methods) != 8:
        raise ValueError(f"Expected 8 RQ1 methods, found {len(ordered_methods)}: {ordered_methods}")
    old_indices = [int(value) for value in args.independent_old_indices.split(",") if value.strip()]
    if len(old_indices) != 4:
        raise ValueError("--independent-old-indices must contain four values.")

    overrides = _parse_method_q_high_overrides(args.method_q_high_overrides)
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(FIGURE_LAYOUT["fig_width"], FIGURE_LAYOUT["fig_height"]),
        facecolor="#FFFFFF",
    )
    flat_axes = axes.flatten()
    rq1_axis_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    for method, axis_index in zip(ordered_methods, rq1_axis_indices):
        with result_files[method].open("r", encoding="utf-8") as f:
            result = json.load(f)
        group_order = result.get("groups", [])
        if not group_order:
            raise ValueError(f"Missing groups in {result_files[method]}")
        grouped = _extract_group_scores(result, args.violin_score_field)
        method_args = argparse.Namespace(**vars(args))
        method_args.q_high = overrides.get(method, args.q_high)
        filtered = _apply_outlier_filter(group_order, grouped, method_args)
        title = f"{METHOD_DISPLAY_LABELS.get(method, method.upper())} Value Distribution"
        _plot_rq1_violin_on_ax(flat_axes[axis_index], group_order, filtered, title, args.violin_trend_stat)
        if axis_index not in {0, 5}:
            flat_axes[axis_index].set_ylabel("")

    grouped_mem, grouped_syn, _ = _load_model_scores(args.rq2_lat_json, old_indices, args)
    # model_label = args.rq2_lat_json.parts[-5] if len(args.rq2_lat_json.parts) >= 5 else "Model"
    _plot_rq2_violin_on_ax(
        axes[0, 4], GROUP_ORDER_SINGLE, grouped_mem, f"Mem Score Distribution", "", args.violin_trend_stat
    )
    _plot_rq2_violin_on_ax(
        axes[1, 4], GROUP_ORDER_SINGLE, grouped_syn, f"Syn Score Distribution", "", args.violin_trend_stat
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
    output = args.output.resolve() if args.output else args.from_result.resolve() / "rq1_trans_code_baseline_2x5.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"[RQ1] saved: {output}")


if __name__ == "__main__":
    main()
