"""四张主图(图内文字用英文,避免无中文字体环境出豆腐块):
  1. UAPR-ORR 权衡前沿(Pareto)
  2. BRS(k) 边界遗忘曲线(全体 + 长时程)
  3. 语义漂移对照(labeled/unlabeled 分组柱状)
  4. 轮数标度(max_rounds vs drift-recall 与 EC)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics as M  # noqa: E402

COLORS = {"naive_executor": "#999999", "contract_prompt": "#1f77b4",
          "static_guardrail": "#ff7f0e", "debate_no_critic": "#2ca02c",
          "debate_no_contract": "#9467bd", "full_system": "#d62728",
          "oracle": "#17becf", "random": "#8c564b"}


def _c(name):
    return COLORS.get(name, "#333333")


def plot_pareto(all_summaries: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    pts = {}
    for name, eps in all_summaries.items():
        x = M.orr(eps)
        y = M.uapr(eps)["overall"]
        if x != x or y != y:
            continue
        pts[name] = (x, y)
        ax.scatter(x, y, s=70, color=_c(name), zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    # Pareto 前沿:UAPR 最大化、ORR 最小化
    front = []
    for n, (x, y) in pts.items():
        if not any((x2 <= x and y2 >= y and (x2, y2) != (x, y)) for x2, y2 in pts.values()):
            front.append((x, y, n))
    front.sort()
    if len(front) >= 2:
        ax.plot([p[0] for p in front], [p[1] for p in front], "--", color="#444", lw=1,
                label="Pareto front", zorder=2)
        ax.legend()
    ax.set_xlabel("ORR (over-refusal rate, safe-clear)")
    ax.set_ylabel("UAPR (unsafe action prevention rate)")
    ax.set_title("Safety-utility trade-off frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {n: pts[n] for n in pts}, [p[2] for p in front]


def plot_brs(all_summaries: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (title, filt) in zip(axes, [
            ("All scenarios", lambda e: True),
            ("Long-horizon (horizon >= 25)", lambda e: e["long_horizon"])]):
        for name, eps in all_summaries.items():
            if name == "random":
                continue
            sel = [e for e in eps if filt(e)]
            if not sel:
                continue
            curve = M.brs_curve(sel)
            ax.plot(range(len(curve)), curve, label=name, color=_c(name), lw=1.8)
        ax.set_xlabel("step k")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BRS(k): fraction of constraints still held")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_drift_contrast(all_summaries: dict, out: Path, agents_order: list[str]):
    names = [n for n in agents_order if n in all_summaries]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.38
    for ax, key, title in [(axes[0], "drift_accept_rate", "Drift acceptance rate"),
                           (axes[1], "avr", "AVR in drift scenarios")]:
        xs = range(len(names))
        lab, unl = [], []
        for n in names:
            dc = M.drift_contrast(all_summaries[n])
            lab.append((dc["labeled"] or {}).get(key, float("nan")))
            unl.append((dc["unlabeled"] or {}).get(key, float("nan")))
        ax.bar([x - width / 2 for x in xs], lab, width, label="labeled (restricted=true)",
               color="#4c72b0")
        ax.bar([x + width / 2 for x in xs], unl, width, label="unlabeled (no surface cue)",
               color="#c44e52")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Static rules catch labeled drift only", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_rounds_scaling(rounds_data: list[dict], out: Path):
    """rounds_data: [{max_rounds, drift_recall, tokens, agent}]"""
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twinx()
    by_agent = defaultdict(list)
    for r in rounds_data:
        by_agent[r["agent"]].append(r)
    for name, rows in by_agent.items():
        rows.sort(key=lambda r: r["max_rounds"])
        xs = [r["max_rounds"] for r in rows]
        ax1.plot(xs, [r["drift_recall"] for r in rows], "-o", color=_c(name),
                 label=f"{name} recall")
        ax2.plot(xs, [r["tokens"] for r in rows], "--s", color=_c(name), alpha=0.5,
                 label=f"{name} tokens")
    ax1.set_xlabel("max debate rounds")
    ax1.set_ylabel("drift recall")
    ax2.set_ylabel("tokens per episode")
    ax1.set_xscale("log", base=2)
    ax1.grid(alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.set_title("Rounds scaling: recall vs verifier tax")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
