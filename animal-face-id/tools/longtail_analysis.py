"""Long-tail recognition (LTR) analysis for the macaque ReID model.

Links per-ID *training support* to per-class test performance to diagnose
whether the model over-prioritises the head (frequent IDs) at the expense of
the tail (rare IDs).

Inputs
------
- data/macaque_faces/splits.json        : train/val/test image lists per ID
- per_class_metrics.csv                 : accuracy / precision / recall / f1 / support (test)

Outputs (artifacts/longtail_analysis/)
-------
- id_distribution_rank.png              : sorted training-image count per ID (the long-tail curve)
- id_distribution_hist.png              : histogram of images-per-ID
- support_vs_accuracy.png               : per-class accuracy vs training support (scatter + trend)
- support_vs_prf.png                    : precision / recall / f1 vs training support
- head_vs_tail_bars.png                 : head/mid/tail grouped performance (macro P/R/F1/acc)
- per_class_accuracy_by_support.png     : per-class accuracy bars, ordered by support, tail highlighted
- longtail_summary.json / .md           : numeric summary + diagnosis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "macaque_faces" / "splits.json"
_DEFAULT_PER_CLASS = ROOT / "artifacts" / "final_eval" / (
    "train_macaque_arcface_macaque-resnet50-arcface_aug2_best_per_class_metrics.csv"
)

_ap = argparse.ArgumentParser(description="Long-tail analysis from a per-class metrics CSV.")
_ap.add_argument("--per-class", type=Path, default=_DEFAULT_PER_CLASS,
                 help="Per-class metrics CSV (from run_final_eval).")
_ap.add_argument("--out-subdir", type=str, default="longtail_analysis",
                 help="Output subdir under artifacts/ (use distinct names per model).")
_args = _ap.parse_args()
PER_CLASS = _args.per_class
OUT = ROOT / "artifacts" / _args.out_subdir
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11})


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
splits = json.loads(SPLITS.read_text())
train_counts = Counter(item["id"] for item in splits["train"])
val_counts = Counter(item["id"] for item in splits["val"])
test_counts = Counter(item["id"] for item in splits["test"])

pc = pd.read_csv(PER_CLASS)
pc = pc.rename(columns={"class_id": "id", "num_samples": "test_support"})
pc["train_support"] = pc["id"].map(train_counts).fillna(0).astype(int)
pc["val_support"] = pc["id"].map(val_counts).fillna(0).astype(int)
pc = pc.sort_values("train_support", ascending=False).reset_index(drop=True)

n_ids = len(pc)
total_train = sum(train_counts.values())

# ----------------------------------------------------------------------
# 2. Imbalance statistics
# ----------------------------------------------------------------------
counts = pc["train_support"].to_numpy()


def gini(x):
    x = np.sort(x.astype(float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


imbalance_ratio = float(counts.max() / max(counts.min(), 1))
gini_coef = float(gini(counts))

# Head / Mid / Tail split by training support terciles of *images* (not IDs),
# i.e. head = most frequent IDs that together are the top third by image mass.
# Simpler, standard LTR convention: split IDs into many/medium/few shots.
MANY, FEW = 30, 15  # >=30 imgs = head ("many"), <15 = tail ("few"), else "mid"


def group_of(c):
    if c >= MANY:
        return "head (>=%d)" % MANY
    if c < FEW:
        return "tail (<%d)" % FEW
    return "mid (%d-%d)" % (FEW, MANY - 1)


pc["group"] = pc["train_support"].map(group_of)
group_order = ["head (>=%d)" % MANY, "mid (%d-%d)" % (FEW, MANY - 1), "tail (<%d)" % FEW]

grp = pc.groupby("group").agg(
    n_ids=("id", "count"),
    mean_train=("train_support", "mean"),
    acc=("accuracy", "mean"),
    precision=("precision", "mean"),
    recall=("recall", "mean"),
    f1=("f1", "mean"),
).reindex(group_order)

# ----------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------
TAIL_COLOR = "#dc2626"
HEAD_COLOR = "#2563eb"
MID_COLOR = "#9ca3af"


def color_for(c):
    if c >= MANY:
        return HEAD_COLOR
    if c < FEW:
        return TAIL_COLOR
    return MID_COLOR


# 3a. Rank-frequency (the long-tail curve)
fig, ax = plt.subplots(figsize=(11, 4.5))
order = np.argsort(-counts)
bar_colors = [color_for(counts[i]) for i in order]
ax.bar(range(n_ids), counts[order], color=bar_colors, width=1.0)
ax.set_yscale("log")
ax.set_xlabel("Individual ID (ranked by # training images)")
ax.set_ylabel("# training images (log)")
ax.set_title("Macaque ID distribution — long-tail (%d IDs, %d train images)" % (n_ids, total_train))
ax.axhline(MANY, color=HEAD_COLOR, ls="--", lw=1, alpha=0.7)
ax.axhline(FEW, color=TAIL_COLOR, ls="--", lw=1, alpha=0.7)
handles = [plt.Rectangle((0, 0), 1, 1, color=HEAD_COLOR),
           plt.Rectangle((0, 0), 1, 1, color=MID_COLOR),
           plt.Rectangle((0, 0), 1, 1, color=TAIL_COLOR)]
ax.legend(handles, group_order, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "id_distribution_rank.png")
plt.close(fig)

# 3b. Histogram of images-per-ID
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(counts, bins=30, color="#6366f1", edgecolor="white")
ax.axvline(FEW, color=TAIL_COLOR, ls="--", lw=1.5, label="tail threshold (<%d)" % FEW)
ax.axvline(MANY, color=HEAD_COLOR, ls="--", lw=1.5, label="head threshold (>=%d)" % MANY)
ax.set_xlabel("# training images per ID")
ax.set_ylabel("# individuals")
ax.set_title("Distribution of training images per individual")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "id_distribution_hist.png")
plt.close(fig)

# 3c. Support vs accuracy scatter + trend
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(pc["train_support"], pc["accuracy"],
           c=[color_for(c) for c in pc["train_support"]], s=28, alpha=0.8, edgecolor="white", linewidth=0.4)
# log-x trend (rolling mean over sorted support)
ssort = pc.sort_values("train_support")
xs = ssort["train_support"].to_numpy()
ys = ssort["accuracy"].rolling(15, min_periods=5, center=True).mean().to_numpy()
ax.plot(xs, ys, color="black", lw=2, label="rolling mean (window=15)")
# correlation
r_acc = float(np.corrcoef(np.log1p(pc["train_support"]), pc["accuracy"])[0, 1])
ax.set_xscale("log")
ax.set_xlabel("# training images per ID (log)")
ax.set_ylabel("Per-class test accuracy (recall)")
ax.set_title("Accuracy vs training support  (Spearman-ish r[log] = %.2f)" % r_acc)
ax.set_ylim(-0.03, 1.03)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "support_vs_accuracy.png")
plt.close(fig)

# 3d. Precision / recall / f1 vs support (rolling means)
fig, ax = plt.subplots(figsize=(7.5, 5))
for col, color in [("precision", "#16a34a"), ("recall", "#dc2626"), ("f1", "#2563eb")]:
    ys = ssort[col].rolling(15, min_periods=5, center=True).mean().to_numpy()
    ax.plot(xs, ys, color=color, lw=2, label=col)
ax.set_xscale("log")
ax.set_xlabel("# training images per ID (log)")
ax.set_ylabel("Score (rolling mean, window=15)")
ax.set_title("Precision / Recall / F1 vs training support")
ax.set_ylim(0, 1.03)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "support_vs_prf.png")
plt.close(fig)

# 3e. Head/Mid/Tail grouped bars
fig, ax = plt.subplots(figsize=(8.5, 5))
metrics = ["acc", "precision", "recall", "f1"]
x = np.arange(len(group_order))
w = 0.2
colors = ["#0ea5e9", "#16a34a", "#dc2626", "#7c3aed"]
for i, (m, c) in enumerate(zip(metrics, colors)):
    ax.bar(x + (i - 1.5) * w, grp[m].to_numpy(), w, label=m, color=c)
ax.set_xticks(x)
ax.set_xticklabels(["%s\n(%d IDs)" % (g, grp.loc[g, "n_ids"]) for g in group_order])
ax.set_ylabel("Mean score")
ax.set_title("Performance by support group — macro avg within group")
ax.set_ylim(0, 1.0)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(OUT / "head_vs_tail_bars.png")
plt.close(fig)

# 3f. Per-class accuracy ordered by support, tail highlighted
fig, ax = plt.subplots(figsize=(13, 4.5))
acc_order = pc["accuracy"].to_numpy()  # pc already sorted by train_support desc
ax.bar(range(n_ids), acc_order, color=[color_for(c) for c in pc["train_support"]], width=1.0)
ax.set_xlabel("Individual ID (ranked by training support, head -> tail)")
ax.set_ylabel("Per-class accuracy")
ax.set_title("Per-class accuracy ordered head -> tail (red = tail IDs)")
ax.set_ylim(0, 1.03)
ax.legend(handles, group_order, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "per_class_accuracy_by_support.png")
plt.close(fig)

# ----------------------------------------------------------------------
# 4. Summary
# ----------------------------------------------------------------------
zero_acc = pc[pc["accuracy"] == 0]
summary = {
    "n_ids": int(n_ids),
    "total_train_images": int(total_train),
    "min_images_per_id": int(counts.min()),
    "max_images_per_id": int(counts.max()),
    "median_images_per_id": float(np.median(counts)),
    "mean_images_per_id": float(counts.mean()),
    "imbalance_ratio_max_over_min": imbalance_ratio,
    "gini_coefficient": gini_coef,
    "top10pct_ids_share_of_images": float(
        np.sort(counts)[::-1][: max(1, n_ids // 10)].sum() / total_train
    ),
    "correlation_logsupport_accuracy": r_acc,
    "n_zero_accuracy_ids": int(len(zero_acc)),
    "zero_accuracy_ids": zero_acc.sort_values("train_support")[
        ["id", "train_support", "test_support"]
    ].to_dict("records"),
    "groups": {
        g: {
            "n_ids": int(grp.loc[g, "n_ids"]),
            "mean_train_images": float(grp.loc[g, "mean_train"]),
            "accuracy": float(grp.loc[g, "acc"]),
            "precision": float(grp.loc[g, "precision"]),
            "recall": float(grp.loc[g, "recall"]),
            "f1": float(grp.loc[g, "f1"]),
        }
        for g in group_order
    },
}
(OUT / "longtail_summary.json").write_text(json.dumps(summary, indent=2))

# enriched per-class table
pc.to_csv(OUT / "per_class_with_support.csv", index=False)

# markdown report
head_f1 = grp.loc[group_order[0], "f1"]
tail_f1 = grp.loc[group_order[2], "f1"]
# Overall metrics computed from THIS per-class CSV (not hardcoded).
macro_f1 = float(pc["f1"].mean())
macro_p = float(pc["precision"].mean())
macro_r = float(pc["recall"].mean())
weighted_acc = float((pc["accuracy"] * pc["test_support"]).sum() / pc["test_support"].sum())
md = f"""# Long-Tail Recognition (LTR) Analysis — Macaque ReID

Source per-class CSV: `{PER_CLASS.name}`
Overall (from this CSV): Top-1≈{weighted_acc:.3f}, Macro-F1={macro_f1:.3f}, Macro-P={macro_p:.3f}, Macro-R={macro_r:.3f}

## Dataset imbalance
- Individuals: **{n_ids}**, training images: **{total_train}**
- Images per ID: min **{counts.min()}**, median **{np.median(counts):.0f}**, mean **{counts.mean():.1f}**, max **{counts.max()}**
- **Imbalance ratio (max/min) = {imbalance_ratio:.1f}x**
- **Gini coefficient = {gini_coef:.3f}**  (0 = balanced, 1 = maximally skewed)
- Top 10% of IDs hold **{summary['top10pct_ids_share_of_images']*100:.0f}%** of all training images

## Does the head dominate? (support -> performance)
- Correlation between log(training support) and per-class accuracy: **r = {r_acc:.2f}**
- Head F1 = **{head_f1:.2f}** vs Tail F1 = **{tail_f1:.2f}**  -> gap of **{(head_f1-tail_f1):.2f}**

| Group | #IDs | mean train imgs | accuracy | precision | recall | f1 |
|-------|-----:|----------------:|---------:|----------:|-------:|---:|
"""
for g in group_order:
    md += "| %s | %d | %.1f | %.2f | %.2f | %.2f | %.2f |\n" % (
        g, grp.loc[g, "n_ids"], grp.loc[g, "mean_train"], grp.loc[g, "acc"],
        grp.loc[g, "precision"], grp.loc[g, "recall"], grp.loc[g, "f1"],
    )
md += f"""
## Failure concentration
- IDs with **0% accuracy**: **{len(zero_acc)}** — mean training images = {zero_acc['train_support'].mean():.1f} (vs dataset mean {counts.mean():.1f})

## Figures
- `id_distribution_rank.png` — the long-tail curve (head=blue, tail=red)
- `id_distribution_hist.png` — histogram of images per ID
- `support_vs_accuracy.png` — accuracy rises with support
- `support_vs_prf.png` — precision/recall/F1 vs support
- `head_vs_tail_bars.png` — grouped head/mid/tail performance
- `per_class_accuracy_by_support.png` — per-class accuracy head->tail
"""
(OUT / "longtail_report.md").write_text(md)

print(md)
print("\nOutputs written to:", OUT)
