#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures (300 dpi) from current-code canonical results -> Desktop."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DESK = os.path.expanduser("~/Desktop")
MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]
METRIC_KEYS = ["acc", "sen", "spe", "prec", "f1", "auc"]
METRIC_LAB = {"acc": "Accuracy", "sen": "Sensitivity", "spe": "Specificity",
              "prec": "Precision", "f1": "F1", "auc": "AUC"}
META = {"Accuracy": "#1f77b4", "Sensitivity": "#ff7f0e", "Specificity": "#d62728",
        "Precision": "#2ca02c", "F1": "#9467bd", "AUC": "#e377c2"}
MRK = {"Accuracy": "o", "Sensitivity": "^", "Specificity": "D", "Precision": "s",
       "F1": "v", "AUC": "*"}
TITLES = {"SVM": "(a) Support Vector Machine (SVM)", "LR": "(b) Logistic Regression (LR)",
          "RF": "(c) Random Forest (RF)", "XGBoost": "(d) Extreme Gradient Boosting (XGBoost)",
          "KNN": "(e) K-Nearest Neighbors (KNN)"}
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": "--"})


def scale_figs():
    csv = os.path.join(HERE, "canonical_scale6.csv")
    df = pd.read_csv(csv)
    df["Scale_mm"] = df["Scale"].str.replace("mm", "").astype(int)
    df["Label"] = df["Metric"].map(METRIC_LAB)
    for mi, m in enumerate(MODELS):
        sub = df[df.Model == m]
        fig, ax = plt.subplots(figsize=(10, 6.5))
        for lab, grp in sub.groupby("Label"):
            g = grp.sort_values("Scale_mm")
            ax.plot(g["Scale_mm"], g["Mean"], color=META[lab], ls="-",
                    marker=MRK[lab], lw=2.4, ms=8, label=lab)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(["0 mm", "1 mm", "2 mm", "3 mm", "4 mm"])
        ax.set_title(TITLES[m])
        ax.set_xlabel("Peritumoral Range"); ax.set_ylabel("Metric Value")
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(DESK, f"fig4{chr(97+mi)}_{m}_ranges.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
    # composite
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for ax, m in zip(axes, MODELS):
        sub = df[df.Model == m]
        for lab, grp in sub.groupby("Label"):
            g = grp.sort_values("Scale_mm")
            ax.plot(g["Scale_mm"], g["Mean"], color=META[lab], ls="-", marker=MRK[lab],
                    lw=2, ms=7, label=lab)
        ax.set_title(TITLES[m], fontsize=11)
        ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["0", "1", "2", "3", "4"])
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Metric Value")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(DESK, "fig4_peritumoral_ranges.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fusion_fig():
    csv = os.path.join(HERE, "canonical_fusion_auc.csv")
    df = pd.read_csv(csv)
    order = ["base(1mm gray)", "+VFA", "+3", "+13", "+21"]
    x = {k: i for i, k in enumerate(order)}
    df["x"] = df.FeatureSet.map(x)
    colors = {"SVM": "#1f77b4", "LR": "#ff7f0e", "RF": "#2ca02c",
              "XGBoost": "#d62728", "KNN": "#9467bd"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in MODELS:
        g = df[df.Model == m].sort_values("x")
        ax.errorbar(g["x"], g["AUC"], yerr=g["SD"], color=colors[m], marker="o",
                    ms=7, lw=2, ls="-", capsize=3, label=m)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Grayscale\n(1 mm)\nN=0", "+ VFA\nN=1", "+ 3 Doppler\n(optim.)\nN=3",
                        "+ 13 Doppler\n(stage4)\nN=13", "+ 21 Doppler\nN=21"], fontsize=9)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Doppler feature set added to 1 mm grayscale radiomics")
    ax.set_ylabel("AUC (10-seed mean ± SD)")
    ax.set_title("Multimodal fusion across Doppler feature sets (current-code results)",
                 fontsize=12)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(DESK, "fig5_doppler_ablation.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    scale_figs()
    fusion_fig()
    print("[OK] canonical figures written to Desktop")
