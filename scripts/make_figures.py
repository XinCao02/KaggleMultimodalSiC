"""Rebuild publication figures solely from frozen experiment outputs.

Python backend; all plotted quantitative data are exported alongside each panel.
No synthetic observations, jitter, significance stars or test-based selection.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from citebench.metrics import protein_correlations

ROOT = Path(".")
OUT = ROOT / "figures"
SOURCE = OUT / "source_data"
SOURCE.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
TEAL = "#157F7A"
GOLD = "#BC792D"
BLUE = "#547BA6"
INK = "#243547"
GREY = "#939EA7"
MODELS = [
    "mean_profile",
    "ridge",
    "residual_ensemble",
    "selected_blend",
    "ssopm_adapted",
]
NAMES = {
    "mean_profile": "Mean profile",
    "ridge": "Ridge",
    "residual_ensemble": "ResMLP ensemble",
    "selected_blend": "ResMLP-Ridge blend",
    "ssopm_adapted": "SS-OPM adapted",
}
COLORS = dict(zip(MODELS, [GREY, BLUE, "#79B3A4", TEAL, GOLD]))
SPLITS = ["temporal", "donor_13176", "donor_31800", "donor_32606", "random"]
SHORT = ["Day 4", "Donor 13176", "Donor 31800", "Donor 32606", "Random"]
width_mm = 183
WIDTH = width_mm / 25.4


def label(ax, letter, title):
    ax.set_title(title, loc="left", pad=10, fontweight="medium")
    ax.text(
        -0.12,
        1.06,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(
        OUT / f"{name}.tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def table(df, name):
    df.to_csv(SOURCE / f"{name}.csv", index=False)


def load_cell_data():
    global meta, mt, y, proteins, idx, yt, pred, wp
    meta = pd.read_csv("data/processed/meta_train.csv")
    mt = pd.read_csv("data/processed/meta_test.csv")
    y = np.load("data/processed/y_train.npy")
    proteins = np.load("data/processed/proteins.npy")
    idx = np.load("results/temporal/split_indices.npz")["test"]
    yt = y[idx]
    pred = np.load("results/temporal/selected_blend/predictions.npy")
    wp = np.load("results/temporal/ssopm_adapted/predictions.npy")


def figure1():
    load_cell_data()
    fig = plt.figure(figsize=(WIDTH, 4.65))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 1.15],
        left=0.09,
        right=0.97,
        bottom=0.16,
        top=0.91,
        hspace=0.57,
        wspace=0.38,
    )
    ax = fig.add_subplot(gs[0, 0])
    label(ax, "a", "Forward prediction protocol")
    ax.axis("off")
    counts = meta.groupby("day").size()
    for i, (day, role, color) in enumerate(
        [(2, "Fit", BLUE), (3, "Select", TEAL), (4, "Test", GOLD)]
    ):
        x = 0.15 + i * 0.34
        ax.text(
            x,
            0.70,
            f"Day {day}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x,
            0.45,
            role,
            ha="center",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=.4", "fc": color + "18", "ec": "none"},
        )
        ax.text(x, 0.17, f"{counts[day]:,} cells", ha="center", fontsize=8)
        if i < 2:
            ax.annotate(
                "",
                xy=(x + 0.25, 0.45),
                xytext=(x + 0.12, 0.45),
                arrowprops={"arrowstyle": "->", "color": GREY},
            )
    ax.text(
        0.5,
        -0.13,
        "Different cells sampled over time",
        ha="center",
        fontsize=7,
        color=INK,
    )
    ax = fig.add_subplot(gs[0, 1])
    label(ax, "b", "Labelled cells by donor and day")
    c = meta.groupby(["donor", "day"]).size().unstack()
    im = ax.imshow(c.values, cmap="Blues", vmin=0, vmax=11000, aspect="auto")
    ax.set_xticks(range(3), [f"Day {d}" for d in c.columns])
    ax.set_yticks(range(3), c.index.astype(str))
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{c.iloc[i,j]:,}",
                ha="center",
                va="center",
                color="white" if c.iloc[i, j] > 7500 else INK,
                fontsize=8,
            )
    table(
        c.reset_index().melt(id_vars="donor", var_name="day", value_name="cells"),
        "fig1b_donor_day_counts",
    )
    ax = fig.add_subplot(gs[1, 0])
    label(ax, "c", "Cell populations change over time")
    ct = pd.crosstab(meta.day, meta.cell_type)
    fra = ct.div(ct.sum(1), axis=0)
    palette = [
        "#2D6A8B",
        "#C47E57",
        "#558C78",
        "#B49BC8",
        "#D2B956",
        "#A1BCC4",
        "#8C8C8C",
    ]
    bottom = np.zeros(3)
    for typ, col in zip(fra.columns, palette):
        ax.bar(range(3), fra[typ], bottom=bottom, color=col, label=typ, width=0.60)
        bottom += fra[typ].values
    ax.set_xticks(range(3), ["Day 2", "Day 3", "Day 4"])
    ax.set_ylabel("Fraction of labelled cells")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        columnspacing=0.9,
        handlelength=1,
    )
    table(
        ct.reset_index().melt(id_vars="day", var_name="cell_type", value_name="cells"),
        "fig1c_cell_type_counts",
    )
    ax = fig.add_subplot(gs[1, 1])
    label(ax, "d", "Original test-file integrity")
    audit = json.loads(Path("results/data_audit.json").read_text())
    n = audit["identical_train_test_profiles"]
    rest = len(mt) - n
    ax.barh(["Original test"], [rest], color=BLUE, height=0.32, label="Remaining")
    ax.barh(
        ["Original test"], [n], left=[rest], color=GOLD, height=0.32, label="Duplicates"
    )
    ax.set_ylim(-0.65, 0.65)
    ax.set_xlim(0, 52000)
    ax.set_xlabel("Cells")
    ax.set_yticks([])
    ax.set_xticks([0, 20000, 40000], ["0", "20,000", "40,000"])
    ax.text(rest / 2, 0, f"{rest:,}", ha="center", va="center", color="white")
    ax.text(
        rest + n / 2, 0, f"{n:,}", ha="center", va="center", color="white", fontsize=7
    )
    ax.text(
        0.01,
        0.85,
        "7,476 exact RNA matches to training",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.text(
        0.01,
        0.69,
        "0 shared cell IDs; 0 within-train duplicates",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False, fontsize=7, ncol=2
    )
    table(pd.DataFrame([audit]), "fig1d_duplicate_audit")
    save(fig, "fig1_data_design")


def architecture(ax):
    """Architecture reflects models.py; shared features are fitted on training cells."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0, 0.98, "a", weight="bold", fontsize=10, va="top")
    ax.text(0.035, 0.98, "RNA-to-protein architectures", fontsize=9, va="top")
    ax.text(
        0.50,
        0.87,
        "Shared RNA features: 128 PCs + 143 selected genes + 4 QC summaries = 275 inputs",
        ha="center",
        fontsize=7.4,
        color=INK,
    )

    def box(x, y, w, h, text, col=TEAL, fs=7.2):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                linewidth=0.7,
                edgecolor=col,
                facecolor=col + "12",
            )
        )
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK
        )

    def arrow(x1, y1, x2, y2, col=GREY):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=col, lw=0.8, shrinkA=1, shrinkB=1),
        )

    ax.text(0.015, 0.80, "ResMLP-Ridge", color=TEAL, weight="bold", fontsize=8)
    box(0.015, 0.45, 0.115, 0.19, "275\nfeatures", BLUE)
    box(
        0.17,
        0.45,
        0.30,
        0.19,
        "Linear 275→512 → 2 residual blocks\nLN / GELU → 140 + linear skip",
    )
    box(0.51, 0.45, 0.15, 0.19, "Center + L2 norm\nMean of 3 seeds")
    box(0.70, 0.45, 0.15, 0.19, "Weighted blend\nw NN + (1−w) ridge")
    box(0.89, 0.45, 0.095, 0.19, "140\nproteins")
    for a, b in [(0.13, 0.17), (0.47, 0.51), (0.66, 0.70), (0.85, 0.89)]:
        arrow(a, 0.545, b, 0.545)
    box(0.51, 0.70, 0.15, 0.09, "Ridge → norm", BLUE)
    arrow(0.072, 0.64, 0.072, 0.745, BLUE)
    arrow(0.072, 0.745, 0.50, 0.745, BLUE)
    arrow(0.66, 0.745, 0.775, 0.745, BLUE)
    arrow(0.775, 0.745, 0.775, 0.64, BLUE)
    ax.text(0.015, 0.335, "SS-OPM adapted", color=GOLD, weight="bold", fontsize=8)
    box(0.015, 0.055, 0.115, 0.19, "275\nfeatures", BLUE)
    box(0.17, 0.055, 0.18, 0.19, "Linear → 2,048\nEncoder → decoder", GOLD)
    box(
        0.39,
        0.055,
        0.22,
        0.19,
        "5 sequential decoder blocks\n6 heads × 128 latent outputs",
        GOLD,
    )
    box(0.65, 0.055, 0.20, 0.19, "Decode latent → 140\nStandardize each head", GOLD)
    box(0.89, 0.055, 0.095, 0.19, "Mean\n6 heads", GOLD)
    for a, b in [(0.13, 0.17), (0.35, 0.39), (0.61, 0.65), (0.85, 0.89)]:
        arrow(a, 0.15, b, 0.15)


def figure2():
    allscores = pd.concat(
        [pd.read_csv(f"results/{s}/summary.csv").assign(protocol=s) for s in SPLITS],
        ignore_index=True,
    )
    table(allscores, "fig2_model_scores")
    fig = plt.figure(figsize=(WIDTH, 5.0))
    arch = fig.add_axes([0.025, 0.53, 0.95, 0.43])
    architecture(arch)
    gs = fig.add_gridspec(
        1, 2, left=0.105, right=0.97, bottom=0.17, top=0.45, wspace=0.40
    )
    ax = fig.add_subplot(gs[0, 0])
    label(ax, "b", "Generalization across holdouts")
    for i, m in enumerate(MODELS):
        vals = [
            float(
                allscores.loc[
                    (allscores.protocol == s) & (allscores.model == m), "cell_pearson"
                ].iloc[0]
            )
            for s in SPLITS
        ]
        ax.plot(
            np.arange(5) + (i - 2) * 0.10,
            vals,
            "o",
            color=COLORS[m],
            label=NAMES[m],
            markersize=5,
        )
    ax.set_xticks(
        range(5), ["Day 4", "Donor\n13176", "Donor\n31800", "Donor\n32606", "Random"]
    )
    ax.set_ylabel("Mean cell-wise Pearson r")
    ax.set_ylim(0.73, 0.95)
    ax.grid(axis="y", alpha=0.18)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.008),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.0,
    )
    ax.set_ylim(0.74, 0.92)
    ax = fig.add_subplot(gs[0, 1])
    label(ax, "c", "ResMLP-Ridge − SS-OPM adapted")
    rows = []
    for i, s in enumerate(SPLITS):
        ci = json.loads(Path(f"results/{s}/paired_comparison.json").read_text())
        rows.append(dict(protocol=s, **ci))
        ax.errorbar(
            i,
            ci["delta"],
            yerr=[[ci["delta"] - ci["low"]], [ci["high"] - ci["delta"]]],
            fmt="o",
            color=TEAL,
            capsize=3,
            markersize=5,
        )
        a = pd.read_csv(f"results/{s}/selected_blend/by_batch.csv")
        b = pd.read_csv(f"results/{s}/ssopm_adapted/by_batch.csv")
        v = a.merge(b, on="batch", suffixes=("_a", "_b"))
        ax.scatter(
            np.full(len(v), i) + np.linspace(-0.13, 0.13, len(v)),
            v.mean_a - v.mean_b,
            s=14,
            color=GREY,
            alpha=0.8,
            zorder=1,
        )
    ax.axhline(0, color=INK, ls="--", lw=0.8)
    ax.set_xticks(
        range(5), ["Day 4", "Donor\n13176", "Donor\n31800", "Donor\n32606", "Random"]
    )
    ax.set_ylabel("Difference in Pearson r")
    ax.grid(axis="y", alpha=0.15)
    table(pd.DataFrame(rows), "fig2c_cluster_intervals")
    save(fig, "fig2_generalization")


def figure3():
    load_cell_data()
    fig = plt.figure(figsize=(WIDTH, 5.3))
    upper = fig.add_gridspec(
        1, 2, left=0.10, right=0.95, bottom=0.57, top=0.91, wspace=0.65
    )
    lower = fig.add_gridspec(
        1, 3, left=0.10, right=0.89, bottom=0.12, top=0.43, wspace=0.40
    )
    ax = fig.add_subplot(upper[0, 0])
    label(ax, "a", "Protein-specific recovery")
    pc = protein_correlations(yt, pred)
    wc = protein_correlations(yt, wp)
    order = np.argsort(pc)
    iso = np.array(["IgG" in p for p in proteins])
    ax.plot(range(140), wc[order], ".", color=GOLD, ms=3, label="SS-OPM adapted")
    ax.plot(range(140), pc[order], ".", color=TEAL, ms=3, label="ResMLP-Ridge blend")
    ax.scatter(
        np.flatnonzero(iso[order]),
        pc[order][iso[order]],
        s=24,
        facecolors="none",
        edgecolors=INK,
        linewidths=0.7,
        label="Isotype control",
        zorder=4,
    )
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xlabel("Protein rank (ResMLP-Ridge blend)")
    ax.set_ylabel("Across-cell Pearson r")
    ax.set_ylim(-0.35, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1)
    table(
        pd.DataFrame(
            {"protein": proteins, "compact_r": pc, "ssopm_r": wc, "isotype": iso}
        ),
        "fig3a_protein_correlations",
    )
    ax = fig.add_subplot(upper[0, 1])
    label(ax, "b", "Accuracy by cell population")
    a = pd.read_csv("results/temporal/selected_blend/by_cell_type.csv").sort_values(
        "size", ascending=False
    )
    b = pd.read_csv("results/temporal/ssopm_adapted/by_cell_type.csv")
    a = a.merge(b, on="cell_type", suffixes=("_a", "_b"))
    ax.plot(a.mean_a, range(len(a)), "o", color=TEAL, ms=4)
    ax.plot(a.mean_b, range(len(a)), "o", color=GOLD, ms=4)
    for i, r in a.iterrows():
        ax.plot([r.mean_a, r.mean_b], [i, i], color=GREY, lw=1, zorder=0)
    ax.set_yticks(
        range(len(a)), [f"{t} (n={n:,})" for t, n in zip(a.cell_type, a.size_a)]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean cell-wise Pearson r")
    ax.grid(axis="x", alpha=0.15)
    table(a, "fig3b_cell_type_scores")
    # Prediction units are not dsb: axes explicitly state their different scales.
    hexbins = []
    for k, name in enumerate(["CD38", "CD71", "CD41"]):
        j = np.flatnonzero(proteins == name)
        if len(j) != 1:
            raise ValueError(f"Expected exact target {name}")
        j = j[0]
        ax = fig.add_subplot(lower[0, k])
        label(ax, chr(99 + k), f"{name}: r = {pc[j]:.2f}")
        hb = ax.hexbin(
            yt[:, j],
            pred[:, j],
            gridsize=35,
            mincnt=1,
            norm=LogNorm(),
            cmap="viridis",
            rasterized=True,
            linewidths=0,
        )
        ax.set_xlabel("Measured dsb value")
        ax.set_ylabel("Predicted relative value" if k == 0 else "")
        ax.tick_params(labelsize=6.8)
        ax.locator_params(nbins=4)
        hexbins.append(hb)
        offsets = hb.get_offsets()
        table(
            pd.DataFrame(
                {
                    "measured_bin_center": offsets[:, 0],
                    "prediction_bin_center": offsets[:, 1],
                    "cells": hb.get_array(),
                }
            ),
            f"fig3_{name}_hexbin",
        )
    shared_norm = LogNorm(vmin=1, vmax=max(float(h.get_array().max()) for h in hexbins))
    for h in hexbins:
        h.set_norm(shared_norm)
    cax = fig.add_axes([0.925, 0.12, 0.012, 0.31])
    cb = fig.colorbar(hexbins[-1], cax=cax, ticks=[1, 10, 100, 1000])
    cb.ax.set_yticklabels(["1", "10", "100", "1,000"])
    cb.ax.tick_params(labelsize=6.5)
    cb.ax.set_title("Cells/bin", fontsize=6.5, pad=8)
    save(fig, "fig3_biological_fidelity")


def figure4():
    fig = plt.figure(figsize=(WIDTH, 4.5))
    gs = fig.add_gridspec(
        1, 2, left=0.11, right=0.97, bottom=0.61, top=0.91, wspace=0.42
    )
    lower = fig.add_gridspec(
        1, 3, left=0.085, right=0.97, bottom=0.15, top=0.43, wspace=0.58
    )
    ax = fig.add_subplot(gs[0, 0])
    label(ax, "a", "Validation learning curves")
    hist = []
    for m, col in [("residual_s42", TEAL), ("ssopm_adapted", GOLD)]:
        h = pd.read_csv(f"results/temporal/{m}/history.csv")
        ax.plot(
            h.epoch,
            h.validation_pearson,
            color=col,
            label="ResMLP" if m.startswith("residual") else "SS-OPM adapted",
        )
        hist.append(h.assign(model=m))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Day-3 validation r")
    ax.legend(frameon=False, loc="lower right")
    table(pd.concat(hist), "fig4a_learning_curves")
    ax = fig.add_subplot(gs[0, 1])
    label(ax, "b", "Feature ablation (seed 42)")
    a = pd.read_csv("results/temporal/ablations/summary.csv")
    full = json.loads(Path("results/temporal/residual_s42/metrics.json").read_text())
    vals = [
        a.loc[a.model == "pca_only", "cell_pearson"].iloc[0],
        a.loc[a.model == "selected_genes_only", "cell_pearson"].iloc[0],
        full["cell_pearson"],
    ]
    ax.bar(range(3), vals, color=[BLUE, GREY, TEAL], width=0.6)
    ax.set_xticks(range(3), ["PCA", "Selected\ngenes", "PCA + genes\n+ QC"])
    ax.set_ylim(0.78, 0.89)
    ax.set_ylabel("Day-4 test r")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=7)
    table(
        pd.DataFrame(
            {
                "features": ["PCA only", "Selected genes only", "PCA + genes + QC"],
                "cell_pearson": vals,
            }
        ),
        "fig4b_ablation",
    )
    infos = []
    for m in ["residual_s42", "residual_s43", "residual_s44", "ssopm_adapted"]:
        infos.append(
            dict(
                model=m,
                **json.loads(Path(f"results/temporal/{m}/training.json").read_text()),
            )
        )
    a = pd.DataFrame(infos)
    table(a, "fig4_compute")
    ax = fig.add_subplot(lower[0, 0])
    label(ax, "c", "Training time")
    vals = [a.iloc[:3].seconds.sum() / 60, a.iloc[3].seconds / 60]
    ax.bar(range(2), vals, color=[TEAL, GOLD], width=0.5)
    ax.set_xticks(range(2), ["ResMLP\n3 seeds", "SS-OPM\nadapted"])
    ax.set_ylabel("Minutes incl. validation")
    ax.set_ylim(0, max(vals) * 1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=7)
    ax = fig.add_subplot(lower[0, 1])
    label(ax, "d", "Parameters")
    vals = [a.iloc[0].parameters / 1e6, a.iloc[3].parameters / 1e6]
    ax.bar(range(2), vals, color=[TEAL, GOLD], width=0.5)
    ax.set_xticks(range(2), ["ResMLP", "SS-OPM\nadapted"])
    ax.set_ylabel("Million parameters")
    ax.set_ylim(0, max(vals) * 1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.2f}", ha="center", fontsize=7)
    ax = fig.add_subplot(lower[0, 2])
    label(ax, "e", "MPS tensor memory")
    memory = pd.read_csv("results/memory/summary.csv")
    groups = [
        memory.loc[memory.model == m, "tensor_max_bytes"].to_numpy() / 2**20
        for m in ["residual", "ssopm"]
    ]
    vals = [np.median(g) for g in groups]
    errors = [
        [v - g.min() for v, g in zip(vals, groups)],
        [g.max() - v for v, g in zip(vals, groups)],
    ]
    ax.bar(range(2), vals, color=[TEAL, GOLD], width=0.5, yerr=errors, capsize=3)
    ax.set_xticks(range(2), ["ResMLP", "SS-OPM\nadapted"])
    ax.set_ylabel("MiB (stage-sampled max.)")
    ax.set_ylim(0, max(g.max() for g in groups) * 1.22)
    for i, (v, g) in enumerate(zip(vals, groups)):
        ax.text(i, g.max() + max(vals) * 0.035, f"{v:.0f}", ha="center", fontsize=7)
    table(memory, "fig4e_memory")
    save(fig, "fig4_ablation_compute")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4]
    )
    args = parser.parse_args()
    for number in args.figures:
        globals()[f"figure{number}"]()
    print("Requested figures exported as PDF, SVG, PNG and TIFF.")
