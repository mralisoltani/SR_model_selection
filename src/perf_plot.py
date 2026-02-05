import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from itertools import cycle
from matplotlib.ticker import ScalarFormatter

models_path = {
    "f1": "f1_100_10.operon",
    "f2": "f2_100_2.operon",
    "f3": "f3_100_1.operon",
    "f4": "f4_100_2.operon",
    "f5": "f5_100_3.operon",
    "f6": "f6_100_2.operon",
    "f7": "f7_100_2.operon",
}

# maximum number of top-k solutions
k_max = 50
results_dict = {}
csv_path = Path("results/report_BFGS_cluster_results_f1_100_10_1.csv")
df = pd.read_csv(csv_path)
df_clean = df.drop(columns=["Unnamed: 0", "dataset"], errors="ignore")
df_clean.set_index("criteria", inplace=True)
criteria = list(df_clean.index)
metrics = list(df_clean.columns)

results_dict = {
    name + "_" + k: pd.DataFrame(index=metrics)
    for name in criteria
    for k in models_path.keys()
}

for k, v in models_path.items():
    for top_k in range(1, k_max + 1):
        csv_path = Path(f"results/report_BFGS_cluster_results_{v[:-7]}_{top_k}.csv")
        df = pd.read_csv(csv_path)
        df_clean = df.drop(columns=["Unnamed: 0", "dataset"], errors="ignore")
        df_clean.set_index("criteria", inplace=True)
        for name in df_clean.index:
            for col in df_clean.columns:
                results_dict[name + "_" + k].loc[col, top_k] = df_clean.loc[name, col]


def _change_points(y):
    y = np.asarray(y, dtype=float)
    return np.where(np.r_[False, np.diff(y) != 0])[0]


def _tie_jitter(Y, frac=0.001):
    """Y: list of 1D arrays (same length). Jitter only where equal at same x."""
    A = np.vstack(Y)  # n_series × n_points
    yr = (np.nanmax(A) - np.nanmin(A)) or 1.0
    eps = frac * yr
    A2 = A.copy()
    for t in range(A.shape[1]):
        vals = A[:, t]
        groups = {}
        for i, v in enumerate(vals):
            groups.setdefault(v, []).append(i)
        for _, idxs in groups.items():
            if len(idxs) > 1:
                k = len(idxs)
                offsets = eps * (np.arange(k) - (k - 1) / 2.0)
                for off, i in zip(offsets, idxs):
                    A2[i, t] += off
    return [A2[i] for i in range(A.shape[0])]


def _end_label_offsets(y_last_list, frac=0.0015):
    """Return per-series small y offsets to dodge overlapping end labels."""
    arr = np.asarray(y_last_list, dtype=float)
    offsets = np.zeros_like(arr, dtype=float)
    scale = (np.nanmax(arr) - np.nanmin(arr)) or 1.0
    tol = 1e-12 * max(1.0, scale)
    used = np.zeros(len(arr), dtype=bool)
    for i in range(len(arr)):
        if used[i]:
            continue
        close = np.where(np.abs(arr - arr[i]) <= tol)[0]
        used[close] = True
        if len(close) > 1:
            k = len(close)
            step = frac * scale
            local = step * (np.arange(k) - (k - 1) / 2.0)
            for j, idx in enumerate(close):
                offsets[idx] = local[j]
    return offsets


LINESTYLES = cycle(["-", "--", "-.", ":"])
MARKERS = cycle(["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"])

METRIC_LABELS = {
    "avg_test_error": r"Average MSE$_{\mathrm{test}}$",
    "precision_at_k": r"Precision at $k$",
    "ground_truth_hit": r"Ground-truth hit",
    "avg_size": r"Average expression size",
}


def _collect_sorted_k_for_benchmark(func, metric, results_dict, criteria):
    """Return (ks_sorted_str, ks_sorted_int) across all criteria for this benchmark."""
    ks_all = set()
    for name in criteria:
        key = f"{name}_{func}"
        if key in results_dict and metric in results_dict[key].index:
            ks_all.update(results_dict[key].columns)
    if not ks_all:
        return [], []
    ks_sorted = sorted(
        ks_all, key=lambda c: int(c) if str(c).isdigit() else float("inf")
    )
    xs_int = [int(c) if str(c).isdigit() else c for c in ks_sorted]
    return ks_sorted, xs_int


def _draw_hit_first_k_bar(
    ax,
    func,
    metric,
    results_dict,
    criteria,
    bar_color="#1f77b4",
    miss_color="#aaaaaa",
    miss_hatch="//",
):
    ks_sorted, xs_int = _collect_sorted_k_for_benchmark(
        func, metric, results_dict, criteria
    )
    if not ks_sorted:
        ax.set_title(f"{func} — {metric} (no data)")
        ax.axis("off")
        return

    first_k = []
    for name in criteria:
        key = f"{name}_{func}"
        if key not in results_dict or metric not in results_dict[key].index:
            first_k.append(None)
            continue
        s = results_dict[key].loc[metric, ks_sorted].astype(float).values
        idx = np.argmax(s == 1) if np.any(s == 1) else None
        first_k.append(xs_int[idx] if idx is not None else None)

    y_idx = np.arange(len(criteria))
    hits_mask = [v is not None for v in first_k]
    miss_mask = [not m for m in hits_mask]
    hit_values = [v if v is not None else 0 for v in first_k]

    if any(hits_mask):
        ax.barh(
            y_idx,
            [hit_values[i] for i in range(len(criteria))],
            color=[bar_color if hits_mask[i] else "none" for i in range(len(criteria))],
            edgecolor="none",
        )
    if any(miss_mask):
        eps = max(0.02 * (max(xs_int) or 1), 0.5)  # tiny visible stub
        ax.barh(
            y_idx,
            [eps if miss_mask[i] else 0 for i in range(len(criteria))],
            color=miss_color,
            edgecolor="none",
            hatch=miss_hatch,
            alpha=0.8,
        )

    for i, v in enumerate(first_k):
        if v is None:
            ax.text(0, i, "×", va="center", ha="left", fontsize=9, color="#333")
        else:
            ax.text(v, i, f"  {v}", va="center", ha="left", fontsize=9)

    ax.set_title(f"{func}", fontsize=11)
    ax.set_yticks(y_idx)
    ax.set_yticklabels(criteria)
    ax.set_xlim(0, (max(xs_int) if xs_int else 1) + 1)
    ax.grid(axis="x", alpha=0.25)


def _draw_one_benchmark(
    ax,
    func,
    metric,
    results_dict,
    criteria,
    k_max,
    unique_dashes=True,
    unique_markers=True,
    alpha=0.9,
    linewidth=1.8,
    outline=True,
    change_point_markers=True,
    steps=False,
    tie_jitter=True,
    tie_jitter_frac=0.001,
    end_labels=True,
    end_label_frac=0.0015,
    hit_view="heatmap",
):
    Functions_name = {
        "f1": r"f$_{\mathrm{1}}$",
        "f2": r"f$_{\mathrm{2}}$",
        "f3": r"f$_{\mathrm{3}}$",
        "f4": r"f$_{\mathrm{4}}$",
        "f5": r"f$_{\mathrm{5}}$",
        "f6": r"f$_{\mathrm{6}}$",
        "f7": r"f$_{\mathrm{7}}$",
    }

    metric_key = metric.lower().replace(" ", "_")
    if metric_key in {"ground_truth_hit", "ground_truth_hit_rate", "hit"}:
        if hit_view == "bar":
            _draw_hit_first_k_bar(ax, func, metric, results_dict, criteria)
            return
        elif hit_view == "line":
            pass
        else:
            ks_sorted, xs = _collect_sorted_k_for_benchmark(
                func, metric, results_dict, criteria
            )
            if not ks_sorted:
                ax.set_title(f"{func} — {metric} (no data)")
                ax.axis("off")
                return
            from matplotlib.colors import ListedColormap

            M = np.full((len(criteria), len(ks_sorted)), np.nan, dtype=float)
            for r, name in enumerate(criteria):
                key = f"{name}_{func}"
                if key in results_dict and metric in results_dict[key].index:
                    vals = results_dict[key].loc[metric, ks_sorted].astype(float)
                    M[r, : len(vals)] = vals.values
            cmap = ListedColormap(["#1f77b4", "#d62728"])  # blue (0), red (1)
            cmap.set_bad(color="#e0e0e0")
            ax.set_title(f"{Functions_name[func]}", fontsize=11)
            step = max(1, len(xs) // 8)
            ax.set_xticks(range(0, len(xs), step))
            ax.set_xticklabels([xs[i] for i in range(0, len(xs), step)])
            ax.set_xlabel("k top models")
            ax.set_yticks(range(len(criteria)))
            ax.set_yticklabels(criteria)
            ax.set_xticks(np.arange(-0.5, len(xs), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(criteria), 1), minor=True)
            ax.grid(which="minor", color="black", linewidth=0.5, alpha=0.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            return

    series_keys = list(criteria)
    x = None
    Y = []

    for name in series_keys:
        key = f"{name}_{func}"
        if key not in results_dict:
            continue
        df_plot = results_dict[key]
        if (metric not in df_plot.index) or df_plot.empty:
            continue

        cols_sorted = sorted(
            df_plot.columns, key=lambda c: int(c) if str(c).isdigit() else float("inf")
        )
        x = np.asarray(
            [int(c) if str(c).isdigit() else c for c in cols_sorted], dtype=float
        )
        y = df_plot.loc[metric, cols_sorted].astype(float).values
        Y.append((name, y))

    if not Y:
        ax.set_title(f"{func} — {metric} (no data)")
        ax.axis("off")
        return

    names, Ys = zip(*Y)

    Ys_adj = _tie_jitter(list(Ys), frac=tie_jitter_frac) if tie_jitter else Ys

    ax.set_title(f"{Functions_name[func]}", fontsize=19)
    ax.set_xticks(range(0, k_max + 1, 5))
    if metric in ("precision_at_k"):
        ax.set_ylim(0, 1)
    elif metric in ("ground_truth_hit"):
        ax.set_ylim(-0.5, 1.5)

    ls_cycle = cycle(["-", "--", "-.", ":"]) if unique_dashes else cycle(["-"])
    mk_cycle = (
        cycle(["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"])
        if unique_markers
        else cycle([None])
    )

    for name, y in zip(names, Ys_adj):
        ls = next(ls_cycle)
        mk = next(mk_cycle)
        markevery = _change_points(y) if change_point_markers else None
        drawstyle = "steps-post" if steps else "default"

        (line,) = ax.plot(
            x,
            y,
            linestyle=ls,
            marker=mk,
            markevery=markevery,
            linewidth=linewidth,
            alpha=alpha,
            label=str(name),
            drawstyle=drawstyle,
        )
        if outline:
            line.set_path_effects(
                [pe.Stroke(linewidth=linewidth + 1.2, foreground="white"), pe.Normal()]
            )

    if end_labels:
        y_last = [y[-1] for y in Ys_adj]
        offs = _end_label_offsets(y_last, frac=end_label_frac)
        xpad = (x[-1] - x[0]) * 0.02 if len(x) > 1 else 0.5
        for name, y, dy in zip(names, Ys_adj, offs):
            ax.text(
                x[-1] + xpad, y[-1] + dy, str(name), va="center", ha="left", fontsize=9
            )
        ax.margins(x=0.15)


def plot_benchmarks_grid_per_metric(
    criteria,
    metrics,
    results_dict,
    models_path,
    k_max,
    unique_dashes=True,
    unique_markers=True,
    alpha=0.9,
    linewidth=1.8,
    outline=True,
    change_point_markers=True,
    steps=False,
    tie_jitter=True,
    tie_jitter_frac=0.001,
    end_labels=True,
    end_label_frac=0.0015,
    legend_scope="figure",
    save_dir="plots",
    dpi=300,
    show=True,
    nrows=2,
    ncols=4,
    hit_view="heatmap",
):
    import os

    os.makedirs(save_dir, exist_ok=True)

    funcs = list(models_path.keys())
    assert nrows * ncols >= len(funcs), "Grid too small for number of benchmarks."

    for metric in metrics:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(7 * ncols, 3.6 * nrows), sharex=True
        )
        axes = np.array(axes).reshape(nrows, ncols)

        for i, func in enumerate(funcs):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            _draw_one_benchmark(
                ax,
                func,
                metric,
                results_dict,
                criteria,
                k_max,
                unique_dashes=unique_dashes,
                unique_markers=unique_markers,
                alpha=alpha,
                linewidth=linewidth,
                outline=outline,
                change_point_markers=change_point_markers,
                steps=steps,
                tie_jitter=tie_jitter,
                tie_jitter_frac=tie_jitter_frac,
                end_labels=end_labels,
                end_label_frac=end_label_frac,
                hit_view=hit_view,
            )

        for j in range(len(funcs), nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")

        for i, ax in enumerate(axes.ravel()):
            r, c = divmod(i, ncols)
            if r * ncols + c >= len(funcs):
                continue
            ax.tick_params(axis="x", which="both", labelbottom=True)

        for c in range(ncols):
            axes[-1, c].set_xlabel("k top models")

        if legend_scope == "figure":
            handles, labels = None, None

            for ax0 in axes.ravel():
                h, l = ax0.get_legend_handles_labels()  # noqa
                if "MSE_train_opt" in l:
                    l[0] = r"MSE$_{\mathrm{train}}$"
                    l[-1] = r"Err$_{\mathrm{in}}$"
                if h:
                    handles, labels = h, l
                    break
            if handles:
                flat = axes.ravel()

                unused_axes = flat[len(funcs) :]

                if len(unused_axes) > 0:
                    leg_ax = unused_axes[0]
                    leg_ax.axis("off")
                    leg_ax.legend(
                        handles,
                        labels,
                        loc="center",
                        frameon=True,
                        title="criterion",
                        ncol=1 if len(labels) <= 6 else 2,
                        fontsize=17,
                        title_fontsize=17,
                    )
                else:
                    fig.legend(
                        handles,
                        labels,
                        title="criterion",
                        loc="lower center",
                        ncol=min(6, len(labels)),
                        frameon=True,
                        bbox_to_anchor=(0.5, 0.12),
                        borderaxespad=0.5,
                        fontsize=17,  # <-- label text size
                        title_fontsize=17,  # <-- title size
                    )

        if metric == "avg_test_error":
            for i, ax in enumerate(axes.ravel()):
                r, c = divmod(i, ncols)
                if r * ncols + c >= len(funcs):
                    continue

                fmt = ScalarFormatter(useMathText=True)
                fmt.set_scientific(True)
                fmt.set_powerlimits((0, 0))  # force 1e±N
                ax.yaxis.set_major_formatter(fmt)
                ax.ticklabel_format(
                    axis="y", style="sci", scilimits=(0, 0), useMathText=True
                )

            fig.canvas.draw()

            for i, ax in enumerate(axes.ravel()):
                r, c = divmod(i, ncols)
                if r * ncols + c >= len(funcs) or c != 0:
                    continue
                off = ax.yaxis.get_offset_text()
                off.set_va("bottom")
                off.set_fontsize(10)
        metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))
        fig.supylabel(metric_label, x=0.01, fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88, right=0.86)  # room for legend/end labels

        # out_path = f"{save_dir}/cluster_benchmarks_grid_{metric}.png"
        # fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        out_path = f"{save_dir}/cluster_benchmarks_grid_{metric}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        # if show:
        #     plt.show()
        # else:
        #     plt.close(fig)


def set_gecco_acm_matplotlib_style(base=16):
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": base,  # default text
            "axes.titlesize": base + 2,  # subplot titles
            "axes.labelsize": base + 1,  # axis labels
            "xtick.labelsize": base,  # tick labels
            "ytick.labelsize": base,
            "legend.fontsize": base,  # legend labels
            "legend.title_fontsize": base + 1,  # legend title
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "savefig.transparent": False,
        }
    )


set_gecco_acm_matplotlib_style()
# plt.rcdefaults()

plot_benchmarks_grid_per_metric(
    criteria,
    metrics,
    results_dict,
    models_path,
    k_max,
    unique_dashes=True,
    alpha=0.9,
    linewidth=3,
    outline=True,
    unique_markers=False,
    change_point_markers=True,
    steps=False,
    tie_jitter=True,
    tie_jitter_frac=0.007,
    end_labels=False,
    legend_scope="figure",
    nrows=4,
    ncols=2,
    save_dir="plots",
    dpi=600,
    show=True,
    hit_view="bar",  # -- "heatmap" | "bar" | "line"
)
