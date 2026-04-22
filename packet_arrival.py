from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

plt.rcParams.update({
    "font.size": 18,          # base font
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.0,
})


# ----------------------------
# Data generation
# ----------------------------
def generate_packet_counts(N: int, lam: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=lam, size=N)


# ----------------------------
# Running statistics
# ----------------------------
def running_mean(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def running_variance(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for k in range(2, len(x) + 1):
        out[k - 1] = np.var(x[:k], ddof=1)
    return out


# ----------------------------
# Frame creation
# ----------------------------
def make_frame(
    counts: np.ndarray,
    lam: float,
    n: int,
    y_max_counts: float,
    hist_y_max: float,
    var_y_max: float,
    output_path: Path,
) -> None:

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1.2, 1, 1]}
    )

    ax_top, ax_hist, ax_stats = axes

    x = np.arange(1, len(counts) + 1)

    # ----------------------------
    # Top: time series
    # ----------------------------
    ax_top.vlines(x, 0, counts, linewidth=0.8)
    ax_top.axvspan(1, n, alpha=0.2)

    ax_top.set_xlim(0.5, len(counts) + 0.5)
    ax_top.set_ylim(0, y_max_counts)

    ax_top.set_ylabel("Packets")
    ax_top.set_xlabel("Time (ms)")

    # annotation
    ax_top.text(0.02, 0.85, f"n = {n}", transform=ax_top.transAxes)

    # ----------------------------
    # Middle: histogram
    # ----------------------------
    window_data = counts[:n]

    min_c = int(np.min(counts))
    max_c = int(np.max(counts))
    bins = np.arange(min_c - 0.5, max_c + 1.5, 1)

    ax_hist.hist(
        window_data,
        bins=bins,
        density=True,
        edgecolor="black",
        linewidth=1.0,
    )

    k = np.arange(min_c, max_c + 1)
    pmf = poisson.pmf(k, mu=lam)

    ax_hist.plot(k, pmf, marker='o')

    ax_hist.set_xlim(min_c - 0.5, max_c + 0.5)
    ax_hist.set_ylim(0, hist_y_max)

    ax_hist.set_ylabel("Probability")
    ax_hist.set_xlabel("number of packets")

    # ----------------------------
    # Bottom: running mean + variance
    # ----------------------------
    rm = running_mean(counts)
    rv = running_variance(counts)

    ax_stats.plot(x[:n], rm[:n], linewidth=1.2, label="Mean")
    ax_stats.plot(x[:n], rv[:n], linewidth=1.2, linestyle="--", label="Variance")

    ax_stats.axhline(lam, linewidth=1.0)
    ax_stats.axhline(lam, linewidth=1.0, linestyle="--")

    ax_stats.set_xlim(0.5, len(counts) + 0.5)
    ax_stats.set_ylim(0, var_y_max)

    ax_stats.set_xlabel("length of averaging window (ms)")
    ax_stats.set_ylabel("Value of estimator")
    ax_stats.text(0.02, 0.15, f"mean = {rm[n]:.2f}, variance = {rv[n]:.2f}", transform=ax_stats.transAxes)

    # minimal legend (optional)
    ax_stats.legend(frameon=False)

    # ----------------------------
    # Layout
    # ----------------------------
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------
# GIF creation
# ----------------------------
def make_gif(counts: np.ndarray, lam: float, output_file: str) -> None:
    tmp_dir = Path("gif_frames")
    tmp_dir.mkdir(exist_ok=True)

    N = len(counts)

    # Precompute stats for axis limits
    rm = running_mean(counts)
    rv = running_variance(counts)

    y_max_counts = max(counts) + 2
    hist_y_max = poisson.pmf(int(lam), lam) * 1.5
    var_y_max = max(np.nanmax(rv), lam) * 1.3

    # frame selection (log spaced = better pacing)
    frame_indices = np.unique(np.logspace(0, np.log10(N), 200).astype(int))

    frame_paths = []

    for i, n in enumerate(frame_indices):
        path = tmp_dir / f"frame_{i:03d}.png"

        make_frame(
            counts=counts,
            lam=lam,
            n=n,
            y_max_counts=y_max_counts,
            hist_y_max=hist_y_max,
            var_y_max=var_y_max,
            output_path=path,
        )

        frame_paths.append(path)

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(output_file, images, duration=0.1)

    print(f"Saved GIF: {output_file}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    lams = [3, 10, 30]
    for lam in lams:
        N = 5000
        seed = 3

        counts = generate_packet_counts(N, lam, seed)

        make_gif(counts, lam, f"packet_hist_evolution_lam_{lam:.0f}.gif")