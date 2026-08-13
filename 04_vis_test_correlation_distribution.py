"""
Prepare results for visualization, including correlation histograms and
example true and predicted maps.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def significance_test_distributions(r_b, r_w, c_b, c_w):
    """Perform a significance test between bin and weighted r_2 and corr"""
    u_r2 = mannwhitneyu(r_b, r_w, alternative="two-sided")
    u_corr = mannwhitneyu(c_b, c_w, alternative="two-sided")
    print(f"R^2 Mann-Whitney U test: U={u_r2.statistic}, p={u_r2.pvalue}")
    if u_r2.pvalue < 0.05:
        print("R^2 distributions are significantly different")
    print(f"Corr Mann-Whitney U test: U={u_corr.statistic}, p={u_corr.pvalue}")
    if u_corr.pvalue < 0.05:
        print("Corr distributions are significantly different")

    return u_r2, u_corr


def load_subject_results(results_dir, sub, out_flag):
    """Load real and null result CSVs for a given subject."""
    real = pd.read_csv(os.path.join(results_dir, f"{sub}_results.csv"))
    null = pd.read_csv(
        os.path.join(results_dir, f"{sub}_results{out_flag}.csv")
    )
    return real, null


def collect_metrics(df, corr_list, r2_list):
    """Accumulate per-subject correlation and R2 values into running lists."""
    corr_list.extend(df["mean_corr_s"].values)
    r2_list.extend(df["mean_r2_s"].values)
    return corr_list, r2_list


def plot_overlaying_density_curves(c_n, c_b, c_w, out_imgs_dir):
    """Create continuous bell-like curves using kernel density estimation"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    kde_null = gaussian_kde(c_n)
    kde_bin = gaussian_kde(c_b)
    kde_wtd = gaussian_kde(c_w)

    # Create x values for smooth curves
    x_min = min(np.min(c_n), np.min(c_b), np.min(c_w))
    x_max = max(np.max(c_n), np.max(c_b), np.max(c_w))
    x_vals = np.linspace(x_min, x_max, 300)

    # Plot smooth density curves
    ax.plot(
        x_vals,
        kde_null(x_vals),
        color="lightgray",
        linewidth=3,
        label="Null Model",
    )
    ax.fill_between(x_vals, kde_null(x_vals), alpha=0.3, color="lightgray")

    ax.plot(
        x_vals,
        kde_bin(x_vals),
        color="mediumorchid",
        linewidth=3,
        label="Binary Model",
    )
    ax.fill_between(x_vals, kde_bin(x_vals), alpha=0.3, color="mediumorchid")

    ax.plot(
        x_vals,
        kde_wtd(x_vals),
        color="mediumseagreen",
        linewidth=3,
        label="Weighted Model",
    )
    ax.fill_between(x_vals, kde_wtd(x_vals), alpha=0.3, color="mediumseagreen")

    # Set x-axis ticks to increments of 0.1
    ax.set_xticks(np.arange(-0.6, 1.1, 0.1))

    # Add median lines
    ax.axvline(
        np.median(c_n),
        color="gray",
        linestyle=":",
        linewidth=2,
        alpha=0.6,
        label="Null Median",
    )
    ax.axvline(
        np.median(c_b),
        color="purple",
        linestyle=":",
        linewidth=2,
        alpha=0.6,
        label="Binary Median",
    )
    ax.axvline(
        np.median(c_w),
        color="darkgreen",
        linestyle=":",
        linewidth=2,
        alpha=0.6,
        label="Weighted Median",
    )

    ax.set_xlabel("Correlation", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title("Distribution of Correlation values", fontsize=18)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    plt.xticks(rotation=35)
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(out_imgs_dir, "correlation_distributions_overlay.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


def main(res_wtd_dir, res_bin_dir, out_imgs_dir, out_flag):
    """
    Main function to load results, collect metrics, perform significance tests,
    and plot overlaying density curves for correlation values.
    Parameters
    ----------
    res_wtd_dir : str
        Directory containing weighted model results.
    res_bin_dir : str
        Directory containing binary model results.
    out_imgs_dir : str
        Directory to save output images.
    out_flag : str
        Flag to identify null model results (e.g., "_null").
    """
    corr_real_bin, corr_null_bin, r2_real_bin, r2_null_bin = [], [], [], []
    corr_real_wtd, corr_null_wtd, r2_real_wtd, r2_null_wtd = [], [], [], []

    for sub in SUBJECTS:
        # 1. Bin model results
        real_bin, null_bin = load_subject_results(res_bin_dir, sub, out_flag)
        collect_metrics(real_bin, corr_real_bin, r2_real_bin)
        collect_metrics(null_bin, corr_null_bin, r2_null_bin)

        # 2. Weighted model results
        real_wtd, null_wtd = load_subject_results(res_wtd_dir, sub, out_flag)
        collect_metrics(real_wtd, corr_real_wtd, r2_real_wtd)
        collect_metrics(null_wtd, corr_null_wtd, r2_null_wtd)

    u_r2, u_corr = significance_test_distributions(
        r2_real_bin, r2_real_wtd, corr_real_bin, corr_real_wtd
    )

    plot_overlaying_density_curves(
        corr_null_wtd, corr_real_bin, corr_real_wtd, out_imgs_dir
    )


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    res_wtd_dir = os.path.join(project_dir, "results_weighted")
    res_bin_dir = os.path.join(project_dir, "results_binary")
    out_imgs_dir = os.path.join(project_dir, "results_visualization")
    os.makedirs(out_imgs_dir, exist_ok=True)
    out_flag = "_nullmodel-shuffle-conditions"
    main(res_wtd_dir, res_bin_dir, out_imgs_dir, out_flag)
