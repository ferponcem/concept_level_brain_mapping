import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import plotting
from scipy.stats import wilcoxon

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def _plot_diff_histogram(diffs, title="", savepath=None):
    """Helper function to plot histogram of differences independently."""
    fig = plt.figure(figsize=(12, 4))
    bins = np.arange(-0.35, 0.4, 0.05)
    plt.hist(
        diffs,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
        color="darkseagreen",  # "palevioletred"
        density=True,
    )
    plt.axvline(0, color="navy", linestyle="--", linewidth=2)
    plt.title(
        f"Histogram of differences {title}",
        fontsize=18,
    )
    plt.xlabel("Difference in Pearson r", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(bins, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


def histogram_results(
    results,
    col_one,
    col_two,
    col_one_tl,
    col_two_tl,
    use_threshold=True,
    savepath=None,
):
    """Plot histograms and print summary statistics of the results DataFrame"""
    if use_threshold:
        col_one += "_thr"
        col_two += "_thr"
        col_one_values = results[col_one].values
        col_two_values = results[col_two].values
        diffs = results[col_two] - results[col_one]
        title_suffix = " (thresholded overlap)"
    else:
        col_one += "_all"
        col_two += "_all"
        col_one_values = results[col_one].values
        col_two_values = results[col_two].values
        diffs = results[col_two] - results[col_one]
        title_suffix = " (all voxels)"

    # Plot a split violin plot of r values
    r_one = col_one_values[~np.isnan(col_one_values)]
    r_two = col_two_values[~np.isnan(col_two_values)]
    data = pd.DataFrame(
        {f"{col_one_tl} Model": r_one, f"{col_two_tl} Model": r_two}
    )
    mean_one = np.mean(r_one)
    mean_two = np.mean(r_two)
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=data,
        palette=["skyblue", "pink"],
        split=True,
        gap=-0.23,
    )
    plt.axhline(
        mean_one, color="blue", linestyle="--", label=f"{col_one_tl} Mean"
    )
    plt.axhline(
        mean_two, color="red", linestyle="--", label=f"{col_two_tl} Mean"
    )
    plt.title(f"Model Correlations{title_suffix}")
    plt.ylabel("Pearson r")

    # Find common range for both histograms
    all_values = np.concatenate([col_one_values, col_two_values])
    all_values = all_values[~np.isnan(all_values)]  # Remove NaN values

    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    bins = np.arange(-0.35, 0.4, 0.05)
    axes[0].hist(
        col_one_values,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
        color="skyblue",
    )
    axes[0].set_title(f"{col_one_tl} model correlations {title_suffix}")
    axes[0].set_xlabel("Pearson r")

    axes[1].hist(
        col_two_values, bins=bins, edgecolor="black", alpha=0.8, color="pink"
    )
    axes[1].set_title(f"{col_two_tl} model correlations {title_suffix}")
    axes[1].set_xlabel("Pearson r")

    diffs = diffs[~np.isnan(diffs)]  # Remove NaN values
    axes[2].hist(
        diffs, bins=bins, edgecolor="black", alpha=0.8, color="purple"
    )
    axes[2].set_title(
        f"Differences in correlation values ({col_two_tl} - {col_one_tl}){title_suffix}",
        fontsize=16,
    )
    axes[2].set_xlabel("Difference in Pearson r")

    # Plot an extra histogram of differences with KDE
    _plot_diff_histogram(
        diffs,
        title=f"({col_two_tl} - {col_one_tl})",
        savepath=f"figs/diff_histogram_{col_two_tl}_{col_one_tl}.png",
    )

    # Summary counts
    col_one_better = np.sum(col_one_values > col_two_values)
    col_two_better = np.sum(col_two_values > col_one_values)
    col_one_large = np.sum(col_one_values > 0.3)
    col_two_large = np.sum(col_two_values > 0.3)
    print(
        f"{col_one_tl} model better: {col_one_better}, ({col_one_large} with r > 0.3)"
    )
    print(
        f"{col_two_tl} model better: {col_two_better}, ({col_two_large} with r > 0.3)"
    )

    if savepath:
        fig.savefig(savepath, dpi=300)
    plt.show()


def plot_example_maps(
    concept,
    term,
    bin_dir,
    wtd_dir,
    meta_dir,
    meta_flag,
    thr_meta,
    thr_bin,
    thr_wtd,
    savepath=None,
):
    """Plot example maps for a given concept/term from binary, weighted and
    neurosynth maps."""
    bin_map = os.path.join(bin_dir, f"mean_{concept}.nii.gz")
    wtd_map = os.path.join(wtd_dir, f"mean_{concept}.nii.gz")
    nsynth_map = os.path.join(meta_dir, f"{meta_flag}_{term}.nii.gz")

    cut_coords = plotting.find_cut_slices(nsynth_map, direction="z", n_cuts=3)
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    for ax, (img, title) in zip(
        axes,
        [
            (nsynth_map, f"Metanalysis"),
            (bin_map, f"Binary"),
            (wtd_map, f"Weighted"),
        ],
    ):
        if title.startswith("Metanalysis"):
            threshold = thr_meta
            case = "meta"
        elif title.startswith("Binary"):
            threshold = thr_bin
            case = "bin"
        else:
            threshold = thr_wtd
            case = "wtd"

        plotting.plot_stat_map(
            img,
            title=title,
            display_mode="z",
            cut_coords=(-10, 0, 48),  # cut_coords,
            # threshold=4 if case == "meta" else 1, # threshold,
            draw_cross=False,
            axes=ax,
            colorbar=True,
            cmap="RdBu_r",
            annotate=False,
            transparency=img,
            vmax=7 if case == "meta" else 3,
        )
        plt.suptitle(f"{term.replace('_', ' ')}", fontsize=16, y=0.95)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


def wilcoxon_test(results, col_one, col_two, alpha=0.05, use_threshold=False):
    """Test significance of difference between binary and weighted model"""
    if use_threshold:
        col_one += "_thr"
        col_two += "_thr"
        r_one = results[col_one].dropna()
        r_two = results[col_two].dropna()
    else:
        col_one += "_all"
        col_two += "_all"
        r_one = results[col_one].dropna()
        r_two = results[col_two].dropna()

    stat, p_value = wilcoxon(r_two, r_one, alternative="greater")
    print(f"Wilcoxon signed-rank test: stat = {stat:.3f}, p = {p_value:.3f}")
    if p_value < alpha:
        print(
            f"=> {col_two} has significantly higher correlations than {col_one}"
        )
    else:
        print("=> No significant difference between models")

    return stat, p_value


def main(results_bin_wtd, bin_maps, wtd_maps, meta_dir, specter_flag):
    """
    Main function to load results, plot histograms, and perform Wilcoxon test.
    Parameters
    ----------
    results_bin_wtd : str
        Path to the CSV file containing binary and weighted model results.
    bin_maps : str
        Directory containing binary model maps.
    wtd_maps : str
        Directory containing weighted model maps.
    specter_dir : str
        Directory containing meta-analysis maps.
    specter_flag : str
        Flag to identify meta-analysis maps (e.g., "specter_nsynth_mkda").
    """
    results_bin_wtd = pd.read_csv(results_bin_wtd)

    histogram_results(
        results_bin_wtd,
        col_one="r_bin",
        col_two="r_wtd",
        col_one_tl="Binary",
        col_two_tl="Weighted",
        use_threshold=False,
    )

    _, _ = wilcoxon_test(
        results_bin_wtd,
        col_one="r_group",
        col_two="r_bin",
        use_threshold=False,
    )

    # Plot example maps for biggest gain, no gain and negative gain cases
    # sentence_comprehension, visual_representation, and narrative_comprehension
    concept = "arithmetic_processing"
    term = concept.replace("_", " ")
    plot_example_maps(
        concept=concept,
        term=concept,
        bin_dir=bin_maps,
        wtd_dir=wtd_maps,
        meta_dir=meta_dir,
        meta_flag=specter_flag,
        thr_meta=results_bin_wtd.loc[
            results_bin_wtd["term"] == term, "thr_meta"
        ].values[0],
        thr_bin=results_bin_wtd.loc[
            results_bin_wtd["term"] == term, "thr_bin"
        ].values[0],
        thr_wtd=results_bin_wtd.loc[
            results_bin_wtd["term"] == term, "thr_wtd"
        ].values[0],
    )


if __name__ == "__main__":
    nimare_dir = "nimare_data"
    wtd_dir = "cognitive_components_subout_regsel_contrasts"
    bin_dir = "cognitive_components_regsel_contrasts"
    meta_dir = "example_data"
    specter_flag = "specter_nsynth_mkda"
    meta_maps = os.path.join(nimare_dir, "nsynth_maps")
    wtd_maps = os.path.join(wtd_dir, "components_mean")
    bin_maps = os.path.join(bin_dir, "components_conj")
    results_bin_wtd = os.path.join(nimare_dir, f"specter_corrs_90.csv")
    main(results_bin_wtd, bin_maps, wtd_maps, meta_maps, specter_flag)
