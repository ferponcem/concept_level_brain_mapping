"""
Script to create visualizations for the comparison of variance explained by
binary vs. weighted design matrices in GLM analyses. Figure 2 in manuscript.
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from nilearn import plotting
from nilearn.image import load_img, math_img, mean_img

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def plot_voxelwise_diff_r2(dir_one, leg_one, dir_two, leg_two, save_path=None):
    """
    Plot voxelwise difference in R² between two models. The difference is
    computed as R²_model_two - R²_model_one, and the mean difference across
    subjects is plotted.
    Parameters
    ----------
    dir_one : str
        Directory containing R² images for the first model.
    leg_one : str
        Legend label for the first model.
    dir_two : str
        Directory containing R² images for the second model.
    leg_two : str
        Legend label for the second model.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    """
    all_diff_imgs = []
    imgs_one = []
    imgs_two = []

    for sub in SUBJECTS:
        if leg_one == "null":
            r2_flag_one = "cc_*_nullr2.nii.gz"
        else:
            r2_flag_one = "cc_*_r2.nii.gz"

        r2_path_one = os.path.join(dir_one, sub)
        r2_path_file_one = glob(os.path.join(r2_path_one, r2_flag_one))[0]
        imgs_one.append(r2_path_file_one)

        r2_flag_two = "cc_*_r2.nii.gz"
        r2_path_two = os.path.join(dir_two, sub)
        r2_path_file_two = glob(os.path.join(r2_path_two, r2_flag_two))[0]
        imgs_two.append(r2_path_file_two)

    mean_img_one = mean_img(imgs_one)
    mean_img_two = mean_img(imgs_two)

    mean_diff_img = math_img(
        "img2 - img1", img1=mean_img_one, img2=mean_img_two
    )

    fig = plt.figure(figsize=(10, 6))

    plotting.plot_stat_map(
        mean_diff_img,
        title=f"Mean voxelwise R² difference: {leg_two} - {leg_one}",
        draw_cross=False,
        vmax=0.5,
        cut_coords=(10, -31, 15),
        colorbar=True,
        threshold=0.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show()


def plot_violin_r2_nullmark(
    dir_bin, bin_leg, dir_wtd, leg_wtd, ttl="", ax=None, save_path=None
):
    """
    Plot the distribution of R² values for each subject, binary vs weighted.
    Add a small horizontal line to mark the null model R² for each subject.
    If `ax` is provided, plot on that axis; otherwise create a new figure.
    Parameters
    ----------
    dir_bin : str
        Directory containing binary model R² images.
    bin_leg : str
        Legend label for binary model.
    dir_wtd : str
        Directory containing weighted model R² images.
    leg_wtd : str
        Legend label for weighted model.
    ttl : str, optional
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    """
    all_data = []
    r2_null_flag = "cc_*_nullr2.nii.gz"
    r2_flag = "cc_*_r2.nii.gz"

    for model_type, folder in [
        (bin_leg, dir_bin),
        (leg_wtd, dir_wtd),
        ("Null", dir_wtd),
    ]:

        if model_type == "Null":
            r2_flag_use = r2_null_flag
        else:
            r2_flag_use = r2_flag

        for sub in SUBJECTS:
            r2_path = os.path.join(folder, sub)
            r2_path_file = glob(os.path.join(r2_path, r2_flag_use))[0]
            img = load_img(r2_path_file)
            data = img.get_fdata()
            valid_data = data[np.isfinite(data) & (data > 0)]
            all_data.extend([(sub, val, model_type) for val in valid_data])

    df = pd.DataFrame(all_data, columns=["Subject", "R2", "Model"])

    # Plot the distribution of R² values for each subject and model type
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    # Create split violin plot for binary vs weighted, with null median markers
    df_split = df[df["Model"] != "Null"].copy()
    df_null = df[df["Model"] == "Null"].copy()

    # Calculate null median for each subject
    null_medians = df_null.groupby("Subject")["R2"].median().reset_index()
    null_medians.columns = ["Subject", "Null_Median"]

    df_split = df[df["Model"].isin(["Binary", "Weighted"])].copy()

    sns.violinplot(
        data=df_split,
        x="Subject",
        y="R2",
        hue="Model",
        split=True,
        inner="box",
        ax=ax,
    )

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # Add horizontal lines for null medians
    for i, subject in enumerate(df_split["Subject"].unique()):
        null_median = null_medians[null_medians["Subject"] == subject][
            "Null_Median"
        ].iloc[0]
        ax.hlines(
            null_median,
            i - 0.4,
            i + 0.4,
            colors="darkred",
            linewidth=2,
            alpha=0.8,
        )

    custom_lines = Line2D(
        [0], [0], color="darkred", linewidth=2, label="Null Median"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(custom_lines)
    labels.append("Null Median")
    ax.legend(handles=handles, labels=labels, fontsize=14)

    ax.set_title(f"{ttl}", fontsize=18)
    ax.set_ylabel("R²", fontsize=16)
    ax.set_xlabel("", fontsize=16, rotation=20)
    ax.grid(True, linestyle="--", alpha=0.5)

    if created_fig:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

    if created_fig:
        plt.show()


def main(bin_dir, wtd_dir):
    """
    Main function to generate visualizations comparing binary and weighted
    models.
    Parameters
    ----------
    bin_dir : str
        Directory containing binary model R² images.
    wtd_dir : str
        Directory containing weighted model R² images.
    """
    plot_violin_r2_nullmark(
        bin_dir,
        "Binary",
        wtd_dir,
        "Weighted",
        ttl="R² Distribution: Binary vs Weighted Model",
        save_path="r2_binary_weighted_comparison.pdf",
    )

    plot_voxelwise_diff_r2(
        bin_dir,
        "Binary",
        wtd_dir,
        "Weighted",
        save_path="voxelwise_r2_diff.pdf",
    )


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(proj_dir, "cognitive_components")
    single_out_dir = os.path.join(proj_dir, "cognitive_components_subout")
