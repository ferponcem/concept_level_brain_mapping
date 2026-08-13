"""
Compute contrast reliability for IBC contrasts.

Reliability is estimated from the similarity between repeated runs of the
same contrast. The analysis is performed on volume data using the
Schaefer100 parcellation.

For each contrast and brain region, cosine similarity is computed between
repeated runs within subjects and summarized across subjects.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ibc_public.utils_data import CONTRASTS, SMOOTH_DERIVATIVES, data_parser
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_roi

from utils_components import append_task_id, flatten, get_mask

DEFAULT_EXCLUDED_TASKS = ("movie_tasks", "retinotopy")
SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def get_task_list(task_file, exclude=DEFAULT_EXCLUDED_TASKS):
    """Get the list of tasks to include in the analysis"""
    with open(task_file) as f:
        tasks = yaml.safe_load(f)
    all_tasks = flatten(list(tasks.values()))
    excluded_tasks = flatten(
        [tasks[group] for group in exclude if group in tasks]
    )
    return [task for task in all_tasks if task not in excluded_tasks]


def average_row_cosine_similarity(X):
    """Compute average cosine distance across the rows of X"""
    n_rows = X.shape[0]
    cds = []
    for i in range(n_rows):
        for j in range(i):
            stat = np.dot(X[i], X[j]) / (
                np.linalg.norm(X[i]) * np.linalg.norm(X[j])
            )
            cds.append(stat)
    return np.median(cds)


def make_vol_data_dict(df, masker, contrasts=None):
    """
    Make a dictionary with the masked voxel data for each subject and contrast.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'subject', 'contrast', and 'path'.
    masker : NiftiMasker
        Fitted masker used to transform the images.
    contrasts : list of str, optional
        List of contrasts to include. If None, all contrasts in df are used.

    Returns
    -------
    sub_data : dict
        Structure sub_data[subject][contrast] = shape (n_maps, n_voxels)
    """
    subjects = df.subject.unique()
    contrasts = contrasts if contrasts is not None else df.contrast.unique()

    sub_data = {sub: {} for sub in subjects}

    for sub in subjects:
        print("Processing subject %s" % sub)
        df_sub = df[df.subject == sub]

        for contrast in contrasts:
            df_con = df_sub[df_sub.contrast == contrast]

            if df_con.empty:
                continue

            X = []
            for path in df_con.path.values:
                arr = masker.transform(path)
                arr = np.nan_to_num(arr)
                X.append(arr)

            if X:
                sub_data[sub][contrast] = np.vstack(X)

    return sub_data


def compute_contrast_scores(contrast, dic, region_indices):
    """Compute median cosine reliability for one contrast."""
    n_regions = len(region_indices)
    region_scores = np.full(n_regions, np.nan)

    if (contrast.startswith("LocalizerAbstraction")) or (
        contrast.startswith("LPPLocalizer")
    ):
        print(
            f"Special case for {contrast}: computing inter-subject reliability"
        )
        data = [
            dic[sub][contrast]
            for sub in SUBJECTS
            if contrast in dic.get(sub, {})
        ]
        data = np.vstack(data)
        for r, idx in enumerate(region_indices):
            region_data_masked = data[:, idx]
            if region_data_masked.shape[0] > 1:
                region_scores[r] = average_row_cosine_similarity(
                    region_data_masked
                )
        return contrast, region_scores

    failed_subjects = []
    for r, idx in enumerate(region_indices):
        region_stat = []
        for sub in SUBJECTS:
            if contrast in dic.get(sub, {}):
                data = dic[sub][contrast][:, idx]
                if data.shape[0] > 1:
                    cos_dist = average_row_cosine_similarity(data)
                    region_stat.append(cos_dist)
                elif sub not in failed_subjects:
                    print(f"Not enough runs {sub}, {contrast}")
                    failed_subjects.append(sub)
        if region_stat:
            region_scores[r] = np.median(region_stat)

    return contrast, region_scores


def count_reliable_regions(score_df, threshold=0.5):
    """Count the number of regions above the threshold for each contrast"""
    reliable_counts = (score_df > threshold).sum(axis=0)
    non_reliable = reliable_counts[reliable_counts == 0]
    return reliable_counts, non_reliable


def plot_roi_check(region_labels, wanted_labels, ibc_masker):
    """Plot the selected ROIs to check consistency"""
    for i in wanted_labels:
        # Create a binary mask for the current region
        region_mask = (region_labels == i).astype(int)
        # Inverse transform the mask back to the original space
        roi_img = ibc_masker.inverse_transform(region_mask)
        # Create a new figure for each plot
        plt.figure(figsize=(6, 6))
        # Plot the ROI
        plot_roi(
            roi_img,
            title=f"Region {i + 1}",
            display_mode="ortho",
            colorbar=False,
            draw_cross=False,
        )
        plt.show()


def plot_high_regions_as_rois(
    score_df,
    ibc_masker,
    region_labels,
    wanted_contrasts=slice(0, 10),
    threshold=0.5,
):
    """Plot regions with high reliability as ROIs.
    Parameters
    ----------
    score_df : pandas.DataFrame
        DataFrame with regions as rows and contrasts as columns.
    ibc_masker : NiftiMasker
        Fitted masker used to reconstruct ROI images.
    region_labels : ndarray
        Atlas label for each voxel in masker space.
    wanted_contrasts : slice or list of str, optional
        Contrasts to plot.
    threshold : float, optional
        Minimum reliability required for a region to be displayed.
    """
    if isinstance(wanted_contrasts, slice):
        contrasts = score_df.columns[wanted_contrasts]
    elif isinstance(wanted_contrasts, list):
        contrasts = [
            cont for cont in wanted_contrasts if cont in score_df.columns
        ]
    else:
        raise ValueError(
            "whic_contrasts must be a slice or a list of column names"
        )
    for cont in contrasts:
        # Find regions with high reliability
        high_reliability = score_df[cont] > threshold
        reliable_regions = score_df.index[high_reliability].tolist()

        if not reliable_regions:
            print(f"No reliable regions found for {cont}")
            continue

        # Combine all reliable regions for the current contrast into one mask
        combined_mask = np.zeros_like(region_labels, dtype=int)
        for idx, region in enumerate(reliable_regions, start=1):
            region_id = int(region.split("_")[1])
            combined_mask[region_labels == region_id] = idx

        # Create a single ROI image for all reliable regions
        combined_roi_img = ibc_masker.inverse_transform(combined_mask)
        # Create a colormap with distinct colors for each region
        unique_regions = np.unique(combined_mask)
        unique_regions = unique_regions[
            unique_regions > 0
        ]  # Exclude background
        n_colors = len(unique_regions) + 1

        tab20b_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
        tab20c_colors = plt.cm.tab20c(np.linspace(0, 1, 20))

        # Drop the last 4 grayish tones from tab20c
        tab20c_colors = tab20c_colors[:-4]

        # Stack them together and trim if needed
        colors = np.vstack([tab20b_colors, tab20c_colors])
        cmap = ListedColormap(colors[:n_colors])

        plot_roi(
            combined_roi_img,
            title=f"Regions with reliable signal {cont}",
            colorbar=True,
            draw_cross=False,
            cmap=cmap,
            vmax=n_colors,  # Set the maximum value for the colormap
        )
        plt.show()


def main(
    project_dir,
    task_file="all_tasks.yml",
    n_jobs=20,
    n_rois=100,
    reliability_threshold=0.5,
):
    """Run the complete contrast reliability analysis.

    Parameters
    ----------
    project_dir : path-like
        Root directory where analysis outputs are stored.
    task_file : path-like, optional
        YAML file defining task groups.
    n_jobs : int, optional
        Number of parallel jobs used for contrast-level computation.
    n_rois : int, optional
        Number of Schaefer atlas regions.
    reliability_threshold : float, optional
        Threshold used to identify reliable regions.
    """

    work_dir = os.path.join(project_dir, "contrast_reliability")
    work_dir.mkdir(parents=True, exist_ok=True)

    score_file = os.path.join(work_dir, "cosine_dist_volume_conts.csv")
    results_yaml = os.path.join(work_dir, "nonreliable_contrast_volume.yml")

    tasks = get_task_list(task_file)
    db = data_parser(
        derivatives=SMOOTH_DERIVATIVES,
        subject_list=SUBJECTS,
        task_list=tasks,
        conditions=CONTRASTS,
    )

    not_ffx = ~db["path"].str.contains("dir-ffx", na=False)
    df_bold = db.loc[not_ffx & db["modality"].str.contains("bold", na=False)]

    db_names = append_task_id(df_bold)
    mask = get_mask()
    masker = NiftiMasker(mask_img=mask, memory="nilearn_cache", verbose=0)
    masker.fit()

    atlas = fetch_atlas_schaefer_2018(n_rois=n_rois)
    atlas_resampled = resample_to_img(
        atlas["maps"], masker.mask_img_, interpolation="nearest"
    )
    region_labels = masker.transform(atlas_resampled).ravel().astype(int)
    region_indices = [
        np.where(region_labels == i + 1)[0] for i in range(n_rois)
    ]

    contrasts = db_names["contrast"].unique().tolist()

    if score_file.exists():
        score_df = pd.read_csv(score_file, index_col=0)
    else:
        data = make_vol_data_dict(
            db_names,
            masker=masker,
            contrasts=contrasts,
        )

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_contrast_scores)(contrast, data, region_indices)
            for contrast in contrasts
        )
        score_dict = {contrast: scores for contrast, scores in results}
        score_df = pd.DataFrame(
            score_dict, index=[f"region_{i+1}" for i in range(n_rois)]
        )
        score_df.to_csv(score_file)

    plot_high_regions_as_rois(
        score_df,
        masker,
        region_labels,
        wanted_contrasts=slice(0, 10),
        threshold=reliability_threshold,
    )

    _, non_reliable = count_reliable_regions(
        score_df,
        threshold=reliability_threshold,
    )

    with open(results_yaml, "w") as f:
        yaml.safe_dump(
            non_reliable.index.tolist(),
            f,
        )

    print(f"Results saved to {work_dir}")


if __name__ == "__main__":
    main(
        project_dir=".",  # Project directory here
    )
