"""Compare the maps for each component from binary and weighted models:
Which one looks more like the neurosynth map?"""

import os
from glob import glob

import numpy as np
import pandas as pd
from joblib import Memory
from nilearn.maskers import NiftiMasker
from scipy.stats import pearsonr

from utils_components import get_mask

memory = Memory(location="nilearn_cache", verbose=1)
SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def get_components_list(work_dir):
    """Get the list of components in a directory"""
    comp_list = glob(os.path.join(work_dir, "mean_*.nii.gz"))
    comp_list = [comp for comp in comp_list if "r2" not in comp]
    return comp_list


def get_threshold(img_data, percentile=95.0):
    """Get the threshold that corresponds to a given percentile of non-zero
    voxels.

    Parameters
    ----------
    img_data : Array of extracted voxel values
        1D array of voxel values.
    Returns
    -------
    threshold : float
        The 95th percentile threshold value.
    """
    non_zero_data = img_data[img_data != 0]
    return float(np.percentile(non_zero_data, percentile))


@memory.cache
def _cached_transform(img_path, masker):
    """
    Load `img_path`, fit a NiftiMasker on `mask_img_path` and return masked
    1D array. This is cached by file paths, so repeated runs re-use the result.
    """
    arr = masker.transform(img_path).ravel()
    return np.asarray(arr, dtype=float)


def compute_corrs(
    meta_map, bin_map, wtd_map, masker, percentile=95, verbose=True
):
    """
    Compute voxelwise spatial correlation between sets of maps.

    Parameters
    ----------
    map1 : Niimg-like
        First z-map. Typically the neurosynth/meta-analytic map.
    map2 : Niimg-like
        Second z-map.
    map3 : Niimg-like
        Third z-map.
    masker : NiftiMasker
        Fitted masker defining the voxel grid and mask.
    percentile : float
        Percentile for thresholding the maps.
    verbose : bool
        Whether to print results.

    Returns
    -------
    r_bin : float
        Pearson correlation between map1 and map2 over suprathreshold voxels.
    r_wtd : float
        Pearson correlation between map1 and map3 over suprathreshold voxels.
    r_bin_all : float
        Pearson correlation between map1 and map2 over all voxels.
    r_wtd_all : float
        Pearson correlation between map1 and map3 over all voxels.
    thr_meta : float
        Threshold applied to map1.
    thr_group : float
        Threshold applied to map2.
    thr_bin : float
        Threshold applied to map3.
    """

    # Vectorize the maps using the masker
    data_meta = _cached_transform(meta_map, masker)
    data_bin = _cached_transform(bin_map, masker)
    data_wtd = _cached_transform(wtd_map, masker)

    # Keep only the positive parts of the maps
    data_meta[data_meta < 0] = 0
    data_bin[data_bin < 0] = 0
    data_wtd[data_wtd < 0] = 0

    # Determine threshold based on wtd map
    thr_meta = get_threshold(data_meta, percentile=percentile)
    thr_bin = get_threshold(data_bin, percentile=percentile)
    thr_wtd = get_threshold(data_wtd, percentile=percentile)

    # Apply thresholding
    mask_bin = (data_meta >= thr_meta) & (data_bin >= thr_bin)
    mask_wtd = (data_meta >= thr_meta) & (data_wtd >= thr_wtd)

    # Check if there are enough suprathreshold voxels
    if np.sum(mask_bin) < 5 or np.sum(mask_wtd) < 5:
        print("⚠️ Not enough overlapping suprathreshold voxels — filling NaNs")
        r_bin, r_wtd = np.nan, np.nan
    else:
        # Compute Pearson correlations over suprathreshold voxels
        r_bin, _ = pearsonr(data_meta[mask_bin], data_bin[mask_bin])
        r_wtd, _ = pearsonr(data_meta[mask_wtd], data_wtd[mask_wtd])

    # Correlation across all voxels
    r_bin_all, _ = pearsonr(data_meta, data_bin)
    r_wtd_all, _ = pearsonr(data_meta, data_wtd)

    if verbose:
        print("----> Voxelwise correlation results")
        print(f"Meta & bin (thr overlap): r = {r_bin:.3f}")
        print(f"Meta & wtd (thr overlap): r = {r_wtd:.3f}")
        print(f"Meta & bin (all voxels): r = {r_bin_all:.3f}")
        print(f"Meta & wtd (all voxels): r = {r_wtd_all:.3f}")

    return r_bin, r_wtd, r_bin_all, r_wtd_all, thr_meta, thr_bin, thr_wtd


def main(wtd_maps, bin_maps, meta_maps, meta_flag, percentile, results_file):
    """
    Main function to compare maps and compute correlations.
    Parameters
    ----------
    wtd_maps : str
        Directory containing weighted model maps.
    bin_maps : str
        Directory containing binary model maps.
    meta_maps : str
        Directory containing meta-analytic maps.
    meta_flag : str
        Flag to identify meta-analytic maps (e.g., "specter_nsynth_mkda").
    percentile : float
        Percentile for thresholding the maps.
    results_file : str
        Path to save the results CSV file.
    """
    mask = get_mask()
    masker = NiftiMasker(mask_img=mask).fit()

    comp_list = get_components_list(wtd_maps)
    components = [
        os.path.basename(f).split("mean_")[1].split(".nii.gz")[0]
        for f in comp_list
    ]

    accounted_components = []
    results = pd.DataFrame(
        columns=[
            "component",
            "term",
            "r_bin_thr",
            "r_wtd_thr",
            "r_bin_all",
            "r_wtd_all",
            "thr_meta",
            "thr_bin",
            "thr_wtd",
        ]
    )

    # 1. Look for exact component matches
    for component in components:
        nsynth_map = os.path.join(meta_maps, f"{meta_flag}_{component}.nii.gz")
        wtd_map = os.path.join(wtd_maps, f"mean_{component}.nii.gz")
        bin_map = os.path.join(bin_maps, f"mean_{component}.nii.gz")
        # If all maps exist, compute correlations
        if os.path.exists(nsynth_map):
            if os.path.exists(wtd_map) and os.path.exists(bin_map):
                (
                    r_bin,
                    r_wtd,
                    r_bin_all,
                    r_wtd_all,
                    thr_meta,
                    thr_bin,
                    thr_wtd,
                ) = compute_corrs(
                    meta_map=nsynth_map,
                    bin_map=bin_map,
                    wtd_map=wtd_map,
                    masker=masker,
                    percentile=percentile,
                )
                new_row = pd.DataFrame(
                    [
                        {
                            "component": component,
                            "term": component.replace("_", " "),
                            "r_bin_thr": round(r_bin, 4),
                            "r_wtd_thr": round(r_wtd, 4),
                            "r_bin_all": round(r_bin_all, 4),
                            "r_wtd_all": round(r_wtd_all, 4),
                            "thr_meta": round(thr_meta, 4),
                            "thr_bin": round(thr_bin, 4),
                            "thr_wtd": round(thr_wtd, 4),
                        }
                    ]
                )
                results = pd.concat([results, new_row], ignore_index=True)
                accounted_components.append(component)

    print(f"{len(accounted_components)}/{len(components)} compared.")
    results.to_csv(results_file, index=False)


if __name__ == "__main__":
    nimare_dir = "nimare_data"
    meta_maps = "example_data"
    wtd_maps = os.path.join("cognitive_components_subout", "components_mean")
    bin_maps = os.path.join("cognitive_components", "components_mean")
    percentile = 90
    meta_flag = "specter_nsynth_mkda"
    results_file = os.path.join(
        nimare_dir, f"specter_corrs_{int(percentile)}.csv"
    )
    main(wtd_maps, bin_maps, meta_maps, meta_flag, percentile, results_file)
