"""
Get IBC condition maps and display its annotations
"""

import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml
from ibc_public.utils_data import (
    ALL_CONTRASTS,
    CONDITIONS,
    CONTRASTS,
    SMOOTH_DERIVATIVES,
    data_parser,
)
from nilearn import datasets
from nilearn.image import (
    load_img,
    math_img,
    new_img_like,
    resample_to_img,
    threshold_img,
)
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.plotting import plot_stat_map


def flatten(li):
    return sum(
        ([x] if not isinstance(x, list) else flatten(x) for x in li), []
    )


def setup_data(
    participants,
    conditions=CONTRASTS,
    wanted_tasks="all",
    exclude=None,
    no_task=[],
):
    """
    Run the data parser to get the paths of the desired contrasts
    while excluding some groups of tasks.
    Op 1: specify "wanted_tasks" to get only the tasks of a specific release
    Op 2: specify "exclude" to exclude a block of tasks of a specific release
    Op 3: specify "no_task" to exclude specific indvidual tasks
    """
    # Prepare list of desired tasks
    tasks_ = yaml.safe_load(open("all_tasks.yml"))
    TASKS = []
    if wanted_tasks == "all":
        wanted_tasks = list(tasks_.keys())

    for release in wanted_tasks:
        TASKS.append(tasks_[release])
        TASKS = flatten(TASKS)

    # Exclude specified tasks
    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        to_exclude = flatten([tasks_[ex] for ex in exclude if ex in tasks_])
        TASKS = [task for task in TASKS if task not in to_exclude]

    # Exclude specified tasks
    if len(no_task) > 0:
        TASKS = [task for task in TASKS if task not in no_task]

    # Database of IBC data paths
    db = data_parser(
        derivatives=SMOOTH_DERIVATIVES,
        subject_list=participants,
        conditions=conditions,
        task_list=TASKS,
    )
    return db


def get_used_contrasts(my_tasks, conditions, all_contrasts_file=ALL_CONTRASTS):
    """
    Format df_contrast with only the contrasts used in the analysis
    """
    df_all_contrasts = pd.read_csv(all_contrasts_file, sep="\t")
    df_contrasts = pd.DataFrame()
    df_list = []

    for task_name in my_tasks:
        cond_list = conditions[conditions.task == task_name].contrast.values
        df1 = df_all_contrasts[
            (df_all_contrasts.task == task_name)
            & (df_all_contrasts.contrast.isin(cond_list))
        ]
        df_list.append(df1)

    df_contrasts = pd.concat(df_list, ignore_index=True)
    df_contrasts = df_contrasts[["task", "contrast", "tags"]]
    return df_contrasts


def get_task_list(
    wanted_tasks="all",
    exclude="none",
    no_task=[],
):
    """
    Return a list of all the tasks in the IBC dataset
    Op 1: specify "wanted_tasks" to get only the tasks of a specific release
    Op 2: specify "exclude" to exclude a block of tasks of a specific release
    Op 3: specify "no_task" to exclude specific indvidual tasks
    """
    tasks_ = yaml.safe_load(open("all_tasks.yml"))
    TASKS = []
    if wanted_tasks == "all":
        wanted_tasks = list(tasks_.keys())
    for release in wanted_tasks:
        TASKS.append(tasks_[release])
        TASKS = flatten(TASKS)

    # Exclude specified blocks of tasks
    if exclude != "none":
        if isinstance(exclude, str):
            exclude = [exclude]
        to_exclude = flatten([tasks_[ex] for ex in exclude])
        TASKS = [task for task in TASKS if task not in to_exclude]

    # Exclude specified tasks
    if len(no_task) > 0:
        TASKS = [task for task in TASKS if task not in no_task]

    return TASKS


def contrasts_and_tags(selected_tasks, df_all_cts):
    """
    Extract contrasts and tags from selected tasks in the all_contrasts.tsv
    and add the task id to avoid repetition of contrast names.
    """
    cts_list = []
    all_tgs = []
    for tk in selected_tasks:
        conts_id = df_all_cts[df_all_cts.task == tk].contrast.tolist()
        new_conts_ids = [tk + "_" + id for id in conts_id]
        cts_list.extend(new_conts_ids)
        all_tgs.extend(df_all_cts[df_all_cts.task == tk].tags.tolist())
    return cts_list, all_tgs


def append_task_id(db):
    """
    Append task to contrast name to avoid issues with repeated contrast names
    """
    db = db.sort_values(by=["subject", "task", "contrast"])
    # Add task to contrast name to avoid repetition of contrast IDs
    db.loc[db.modality == "bold", "contrast"] = (
        db.loc[db.modality == "bold", "task"]
        + "_"
        + db.loc[db.modality == "bold", "contrast"]
    )
    return db


def tags(tags_lists):
    """
    Extract all tags of a set of contrasts from all_contrasts.tsv,
    clean labels and return their unique tags.
    """
    tags_lists = [tl.replace("'", "") for tl in tags_lists]
    tags_lists = [list(tg.strip("][]").split(",")) for tg in tags_lists]
    # Some cleaning:
    # 1. remove extra-spaces;
    # 2. replace spaces between words by underscores
    tags_clean = []
    for tags_list in tags_lists:
        tag_clean = []
        for tc in tags_list:
            tc = tc.replace(" ", "_")
            if tc == "_":
                continue
            if tc[0] == "_":
                tc = tc[1:]
            if tc[-1] == "_":
                tc = tc[:-1]
            tag_clean.append(tc)
        tags_clean.append(tag_clean)
    # Get array with unique tags
    tags_flatten = [item for sublist in tags_clean for item in sublist]
    utags_array = np.array(tags_flatten)
    utags_array = np.unique(utags_array)
    unique_tags = utags_array.tolist()
    return tags_clean, tags_flatten, unique_tags


def occurences_matrix(tgs_clean, unique_tgs, contrasts):
    """
    Create occurrences matrix of cognitive components
    Return data-frame
    """
    occur_mtx = []
    for tlist in tgs_clean:
        occur = []
        for component in unique_tgs:
            if component in tlist:
                occur.append(1)
            else:
                occur.append(0)
        occur_mtx.append(occur)
    data_frame = pd.DataFrame(occur_mtx, columns=unique_tgs, index=contrasts)
    return data_frame


def plot_occurrences(df, display_labels=False):
    """
    Plot the occurrences matrix
    """
    plt.figure(figsize=(12, 15))
    plt.imshow(df, cmap="binary_r", aspect="equal")
    # Set gridlines at the boundaries of the cells
    plt.gca().set_xticks(
        [x - 0.5 for x in range(1, df.shape[1])], minor=True
    )  # Minor ticks for vertical gridlines
    plt.gca().set_yticks(
        [y - 0.5 for y in range(1, df.shape[0])], minor=True
    )  # Minor ticks for horizontal gridlines
    plt.grid(
        visible=True,
        which="minor",
        color="gray",
        linewidth=0.5,
        linestyle="--",
    )
    if display_labels:
        plt.xticks(
            range(df.shape[1]),
            df.columns,
            rotation=45,
            ha="right",
            fontsize=10,
        )
        plt.yticks(range(df.shape[0]), df.index, fontsize=10)
    plt.show()


def check_design_mtx(db):
    """
    Check for exact duplications and linear correlations
    """
    for i, cc1 in enumerate(np.array(db.values.T.tolist())):
        for j, cc2 in enumerate(np.array(db.values.T.tolist())):
            if np.array_equal(cc1, cc2) and i < j:
                print("duplication: %s, %s" % (db.columns[i], db.columns[j]))
            correlation, p = st.pearsonr(cc1, cc2)
            if abs(correlation) > 0.5 and p < 0.001 and i < j:
                print(correlation, p)
                print("correlation: %s, %s" % (db.columns[i], db.columns[j]))


def get_mask(three_mm=False):
    """
    Get the grey matter mask common to all IBC subjects
    """
    import ibc_public

    _package_directory = os.path.dirname(
        os.path.abspath(ibc_public.utils_data.__file__)
    )
    if three_mm:
        mask_gm = os.path.join(
            _package_directory, "../ibc_data", "gm_mask_3mm.nii.gz"
        )
    else:
        mask_gm = os.path.join(
            _package_directory, "../ibc_data", "gm_mask_1_5mm.nii.gz"
        )
    return mask_gm


def __transform_contrast(c):
    if c == "fixation":
        return c
    elif c.startswith("saccade"):
        return "saccade-fixation"
    elif c.startswith("tongue"):
        return "tongue-fixation"
    else:
        return f"{c}-fixation"


def modify_moto_conditions(cond_list=CONDITIONS):
    """
    Add the existing maps to the list of motor conditions and replace the
    existing strings with the new ones that have "-fixation" appended.
    """
    conditions = cond_list.copy()

    # Mask motor conditions
    mask = conditions["task"] == "Moto"

    # Apply transformation only to motor contrasts
    conditions.loc[mask, "contrast"] = conditions.loc[mask, "contrast"].map(
        __transform_contrast
    )

    # Drop duplicates but keep one copy of each
    conditions = conditions.drop_duplicates(
        subset=["task", "contrast"]
    ).reset_index(drop=True)

    return conditions


def ibc_mask_schaefer(ibc_masker, n_rois=100, verbose=False):
    """
    Get region labels and voxel indices for the Schaefer atlas in the
    ibc mask space.

    Parameters
    ----------
    ibc_masker : NiftiMasker object
        Fitted NiftiMasker with the IBC brain mask.
    n_rois : int, optional
        Number of regions in the Schaefer atlas (default is 100).
    verbose : bool, optional
        If True, print additional information (default is False).

    Returns
    -------
    region_labels : array
        Array of region labels for each voxel in the IBC mask.
    unique_labels : array
        Array of unique region labels present in the IBC mask.
    region_indices : list of arrays
        List where each element contains the voxel indices for a specific region.
    """
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    atlas_res = resample_to_img(
        schaefer["maps"],
        ibc_masker.mask_img_,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    # Apply mask to atlas → gives region ID for each voxel in mask
    region_labels = ibc_masker.transform(atlas_res).ravel().astype(int)
    unique_labels = np.unique(region_labels)
    unique_labels = unique_labels[unique_labels > 0]

    # Build indices per region
    region_indices = [np.where(region_labels == r)[0] for r in unique_labels]

    if verbose:
        print("Regions found:", unique_labels)

    return region_labels, unique_labels, region_indices


def mask_data_by_reliable_regions(
    whole_brain_mask,
    fmri_files,
    X,
    n_rois=100,
    noise_weight=0.1,
    use_contrasts=False,
    return_array=False,
    plot=False,
):
    """
    Mask fMRI data by reliability per region.

    Parameters
    ----------
    whole_brain_mask : str
        Path to the whole brain mask NIfTI file.
    fmri_files : list of str
        List of paths to fMRI NIfTI files for each condition.
    X : DataFrame
        Design matrix with conditions as index.
    n_rois : int, optional
        Number of regions in the Schaefer atlas (default is 100).
    noise_weight : float, optional
        Weight to decrease the signal of voxels outside reliable regions
        (default is 0.1).
    use_contrasts : bool, optional
        If True, use contrasts list; if False, use conditions list
        (default is False).
    return_array : bool, optional
        If True, return data as a NumPy array; if False, return as NIfTI images
        (default is False).
    plot : bool, optional
        If True, plot the masked condition maps (default is False).

    Returns
    -------
    Y_data : array or list of Nifti1Image
        Masked fMRI data, either as a NumPy array or a list of NIfTI images.
    """

    # Load reliability information
    if use_contrasts:
        cos_dist_file = "cosine_dist_volume_conts.csv"
    else:
        cos_dist_file = "cosine_dist_volume_conds.csv"

    # Extract voxels from reliable regions
    region_info = pd.read_csv(
        os.path.join("contrast_reliability", cos_dist_file),
        index_col=0,
    )
    # Schaefer parcellation
    ibc_masker = NiftiMasker(
        mask_img=whole_brain_mask,
        memory="nilearn_cache",
        verbose=0,
    ).fit()
    region_labels, unique_labels, _ = ibc_mask_schaefer(
        ibc_masker, n_rois=n_rois
    )

    Y_data = []
    for idx, condition in enumerate(X.index):
        # Check that condition exists in region_info
        if condition not in region_info.columns:
            raise ValueError(f"Condition {condition} not found in region info")
        # Get mask of reliable regions for this condition
        reliable_mask = region_info[condition].values >= 0.5
        reliable_region_ids = unique_labels[reliable_mask]
        # Transform and mask data
        condition_data = ibc_masker.transform(fmri_files[idx]).ravel()
        voxel_mask = np.isin(region_labels, reliable_region_ids)
        # Decrease the weight of voxels outside reliable regions
        condition_data[~voxel_mask] *= noise_weight

        if return_array:
            Y_data.append(condition_data)
        else:
            cond_data_img = ibc_masker.inverse_transform(condition_data)
            if plot:
                plot_stat_map(
                    cond_data_img, title=f"{condition}", draw_cross=False
                )
            Y_data.append(cond_data_img)

    assert len(Y_data) == len(fmri_files), "Mismatch in number of conditions"

    return np.array(Y_data) if return_array else Y_data


def check_matrix_conditioning(X):
    """
    Check the conditioning of the design matrix X by computing its
    condition number.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = np.linalg.matrix_rank(X)
    n_cols = X.shape[1]
    condition_number = s[0] / s[-1]

    smallest = np.argmin(s)
    contributing_regs = X.columns[np.abs(Vt[smallest]) > 0.1].tolist()
    print("Smallest singular value:", s[smallest])
    print("Contributing regressors:", contributing_regs)
    print(f"Condition number of the design matrix: {condition_number}")
    print(f"Rank of the design matrix: {rank} / {n_cols}")

    # Plot on a log scale
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(s) + 1), np.log10(s), marker="o")
    plt.xlabel("Singular value index")
    plt.ylabel("log10(singular value)")
    plt.title("Singular value spectrum of design matrix")
    plt.grid(True)
    plt.show()

    return contributing_regs, condition_number


def get_components_by_task(task):
    """Get the cognitive components anotated on a given task

    Parameters
    ----------
        task (str): Name of the IBC task.

    Returns
    -------
        unique_tags (list): List of unique cognitive components for the task.
    """

    all_contrasts = pd.read_csv(ALL_CONTRASTS, sep="\t")
    all_contrasts_task = all_contrasts[all_contrasts["task"] == task]
    tags = all_contrasts_task["tags"].tolist()
    flat_tags = [tag for sublist in tags for tag in ast.literal_eval(sublist)]
    seen = set()
    unique_tags = [t for t in flat_tags if not (t in seen or seen.add(t))]

    return unique_tags


def binarize_img(img):
    """Convert all non-zero voxels to 1 (i.e., binary mask)."""
    return math_img("img > 0", img=img)


def create_component_noi(component, comps_dir, thr):
    """Create a mask for a given component(s)"""
    component_file = os.path.join(comps_dir, f"mean_{component}.nii.gz")
    # 1. Mask-out the zeors
    data = load_img(component_file).get_fdata()
    nonzero_data = data[data > 0]
    thr_val = np.percentile(nonzero_data, float(thr.strip("%")))
    # 2. Threshold the masked component
    th_img = threshold_img(component_file, threshold=thr_val, two_sided=False)
    # 3. Binarize the component
    component_as_noi = binarize_img(th_img)
    return component_as_noi


def get_schaefer_atlas(comp_mask=None, n_rois=1000, res_mm=1, report=False):
    """Fetch the Schaefer atlas and get the number of regions that overlap with
    a given mask."""
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, resolution_mm=res_mm
    )
    atlas = schaefer.maps

    # If no mask is provided, return the full atlas masker
    if comp_mask is None:
        masker = NiftiLabelsMasker(atlas, standardize=False).fit()
        filtered_atlas = atlas
    # If a mask is provided, filter the atlas regions
    else:
        atlas_res = resample_to_img(
            atlas,
            comp_mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_data = atlas_res.get_fdata()

        comp_mask_data = comp_mask.get_fdata()

        region_labels = np.unique(atlas_data)
        region_labels = region_labels[region_labels != 0]

        keep_labels = []
        for label in region_labels:
            region_img = atlas_data == label
            overlap = np.logical_and(region_img, comp_mask_data)
            if any(overlap.flatten()):
                keep_labels.append(int(label))

        filtered_data = (np.isin(atlas_data, keep_labels) * atlas_data).astype(
            np.int32
        )
        filtered_atlas = new_img_like(atlas_res, filtered_data)
        masker = NiftiLabelsMasker(filtered_atlas, standardize=False).fit()

    if report:
        masker.generate_report()

    return masker, filtered_atlas


def sort_regions_by_contribution(component, n_rois=1000, res_mm=1, top_k=100):
    """Get the regions in an atlas sorted by their values in a given
    component/noi.

    Parameters
    ----------
    component : Niimg-like
        Path to the component image or Niimg object.
    n_rois : int, optional
        Number of regions in the Schaefer atlas. Default is 1000.
    res_mm : int, optional
        Resolution of the Schaefer atlas in mm. Default is 1.
    top_k : int, optional
        Number of top regions to return. Default is 100.

    Returns
    -------
    new_masker : NiftiLabelsMasker
        A NiftiLabelsMasker fitted on the filtered atlas containing only the
        top_k regions.
    filtered_atlas : Niimg-like
        A Nifti image containing only the top_k regions from the atlas.
    """

    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, resolution_mm=res_mm
    )
    atlas = schaefer.maps
    atlas_res = resample_to_img(
        atlas,
        component,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    masker = NiftiLabelsMasker(atlas_res, lut=schaefer.lut, standardize=False)
    masker.fit()

    # Extract region values: the output is an array of shape (1, n_regions)
    region_values = masker.transform(component).ravel()

    # Get labels, gives an array of shape (n_regions + 1,) 0 is the background
    labels = masker.labels_
    labels = labels[1:]  # remove background label
    labels = np.array(labels)

    # Sort the region values in descending order and get their labels
    sorted_idx = np.argsort(region_values)[::-1]
    top_labels = labels[sorted_idx[:top_k]]

    # Filter the atlas to keep only the top regions
    atlas_data = atlas_res.get_fdata().astype(int)
    filtered_data = (np.isin(atlas_data, top_labels) * atlas_data).astype(
        np.int32
    )
    filtered_atlas = new_img_like(atlas_res, filtered_data)

    # Fit a new masker on the filtered atlas
    new_masker = NiftiLabelsMasker(filtered_atlas, standardize=False).fit()

    return new_masker, filtered_atlas


def get_union_top_regions(component_1, component_2, n_rois=1000, top_k=100):
    """
    Get the union of top_k regions from two components.
    Parameters
    ----------
    component_1 : Niimg-like
        Path to the first component image or Niimg object.
    component_2 : Niimg-like
        Path to the second component image or Niimg object.
    n_rois : int, optional
        Number of regions in the Schaefer atlas. Default is 1000.
    top_k : int, optional
        Number of top regions to consider for union. Default is 100.
    Returns
    -------
    union_masker : NiftiLabelsMasker
        Masker fitted on the union of the top_k regions from both components.
    union_atlas : Niimg-like
        A Nifti image containing the union of the top_k regions from both
        components.
    """
    # Take half of the top_k from each component to create a union of
    # approximately top_k regions
    top_k = np.ceil(top_k / 2).astype(int)

    # Get the top_k regions for each component separately
    masker_1, _ = sort_regions_by_contribution(
        component=component_1, n_rois=n_rois, top_k=top_k
    )
    masker_2, _ = sort_regions_by_contribution(
        component=component_2, n_rois=n_rois, top_k=top_k
    )

    # Get the labels of the top regions for both components
    labels_1_set = {label for label in masker_1.labels_ if label != 0}
    labels_2_set = {label for label in masker_2.labels_ if label != 0}

    # Get the union of the top region labels
    union_labels = labels_1_set.union(labels_2_set)

    # Filter the atlas to keep only the union of the top regions
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, resolution_mm=1
    )
    masker = NiftiLabelsMasker(
        schaefer.maps, lut=schaefer.lut, standardize=False
    ).fit()
    atlas_data = masker.labels_img_.get_fdata().astype(int)
    union_data = (np.isin(atlas_data, list(union_labels)) * atlas_data).astype(
        np.int32
    )
    union_atlas = new_img_like(masker.labels_img_, union_data)
    union_masker = NiftiLabelsMasker(union_atlas, standardize=False).fit()

    return union_masker, union_atlas


def test_images_equal(image1_path, image2_path, corr_threshold=0.99):
    """Test if two images are the same by computing the Pearson correlation
    between their voxel values.
    Parameters
    ----------
    image1_path : str
        Path to the first image file (NIfTI format).
    image2_path : str
        Path to the second image file (NIfTI format).
    Returns
    -------
    None
        Prints whether the images are essentially the same or different, along
        with the correlation value if they are different.
    """
    # Load the images
    img1 = load_img(image1_path)
    img2 = load_img(image2_path)

    # Get the data arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Flatten the data arrays to 1D
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    # Compute the Pearson correlation coefficient
    correlation, _ = st.pearsonr(flat_data1, flat_data2)

    # Check if the correlation is close to 1 (indicating identical images)
    if correlation > corr_threshold:
        print("The images are essentially the same.")
    else:
        print("The images are different, correlation:", correlation)


def _conjunction_inference_from_z_values(
    Z, u=0.5, verbose=False, two_tailed=False
):
    """Returns a conjunction z-values against the null hypothesis:
    A proportion of less than u values per row of Z display an effect
    Stouffer approach is used.

    Parameters
    ----------
    Z: arraz of shape (n_tests, n)
       Presumably z values
    u: int in [0, 1]
       desired proportion level
    """
    if (u >= 1) or (u < 0):
        raise ValueError("u should be between 0 and 1")
    p = int((1 - u) * Z.shape[1])
    if two_tailed:
        signZ = np.sign(np.mean(Z, 1))
        Z_ = np.sort(Z.T * signZ, 0).T
        stat = np.maximum(0, np.sum(Z_[:, :p], 1) / np.sqrt(p))
        return signZ * stat

    Z_ = np.sort(Z, 1)
    if verbose:
        print(p)
    return np.sum(Z_[:, :p], 1) / np.sqrt(p)


def conjunction_inference_from_z_images(z_images, masker, u=0.5):
    """Mass univariate test for the null hypothesis:
    A proportion of less than u values per row of Z display an effect
    Stouffer approach is used.

    Parameters
    -----------
    z_images: nims,
              Z images provided as input
    u: int in [0, 1]
       desired proportion level
    """
    Z = masker.transform(z_images)
    conj = _conjunction_inference_from_z_values(Z.T, u)
    conj_img = masker.inverse_transform(conj)
    return conj_img
