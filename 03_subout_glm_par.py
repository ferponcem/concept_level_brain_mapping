"""
Script that creates the cognitive-components maps of the IBC tasks taking as
design matrix a sparse matrix from the weighting process. The script runs a
second-level GLM for each subject, where the design matrix is the average of the
weights across all other subjects.
"""

import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from ibc_public.utils_data import CONTRASTS
from joblib import Parallel, delayed
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import math_img
from nilearn.maskers import NiftiMasker

from utils_components import (
    append_task_id,
    conjunction_inference_from_z_images,
    get_mask,
    mask_data_by_reliable_regions,
    setup_data,
)

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
DEFAULT_EXCLUDED_TASKS = ("movie_tasks", "retinotopy")


def prepare_data(db, proj_dir, dir_name):
    """
    Set up some paths and general variables
    """
    write_dir = os.path.join(proj_dir, dir_name)
    os.makedirs(write_dir, exist_ok=True)
    mask = get_mask()
    # append the task name so that the contrast names are unique
    db_renamed = append_task_id(db)
    return db_renamed, mask, write_dir


def get_des_mat_subsout(dir_name, sub):
    """
    Load the weighting/design matrix computed by averaging the weights across
    all subjects except the specified subject
    """
    mat_path = os.path.join(
        dir_name, f"avg_A_stack_{sub.split('-')[-1]}out.csv"
    )
    df = pd.read_csv(mat_path, index_col=0)
    return df


def individual_analysis(
    subject,
    whole_brain_mask,
    ibc_db,
    wdir,
    suffix="",
    use_contrasts=False,
    get_r2=True,
    run_null=False,
):
    """Run second-level GLM analysis for one subject.
    Parameters
    ----------
    subject : str
        Subject identifier (e.g., 'sub-01').
    whole_brain_mask : str
        Path to the whole brain mask NIfTI file.
    ibc_db : pd.DataFrame
        DataFrame containing the IBC dataset information.
    wdir : str
        Working directory where results will be saved.
    suffix : str, optional
        Suffix to append to the output directory name. Default is empty string.
    use_contrasts : bool, optional
        Whether to use contrasts in the analysis. Default is False.
    get_r2 : bool, optional
        Whether to compute R^2 maps for each subject. Default is True.
    run_null : bool, optional
        Whether to run a null model for each subject. Default is False.
    """
    if use_contrasts:
        suffix += "_contrasts"

    # particular to this script, the design matrix is corss-avg weight matrix
    # result from 03_leave_sub_out_average_weighted_matrices.py
    contrasts_df = get_des_mat_subsout(
        "weighting_components" + suffix, subject
    )
    comp_names = contrasts_df.columns
    n_comp = len(comp_names)

    subject_dir = os.path.join(wdir, subject)
    os.makedirs(subject_dir, exist_ok=True)

    dbs = ibc_db[
        (ibc_db.modality == "bold")
        & (ibc_db.subject == subject)
        & (ibc_db.acquisition == "ffx")
        & (ibc_db.session != "ses-00")
    ]

    # Debugging: conflict with mathlang and sub-08
    # remove maps from mathlang ses-24
    if subject == "sub-08":
        dbs = dbs[~((dbs.task == "MathLanguage") & (dbs.session == "ses-24"))]

    valid_dbs = dbs[dbs.contrast.isin(contrasts_df.index)]
    fmri_files = valid_dbs.path.tolist()  # list(dbs.path.values)

    # Create and check design matrix
    X = contrasts_df.loc[contrasts_df.index.intersection(dbs.contrast)]
    if X.index.to_list() == valid_dbs.contrast.to_list():
        print(f"{subject} design matrix:", X.shape)
    else:
        raise ValueError("Design matrix and contrasts do not match")

    # Check matrix conditioning:
    # 1. Remove all-zero columns
    X = X.T.drop_duplicates().T
    all_zero_cols = X.columns[(X == 0).all()]
    if len(all_zero_cols) > 0:
        print(
            f"Warning: {subject} matrix has {len(all_zero_cols)} all-zero cols.",
            "They will be removed.",
        )
        X = X.drop(columns=all_zero_cols)

    # 2. Subject-specific drops for collinear columns
    if use_contrasts:
        sub_drops = [
            "sub-01",
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-09",
            "sub-13",
        ]
    else:
        sub_drops = ["sub-01", "sub-05", "sub-07", "sub-13"]

    if subject in sub_drops:
        with open("sub_specific_drops.yaml") as f:
            drops = yaml.safe_load(f)
        drop_cols = drops[f"{subject}_{suffix}_sparse"]
        print(f" - Dropping {len(drop_cols)} correlated columns for {subject}")
        X = X.drop(columns=drop_cols)

    print(f"New design matrix shape: {X.shape}")
    comp_names = X.columns
    n_comp = len(comp_names)

    _, s, _ = np.linalg.svd(X, full_matrices=False)
    if np.min(s) < 0.1:
        raise ValueError(f"Design matrix for {subject} is ill-conditioned")

    # ==================== Mask data by reliable regions =====================
    Y = mask_data_by_reliable_regions(
        whole_brain_mask, fmri_files, X, use_contrasts=use_contrasts
    )
    # ========================================================================

    # Fit the GLM
    glm = SecondLevelModel(
        mask_img=whole_brain_mask, n_jobs=1, minimize_memory=False
    )

    if run_null:
        # nilearn's glm requires computing contrasts to get the r^2 map, so we
        # compute the two first contrasts but don't save them
        print(f"{subject} Fitting NULL model...")
        rand_X = X.apply(np.random.permutation)
        glm.fit(Y, design_matrix=rand_X)
        for component in np.arange(2):
            zmap = glm.compute_contrast(
                np.eye(n_comp)[component], output_type="z_score"
            )
            name = comp_names[component]
            print(f"{subject} {name} done (NULL)")
        r_sqr = glm.r_square
        nib.save(
            r_sqr, os.path.join(subject_dir, f"cc_{n_comp}_nullr2.nii.gz")
        )
        print(f"{subject} done (NULL)")
        return

    # Standard fit
    glm.fit(Y, design_matrix=X)
    for component in np.arange(n_comp):
        zmap = glm.compute_contrast(
            np.eye(n_comp)[component], output_type="z_score"
        )
        name = comp_names[component]
        nib.save(zmap, os.path.join(subject_dir, f"cc_{name}.nii.gz"))
        print(f"==============> {subject} {name} done")

    # Get the R^2 map to check the explained variance
    if get_r2:
        r_sqr = glm.r_square
        nib.save(r_sqr, os.path.join(subject_dir, f"cc_{n_comp}_r2.nii.gz"))
    print(f"Subject {subject} done")


def second_level_conjunction(conj_dir, participants, mask, comp_names, u=0.25):
    """
    Run a conjunction analysis across subjects for each component.
    """
    masker = NiftiMasker(mask_img=mask, smoothing_fwhm=None).fit()
    for term in comp_names:
        cj_imgs = [
            os.path.join(conj_dir, os.pardir, participant, f"cc_{term}.nii.gz")
            for participant in participants
        ]
        cj_imgs = [img for img in cj_imgs if os.path.exists(img)]
        print(f"Conjunction for {term}: {len(cj_imgs)} images found")
        cj_coco = conjunction_inference_from_z_images(cj_imgs, masker, u=u)
        nib.save(cj_coco, os.path.join(conj_dir, f"coco_cj_{term}.nii.gz"))


def compute_components(
    db,
    proj_dir,
    dir_name,
    second_level="cj",
    sfx="",
    use_contrasts=False,
    get_r2=True,
    run_null=False,
):
    """
    1. Prepare cognitive components and design matrix
    2. Individual GLM analysis
    3. Second-level conjunction analysis
    Parameters
    ----------
    db : pd.DataFrame
        DataFrame containing the IBC dataset information.
    proj_dir : str
        Path to the project directory (where the script is located).
    dir_name : str
        Name of the output directory for storing results.
    second_level : str, optional
        Second-level analysis to perform. Default is "cj" for conjunction.
    sfx : str, optional
        Suffix to append to the output directory name. Default is empty string.
    use_contrasts : bool, optional
        Whether to use contrasts in the analysis. Default is False.
    get_r2 : bool, optional
        Whether to compute R^2 maps for each subject. Default is True.
    run_null : bool, optional
        Whether to run a null model for each subject. Default is False.
    """
    db_names, mask_gm, write_dir = prepare_data(db, proj_dir, dir_name)
    subject_list = db_names.subject.unique()

    Parallel(n_jobs=16)(
        delayed(individual_analysis)(
            subject,
            mask_gm,
            db_names,
            write_dir,
            suffix=sfx,
            use_contrasts=use_contrasts,
            get_r2=get_r2,
            run_null=run_null,
        )
        for subject in subject_list
    )

    components = glob(os.path.join(write_dir, "sub-04", "cc_*.nii.gz"))
    components = [c for c in components if "r2" not in c]
    component_names = [
        os.path.basename(comp).split("cc_")[-1].split(".")[0]
        for comp in components
    ]

    # Across-subject analysis
    if second_level is not None:
        group_dir = os.path.join(write_dir, second_level)
        if not os.path.exists(group_dir):
            os.mkdir(group_dir)

    if second_level == "cj":
        second_level_conjunction(
            group_dir, subject_list, mask_gm, component_names
        )


def main(project_dir, use_contrasts, maps_list, out_dir_name):
    """
    Main function to set up data and compute cognitive components.
    Parameters
    ----------
    project_dir : str
        Path to the project directory (where the script is located).
    use_contrasts : bool
        Whether to use contrasts in the analysis.
    maps_list : list
        List of maps to include in the analysis.
    out_dir_name : str
        Name of the output directory for storing results.
    """
    db = setup_data(
        SUBJECTS,
        exclude=DEFAULT_EXCLUDED_TASKS,
        conditions=maps_list,
    )
    compute_components(
        db,
        proj_dir=project_dir,
        dir_name=out_dir_name,
        second_level="cj",
        sfx="",
        use_contrasts=use_contrasts,
        get_r2=True,
        run_null=False,
    )


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    use_contrasts = True
    maps_list = CONTRASTS
    out_dir_name = "cognitive_components_subout"
    main(project_dir, use_contrasts, maps_list, out_dir_name)
