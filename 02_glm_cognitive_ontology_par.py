"""
Compute cognitive-components maps of the IBC tasks.
Process subjects parallely.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
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


def _run_subject_analysis(
    subject,
    whole_brain_mask,
    ibc_db,
    contrasts_df,
    wdir,
    comp_names,
    sfx="",
    use_contrasts=False,
    get_r2=True,
    run_null=False,
):
    """Run individual GLM analysis for a single subject."""

    if use_contrasts:
        sfx += "_contrasts"

    subject_dir = os.path.join(wdir, subject)
    os.makedirs(subject_dir, exist_ok=True)

    # Get the subject-specific database
    dbs = ibc_db[
        (ibc_db.modality == "bold")
        & (ibc_db.subject == subject)
        & (ibc_db.acquisition == "ffx")
        & (ibc_db.session != "ses-00")
    ]
    if subject == "sub-08":
        dbs = dbs[~((dbs.task == "MathLanguage") & (dbs.session == "ses-24"))]

    # Align the design matrix and the available contrasts for the subject
    valid_dbs = dbs[dbs.contrast.isin(contrasts_df.index)]
    valid_contrasts = valid_dbs.contrast.tolist()

    # Get the design matrix
    X = contrasts_df.T[valid_contrasts].T

    # Some subjects have missing data, drop duplicates (zero columns)
    X = X.T.drop_duplicates().T

    # Subject-specific regressor drops, only for contrasts (for now)
    if use_contrasts:
        if subject in ["sub-01", "sub-07", "sub-13"]:
            with open("sub_specific_drops.yaml") as f:
                drops = yaml.safe_load(f)
            drop_cols = drops[f"{subject}_{sfx}_bin"]
            X = X.drop(columns=drop_cols)

    # Some subjects have less data, hence some columns are all zero
    cols_all_zero = X.columns[(X == 0).all(axis=0)].tolist()
    if cols_all_zero:
        print(f"⚠️ {subject} has all-zero columns: {cols_all_zero}")
        X = X.drop(columns=cols_all_zero)

    _, s, _ = np.linalg.svd(X.values)
    if np.min(s) < 0.1:
        raise ValueError(
            f"Matrix for {subject} is ill-conditioned. Min s-val: {np.min(s)}"
        )

    # Drop from comp_names
    comp_names = [n for n in comp_names if n in X.columns]
    n_comp = len(comp_names)
    print(f"New number of components: {n_comp}")
    X.to_csv(os.path.join(subject_dir, "design_matrix.tsv"), sep="\t")
    print(f"{subject} design matrix:", X.shape)

    # Shrink valid_dbs to available contrasts
    valid_dbs = valid_dbs[valid_dbs.contrast.isin(X.index)]

    # Get the fMRI files path
    fmri_files = valid_dbs.path.tolist()

    # Model initialization
    glm = SecondLevelModel(
        mask_img=whole_brain_mask, n_jobs=1, minimize_memory=False
    )

    # ========== Mask data by reliable regions ===============================
    Y_data = mask_data_by_reliable_regions(
        whole_brain_mask, fmri_files, X, use_contrasts=use_contrasts
    )
    # ========================================================================

    if run_null:
        print("Fitting NULL model !!!!!!!!!!!!!!!!!!")
        rand_X = X.apply(np.random.permutation)
        glm.fit(Y_data, design_matrix=rand_X)
        for component in np.arange(2):
            zmap = glm.compute_contrast(
                np.eye(n_comp)[component], output_type="z_score"
            )
            name = comp_names[component]
            print(f"==============> Component {name} done")
        r_sqr = glm.r_square
        nib.save(
            r_sqr, os.path.join(subject_dir, f"cc_{n_comp}_nullr2.nii.gz")
        )
        print(f"Subject {subject} done")
        return

    glm.fit(Y_data, design_matrix=X)
    for component in np.arange(n_comp):
        zmap = glm.compute_contrast(
            np.eye(n_comp)[component], output_type="z_score"
        )
        name = comp_names[component]
        nib.save(zmap, os.path.join(subject_dir, f"cc_{name}.nii.gz"))
        print(f"==============> Component {name} done")

    if get_r2:
        r_sqr = glm.r_square
        nib.save(r_sqr, os.path.join(subject_dir, f"cc_{n_comp}_r2.nii.gz"))

    print(f"Subject {subject} done")


def individual_analysis(
    whole_brain_mask,
    participants,
    ibc_db,
    contrasts_df,
    wdir,
    n_comp,
    comp_names,
    sfx="",
    use_contrasts=False,
    get_r2=False,
    run_null=False,
    n_jobs=-1,
):
    """Launch individual analysis in parallel."""

    Parallel(n_jobs=n_jobs)(
        delayed(_run_subject_analysis)(
            subject,
            whole_brain_mask,
            ibc_db,
            contrasts_df,
            wdir,
            n_comp,
            comp_names,
            sfx,
            use_contrasts,
            get_r2,
            run_null,
        )
        for subject in participants
    )


def second_level_conjunction(
    conj_dir, participants, mask, comp_names, cj_u=0.25
):
    """Compute group conjunction analysis."""
    masker = NiftiMasker(mask_img=mask, smoothing_fwhm=None).fit()
    for term in comp_names:
        cj_imgs = [
            os.path.join(conj_dir, os.pardir, participant, f"cc_{term}.nii.gz")
            for participant in participants
        ]
        cj_imgs = [img for img in cj_imgs if os.path.exists(img)]
        print(f"Computing conjunction for {term}, {len(cj_imgs)} images")
        cj_coco = conjunction_inference_from_z_images(cj_imgs, masker, u=cj_u)
        nib.save(cj_coco, os.path.join(conj_dir, f"coco_cj_{term}.nii.gz"))
        plotting.plot_stat_map(
            cj_coco,
            colorbar=False,
            threshold=3.1,
            vmax=8,
            display_mode="z",
            cut_coords=[-15, -4, 10, 43, 50, 56],
            title=term,
            output_file=os.path.join(conj_dir, f"coco_cj_{term}.png"),
        )
        positive_cj_coco = math_img("img * (img > 0)", img=cj_coco)
        plotting.plot_glass_brain(
            positive_cj_coco,
            colorbar=True,
            threshold=3.1,
            vmax=8,
            title=term,
            output_file=os.path.join(conj_dir, f"gcoco_cj_{term}.png"),
        )


# %%
def compute_components(
    db,
    dir_name,
    second_level,
    dmtx_file,
    suffix="",
    use_contrasts=False,
    get_r2=True,
    run_null=False,
):
    """
    1. Prepare cognitive components and design matrix
    2. Individual GLM analysis
    3. Group analysis, conjunction or random-effects analysis
    Parameters
    ----------
    db : pandas.DataFrame
        DataFrame containing paths to the selected IBC data.
    dir_name : str
        Name of the directory to store results.
    second_level : str
        Type of group analysis to perform: 'cj' for conjunction,
        'rfx' for random-effects.
    design_mtx_file : str
        Name of the design matrix file (TSV) to use.
    use_contrasts : bool, optional
        If True, use contrasts list; if False, use conditions list
        (default is False).
    get_r2 : bool, optional
        If True, compute and save R² maps for each subject
        (default is True).
    run_null : bool, optional
        If True, run a null model with permuted design matrix
        (default is False).
    Returns
    -------
    None
    """
    # Output main directory, subject list and mask
    os.makedirs(dir_name, exist_ok=True)
    subject_list = db.subject.unique().tolist()
    mask_gm = get_mask()

    # Read design matrix
    df = pd.read_csv(os.path.join(dir_name, dmtx_file), sep="\t", index_col=0)

    # Format database to append task IDs to contrast names
    db = append_task_id(db)
    # Get the components used in the design matrix after conditioning
    component_names = [col.replace(" ", "_") for col in df.columns]
    n_components = len(component_names)

    individual_analysis(
        mask_gm,
        subject_list,
        db,
        df,
        dir_name,
        n_components,
        component_names,
        sfx=suffix,
        use_contrasts=use_contrasts,
        get_r2=get_r2,
        run_null=run_null,
        n_jobs=12,
    )

    # Across-subject analysis
    group_dir = os.path.join(dir_name, second_level)
    os.makedirs(group_dir, exist_ok=True)
    if second_level == "cj":
        second_level_conjunction(
            group_dir, subject_list, mask_gm, component_names
        )


def main(
    maps_list=CONTRASTS,
    components_dir="cognitive_components",
    second_level="cj",
    dmat_file="design_matrix.tsv",
    suffix="",
    use_contrasts=True,
    get_r2=False,
    run_null=True,
):
    components_dir = Path(components_dir)
    components_dir.mkdir(parents=True, exist_ok=True)
    db = setup_data(
        SUBJECTS, exclude=DEFAULT_EXCLUDED_TASKS, maps_list=maps_list
    )
    compute_components(
        db,
        components_dir,
        second_level=second_level,
        dmtx_file=dmat_file,
        suffix=suffix,
        use_contrasts=use_contrasts,
        get_r2=get_r2,
        run_null=run_null,
    )


if __name__ == "__main__":
    main()
