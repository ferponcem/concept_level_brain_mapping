"""
Parcellates the fMRI data for all IBC conditions into the Schaefer atlas.
"""

import os

import numpy as np
from ibc_public.utils_data import CONDITIONS, CONTRASTS
from joblib import Memory, Parallel, delayed
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

from utils_components import setup_data

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def process_file(sub, file, out_sub, masker):
    """
    Process a single fMRI file: extract the time series using the provided
    masker and save it to the output directory.
    """
    # Preparing output file
    task_name = file.split("task-")[-1].split("_")[0]
    if "run" in file:
        run_id = file.split("run-")[-1].split("/")[0]
    else:
        run_id = "dir-" + file.split("dir-")[-1].split("/")[0]

    out_file = os.path.join(
        out_sub,
        sub
        + "_"
        + task_name
        + "_"
        + run_id
        + "_"
        + os.path.basename(file).replace(".nii.gz", ".npy"),
    )

    # Actual time series extraction
    if not os.path.exists(out_file):
        print(f" =========> Processing {sub} {file.split('/')[-1]}")
        time_series = masker.transform(file)
        with open(out_file, "wb") as ts_file:
            np.save(ts_file, time_series)
    else:
        print(f"File already exists: {out_file}")


def process_subject(sub, nii_files, main_dir, schaefers_name, masker):
    """
    Helper to process all files for a given subject in parallel.
    """
    # Preparing output directory
    out_sub = os.path.join(main_dir, f"{sub}_{schaefers_name}")
    if not os.path.exists(out_sub):
        os.makedirs(out_sub)

    # Process each file in parallel
    Parallel(
        n_jobs=20,
        verbose=5,
    )(delayed(process_file)(sub, file, out_sub, masker) for file in nii_files)


def main(main_dir, cache_dir, n_rois, maps_list=CONDITIONS):
    """
    Main function to parcellate fMRI data for all IBC conditions/contrasts into
    the Schaefer atlas.
    Parameters
    ----------
    main_dir : str
        Directory where the parcellated data will be saved. A subject-specific
        subdirectory will be created for each subject.
    cache_dir : str
        Directory for caching intermediate results.
    n_rois : int
        Number of regions of interest in the Schaefer atlas (e.g., 1000)
    maps_list : list of str, optional
        List of conditions or contrasts to process. Defaults to CONDITIONS.
    """
    schaefers = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    schaefers_name = f"schaefer{n_rois}"
    masker = NiftiLabelsMasker(
        labels_img=schaefers["maps"],
        standardize=False,
        memory=Memory(location=cache_dir),
        verbose=5,
    )
    masker.fit()

    db = setup_data(
        SUBJECTS,
        exclude=["movie_tasks", "retinotopy"],
        conds_of_interest=maps_list,
    )
    db = db[(db.acquisition == "ap") | (db.acquisition == "pa")]
    db = db[db.session != "ses-00"]

    for sub in SUBJECTS:
        # Collecting fmri maps
        nii_files = db[db["subject"] == sub]["path"].values

        # Preparing output directory
        out_sub = os.path.join(main_dir, f"{sub}_{schaefers_name}_contrasts")
        if not os.path.exists(out_sub):
            os.makedirs(out_sub)

        process_subject(sub, nii_files, main_dir, schaefers_name, masker)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(project_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    n_rois = 1000
    maps = CONDITIONS  # or CONTRASTS, depending on what you want to process
    main(project_dir, cache_dir, n_rois, maps)
