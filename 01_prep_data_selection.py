"""Prepare IBC data and generate a cognitive-component design matrix.

This script:

1. Selects IBC contrast maps for a set of participants and tasks.
2. Removes conditions/contrasts known to be unreliable.
3. Builds a contrast-by-component design matrix from tag occurrences.
4. Checks the resulting matrix for rank deficiency and poor conditioning.
5. Saves the design matrix for subsequent analyses.

The script relies on the IBC data parser and the cognitive-component
utilities provided by this project.
"""

SUBJECTS = ["sub-%02d" % i for i in [4]]  # One subject with full data as ref
DEFAULT_EXCLUDED_TASKS = ("movie_tasks", "retinotopy")

import os

import numpy as np
import yaml
from ibc_public.utils_data import CONTRASTS

from utils_components import (
    append_task_id,
    check_design_mtx,
    contrasts_and_tags,
    get_used_contrasts,
    occurences_matrix,
    setup_data,
    tags,
)


# Utilities ------------------------------------------------------------------
def load_nonreliable_conditions(file_path):
    """
    Load non-reliable conditions from a yaml file.
    """
    try:
        with open(file_path, "r") as f:
            non_reliable_conds = yaml.safe_load(f)
        if non_reliable_conds is None:
            non_reliable_conds = []
        return non_reliable_conds
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def filter_db_by_conditions(db, non_reliable_conds=None, use_contrast=False):
    """
    Filter the database to exclude non-reliable conditions.
    """
    if non_reliable_conds is None:
        if use_contrast:
            non_reliable_conds = load_nonreliable_conditions(
                "nonreliable_contrast_volume.yml"
            )
        else:
            non_reliable_conds = load_nonreliable_conditions(
                "nonreliable_conditions_volume.yml"
            )
    else:
        assert isinstance(
            non_reliable_conds, list
        ), "non_reliable_conds should be a list of condition names."

    rem_idx = []
    for cond in non_reliable_conds:
        cond_parts = cond.split("_", 1)  # Split only on the first underscore
        task_name = cond_parts[0]
        cond_name = cond_parts[1]

        rem_ = db[
            (db.task == task_name) & (db.contrast == cond_name)
        ].index.tolist()
        rem_idx.extend(rem_)

    filtered_db = db.drop(index=rem_idx).reset_index(drop=True)
    return filtered_db


# Design matrix --------------------------------------------------------------


def occurences_to_designmat(
    db, all_contrasts_file, write_dir, maps=CONTRASTS, verbose=True, out_sfx=""
):
    """
    Create a design matrix based on the occurrence of tags in the contrasts.
    Each row = contrast, each column = tag.
    Pays attention to matrix conditioning and rank.
    """
    # Filter out non-fmri maps
    db_names = append_task_id(db)
    exclude = {"gm", "highres_gm", "highres_t1_bet", "t1", "t1_bet", "wm"}
    unique_maps = [m for m in db_names.contrast.unique() if m not in exclude]

    # Filter all maps to only those present in the db
    maps["full_name"] = maps["task"] + "_" + maps["contrast"]
    maps = maps[maps["full_name"].isin(unique_maps)].reset_index(drop=True)

    # Gather contrasts and tags occurrences
    my_tasks = [task for task in db.task.unique() if task != ""]
    df_contrasts = get_used_contrasts(my_tasks, maps, all_contrasts_file)
    contrasts_list, all_tags = contrasts_and_tags(my_tasks, df_contrasts)
    components_clean, _, unique_components = tags(all_tags)
    df = occurences_matrix(components_clean, unique_components, contrasts_list)

    # Drop duplicated columns and version-specific exclusions
    df = df.T.drop_duplicates().T
    with open("sub_specific_drops.yaml", "r") as file:
        all_drops = yaml.safe_load(file) or {}
    df = df.drop(columns=all_drops[f"all_{out_sfx}"], errors="ignore")

    # Remove all-zero rows/cols
    df = df.loc[~(df == 0).all(axis=1), (df != 0).any(axis=0)]

    # Rank and conditioning checks
    _, s, _ = np.linalg.svd(df.values, 0)
    rank = np.linalg.matrix_rank(df.values)
    print("SINGULAR VALUES: ", s)
    if np.any(s < 0.1) or rank < df.shape[1]:
        check_design_mtx(df)
        raise ValueError("Matrix is not full rank or near-singular")

    if verbose:
        print(f"Shape: {df.shape}, Rank: {rank}, Cond: {s.max()/s.min():.2f}")

    df.to_csv(
        os.path.join(write_dir, f"design_matrix_{out_sfx}.tsv"), sep="\t"
    )

    return df


def print_visual_check(df):
    """
    Check that the contrasts do not have redundant or/and orthogonal tags.
    """
    for col in df.columns:
        # print whenever there is a 1
        rows_with_1 = df.index[df[col] == 1].tolist()
        if len(rows_with_1) > 1:
            print(f"{col} in {rows_with_1}")


# Main -----------------------------------------------------------------------
def main(
    project_dir,
    write_dir_name,
    all_cont_file,
    maps_list=CONTRASTS,
    output_suffix="",
    subjects=SUBJECTS,
):
    """Run the data preparation and generates (and saves) the design-matrix.
    Parameters
    ----------
    project_dir : str
        Path to the project directory.
    write_dir_name : str
        Name of the directory where results will be saved.
    all_cont_file : str
        Path to the file containing all contrasts.
    maps_list : list, optional
        List of maps to be used, by default CONTRASTS.
    output_suffix : str, optional
        Suffix to be added to the output file name, by default "".
    subjects : list, optional
        List of subjects to be included, by default SUBJECTS.
    """
    res_dir = os.path.join(project_dir, write_dir_name)
    res_dir.mkdir(parents=True, exist_ok=True)
    db = setup_data(subjects, exclude=DEFAULT_EXCLUDED_TASKS, maps=maps_list)
    db_cleaned = filter_db_by_conditions(db, use_contrast=True)
    df = occurences_to_designmat(
        db_cleaned,
        all_cont_file,
        res_dir,
        maps=maps_list,
        out_sfx=output_suffix,
    )
    print_visual_check(df)

    print(
        f"Design matrix generated successfully: "
        f"{df.shape[0]} contrasts x {df.shape[1]} components. "
    )


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    write_dir_name = "results"
    all_cont_file = os.path.join(project_dir, "all_contrasts.tsv")
    main(project_dir, write_dir_name, all_cont_file)
