"""
Create the features matrix, using the occurance of the utils_conditions_tags
"""

import glob
import os

import numpy as np
import pandas as pd
from ibc_public.utils_data import CONDITIONS

from utils_components import (
    contrasts_and_tags,
    get_task_list,
    get_used_contrasts,
    occurences_matrix,
    tags,
)

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def create_condition_list(wanted_tasks, conditions=CONDITIONS):
    """
    Keep the entries of the conditions that are in the wanted_tasks
    """
    conditions = conditions[conditions["task"].isin(wanted_tasks)]
    conditions = conditions[["task", "contrast"]]
    return conditions.reset_index(drop=True)


def prepare_Y(sub, parcell_dir, conds_list):
    """
    Prepare the Y matrix for the model: n_conds x 1000 parcels
    """
    # count to give each map a unique id
    map_count = 0
    # Create a dataframe to store the Y matrix
    numbered_columns = ["region_" + str(i) for i in range(1, 1001)]
    Y = pd.DataFrame(
        columns=["task", "condition", "dir", "condition_id", "map_id"]
        + numbered_columns
    )
    for i, cond in conds_list.iterrows():
        # get the task-condition pair
        task = cond["task"]
        condition = cond["contrast"]
        cond_id = i + 1
        # Load the parcellated data for the condition
        cond_maps = glob.glob(
            os.path.join(parcell_dir, f"{sub}_{task}*{condition}.npy")
        )
        for j, cond_map in enumerate(cond_maps):
            # Load the parcellated data
            data = np.load(cond_map)
            map_count += 1
            # Prepare the metadata for the row
            row = {
                "task": task,
                "condition": condition,
                "dir": cond_map.split("dir-")[-1].split("_")[0],
                "condition_id": cond_id,
                "map_id": map_count,
            }
            # Add the parcel data
            row.update(
                {
                    "region_" + str(k + 1): value
                    for k, value in enumerate(data.flatten())
                }
            )
            # Append the row to the DataFrame
            Y = pd.concat([Y, pd.DataFrame([row])], ignore_index=True)

    return Y


def get_tags(tasks, conditions):
    """
    Get the tags for the features matrix
    """
    df_conds = get_used_contrasts(tasks, conditions)
    conds_newname, tags_list = contrasts_and_tags(tasks, df_conds)
    tags_clean, _, unique_tags = tags(tags_list)
    tags_df = occurences_matrix(tags_clean, unique_tags, conds_newname)
    return tags_df


def create_general_X(conditions_df):
    """
    Create a general X matrix with all the tags for the conditions
    """
    tasks_ = conditions_df["task"].unique()
    tags_mat = get_tags(tasks_, conditions_df)
    return tags_mat


def prepare_X(Y_structure, conds_df):
    """
    Prepare the X matrix for the model: n_conds x 151 features (tags)
    """
    X = Y_structure.copy()
    tasks_ = X["task"].unique()
    conds_ = X["condition"].unique()
    conds_list = conds_df[conds_df["contrast"].isin(conds_)]
    tags_mat = get_tags(tasks_, conds_list)

    # fill the X matrix with the tags
    X = pd.concat([X, pd.DataFrame(columns=tags_mat.columns)], axis=1)
    for i, row in X.iterrows():
        # find the tags for the condition
        cont_id = row["task"] + "_" + row["condition"]
        tags = tags_mat.loc[cont_id]
        X.loc[i, tags.index] = tags.values

    return X


def main(main_dir, out_dir, task_to_exclude, parcels_dir_wildcard):
    """
    Main function to prepare the features matrix (X) and the target matrix (Y)
    for each subject, excluding the specified tasks.
    Parameters
    ----------
    main_dir : str
        The main directory where the parcellated data is stored.
    out_dir : str
        The output directory where the X and Y matrices will be saved.
    task_to_exclude : list of str
        List of tasks to exclude from the analysis.
    parcels_dir_wildcard : str
        Wildcard string to identify the parcellated data directories
        (e.g., "_schaefer1000").
    """
    TASKS = get_task_list(no_task=task_to_exclude)
    conditions_df = create_condition_list(TASKS, CONDITIONS)

    X_main = create_general_X(conditions_df)
    X_main.to_csv(os.path.join(out_dir, "X_main.csv"), index=False)

    for sub in SUBJECTS:
        # Load the parcellated data
        parcell_dir = os.path.join(main_dir, f"{sub}{parcels_dir_wildcard}")

        # Get and save the Y matrix
        Y = prepare_Y(sub, parcell_dir, conditions_df)
        Y.to_csv(os.path.join(out_dir, f"{sub}_Y.csv"), index=False)

        # Get and save the X matrix keeping the same metadata as Y
        Y_structure = Y[
            ["task", "condition", "dir", "condition_id", "map_id"]
        ].copy()
        X = prepare_X(Y_structure, conditions_df)
        X.to_csv(os.path.join(out_dir, f"{sub}_X.csv"), index=False)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    parcels_dir_wildcard = "_schaefer1000"
    # These are the tasks that we want to exclude from the analysis, as their
    # condition maps are not "reliable".
    task_to_exclude = [
        "Abstraction",
        "Self",
        "Attention",
        "MVIS",
        "OptimismBias",
        "FingerTapping",
    ]
    main(project_dir, out_dir, task_to_exclude, parcels_dir_wildcard)
