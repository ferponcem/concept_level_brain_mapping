"""
Create the features matrix, using the occurance of the tags and keeping the
coefficients of the weighted matrices for each subject.
"""

import glob
import os

import numpy as np
import pandas as pd
from ibc_public.utils_data import CONDITIONS

from utils_components import get_task_list

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


def prepare_X(sub, Y_structure, weighted_mats_dir):
    """
    Prepare the X matrix for the model: n_conds x features (tags)
    """
    X = Y_structure.copy()
    sub_num = sub.split("-")[-1]
    tags_mat = pd.read_csv(
        os.path.join(
            weighted_mats_dir,
            f"avg_A_stack_{sub_num}out.csv",
        ),
        index_col=0,
    )

    # Copy the headers of tags_mat to X
    X = pd.concat([X, pd.DataFrame(columns=tags_mat.columns)], axis=1)

    # for i, _ in tags_mat.iloc[120:130].iterrows():
    for i, _ in tags_mat.iterrows():
        task_ = i.split("_")[0]
        condition_ = "_".join(i.split("_")[1:])

        # Find where to put these tags in X
        indices = X[
            (X["task"] == task_) & (X["condition"] == condition_)
        ].index

        for idx in indices:
            if tags_mat.loc[i].values.ndim == 1:
                X.loc[idx, tags_mat.columns] = tags_mat.loc[i].values
            else:
                # special case Mario: two same id rows but slightly diff values
                # for now, quick fix to take the first row only
                X.loc[idx, tags_mat.columns] = tags_mat.loc[i].values[0]

    # Find empty rows in X and remove them: data not available
    empty_rows = X[X[tags_mat.columns].isnull().all(axis=1)].index
    if len(empty_rows) > 0:
        X = X.drop(empty_rows).reset_index(drop=True)

    # Find empty columns or all zeros and remove them: tags not used
    empty_cols = []
    for col in tags_mat.columns:
        if (
            (np.sum(X[col].values) == 0) | (X[col].isnull().all(axis=0))
        ).any():
            empty_cols.append(col)

    X = X.drop(columns=empty_cols)

    return X


def adjust_Y(X, Y):
    """
    Adjust Y to have the same rows as X
    """
    merged = pd.merge(
        X[["task", "condition", "dir", "condition_id", "map_id"]],
        Y,
        on=["task", "condition", "dir", "condition_id", "map_id"],
        how="inner",
    )
    return merged.reset_index(drop=True)


def main(data_dir, out_dir, weighted_mats_dir, task_to_exclude, parcel_flag):
    """
    Main function to prepare the features matrix (X) and the target matrix (Y)
    1. Load the saved Y matrix for each subject, or prepare it if not available.
    2. Prepare the X matrix for each subject using the weighted matrices.
    3. Adjust the Y matrix to have the same rows as the X matrix.
    4. Save the X and adjusted Y matrices to the output directory.
    Parameters
    ----------
    data_dir : str
        Directory where the parcellated data is stored.
    out_dir : str
        The output directory where the X and Y matrices will be saved.
    weighted_mats_dir : str
        Directory where the weighted matrices are stored.
    task_to_exclude : list of str
        List of tasks to exclude from the analysis.
    parcel_flag : str
        Flag indicating the parcellation scheme used (e.g., "schaefer1000").
    """
    TASKS = get_task_list(exclude=task_to_exclude)
    conditions_df = create_condition_list(TASKS, CONDITIONS)

    for sub in SUBJECTS:
        # Load the saved Y matrix and save a copy to out_dir
        Y_file = os.path.join(out_dir, f"{sub}_Y.csv")
        if os.path.exists(Y_file):
            Y = pd.read_csv(Y_file)
        else:
            parcell_dir = os.path.join(data_dir, f"{sub}_{parcel_flag}")
            Y = prepare_Y(sub, parcell_dir, conditions_df)

        # Get and save the X matrix keeping the same metadata as Y
        Y_structure = Y[
            ["task", "condition", "dir", "condition_id", "map_id"]
        ].copy()
        X = prepare_X(sub, Y_structure, weighted_mats_dir)
        X.to_csv(os.path.join(out_dir, f"{sub}_X.csv"), index=False)

        # Adjust Y to have the same rows as X
        Y_adjusted = adjust_Y(X, Y)
        Y_adjusted.to_csv(os.path.join(out_dir, f"{sub}_Y.csv"), index=False)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "parcelated_data")
    out_dir = os.path.join(project_dir, "data_weighted")
    os.makedirs(out_dir, exist_ok=True)
    parcel_flag = "schaefer1000"
    task_to_exclude = [
        "Abstraction",
        "Self",
        "Attention",
        "MVIS",
        "OptimismBias",
        "FingerTapping",
    ]
    main(data_dir, out_dir, project_dir, task_to_exclude, parcel_flag)
