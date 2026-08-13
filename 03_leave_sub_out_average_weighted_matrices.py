"""
Create average of weighted matrices from all subjects leaving one out at a time.
"""

import os

import numpy as np
import pandas as pd

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def format_mats_with_columns(A_mat, scode_ini, index_ini):
    """
    Format the sparse code matrix with information on columns and index
    """
    columns = scode_ini.columns
    idx = index_ini.astype(str)
    sub_df = pd.DataFrame(data=A_mat, columns=columns, index=idx)
    return sub_df


def _rename_duplicates(index):
    # Create a mapping from original to temporary names
    seen = {}
    new_index = []
    for item in index:
        if item in seen:
            count = seen[item]
            seen[item] = count + 1
            new_index.append(f"{item}_rep{count}")
        else:
            seen[item] = 1
            new_index.append(item)
    return new_index, seen


def _restore_names(index):
    # Restore original names by removing the suffix
    original_index = [item.split("_rep")[0] for item in index]
    return original_index


def make_matrices_unisize(subs_df, ref_sub="sub-04"):
    """
    Make all matrices the same size by adding NaNs to the missing cells
    The goal is to have sub-04 as reference and match the rest to this one
    """
    ref_df = subs_df[ref_sub]
    ref_columns = ref_df.columns
    ref_index = ref_df.index

    new_subs_df = {}

    for sub, mat in subs_df.items():
        if mat.shape != ref_df.shape:
            # Rename duplicates
            temp_index, seen = _rename_duplicates(mat.index)
            temp_mat = mat.copy()
            temp_mat.index = temp_index

            # Reindex
            temp_mat = temp_mat.reindex(columns=ref_columns, index=ref_index)

            # Restore original names
            temp_mat.index = _restore_names(temp_mat.index)

            new_subs_df[sub] = temp_mat
        else:
            new_subs_df[sub] = mat

    return new_subs_df


def average_matrices(mats_samesize):
    """
    Get the mean sparce matrix of all subjects
    """
    stack_mats = np.stack([mat.to_numpy() for mat in mats_samesize.values()])
    avg_mat = np.nanmean(stack_mats, axis=0)
    ref_sub = "sub-04" if "sub-04" in mats_samesize else "sub-06"
    avg_df = pd.DataFrame(
        data=avg_mat,
        columns=mats_samesize[ref_sub].columns,
        index=mats_samesize[ref_sub].index,
    )
    return avg_df


def main(project_dir, mats_dir_name):
    """
    Average matrices from all subjects leaving one out at a time. Saves
    the leave-one-out average matrices in the mats_dir_name directory.

    Parameters
    ----------
    project_dir : str
        Path to the project directory.
    mats_dir_name : str
        Name of the directory containing the matrices.
    """

    mats_dir = os.path.join(project_dir, mats_dir_name)
    subs_mats = {}
    ref_sub = "sub-04"  # Reference subject to match shapes
    for sub in SUBJECTS:
        # 0. Formatting
        sub_dir = os.path.join(mats_dir, sub)

        # 1. Get each subject's matrix, sparse code and index
        A_mat = np.load(os.path.join(sub_dir, "A_stack_norm.npy"))
        scode_init = pd.read_csv(
            os.path.join(sub_dir, "scode_init.csv"), index_col=0
        )
        index_init = scode_init.index.astype(str)

        # 2. Get columns of each subject's matrix and create a dataframe
        sub_df = format_mats_with_columns(A_mat, scode_init, index_init)
        subs_mats[sub] = sub_df

    # 3. Make all matrices the same size
    mats_samesize = make_matrices_unisize(subs_mats, ref_sub)

    # 4.a Average the matrices
    avg_mat_df = average_matrices(mats_samesize)
    # 5.a Save average matrix
    filepath = os.path.join(mats_dir, "avg_A_stack.csv")
    avg_mat_df.to_csv(filepath)

    # 4.b Remove a subject from the analysis
    for sub in mats_samesize.keys():
        mats_to_avg = {k: v for k, v in mats_samesize.items() if k != sub}
        # 5.b Average matrices
        avg_mat_df = average_matrices(mats_to_avg)
        # 6.b Save average matrix
        filepath = os.path.join(
            mats_dir, f"avg_A_stack_{sub.split('-')[-1]}out.csv"
        )
        avg_mat_df.to_csv(filepath)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    mats_dir_name = "weighting_components"
    main(project_dir, mats_dir_name)
