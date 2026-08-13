"""
Weighting procedure to derive a sparse representation and a better
initialization for the design matrix.
"""

import os

import numpy as np
import pandas as pd
import yaml
from ibc_public.utils_data import CONTRASTS
from joblib import Parallel, delayed
from nilearn.maskers import NiftiMasker
from sklearn.decomposition import SparseCoder
from sklearn.linear_model import LinearRegression

from utils_components import (
    append_task_id,
    check_design_mtx,
    get_mask,
    mask_data_by_reliable_regions,
    setup_data,
)

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
DEFAULT_EXCLUDED_TASKS = ("movie_tasks", "retinotopy")


def init_data_prep(db):
    """
    Perfom all common steps to prepare the IBC data, checks the design matrix
    conditioning and adds the task_id to the database.
    Returns:
    - db: dataframe with the paths of the desired contrasts
    - df: design matrix
    - masker: masker object
    - write_dir: directory to save the results
    """
    mask = get_mask()
    masker = NiftiMasker(mask_img=mask).fit()

    db_renamed = append_task_id(db)

    return db_renamed, masker


def df_load(db, dmat_file, suf="", use_contrasts=True):
    """
    Load the design matrix and drop specific columns depending on the
    subjects in the database.
    Returns:
    - df: design matrix
    """
    df = pd.read_csv(os.path.join(dmat_file), sep="\t", index_col=0)

    # Extra drops for specific subjects with less data
    if use_contrasts:
        suf = suf + "_contrasts"
        with open("sub_specific_drops.yaml", "r") as file:
            all_drops = yaml.safe_load(file) or {}
        for subject in {"sub-01", "sub-07", "sub-13"} & set(
            db.subject.unique()
        ):
            df = df.drop(
                columns=all_drops.get(f"{subject}_{suf}_bin", []),
                errors="ignore",
            )

    return df


def check_matrix_conditioning(df):
    """
    Check the design matrix conditioning using SVD.
    Raises an error if singular values are below a threshold.
    """
    # Drop duplicates and all-zero columns
    df = df.T.drop_duplicates().T

    # Check for all-zero rows
    rows_all_zeros = df.index[(df == 0).all(axis=1)].tolist()
    if rows_all_zeros:
        raise ValueError(f"Rows with all zeros found: {rows_all_zeros}")

    # Drop all-zero columns
    cols_all_zeros = df.columns[(df == 0).all(axis=0)].tolist()
    if cols_all_zeros:
        print(f"Removing cols with all zeros: {cols_all_zeros}")
        df = df.drop(columns=cols_all_zeros)

    # Check design matrix conditioning with SVD
    _, s, _ = np.linalg.svd(df.values, 0)
    print("SINGULAR VALUES: ", s)
    if np.any(s < 0.1):
        check_design_mtx(df)
        raise ValueError("Singular values smaller than 0.1!!!")

    return df


def compute_loss(Y, A, D, alpha):
    """
    Compute the objective function.
    Parameters:
    - Y: [n_samples, n_voxels] data matrix
    - A: [n_samples, n_components] design matrix
    - D: [n_components, n_voxels] dictionary matrix
    - alpha: regularization parameter
    Returns:
    - objective: objective function value
    """
    # Compute residual error || Y - A D ||^2
    recons_error = np.linalg.norm(Y - np.dot(A, D)) ** 2
    # Compute sparsity penalty || A ||_1
    sparsity_penalty = np.linalg.norm(A, ord=1)
    return recons_error + (alpha * sparsity_penalty)


def get_D_regression(Y, A):
    """
    Learn the matrix D using a linear regression model.
    Parameters:
    - Y: (n_samples, n_voxels) observed data
    - A: (n_samples, n_components) sparse codes
    Returns:
    - D: (n_components, n_voxels) dictionary matrix
    """
    Y, A = np.atleast_2d(Y, A)
    reg = LinearRegression(fit_intercept=False, n_jobs=1)
    reg.fit(A, Y)
    D = reg.coef_.T
    return D


def normalize_mat_rows(D, norm_type="l2"):
    """
    Normalize the matrix D by normalizing each row (component) using either
    L1 or L2 norm.
    """
    if norm_type == "l2":
        row_norms = np.linalg.norm(D, ord=2, axis=1, keepdims=True)
    elif norm_type == "l1":
        row_norms = np.linalg.norm(D, ord=1, axis=1, keepdims=True)
    else:
        raise ValueError("Invalid norm type. Use 'l1' or 'l2'.")

    row_norms[row_norms == 0] = 1e-10  # Avoid division by zero
    D_norm = D / row_norms  # Normalize each row

    return D_norm


def estimate_sparse_code(y, D_n, alpha=0.001):
    """
    Compute the sparse code a for a given y and learnt matrix D by using
    SK's sparse coder.
    Parameters:
    - y: (1, n_voxels) data vector
    - D_n: (n_components, n_voxels) masked dictionary matrix
    - alpha: sparsity regularization parameter
    Returns:
    - a: (1, n_components) sparse code
    """
    # Check if y is one-dimensional and reshape if necessary
    if y.ndim == 1:
        y = y.reshape(1, -1)

    coder = SparseCoder(
        dictionary=D_n,
        transform_algorithm="lasso_cd",
        transform_alpha=alpha,
        positive_code=True,
        n_jobs=1,
    )
    a = coder.transform(y)

    return a


def update_A(Y, A, D, alpha=0.001):
    """
    Update each row of A using a sparse coder, considering only the active
    components.

    Parameters:
    - Y: (n_samples, n_voxels) observed data
    - A: (n_samples, n_components) sparse codes
    - D: (n_components, n_voxels) dictionary matrix
    - alpha: sparsity regularization parameter

    Returns:
    - Updated A with row-wise sparse coding
    """
    A_new = np.zeros_like(A)

    for i in range(A.shape[0]):
        # Choose a single sample (1, n_voxels/n_components)
        y_i = Y[i]
        a_i = A[i]

        # Get indices of nonzero components in a_i to mask D
        active_components = np.where(a_i != 0)[0]
        if len(active_components) > 0:
            # Select only the active components in matrix D
            D_masked = np.zeros_like(D)
            D_masked[active_components] = D[active_components]

            # Update A with the new sparse code
            a_i_new = estimate_sparse_code(y_i, D_masked, alpha)
            A_new[i, :] = a_i_new
        else:
            ValueError("No active components for sample ", i)

    return A_new


def apply_delta_normalization(A, D):
    """
    Normalize A and D using a diagonal matrix Gamma based on the column L2
    norms of A.
    Ensures Y = A * Gamma * Gamma^{-1} * D.
    """
    # Column norms of A
    col_norms = np.linalg.norm(A, ord=2, axis=0)
    col_norms[col_norms == 0] = 1e-10
    # Normalize A and adjust D accordingly
    A_norm = A / col_norms
    D_norm = D * col_norms[:, np.newaxis]

    return A_norm, D_norm


def process_subject(
    sub,
    db_names,
    dmat_file,
    masker,
    out_dir,
    suf,
    n_iterations=30,
    use_contrasts=True,
    sc_alpha=0.001,
):
    """
    Process a single subject: prepare data, initialize sparse code, and run
    weighting iterations.
    Saves the results in the specified output directory.
    """
    sub_res_dir = os.path.join(out_dir, sub)
    os.makedirs(sub_res_dir, exist_ok=True)
    dbs = db_names.query(
        "subject == @sub and modality == 'bold' and acquisition == 'ffx' and session != 'ses-00'"
    )
    if sub == "sub-08":
        dbs = dbs[~((dbs.task == "MathLanguage") & (dbs.session == "ses-24"))]

    # Preparing data
    df = df_load(dbs, dmat_file, suf=suf, use_contrasts=use_contrasts)
    valid_dbs = dbs[dbs.contrast.isin(df.index)]
    valid_contrasts = valid_dbs.contrast.tolist()
    fmri_files = valid_dbs.path.tolist()

    # Design matrix restricted to subject's available contrasts
    sparse_code_init = df.T[valid_contrasts].T
    sparse_code_init = check_matrix_conditioning(sparse_code_init)
    print(sub, sparse_code_init.shape)  # Debugging
    sparse_code_init.to_csv(
        os.path.join(sub_res_dir, "scode_init.csv"), index=True
    )

    # Y = masker.transform(fmri_files)
    Y = mask_data_by_reliable_regions(
        masker.mask_img,
        fmri_files,
        sparse_code_init,
        return_array=True,
        use_contrasts=use_contrasts,
    )
    A = sparse_code_init.to_numpy()

    # First matrix D estimation and sparse code update
    D = get_D_regression(Y, A)
    D_N = normalize_mat_rows(D, norm_type="l2")
    A_stack = update_A(Y, A, D_N, alpha=sc_alpha)

    losses = []
    loss = compute_loss(Y, A_stack, D_N, alpha=sc_alpha)
    losses.append(loss)
    print(f"Initial loss: {loss}")

    # Iterative updates of D and A
    for i in range(n_iterations):
        D = get_D_regression(Y, A_stack)
        D_N = normalize_mat_rows(D, norm_type="l2")
        np.save(os.path.join(sub_res_dir, f"D_{i}.npy"), D_N)

        A_stack = update_A(Y, A_stack, D_N, alpha=sc_alpha)
        np.save(os.path.join(sub_res_dir, f"A_stack_{i}.npy"), A_stack)

        loss = compute_loss(Y, A_stack, D_N, alpha=sc_alpha)
        print(f"Iteration {i + 1}, loss: {loss}")
        losses.append(loss)
        np.save(os.path.join(sub_res_dir, "losses.npy"), losses)

    # Final normalization of A and D using a diagonal matrix Delta
    A_stack_norm, D_norm = apply_delta_normalization(A_stack, D_N)
    np.save(os.path.join(sub_res_dir, "A_stack_norm.npy"), A_stack_norm)
    np.save(os.path.join(sub_res_dir, "D_norm.npy"), D_norm)

    print(f"Subject {sub} done!")


def main(
    project_dir,
    dmat_file,
    weights_dir="weighting_components",
    n_iterations=30,
    use_contrasts=True,
    maps_list=CONTRASTS,
    suffix="",
    n_jobs=12,
):
    """
    Main function to run the weighting process for all subjects in parallel.
    Parameters
    ----------
    project_dir : str
        Path to the project directory.
    dmat_file : str
        Path to the design matrix file.
    weights_dir : str, optional
        Directory to save the results, by default "weighting_components".
    n_iterations : int, optional
        Number of pipeline iterations, by default 30.
    use_contrasts : bool, optional
        Whether to use contrasts in the design matrix, by default True.
    maps_list : list, optional
        List of maps to be used, by default CONTRASTS.
    suffix : str, optional
        Suffix to be added to the output file name, by default "".
    n_jobs : int, optional
        Number of parallel jobs to run, by default 12.
    """
    out_dir = os.path.join(project_dir, weights_dir)
    os.makedirs(out_dir, exist_ok=True)
    db = setup_data(
        SUBJECTS, exclude=DEFAULT_EXCLUDED_TASKS, conditions=maps_list
    )
    db_names, masker = init_data_prep(db)
    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(
            sub,
            db_names,
            dmat_file,
            masker,
            out_dir,
            suf=suffix,
            n_iterations=n_iterations,
            use_contrasts=use_contrasts,
        )
        for sub in SUBJECTS
    )


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    use_contrasts = False
    dmat_file = os.path.join(
        project_dir, "cognitive_components", "design_matrix_regsel.tsv"
    )
    main(
        project_dir,
        dmat_file,
        weights_dir="weighting_components",
        n_iterations=30,
        use_contrasts=use_contrasts,
        maps_list=CONTRASTS,
        suffix="",
        n_jobs=12,
    )
