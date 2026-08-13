"""
Subject-level pipeline for training and testing the encoding model.
See Walters et al., 2022, for details.
"""

import argparse
import os

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeavePGroupsOut

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]
memory = Memory(location="cache_dir", verbose=0)


def GroupKFold_random_splits(n_splits, groups, rng):
    """
    Implements functionality of sklearn's GroupKFold, but with random
    assignment of groups to folds (GroupKFold does not do this).
    Requires an rng.
    Output: list of (train, test) splits as lists of indices
    """
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    rng.shuffle(unique)
    splits_ix = []
    for split in np.array_split(unique, n_splits):
        mask = np.isin(groups, split)
        train, test = ix[~mask], ix[mask]
        splits_ix.append((train, test))
    return splits_ix


def _derangement(arr, rng):
    """Return a deranged permutation of arr (no element stays in same pos)"""
    while True:
        perm = rng.permutation(arr)
        if not np.any(perm == arr):  # check no fixed points
            return perm


def generate_null_features(X, features, seed):
    """
    Generate null features based on specified null model.
    """
    rng = np.random.default_rng(seed)
    X_null = X.copy()

    unique_conds = X["condition"].unique()
    permuted = _derangement(unique_conds, rng)

    for orig_cond, shuffled_cond in zip(unique_conds, permuted):
        cond_idx = X.index[X["condition"] == orig_cond].tolist()
        candidate_rows = X.loc[X["condition"] == shuffled_cond, features]

        if candidate_rows.shape[0] == 0:
            # shuffled_cond not present for this subject —
            # fall back to random sample
            sampled_row = (
                X[features]
                .sample(n=1, random_state=rng.bit_generator)
                .iloc[0]
                .values
            )
        else:
            # pick one representative row from that condition (random choice
            # if there are multiple)
            sampled_row = (
                candidate_rows.sample(n=1, random_state=rng.bit_generator)
                .iloc[0]
                .values
            )

        # assign sampled feature vector to all rows of the original condition
        X_null.loc[cond_idx, features] = sampled_row

    return X_null


def compute_metrics(y_true, y_pred):
    """
    Returns a dict of useful metrics computed vectorized.
    """
    # Per-map correlations: for each map pair
    n_maps = y_true.shape[0]
    corr_sug = []
    r2_sug = []
    mse_sug = []
    for m in range(n_maps):
        corr_i = np.corrcoef(y_true[m], y_pred[m])[0, 1]
        corr_sug.append(corr_i)
        r2_i = r2_score(y_true[m], y_pred[m])
        r2_sug.append(r2_i)
        mse_i = np.mean((y_true[m] - y_pred[m]) ** 2)
        mse_sug.append(mse_i)
    corr_s = np.mean(corr_sug)
    r2_s = np.mean(r2_sug)
    mse_s = np.mean(mse_sug)

    return {
        "correlations": corr_sug,
        "mean_corr_s": corr_s,
        "r2_scores_s": r2_sug,
        "mean_r2_s": r2_s,
        "mse_s": mse_sug,
        "mean_mse_s": mse_s,
    }


@memory.cache
def run_single_split(X, y, train_idx, test_idx, features, alphas):
    """
    Runs one CV split. Cached so recomputing is avoided.
    """
    train_X = X.iloc[train_idx].filter(features).values
    train_y = y.iloc[train_idx].filter(regex="region").values

    test_X = X.iloc[test_idx].filter(features).values
    test_y = y.iloc[test_idx].filter(regex="region").values

    groups = np.array(X.condition_id)
    train_groups = groups[train_idx]

    cv = GroupKFold_random_splits(
        n_splits=10, groups=train_groups, rng=np.random.default_rng(0)
    )
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(train_X, train_y)

    y_pred = model.predict(test_X)
    metrics = compute_metrics(test_y, y_pred)

    return {
        "alpha": model.alpha_,
        "y_true": test_y,
        "y_pred": y_pred,
        **metrics,
    }


# SUBJECT-LEVEL PIPELINE
def process_subject(
    sub,
    X,
    y,
    features,
    results_dir,
    out_flag="",
    alphas=np.logspace(-3, 3, 20),
):
    """
    Process a single subject and return a DataFrame saved to
    results/sub-XX_results.csv.
    """
    rows = []

    groups = np.array(X.condition_id)
    splitter_outloop = LeavePGroupsOut(n_groups=1)
    splits = list(splitter_outloop.split(X, y, groups=groups))

    for fold, (train_idx, test_idx) in enumerate(splits):
        result = run_single_split(X, y, train_idx, test_idx, features, alphas)
        result["fold"] = fold
        rows.append(result)

    df = pd.DataFrame(rows)
    out_file = os.path.join(results_dir, f"{sub}_results{out_flag}.csv")
    df.to_csv(out_file, index=False)

    return df


# DISPATCH FOR ALL SUBJECTS
def run_all_subjects(subjects_data, results_dir, out_flag, n_jobs=-1):
    """
    Run the full pipeline for all subjects in parallel.
    subjects_data: { "sub-01": (X, y, features), "sub-02": (X, y, features), ... }
    """
    alphas = np.array([1e-3, 1e-2, 1e-1] + list(np.arange(1, 11, dtype=float)))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(
            sub, X, y, features, results_dir, out_flag, alphas
        )
        for sub, (X, y, features) in subjects_data.items()
    )

    return results


def main(data_dir, null_model, seed, results_dir, out_flag):
    """
    Run and save the results for all subjects, optionally using a null model.
    Parameters
    ----------
    data_dir : str
        Directory containing the input data files (X and Y for each subject).
    null_model : str
        Null model to use: "None" or "shuffle-conditions".
    seed : int
        Random seed for reproducibility.
    results_dir : str
        Directory to save the results.
    out_flag : str
        Suffix to append to the output filenames (e.g., "_nullmodel-shuffle-
        conditions").
    """
    metadata_cols = ["task", "condition", "dir", "condition_id", "map_id"]
    subjects_data = {}

    for sub in SUBJECTS:
        Y = pd.read_csv(os.path.join(data_dir, f"{sub}_Y.csv"))
        X = pd.read_csv(os.path.join(data_dir, f"{sub}_X.csv"))

        features = [f for f in X.columns.tolist() if f not in metadata_cols]

        if null_model != "None":
            X_null = generate_null_features(X, features, seed)
            subjects_data[sub] = (X_null, Y, features)
        else:
            subjects_data[sub] = (X, Y, features)

    _ = run_all_subjects(subjects_data, results_dir, out_flag, n_jobs=15)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")  # <--- here
    results_dir = os.path.join(project_dir, "results_refactored_bin")  # <---
    os.makedirs(results_dir, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Script for getting CEM one subject"
    )
    parser.add_argument(
        "-null_model",
        type=str,
        required=True,
        help="Null model to use: None or shuffle-conditions",
    )

    args = parser.parse_args()
    null_model = args.null_model

    if null_model != "None":
        out_flag = f"_nullmodel-{null_model}"
    else:
        out_flag = ""

    seed = int(0)

    main(
        data_dir,
        null_model,
        seed=0,
        results_dir=results_dir,
        out_flag=out_flag,
    )
