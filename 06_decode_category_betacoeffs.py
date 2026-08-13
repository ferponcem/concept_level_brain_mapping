"""
Classify two different categories of trials (e.g., Faces vs Bodies) using
beta coefficients from different cognitive components as features.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.image import new_img_like, resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Which type of analysis to run: either using the top_k regions from each
# component separately (components) or using the union of top_k regions from
# first two components (union)
MODE = "union"
WITHIN_SUB = False  # whether to run within-subject classification
SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def update_results_collection(results_file, new_results):
    """
    Update the results collection with new results.
    Parameters
    ----------
    results_collection : str
        Path to the existing results collection CSV file.
    new_results : DataFrame
        New results to append to the collection.
    Returns
    -------
    None
    """
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        updated_results = pd.concat([existing_results, new_results])
    else:
        updated_results = new_results

    updated_results.to_csv(results_file, index=False)


def sort_regions_by_contribution(component, n_rois=1000, res_mm=1, top_k=100):
    """
    Get the regions in an atlas sorted by their values in a given
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

    # Get labels, gives an array of shape (n_regions + 1,) where 0 is the background
    labels = masker.labels_
    labels = labels[1:]  # remove background label
    labels = np.array(labels)

    # sort the region values in descending order and get the corresponding labels
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


def within_subject_classification(sub, ts_cat, ts_nocat):
    """
    Perform within-subject classification for a single subject.

    Parameters
    ----------
    sub : str
        Subject ID (e.g., "sub-01").
    ts_cat : array-like
        Time series data for the target category trials
        (shape: n_cat_trials x n_features).
    ts_nocat : array-like
        Time series data for the other category trials
        (shape: n_nocat_trials x n_features).

    Returns
    -------
    mean_sub_score : float
        Mean classification accuracy across cross-validation folds for the
        subject.
    std_sub_score : float
        Standard deviation of classification accuracy across folds for the
        subject.
    sem_Sub_score : float
        Standard error of the mean for classification accuracy across folds
        for the subject.
    """
    X = np.vstack([ts_cat, ts_nocat])
    y = np.array(["cat"] * len(ts_cat) + ["nocat"] * len(ts_nocat))

    cv_sub = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sub_scores = []

    for train_idx, test_idx in cv_sub.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearSVC(dual="auto"),
        )
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        sub_scores.append(accuracy_score(y_test, predictions))

    mean_sub_score = np.mean(sub_scores)
    std_sub_score = np.std(sub_scores)
    sem_Sub_score = std_sub_score / np.sqrt(len(sub_scores))

    print(f"{sub} within-subject accuracy: {mean_sub_score}")

    return mean_sub_score, std_sub_score, sem_Sub_score


def run_classification(
    subjects,
    work_dir,
    masker,
    target_category,
    other_category,
    n_regs,
    comp_name=None,
):
    """
    Run classification between target category and other category using
    beta coefficients from the atlas as features.
    1) Extract time series from the atlas for each subject.
    2) Classify cat vs nocat trials using raw beta values.

    Parameters
    ----------
    subjects: list of str
        list of subject IDs
    work_dir: str
        working directory containing subject data
    masker: Niimg-like
        NiftiLabelsMasker object for extracting time series
    target_category: str
        name of the target category (e.g., "Faces")
    other_category: str
        name of the other category (e.g., "Bodies")
    n_regs: int
        number of regions in the atlas (for reporting)
    comp_name: str, optional
        name of the component (for reporting)

    Returns
    -------
    mean_accuracy: float
        mean classification accuracy across cross-validation folds
    std_accuracy: float
        standard deviation of classification accuracy across folds
    sem_accuracy: float
        standard error of the mean for classification accuracy across folds
    """

    time_series_cat = []
    time_series_nocat = []
    group = []

    # Extract time series for a given subject
    for sub in subjects:
        subject_dir = os.path.join(work_dir, sub)
        trial_data = pd.read_csv(
            os.path.join(subject_dir, "{}_trials.csv".format(sub))
        )

        # Process the in-category trials
        mask_cat = trial_data.trial_type.str.contains(
            target_category, case=False
        )
        cat = trial_data[mask_cat].names
        runs_cat = trial_data[mask_cat].runs
        func_files_cat = [
            os.path.join(
                subject_dir,
                "{}_{}_run-{}_z-map.nii.gz".format(sub, trial, run),
            )
            for trial, run in zip(cat, runs_cat)
        ]
        # run_cat_labels = runs_cat.values.tolist()

        # Process the not-in-category trials
        mask_nocat = trial_data.trial_type.str.contains(
            other_category, case=False
        )
        nocat = trial_data[mask_nocat].names
        runs_nocat = trial_data[mask_nocat].runs
        func_files_nocat = [
            os.path.join(
                subject_dir,
                "{}_{}_run-{}_z-map.nii.gz".format(sub, trial, run),
            )
            for trial, run in zip(nocat, runs_nocat)
        ]
        # run_nocat_labels = runs_nocat.values.tolist()

        time_series_cat.append(masker.transform(func_files_cat))
        time_series_nocat.append(masker.transform(func_files_nocat))
        group.append(sub)

    print(time_series_cat[0].shape, time_series_nocat[0].shape)

    if WITHIN_SUB:
        ind_res = {}
        for i, sub in enumerate(subjects):
            sub_acc, sub_std, sub_sem = within_subject_classification(
                sub, time_series_cat[i], time_series_nocat[i]
            )
            ind_res[sub] = {
                "mean_accuracy": sub_acc,
                "std_accuracy": sub_std,
                "sem_accuracy": sub_sem,
            }
        ind_res_df = pd.DataFrame.from_dict(ind_res, orient="index")
        # Formating help to avoid a super long string in the filename
        c_name = comp_name if comp_name else "no_mask"
        cat_ids = target_category + "_" + other_category
        top_r = len(masker.labels_) - 1
        ind_res_path = os.path.join(
            work_dir,
            f"{c_name}_sch{n_regs}_top{top_r}_{cat_ids}_within_sub_res.csv",
        )
        update_results_collection(ind_res_path, ind_res_df)

    # Collecting all time series together for classification
    pooled_data = time_series_cat + time_series_nocat

    # Create labels and groups for classification
    labels = ["cat"] * len(time_series_cat) + ["nocat"] * len(
        time_series_nocat
    )
    groups = np.array(group * 2).tolist()

    # Trial classification
    scores = []
    _, classes = np.unique(labels, return_inverse=True)
    cv = LeaveOneGroupOut()

    # Convert splits to a list (debugging purposes)
    splits = list(cv.split(pooled_data, classes, groups=groups))

    for train_idx, test_idx in splits:
        # Need to expand the indices to account for trials within
        # each subject, especially for constructing y
        # Probably should redesign this part
        X_train, y_train, groups_train = [], [], []
        for k in train_idx:
            data = pooled_data[k]
            label = classes[k]
            group = groups[k]

            n_trials = data.shape[0]

            X_train.append(data)
            y_train.extend([label] * n_trials)
            groups_train.extend([group] * n_trials)

        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        groups_train = np.array(groups_train)

        X_test, y_test, groups_test = [], [], []
        for h in test_idx:
            data = pooled_data[h]
            label = classes[h]
            group = groups[h]

            n_trials = data.shape[0]

            X_test.append(data)
            y_test.extend([label] * n_trials)
            groups_test.extend([group] * n_trials)

        X_test = np.vstack(X_test)
        y_test = np.array(y_test)
        groups_test = np.array(groups_test)

        # Classifier object, fit and predict
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearSVC(dual="auto"),
        )

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predictions))

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    sem_accuracy = std_accuracy / np.sqrt(len(scores))

    print(f"{comp_name if comp_name else 'no_mask'} accuracy: {mean_accuracy}")

    return mean_accuracy, std_accuracy, sem_accuracy


def process_component(
    target1, target2, comps_dir, noi_name, work_dir, n_regs=1000, top_k=100
):
    """
    Helper to process a single component for classification.
    Parameters
    ----------
    comps_dir : str
        Directory containing the component images.
    noi_name : str
        Name of the component of interest (e.g., "face_perception").
    n_regs : int, optional
        Number of regions in the Schaefer atlas. Default is 1000.
    top_k : int, optional
        Number of top regions to consider for classification. Default is 100.
    Returns
    -------
    dict
        A dictionary containing the classification results for the component
        with keys: 'n_rois', 'top_k', 'component', 'mean_accuracy',
        'std_accuracy', 'sem_accuracy'.
    """
    noi = os.path.join(comps_dir, f"mean_{noi_name}.nii.gz")
    masker, _ = sort_regions_by_contribution(
        component=noi, n_rois=n_regs, top_k=top_k
    )

    mean_acc, std_acc, sem_acc = run_classification(
        SUBJECTS,
        work_dir,
        masker,
        target_category=target1,
        other_category=target2,
        n_regs=n_regs,
        comp_name=noi_name,
    )
    return {
        "n_rois": n_regs,
        "top_k": top_k,
        "component": noi_name,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "sem_accuracy": sem_acc,
    }


def main(target1, target2, n_regs, k_regs, comps_dir, work_dir, comp_oi):
    for top_k in k_regs:
        print(f"\n====== Running classifications for top_k={top_k} ======\n")
        scores_list = []

        if MODE == "components":
            scores_list = Parallel(n_jobs=10)(
                delayed(process_component)(
                    target1,
                    target2,
                    comps_dir,
                    noi_name,
                    work_dir,
                    n_regs,
                    top_k,
                )
                for noi_name in comp_oi
            )

        elif MODE == "union":
            special_comps = comp_oi[:2]  # only take the first two components
            union_masker, _ = get_union_top_regions(
                component_1=os.path.join(
                    comps_dir, f"mean_{special_comps[0]}.nii.gz"
                ),
                component_2=os.path.join(
                    comps_dir, f"mean_{special_comps[1]}.nii.gz"
                ),
                n_rois=n_regs,
                top_k=top_k,
            )
            mean_acc, std_acc, sem_acc = run_classification(
                SUBJECTS,
                work_dir,
                union_masker,
                target_category=target1,
                other_category=target2,
                n_regs=n_regs,
                comp_name="union",
            )
            scores_list.append(
                {
                    "n_rois": n_regs,
                    "top_k": top_k,
                    "component": "union",
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "sem_accuracy": sem_acc,
                }
            )

            # Add the rest of the components separately as well for comparison
            scores_list += Parallel(n_jobs=10)(
                delayed(process_component)(
                    target1,
                    target2,
                    comps_dir,
                    noi_name,
                    work_dir,
                    n_regs,
                    top_k,
                )
                for noi_name in comp_oi[2:]
            )

        # Run classification with the whole atlas (no mask)
        # Only do it once
        if top_k == k_regs[0]:
            print("--> Classification with whole atlas (no mask)")
            schaefer = datasets.fetch_atlas_schaefer_2018(
                n_rois=n_regs, resolution_mm=1
            )
            masker = NiftiLabelsMasker(schaefer.maps, standardize=False).fit()
            mean_acc, std_acc, sem_acc = run_classification(
                SUBJECTS,
                work_dir,
                masker,
                target_category=target1,
                other_category=target2,
                n_regs=n_regs,
                comp_name=None,
            )
            scores_list.append(
                {
                    "n_rois": n_regs,
                    "top_k": n_regs,
                    "component": "no_mask",
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "sem_accuracy": sem_acc,
                }
            )

        scores_df = pd.DataFrame(scores_list)

        plt.figure(figsize=(6, 4))
        plt.barh(
            scores_df["component"],
            scores_df["mean_accuracy"],
            xerr=scores_df["sem_accuracy"],  # or std_accuracy
            capsize=5,
            color="skyblue",
        )
        plt.xlabel("Classification accuracy")
        plt.axvline(0.5, linestyle="--", color="gray", label="Chance")
        plt.title(
            f"{target1} vs {target2} Trial Clf with Coeffs ({top_k}/{n_regs} regions)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                work_dir,
                f"union_sch{n_regs}_top{top_k}_{target1}_{target2}_beta_clf.png",
            )
        )

        # Save the results
        results_file = os.path.join(
            work_dir, f"union_betacoefs_{target1}_{target2}_clf.csv"
        )
        update_results_collection(results_file, scores_df)


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(proj_dir, "facebody")
    components_dir = "cognitive_components_subout/components_mean"
    comps_dir = os.path.join(proj_dir, components_dir)

    target1 = "Faces"
    target2 = "Characters"
    # targets_2 = ["Bodies", "Places", "Objects", "Characters", "Faces"]
    n_regs = 400
    k_regs = [5, 10, 20, 30, 40]  # 400
    # k_regs = [10, 20, 40, 60, 80, 100]  # 1000

    comp_oi = [
        "face_perception",
        "visual_letter_recognition",
        "facial_age_recognition",
        "visual_working_memory",
        "arithmetic_processing",
    ]

    main(target1, target2, n_regs, k_regs, comps_dir, work_dir, comp_oi)
