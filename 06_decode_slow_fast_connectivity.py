"""
Script to classify slow vs fast trials using connectivity measures derived from
fMRI data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ibc_public.utils_data import SUBJECTS
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import new_img_like, resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import find_parcellation_cut_coords, plot_connectome
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC


def update_results_collection(results_file, new_results):
    """Update the results collection with new results.

    Parameters
    ----------
    results_collection : str
        Path to the existing results collection CSV file.
    new_results : DataFrame
        New results to append to the collection.
    """
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        updated_results = pd.concat(
            [existing_results, new_results], ignore_index=True
        )
    else:
        updated_results = new_results

    updated_results.to_csv(results_file, index=False)


def sort_regions_by_contribution(component, n_rois=1000, res_mm=1, top_k=100):
    """Get the regions in an atlas sorted by their values in a given
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


def run_classification(
    subjects, work_dir, masker, comp_name=None, dur_thr=0.8
):
    """
    Run classification between slow and fast trials using connectivity measures.
    1) Extract time series from the atlas for each subject.
    2) Classify slow vs fast trials using connectivity measures.

    Parameters
    ----------
    subjects: list of str
        list of subject IDs
    work_dir: str
        working directory containing subject data
    masker: Niimg-like
        NiftiLabelsMasker object for extracting time series
    comp_name: str, optional
        name of the component (for reporting)
    dur_thr: float, optional
        duration threshold to separate slow and fast trials, default is 0.8s

    Returns
    -------
    mean_accuracy: float
        mean classification accuracy across cross-validation folds
    std_accuracy: float
        standard deviation of classification accuracy across folds
    sem_accuracy: float
        standard error of the mean for classification accuracy across folds
    """
    time_series_slow = []
    time_series_fast = []
    group = []

    # Extract time series for each subject
    for sub in subjects:
        sub_dir = os.path.join(work_dir, sub)
        trial_data = pd.read_csv(
            os.path.join(sub_dir, "{}_trials.csv".format(sub))
        )

        # Process slow trials
        mask_slow = trial_data.duration > dur_thr
        slow = trial_data[mask_slow].names
        runs_slow = trial_data[mask_slow].runs
        func_files_slow = [
            os.path.join(
                sub_dir, "{}_{}_run-{}_z-map.nii.gz".format(sub, trial, run)
            )
            for trial, run in zip(slow, runs_slow)
        ]

        # Process fast trials
        mask_fast = trial_data.duration < dur_thr
        fast = trial_data[mask_fast].names
        runs_fast = trial_data[mask_fast].runs
        func_files_fast = [
            os.path.join(
                sub_dir, "{}_{}_run-{}_z-map.nii.gz".format(sub, trial, run)
            )
            for trial, run in zip(fast, runs_fast)
        ]

        time_series_slow.append(masker.transform(func_files_slow))
        time_series_fast.append(masker.transform(func_files_fast))
        group.append(sub)

    print(time_series_slow[0].shape, time_series_fast[0].shape)

    # Collecting data for classification
    pooled_data = time_series_slow + time_series_fast

    # Initialize the connectome
    conn_vis = ConnectivityMeasure(kind="tangent", vectorize=False)
    conn_vis.fit(pooled_data)

    # Visualize the mean connectome
    coords = find_parcellation_cut_coords(masker.labels_img_)
    mean_conn = conn_vis.mean_

    # Save the mean connectome for later visualization
    pd.DataFrame(mean_conn).to_csv(
        os.path.join(
            work_dir,
            f"mean_connectome_{comp_name if comp_name else 'no_mask'}.csv",
        ),
        index=False,
    )

    plot_connectome(
        mean_conn,
        coords,
        title=f"Mean Connectome - {comp_name if comp_name else 'no_mask'}",
        output_file=os.path.join(
            work_dir,
            f"mean_connectome_{comp_name if comp_name else 'no_mask'}.png",
        ),
    )

    # Create labels and groups for classification
    labels = ["slow"] * len(time_series_slow) + ["fast"] * len(
        time_series_fast
    )
    groups = np.array(group * 2).tolist()

    # Connectome classification
    scores = []
    kind = "tangent"
    _, classes = np.unique(labels, return_inverse=True)
    cv = LeaveOneGroupOut()

    # Convert splits to a list (debugging purposes)
    splits = list(cv.split(pooled_data, classes, groups=groups))

    for train_index, test_index in splits:
        connectome = ConnectivityMeasure(
            kind=kind, vectorize=True, standardize="zscore_sample"
        )

        # Fit and transform training data
        train_data = [pooled_data[k] for k in train_index]
        X_train = connectome.fit_transform(train_data)
        y_train = classes[train_index]

        # transform the test data
        test_data = [pooled_data[h] for h in test_index]
        X_test = connectome.transform(test_data)
        y_test = classes[test_index]

        # fit the classifier
        clf = LinearSVC().fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # compute the accuracy
        scores.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    sem_accuracy = std_accuracy / np.sqrt(len(scores))

    print(f"{comp_name if comp_name else 'no_mask'} accuracy: {mean_accuracy}")

    return mean_accuracy, std_accuracy, sem_accuracy


def main(
    k_regs, n_regs, comp_oi, duration_threshold, subjects, comps_dir, work_dir
):
    """
    Main function to run classifications for different components and top_k regions.
    Parameters
    ----------
    k_regs : list of int
        List of top_k values to test.
    n_regs : int
        Number of regions in the Schaefer atlas.
    comp_oi : list of str
        List of component names to classify.
    duration_threshold : float
        Duration threshold to separate slow and fast trials.
    subjects : list of str
        List of subject IDs.
    comps_dir : str
        Directory containing component images.
    work_dir : str
        Working directory containing subject data and where results will be saved.
    """
    # List to store accuracy results
    for top_k in k_regs:
        print(f"\n====== Running classifications for top_k={top_k} ======\n")

        noi_acc = []

        # Run classifications in parallel
        for noi_name in comp_oi:
            print(f"--> Classification with {noi_name}")
            noi = os.path.join(comps_dir, f"mean_{noi_name}.nii.gz")
            masker, _ = sort_regions_by_contribution(
                component=noi, n_rois=n_regs, res_mm=1, top_k=top_k
            )
            print(
                f"Number of regions used for {noi_name}: {masker.n_elements_}"
            )

            mean_acc, std_acc, sem_acc = run_classification(
                subjects,
                work_dir,
                masker,
                comp_name=noi_name,
                dur_thr=duration_threshold,
            )

            noi_acc.append(
                {
                    "n_rois": n_regs,
                    "top_k": top_k,
                    "component": noi_name,
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "sem_accuracy": sem_acc,
                }
            )

        if top_k == k_regs[0]:  # Only do this once
            # Run classification with the whole atlas (no mask)
            print("--> Classification with whole atlas (no mask)")
            schaefer = datasets.fetch_atlas_schaefer_2018(
                n_rois=n_regs, resolution_mm=1
            )
            masker = NiftiLabelsMasker(schaefer.maps, standardize=False).fit()
            mean_acc, std_acc, sem_acc = run_classification(
                subjects,
                work_dir,
                masker,
                comp_name=None,
                dur_thr=duration_threshold,
            )
            noi_acc.append(
                {
                    "n_rois": n_regs,
                    "top_k": masker.n_elements_,
                    "component": "no_mask",
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "sem_accuracy": sem_acc,
                }
            )

        # Collect results
        noi_acc = pd.DataFrame(noi_acc)

        # Plotting the results
        plt.figure(figsize=(6, 4))
        plt.barh(
            noi_acc["component"],
            noi_acc["mean_accuracy"],
            xerr=noi_acc["sem_accuracy"],  # or std_accuracy
            capsize=5,
            color="skyblue",
        )
        plt.xlabel("Classification accuracy")
        plt.axvline(0.5, linestyle="--", color="gray", label="Chance")
        plt.title(f"Slow vs Fast Trial Clf with Conn ({top_k}/{n_regs} regs)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                work_dir, f"up_sch{n_regs}_top{top_k}_slow_fast_conn_clf.png"
            )
        )

        # Save the results to a csv to later plot all together
        results_file = os.path.join(
            work_dir, f"updated_connectome_slow_fast_clf.csv"
        )
        update_results_collection(results_file, noi_acc)


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(proj_dir, "mcse")
    components_dir = "cognitive_components_subout/components_mean"
    comps_dir = os.path.join(proj_dir, components_dir)
    subjects = [sub for sub in SUBJECTS if sub != "sub-02"]

    # which atlas to use
    comp_oi = [
        "visual_search",
        "visual_working_memory",
        "arithmetic_processing",
    ]

    # Threshold for slow/fast trials
    duration_threshold = 0.8
    n_regs = 400
    k_regs = [5, 10, 20, 30, 40]  # 400
    # k_regs = [10, 20, 40, 60, 80, 100]  # 1000

    main(
        k_regs=k_regs,
        n_regs=n_regs,
        comp_oi=comp_oi,
        duration_threshold=duration_threshold,
        subjects=subjects,
        comps_dir=comps_dir,
        work_dir=work_dir,
    )
