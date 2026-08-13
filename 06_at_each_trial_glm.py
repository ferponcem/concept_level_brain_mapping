"""
GLM analysis for MCSE data with reaction times
Steps:
1. Get the data: fmri, motion and event files
2. Define the design matrice/models
3. Fit and write outputs
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from ibc_public.utils_data import DERIVATIVES, THREE_MM, get_subject_session
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)
from nilearn.image import high_variance_confounds, load_img

from utils_components import get_mask


def load_bold_and_motion_files(subject, session, direction, derivatives, task):
    """Load BOLD and motion files for a given subject, session, and direction."""
    bold_path = glob(
        os.path.join(
            derivatives,
            subject,
            session,
            "func",
            f"wrdc{subject}_{session}_task-{task}_dir-{direction}_*_bold.nii.gz",
        )
    )
    if not bold_path:
        raise FileNotFoundError(
            f"No BOLD file found for subject {subject}, session {session}, "
            f"task {task}, direction {direction}."
        )
    motion_path = glob(
        os.path.join(
            DERIVATIVES,
            subject,
            session,
            "func",
            f"rp_dc{subject}_{session}_task-{task}_dir-{direction}_*_bold.txt",
        )
    )
    return bold_path, motion_path


def flatten_list(nested_list):
    return [
        item
        for sublist in nested_list
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]


def extract_run_info(filepath):
    """Extract run information from the filename."""
    filename = os.path.basename(filepath)
    run_match = (
        filename.split("_run-")[1].split("_")[0] if "_run-" in filename else ""
    )
    return int(run_match) if run_match.isdigit() else 0


def sort_by_run(files):
    """Sort files by run number extracted from the filename."""
    sorted_files = sorted(
        [(f, extract_run_info(f)) for f in files], key=lambda x: x[1]
    )
    return [item[0] for item in sorted_files]


def load_events(task_dir, subject, task_id):
    """Load event files for a given subject and task."""
    wc = f"{task_dir}/event_files/{subject}_task-{task_id}_run*_events.tsv"
    return sorted(glob(wc))


def create_design_matrix(frame_times, event_df, confounds):
    """Create a design matrix for a given set of events and confounds."""
    dmtx = make_first_level_design_matrix(
        frame_times,
        events=event_df,
        hrf_model="spm",  # + derivative
        add_regs=confounds,
        high_pass=1.0 / 128,
    )
    return dmtx


def main(
    current_session,
    current_task,
    task_id,
    TR,
    gm_mask,
    derivatives,
    task_dir,
    write_dir,
):
    """
    Main function to process each subject and run trial-wise GLM analysis.
    Parameters
    ----------
    current_session : str
        The session identifier (e.g., 'scene').
    current_task : str
        The task identifier (e.g., 'FaceBody').
    task_id : str
        The task ID used in file naming (e.g., 'face-body').
    TR : float
        Repetition time of the fMRI scans.
    gm_mask : str
        Path to the gray matter mask image.
    derivatives : str
        Path to the derivatives directory containing preprocessed data.
    task_dir : str
        Directory containing event files for the task.
    write_dir : str
        Directory where output files will be saved.
    """
    subject_sessions = get_subject_session(current_session)
    for subject_session in subject_sessions:
        subject, session = subject_session
        print(
            f"---------------------- Processing {subject} --------------------"
        )

        # Load data: BOLD maps, motion parameters and event files
        bold = []
        motions = []
        for direction in ["pa", "ap"]:
            bold_path, motion_path = load_bold_and_motion_files(
                subject, session, direction, derivatives, current_task
            )
            bold.append(bold_path)
            motions.append(motion_path)
        # Flatten bold and motions lists
        bold = flatten_list(bold)
        motions = flatten_list(motions)
        # Get event files
        events = load_events(task_dir, subject, task_id)
        # Sort all lists by run number
        bold = sort_by_run(bold)
        motions = sort_by_run(motions)
        events = sort_by_run(events)
        subject_dir = os.path.join(write_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)

        q = 0
        run = 1  # Initialize run counter
        names = []
        trial_types = []
        durations = []
        runs = []

        # For every run:
        for event, motion, img in zip(events, motions, bold):
            n_scans = load_img(img).shape[3]
            confounds = high_variance_confounds(
                img, mask_img=gm_mask, n_confounds=5, percentile=5
            )
            motion_parameters = np.loadtxt(motion)
            confounds = np.hstack((confounds, motion_parameters))
            frame_times = TR * np.arange(0, n_scans)
            event_df = pd.read_csv(event, delimiter="\t")

            for trial in range(len(event_df)):
                df_ = event_df.copy()
                name = "trial_%03d" % trial
                # Rename the current trial to add it as a regressor
                df_.loc[trial, "trial_type"] = name
                df_.duration = 6.0  # for now use a standard duration
                dmtx = create_design_matrix(frame_times, df_, confounds)
                model = FirstLevelModel(mask_img=gm_mask, smoothing_fwhm=5)
                model.fit(img, design_matrices=dmtx)
                z_score = model.compute_contrast(name)
                name_ = name.replace("trial_", "trial-")
                run_ = "run-%d" % run
                z_score.to_filename(
                    os.path.join(
                        subject_dir,
                        "{}_{}_{}_z-map.nii.gz".format(subject, name_, run_),
                    )
                )
                q += 1  # trial count up
                names.append(name.replace("trial_", "trial-"))
                trial_types.append(event_df.trial_type[trial])
                durations.append(event_df.duration[trial])
                runs.append(run)

            run += 1

        df = pd.DataFrame(
            {
                "names": names,
                "trial_type": trial_types,
                "duration": durations,
                "runs": runs,
            }
        )
        output_csv = os.path.join(subject_dir, "{}_trials.csv".format(subject))
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    derivatives = THREE_MM
    gm_mask = get_mask(three_mm=True)
    TR = 2.0
    current_session = "scene"
    current_task = "FaceBody"
    task_id = "face-body"
    task_dir = "facebody"  # ideally unnecesary, same as task_id
    main(
        current_session,
        current_task,
        task_id,
        TR,
        gm_mask,
        derivatives,
        task_dir,
        project_dir,
    )
