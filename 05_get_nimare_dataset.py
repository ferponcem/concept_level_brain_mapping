"""
Get dataset from Neurosynth via NiMARE.
"""

import os

import numpy as np
from nimare.dataset import Dataset
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset


def get_neurosynth_dataset(out_dir="nimare_data"):
    """Get Neurosynth dataset."""
    # Download and extract the dataset
    files = fetch_neurosynth(
        data_dir=out_dir,
        vocab="terms",
    )
    print(files)
    neurosynth_db = files[0]

    # Convert to NiMARE dataset
    neurosynth_dset = convert_neurosynth_to_dataset(
        coordinates_file=neurosynth_db["coordinates"],
        metadata_file=neurosynth_db["metadata"],
        annotations_files=neurosynth_db["features"],
    )

    return neurosynth_dset


def main(dset_file, out_dir="nimare_data"):
    """Main function to get the Neurosynth dataset."""

    if os.path.exists(dset_file):
        dset = Dataset.load(dset_file)
    else:
        dset = get_neurosynth_dataset(out_dir=out_dir)
        dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))


if __name__ == "__main__":
    out_dir = "nimare_data"
    os.makedirs(out_dir, exist_ok=True)
    dset_file = os.path.join(out_dir, "neurosynth_dataset.pkl.gz")
    main(dset_file, out_dir=out_dir)
