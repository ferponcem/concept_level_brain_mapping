"""Plot distributions of components and contrasts in a design matrix.

This script generates two histograms from a contrast-by-component design
matrix:

1. Number of active components per contrast.
2. Number of active contrasts per component.

The resulting figure can be displayed interactively and optionally saved
to disk.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histogram_concepts_per_contrasts(dmat, ax=None):
    """Plot the number of active components for each contrast.

    A component is considered active for a contrast when its design-matrix
    value is non-zero.

    Parameters
    ----------
    dmat : pandas.DataFrame
        Contrast-by-component design matrix. Rows correspond to contrasts
        and columns correspond to components.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the histogram. If ``None``, a new figure and
        axes are created.
    """

    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False

    for_hist = {}
    for row in dmat.iterrows():
        n_active_components = (row[1] != 0).sum()
        for_hist[row[0]] = int(n_active_components)
    ax.hist(
        list(for_hist.values()),
        bins=np.arange(0.5, 7.5, 1),
        rwidth=0.9,
        color="skyblue",
    )
    ax.set_xticks(np.arange(1, 8, 1))
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("Number of components", fontsize=13)
    ax.set_ylabel("Number of contrasts", fontsize=13)
    ax.set_title("Active components per contrast", fontsize=14)

    if standalone:
        plt.show()


def histogram_contrasts_per_concepts(dmat, ax=None):
    """Plot the number of active contrasts for each component.

    A contrast is considered active for a component when its design-matrix
    value is non-zero.

    Parameters
    ----------
    dmat : pandas.DataFrame
        Contrast-by-component design matrix. Rows correspond to contrasts
        and columns correspond to components.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the histogram. If ``None``, a new figure and
        axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False
    for_hist = {}
    for col in dmat.columns:
        n_active_contrasts = (dmat[col] != 0).sum()
        for_hist[col] = int(n_active_contrasts)

    ax.hist(
        list(for_hist.values()),
        bins=np.arange(0.5, 20.5, 1),
        rwidth=0.9,
        color="lightskyblue",
    )
    ax.set_xticks(np.arange(1, 21, 1))
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("Number of contrasts", fontsize=13)
    ax.set_ylabel("Number of components", fontsize=13)
    ax.set_title("Active contrasts per component", fontsize=14)

    if standalone:
        plt.show()


def main(proj_dir, dmat_dir, dmat_file, figs_dir, plot_together=True):
    """Main function to generate histograms from the design matrix.

    Parameters
    ----------
    proj_dir : str
        Path to the project directory.
    cc_dir : str
        Path to the cognitive components directory (after design matrix
        generation).
    dmat_file : str
        Path to the design matrix file (TSV format).
    figs_dir : str
        Path to the directory where figures will be saved.
    """
    dmat_dir = os.path.join(proj_dir, dmat_dir)
    dmat_file = os.path.join(dmat_dir, dmat_file)
    figs_dir = os.path.join(proj_dir, "figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    dmat = pd.read_csv(dmat_file, sep="\t", index_col=0)
    # Plot separately
    histogram_concepts_per_contrasts(dmat)
    histogram_contrasts_per_concepts(dmat)

    # Or together
    if plot_together:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        histogram_concepts_per_contrasts(dmat, ax=axes[0])
        histogram_contrasts_per_concepts(dmat, ax=axes[1])
        plt.tight_layout()
        plt.savefig(
            os.path.join(figs_dir, "histograms_concepts_contrasts.png"),
            dpi=300,
            transparent=True,
        )
        plt.show()
    else:  # Plot separately
        histogram_concepts_per_contrasts(dmat)
        histogram_contrasts_per_concepts(dmat)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dmat_dir = "cognitive_components"
    dmat_file = "design_matrix.tsv"
    figs_dir = "figs"
    main(project_dir, dmat_dir, dmat_file, figs_dir)
