"""
Load subject weighted matrices and plot them as a heatmap and the loss
progression.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def plot_matrix(matrix, title):
    """
    Plot a heatmap of the given matrix.
    """
    sns.heatmap(matrix, annot=False, cmap="icefire", center=0)
    plt.title(title)
    plt.show()


def plot_loss(loss, title):
    """
    Plot the progression of the loss function across iterations.
    """
    plt.plot(loss)
    plt.title(title)
    plt.show()


def plot_losses_together(losses_dict):
    """
    Plot all losses in a single plot
    """
    plt.figure(figsize=(10, 6))
    for sub, loss in losses_dict.items():
        plt.plot(loss, label=sub)
    plt.title("Loss Evolution Across Subjects")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main(res_dir):
    """
    Main function to load and plot the matrices and losses for each subject.
    Parameters
    ----------
    res_dir : str
        Directory containing the results for each subject.
    """

    losses_per_subject = {}

    for sub in SUBJECTS:
        sub_dir = os.path.join(res_dir, sub)
        # 1. Plot last iteration of A_stack: new design matrix
        A_stack = np.load(os.path.join(sub_dir, "A_stack_norm.npy"))
        plot_matrix(A_stack, f"A_stack_{sub}")

        # 2. plot evolution of objective function across iterations
        loss = np.load(os.path.join(sub_dir, "losses.npy"))
        plot_loss(loss, f"Loss {sub}")
        # Collect losses for plotting together
        losses_per_subject[sub] = loss

        # 3. Plot the A_stack matrix
        plot_matrix(A_stack, f"A_stack_{sub}")

    # 4. Plot all losses together
    plot_losses_together(losses_per_subject)


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(proj_dir, "weighting_components")
    main(res_dir)
