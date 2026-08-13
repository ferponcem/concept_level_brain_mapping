"""
Average/run conjunction component maps for all subjects
"""

import os
from glob import glob

from nilearn.image import mean_img
from nilearn.maskers import NiftiMasker

from utils_components import conjunction_inference_from_z_images, get_mask

SUBJECTS = ["sub-%02d" % i for i in [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]


def get_components_list(work_dir):
    """Get the list of components in a directory"""
    comp_list = glob(os.path.join(work_dir, "cc_*.nii.gz"))
    comp_list = [comp for comp in comp_list if "r2" not in comp]
    return comp_list


def subject_components(sub_dir, component):
    """Find the component for every subject directory"""
    comp_sub = os.path.join(sub_dir, f"cc_{component}.nii.gz")
    if os.path.exists(comp_sub):
        return comp_sub
    else:
        print(f"Component {component} not found in {sub_dir}")
        return None


def mean_component(imgs, dir_out):
    """Compute the mean of a list of images and save it"""
    comp = imgs[0].split("cc_")[1].split(".")[0]
    mean_res = mean_img(imgs)
    mean_res.to_filename(os.path.join(dir_out, f"mean_{comp}.nii.gz"))
    return mean_res


def conjunction_component(imgs, dir_out, masker):
    """Compute the conjunction of a list of images and save it"""
    comp = imgs[0].split("cc_")[1].split(".")[0]
    conj_res = conjunction_inference_from_z_images(imgs, masker=masker, u=0.5)
    conj_res.to_filename(os.path.join(dir_out, f"conjunction_{comp}.nii.gz"))
    return conj_res


def main(proj_dir, components_dir, ref_sub, analysis_name, out_dir_name):
    """
    Main function to compute mean or conjunction of components across subjects
    Parameters
    ----------
    proj_dir : str
        Path to the project directory
    components_dir : str
        Path to the components directory
    ref_sub : str
        Reference subject to get the list of components
    analysis_name : str
        Type of analysis to perform: "mean" or "conjunction"
    out_dir_name : str
        Name of the output directory to save results
    """
    out_dir = os.path.join(proj_dir, components_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(proj_dir, components_dir, ref_sub)
    comp_list = get_components_list(work_dir)

    for comp in comp_list:
        component = comp.split("cc_")[1].split(".")[0]
        imgs = []
        for sub in SUBJECTS:
            sub_dir = os.path.join(proj_dir, components_dir, sub)
            comp_sub = subject_components(sub_dir, component)
            imgs.append(comp_sub)
        imgs = [img for img in imgs if img is not None]

        if analysis_name == "conjunction":
            mask = get_mask()
            masker = NiftiMasker(mask_img=mask).fit()
            conj_comp_img = conjunction_component(imgs, out_dir, masker)
            print(f"✅ Component {component} conjunction done")
        else:  # mean
            mean_comp_img = mean_component(imgs, out_dir)
            print(f"✅ Component {component} averaged")


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    components_dir = "cognitive_components"
    ref_sub = "sub-04"
    analysis_name = "conjunction"  # or "mean"
    out_dir_name = "components_conj"
    main(project_dir, components_dir, ref_sub, analysis_name, out_dir_name)
