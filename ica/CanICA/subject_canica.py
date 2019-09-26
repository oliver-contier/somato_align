#!/usr/bin/env python3

#SBATCH --job-name=subject_canica
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

"""
Run Nilearn's CanICA on subject/condition level and parallelize through slurm for all data.

Online-Example on interfacing slurm in python script:
https://rcc.uchicago.edu/docs/tutorials/kicp-tutorials/running-jobs.html

# since each cycle lasted 25.6 seconds, that might be a good cutoff for high-pass filtering
# 1 rep / 25.6 sec = 0.0390625 Hz
# two cycles = 0.01953125 Hz
"""

import os
import glob
from os.path import join as pjoin

import numpy as np
from nilearn.decomposition import CanICA
from nilearn.image import load_img, mean_img
from nilearn.plotting import plot_prob_atlas


def run_canica_subject(sub_id,
                       cond_id='D1_D5',
                       ds_dir='/data/BnB_USER/oliver/somato',
                       out_basedir='/home/homeGlobal/oli/somato/scratch/ica/CanICA',
                       ncomps=50, smoothing=3, caa=True, standard=True, detr=True, highpass=.01953125, tr=2.,
                       masktype='epi', ninit=10, seed=42, verb=10):
    """
    Run Nilearn's CanICA on a single condition of a single subject.
    """
    # load example image
    bold_file = pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
    bold_img = load_img(bold_file)
    # paths to output
    out_dir = pjoin(out_basedir, sub_id, cond_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_comp_nii = pjoin(out_dir, 'components.nii.gz')
    out_components_arr = pjoin(out_dir, 'components.npy')
    out_png = pjoin(out_dir, 'components_probatlas.png')

    # set up ica
    ica = CanICA(n_components=ncomps, smoothing_fwhm=smoothing, do_cca=caa, standardize=standard, detrend=detr,
                 mask_strategy=masktype, high_pass=highpass, t_r=tr,
                 n_init=ninit, random_state=seed, verbose=verb)
    # more interesting arguments
    # mask_strategy='mni_something, mask_args=see nilearn.masking.compute_epi_mask, threshold=3.

    # fit ica
    ica.fit(bold_img)

    # save components as 4d nifti
    components_img = ica.components_img_
    components_img.to_filename(out_comp_nii)
    # plot components as prob atlas and save plot
    g = plot_prob_atlas(components_img, bg_img=mean_img(bold_img))
    g.savefig(out_png, dpi=300)
    # save components as 2d np array
    components_arr = ica.components_
    np.save(out_components_arr, components_arr)
    # save automatically generated epi mask
    if masktype == 'epi':
        mask_img = ica.mask_img_
        out_mask_img = pjoin(out_dir, 'mask_img.nii.gz')
        mask_img.to_filename(out_mask_img)

    return ica  # return ica object for later use


def parallelize_canica(i, datadir='/data/BnB_USER/oliver/somato'):
    """
    Parallelize the function run_canica_subject with slurm (so that only this script has to be executed).
    perform work associated with step i.
    """

    # list of all subjects and conditions twice (should be 24)
    subject_ids = [os.path.basename(absp) for absp in glob.glob(pjoin(datadir, '*'))] * 2
    if i >= 12:
        cond = 'D5_D1'
    else:
        cond = 'D1_D5'
    run_canica_subject(sub_id=subject_ids[i], cond_id=cond)


if __name__ == '__main__':
    import multiprocessing
    import sys

    # necessary to add cwd to path when script run by slurm (since it executes a copy)
    sys.path.append(os.getcwd())

    # get number of cpus available to job
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    # create pool of ncpus workers
    pool = multiprocessing.Pool(ncpus)

    # apply my ica function in parallel
    pool.map(parallelize_canica, range(24))
