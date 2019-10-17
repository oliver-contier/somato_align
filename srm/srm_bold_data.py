#!/usr/bin/env python3

"""
Script to run brainiak's shared response model on our data.

We'll start with only the first run. Later, we might want to think about concatinating runs in a sensible fashion.

TODO: Generate and apply brain masks with BET.
"""

import glob
import os
import pickle
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.srm import SRM
from brainiak.funcalign.rsrm import RSRM
from brainiak.io import load_images
from scipy import stats


def grab_bold_data(cond_id='D1_D5',
                   ds_dir='/data/BnB_USER/oliver/somato',
                   testsubs=False,
                   use_filtered_from_workdir='/home/homeGlobal/oli/somato/scratch/ica/MELODIC/melodic_wf_workdir/'
                                             'subject_lvl/somato_melodic_wf/'):
    """
    Get file paths for our bold data for a given run.
    If you want to use raw bold-data, set use_filtered_from_workdir to False.
    Else, give working directory of melodic workflow.
    """
    print('grabbing data')
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_dir + '/*')]
    if testsubs:
        sub_ids = sub_ids[:testsubs]
    if use_filtered_from_workdir:
        boldfiles = [pjoin(use_filtered_from_workdir, 'bpf', 'mapflow', '_bpf%i' % idx, 'data_brain_smooth_filt.nii.gz')
                     for idx in range(0, len(sub_ids)*2, 2)]
    else:
        boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
                     for sub_id in sub_ids]
    return boldfiles, sub_ids


def boldfiles_to_arrays(boldfiles,
                        z_score=True):
    """
    Load bold data into list of arrays (bc that's what brainiak wants)
    """
    print('loading boldfiles')
    # create generator that returns nibabel nifti instances
    nibs_gen = load_images(boldfiles)
    # get numpy array data from each those instances (might take a while)
    print('converting to np arrays')
    bold_arrays_list = [np.reshape(nib_instance.get_fdata(), (1327104, 256))  # TODO: don't hard code this
                        for nib_instance in nibs_gen]
    print('z-scoring')
    if z_score:
        zscored = []
        for bold_array in bold_arrays_list:
            zs = stats.zscore(bold_array, axis=1, ddof=1)
            zs = np.nan_to_num(zs)
            zscored.append(zs)
            # TODO: put z-scoring in preprocessing pipeline later
        bold_arrays_list = zscored

    return bold_arrays_list


def train_srm(training_data,
              use_robust_srm=True,
              pickle_outdir='/home/homeGlobal/oli/somato/scratch/srm',
              n_comps=50,
              n_iters=20):
    # set up
    print('setting up srm')
    if use_robust_srm:
        srm = RSRM(n_iter=n_iters, features=n_comps)
        outpickle = pjoin(pickle_outdir, 'rsrm.p')
    else:
        srm = SRM(n_iter=n_iters, features=n_comps)
        outpickle = pjoin(pickle_outdir, 'srm.p')
    # fit
    print('fitting srm')
    srm.fit(training_data)
    # save srm as pickle
    print('saving srm as pickle')
    if pickle_outdir:
        if not os.path.exists(pickle_outdir):
            os.makedirs(pickle_outdir)
        with open(outpickle, 'wb') as f:
            pickle.dump(srm, f)
    return srm


if __name__ == '__main__':
    bold_files, subids = grab_bold_data()  # testsubs=3)
    bold_arrays = boldfiles_to_arrays(bold_files)
    srm_ = train_srm(bold_arrays, use_robust_srm=True,
                     pickle_outdir='/home/homeGlobal/oli/somato/scratch/srm_filtered')  # , n_comps=5, n_iters=5)
    print('done')
