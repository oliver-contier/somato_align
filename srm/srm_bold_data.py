#!/usr/bin/env python3

"""
Script to run brainiak's shared response model on our data.

We'll start with only the first run. Later, we might want to think about concatinating runs in a sensible fashion.

TODO: Generate and apply brain masks with BET.
"""

import glob
import os
import pickle
import time
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.rsrm import RSRM
from brainiak.funcalign.srm import SRM
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
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_dir + '/*')]
    if testsubs:
        sub_ids = sub_ids[:testsubs]
    if use_filtered_from_workdir:
        boldfiles = [pjoin(use_filtered_from_workdir, 'bpf', 'mapflow', '_bpf%i' % idx, 'data_brain_smooth_filt.nii.gz')
                     for idx in range(0, len(sub_ids) * 2, 2)]
    else:
        boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
                     for sub_id in sub_ids]
    return boldfiles, sub_ids


def boldfiles_to_arrays(boldfiles,
                        z_score=True):
    """
    Load bold data into list of arrays (bc that's what brainiak wants)
    """
    # create generator that returns nibabel nifti instances
    nibs_gen = load_images(boldfiles)
    # get numpy array data from each those instances (might take a while)
    bold_arrays_list = [np.reshape(nib_instance.get_fdata(), (1327104, 256))  # TODO: don't hard code this
                        for nib_instance in nibs_gen]
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
              n_comps=50,
              n_iters=20,
              printruntime=True):
    """
    Fit srm on training data
    """

    # TODO: delete after test
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    if use_robust_srm:
        srm = RSRM(n_iter=n_iters, features=n_comps, comm=mpicomm)
    else:
        srm = SRM(n_iter=n_iters, features=n_comps, comm=mpicomm)
    # fit
    if printruntime:
        start = time.time()
    srm.fit(training_data)
    if printruntime:
        elapsed = start - time.time()
        print('fitting srm took: ', elapsed)
    return srm


def save_srm_as_pickle(srm_instance,
                       robust_srm=True,
                       pickle_outdir='/home/homeGlobal/oli/somato/scratch/srm'):
    """
    take fitted srm instance and save as pickle. If robust srm was used, file will be called "rsrm.p" instad of "srm.p"
    """
    if robust_srm:
        outpickle = pjoin(pickle_outdir, 'rsrm.p')
    else:
        outpickle = pjoin(pickle_outdir, 'srm.p')
    if not os.path.exists(pickle_outdir):
        os.makedirs(pickle_outdir)
    with open(outpickle, 'wb') as f:
        pickle.dump(srm_instance, f)
    return outpickle


def run_srm_pipeline(dsdir='/data/BnB_USER/oliver/somato',
                     use_filtered_wd='/home/homeGlobal/oli/somato/scratch/ica/MELODIC/melodic_wf_workdir/'
                         'subject_lvl/somato_melodic_wf/',
                     whichcond='D1_D5',
                     test_subs=False,
                     zscore=True,
                     robustsrm=True,
                     print_runtime=True,
                     ncomps=50,
                     niters=20,
                     pickleoutdir='/home/homeGlobal/oli/somato/scratch/srm'):
    """
    SRM Pipeline.
    """
    print('grabbing bold data')
    bold_files, subids = grab_bold_data(cond_id=whichcond, ds_dir=dsdir, testsubs=test_subs,
                                        use_filtered_from_workdir=use_filtered_wd)
    print('converting to np arrays')
    bold_arrays = boldfiles_to_arrays(bold_files, z_score=zscore)
    print('fitting srm')
    fitted_srm = train_srm(bold_arrays, use_robust_srm=robustsrm, n_comps=ncomps, n_iters=niters,
                           printruntime=print_runtime)
    print('saving to pickle')
    pickle_path = save_srm_as_pickle(fitted_srm, robust_srm=robustsrm, pickle_outdir=pickleoutdir)
    print('done!')
    return None


if __name__ == '__main__':
    run_srm_pipeline(test_subs=2, robustsrm=False, ncomps=3, niters=2,
                     pickleoutdir='/home/homeGlobal/oli/somato/scratch/srm_mpitest')
