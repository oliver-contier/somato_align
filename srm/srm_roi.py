#!/usr/bin/env python3

"""
# TODO: docstring

use nilearn.masking.unmask to retransform masked voxels into whole-brain space.

troubleshooting brainiak installation problems:
python3 -m pip install --no-use-pep517 brainiak
"""

import os
import pickle
import time
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.rsrm import RSRM
from brainiak.funcalign.srm import SRM
from nilearn.image import load_img
from nilearn.masking import apply_mask
from scipy import stats


def grab_subject_ids(ds_dir='/data/BnB_USER/oliver/somato/scratch/dataset',
                     testsubs=False,
                     exclude_subs=()):
    """
    Get list of all subject IDs.
    """
    import os
    import glob
    sub_ids = [os.path.basename(subdir) for subdir in glob.glob(ds_dir + '/*')]
    if testsubs:
        sub_ids = sub_ids[:testsubs]
    if exclude_subs:
        sub_ids = [subid for subid in sub_ids if subid not in exclude_subs]
    print('subject ids :', sub_ids)
    return sub_ids


def datagrabber(roi_glm_workdir='/data/project/somato/scratch/roi_glm/workdirs/',
                prepped_dsdir='/data/project/somato/scratch/dataset',
                testsubs=False,
                excludesubs=()):
    """
    # grab file names for
    # filtered bold data and roi masks from roi_glm output
    """
    sub_ids = grab_subject_ids(testsubs=testsubs, ds_dir=prepped_dsdir, exclude_subs=excludesubs)
    run1_data, run2_data, run3_data, run4_data, = [], [], [], []
    run1_masks, run2_masks, run3_masks, run4_masks = [], [], [], []
    for sub_id in sub_ids:
        sub_wf_dir = pjoin(roi_glm_workdir, 'subject_ffx_wfs', 'subject_%s_ffx_workdir' % sub_id,
                           'subject_%s_wf' % sub_id)
        # file names for filtered bold files
        for run_idx, rundata in enumerate([run1_data, run2_data, run3_data, run4_data]):
            fname = pjoin(sub_wf_dir, 'bpf', 'mapflow', '_bpf%i' % run_idx, 'data_brain_smooth_filt.nii.gz')
            if not os.path.exists(fname):
                raise IOError('filtered nifti does not exist : ', fname)
            rundata.append(fname)
        # file names for masks
        for run_idx, runmasks in enumerate([run1_masks, run2_masks, run3_masks, run4_masks]):
            fname = pjoin(sub_wf_dir, 'binarize_roi', 'mapflow',
                          '_binarize_roi%i' % run_idx, 'zfstat1_threshold_maths_maths.nii.gz')
            if not os.path.exists(fname):
                raise IOError('mask file does not exist : ', fname)
            runmasks.append(fname)
        # run1_data.append(pjoin(sub_wf_dir, 'bpf', 'mapflow', '_bpf0', 'data_brain_smooth_filt.nii.gz'))
        # run2_data.append(pjoin(sub_wf_dir, 'bpf', 'mapflow', '_bpf1', 'data_brain_smooth_filt.nii.gz'))
        # run3_data.append(pjoin(sub_wf_dir, 'bpf', 'mapflow', '_bpf2', 'data_brain_smooth_filt.nii.gz'))
        # run4_data.append(pjoin(sub_wf_dir, 'bpf', 'mapflow', '_bpf3', 'data_brain_smooth_filt.nii.gz'))
        # run1_masks.append(pjoin(sub_wf_dir, 'binarize_roi', 'mapflow',
        #                         '_binarize_roi0', 'zfstat1_threshold_maths_maths.nii.gz'))
        # run2_masks.append(pjoin(sub_wf_dir, 'binarize_roi', 'mapflow',
        #                         '_binarize_roi1', 'zfstat1_threshold_maths_maths.nii.gz'))
        # run3_masks.append(pjoin(sub_wf_dir, 'binarize_roi', 'mapflow',
        #                         '_binarize_roi2', 'zfstat1_threshold_maths_maths.nii.gz'))
        # run4_masks.append(pjoin(sub_wf_dir, 'binarize_roi', 'mapflow',
        #                         '_binarize_roi3', 'zfstat1_threshold_maths_maths.nii.gz'))
    return run1_data, run2_data, run3_data, run4_data, run1_masks, run2_masks, run3_masks, run4_masks


def load_data(run1_data, run2_data, run1_masks, run2_masks,
              whichrun=1,
              force_mask_run1=False,
              zscore=True,
              nan2num=True):
    """
    Load the masked data for a given run in array form
    to suit brainiak input.
    """
    if whichrun == 1:
        run_data, run_masks = run1_data, run1_masks
    elif whichrun == 2:
        run_data, run_masks = run2_data, run2_masks
    else:
        raise IOError('did not recognize argument %s for whichrun' % str(whichrun))
    if force_mask_run1:
        run_masks = run1_masks
    print('loading data')
    if zscore:
        run_arrs = [
            stats.zscore(  # load image, apply mask, z-score
                apply_mask(load_img(data), mask_img=mask).T,
                axis=1, ddof=1)
            for data, mask in zip(run_data, run_masks)
        ]
    else:
        run_arrs = [apply_mask(load_img(data), mask_img=mask).T
                    for data, mask in zip(run_data, run_masks)]
    if nan2num:
        run_arrs = [np.nan_to_num(bold_array) for bold_array in run_arrs]
    return run_arrs


def train_srm(training_data,
              use_robust_srm=True,
              n_comps=10,
              n_iters=4,
              printruntime=True):
    """
    Fit srm on training data
    """
    if use_robust_srm:
        srm = RSRM(n_iter=n_iters, features=n_comps)
    else:
        srm = SRM(n_iter=n_iters, features=n_comps)
    # fit
    if printruntime:
        start = time.time()
    srm.fit(training_data)
    if printruntime:
        elapsed = time.time() - start
        print('fitting srm took: ', elapsed)
    return srm


def save_srm_as_pickle(srm_instance,
                       robust_srm=True,
                       pickle_outdir='/home/homeGlobal/oli/somato/scratch/srm_roi'):
    """
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


if __name__ == '__main__':
    run1_data, run2_data, run1_masks, run2_masks = datagrabber(testsubs=3)
    run_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks)
    srm = train_srm(training_data=run_arrs)
    # outpickle = save_srm_as_pickle(srm_instance=srm)
