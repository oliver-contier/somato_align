#!/usr/bin/env python3

import runpy
import time
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.rsrm import RSRM

# import io functions from my other srm script
file_globals = runpy.run_path('srm_roi.py')
datagrabber = file_globals['datagrabber']
load_data = file_globals['load_data']
grab_subject_ids = file_globals['grab_subject_ids']


def project_data_crossval(run1_arrs,
                          run2_arrs,
                          k=10,
                          niter=20,
                          outdir='/data/BnB_USER/oliver/somato/scratch/crossval_projection'):
    """
    Project subjects bold data onto shared response space in a cross-validated way. For every subject, an srm is
    trained on the remaining subjects. this srm is trained on the data from run one, and the data from the test
    subject's other run is than projected onto this template. Data is saved in numpy's npy.
    """
    # iterate over runs used for training / testing
    for trainrun_idx in range(2):
        # select run used for training and test
        training_arrs = (run1_arrs, run2_arrs)[trainrun_idx]
        test_arrs = (run1_arrs, run2_arrs)[abs(trainrun_idx - 1)]
        # prepare results arrays (for projected data and estimated srm time series)
        results_array = np.zeros(shape=(len(training_arrs), k, training_arrs[0].shape[1]))
        trained_srms_array = np.zeros(shape=(len(run1_arrs), k, training_arrs[0].shape[1]))
        # iterate over testsubjects
        for testsub_idx in range(len(training_arrs)):
            start = time.time()
            print('starting run %i subject %i' % (trainrun_idx, testsub_idx))
            trainsubs_traindata = [x for i, x in enumerate(training_arrs) if i != testsub_idx]  # select training data
            testsub_traindata = training_arrs[testsub_idx]
            srm = RSRM(n_iter=niter, features=k)  # train srm on training subject's training data
            srm.fit(trainsubs_traindata)
            w, s = srm.transform_subject(testsub_traindata)  # estimate test subject's bases
            # reattach weight matrix and individual term to srm instance
            # (to allow transforming test run with builtin brainiak function)
            srm.w_.insert(testsub_idx, w)
            srm.s_.insert(testsub_idx, s)
            projected_data, ind_terms = srm.transform(test_arrs)  # project test run into shared space
            testsub_proj = projected_data[testsub_idx]  # select projected data from test subject
            results_array[testsub_idx] = testsub_proj  # append result to our results arrays
            trained_srms_array[testsub_idx] = srm.r_
            elapsed = time.time() - start
            print('this round took: ', elapsed)
        # save results array for this run
        proj_outpath = pjoin(outdir, 'proj_run%i_is_train.npy' % (trainrun_idx + 1))
        with open(proj_outpath, 'wb') as outf:
            np.save(outf, results_array)
        trained_srms_outpath = pjoin(outdir, 'trainedsrms_run%i_is_train.npy' % (trainrun_idx + 1))
        with open(trained_srms_outpath, 'wb') as outf:
            np.save(outf, trained_srms_array)
    print('done!')
    return None


if __name__ == '__main__':
    run1_data, run2_data, run1_masks, run2_masks = datagrabber()
    print('loading run 1 data')
    run1_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=1,
                          force_mask_run1=True, zscore=True, nan2num=True)
    print('loading run 2 data')
    run2_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=2,
                          force_mask_run1=True, zscore=True, nan2num=True)
    project_data_crossval(run1_arrs, run2_arrs)
