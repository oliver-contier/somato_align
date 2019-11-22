#!/usr/bin/env python3

"""
1. load data from run 1
2. train SRM on data from first run (leave one subject out)
3. load data from run 2
4. Project test subject's data from run 2 into shared space
5.
"""

import runpy

import numpy as np
import time
from os.path import join as pjoin
from brainiak.funcalign.rsrm import RSRM
from scipy import signal, stats

# import io functions from my other srm script
file_globals = runpy.run_path('srm_roi.py')
datagrabber = file_globals['datagrabber']
load_data = file_globals['load_data']
grab_subject_ids = file_globals['grab_subject_ids']


def get_digit_indices(n_cycles=20,
                      vols=256,
                      vols_per_digit=2.56):
    """
    Produce two boolean arrays, one for each run.
    Each array has shape (n_digits, n_volumes).
    Use these to select samples in our classification task
    """
    # TODO: I'm VERY UNSURE if these indices are correct ...

    vols_per_digit_upsampled = int(vols_per_digit * 100)
    digits_run1 = []
    for didx in range(1, 6):
        # create series of 1s for the first finger stimulation
        finger_signal = signal.boxcar(vols_per_digit_upsampled)
        # add zeros before and after accordingly to form first cycle.
        post_padded = np.append(finger_signal, [0] * vols_per_digit_upsampled * (5 - didx))
        first_cycle = np.insert(post_padded, obj=0, values=[0] * vols_per_digit_upsampled * (didx - 1))
        all_cycles = np.tile(first_cycle, n_cycles)  # repeat to get all cycles
        # resample to volume space (downsampling leads to fuzzy edges)
        digit_signal = signal.resample(all_cycles, vols)
        # make a boolean array
        digit_bool = digit_signal > .5
        digits_run1.append(digit_bool)
    digits_run1 = np.array(digits_run1)
    digits_run2 = np.flip(digits_run1, axis=0)
    return digits_run1, digits_run2


def compute_corrmat_digits(test_data,
                           test_digits_arr,
                           train_digits_arr,
                           trained_srm,
                           average_axis=1):
    """
    Compute a correlation matrix between the digit segments in a test subject's data from the test run
    and the respective trained shared responses.

    data is z-scored and averaged beforehand.
    average_axis=1 indicates averaging over all selected samples of each component,
    which is what I think is intended in the original SRM time segment matching studies.
    """
    corr_mtx = np.zeros(shape=(5, 5))
    for i in range(5):
        for j in range(5):
            data = test_data[:, test_digits_arr[i, :]]
            data = np.average(stats.zscore(data, axis=0, ddof=1), axis=average_axis)
            features = trained_srm.r_[:, train_digits_arr[j, :]]
            features = np.average(stats.zscore(features, axis=0, ddof=1), axis=average_axis)
            corr = np.corrcoef(data, features)[0][1]
            corr_mtx[i, j] = corr
    return corr_mtx


def corrmat_accuracies(corr_mat):
    """
    Take correlation matrix calculated with compute_corrmat_digits and
    return list of accuracies with each element representing one digit.
    """
    accuracies = []
    for i in range(5):
        row = corr_mat[i, :]
        if np.argmax(row) == i:
            accuracies.append(1)
        else:
            accuracies.append(0)
    return accuracies


def run_crossval_classification_given_k(run1_arrs,
                                        run2_arrs,
                                        digits_run1,
                                        digits_run2,
                                        k=3,
                                        niter=20,
                                        outdir='/data/BnB_USER/oliver/somato/scratch/digit_classification'):
    """
    # TODO: In order to transform data from run 2 to run 1, the ROI masks from both runs have to have the same
       number of voxels. Maybe we should take the union of both masks for this?
    """

    # prepare empty results array
    acc_results = np.zeros(shape=(2, len(run1_arrs), 5))  # shape (nruns, nsubs, ndigits)

    for trainrun_idx in range(2):  # iterate over runs
        # select run used for training and test and according digit indices
        training_arrs = (run1_arrs, run2_arrs)[trainrun_idx]
        test_arrs = (run1_arrs, run2_arrs)[abs(trainrun_idx - 1)]
        train_digits = (digits_run1, digits_run2)[trainrun_idx]
        test_digits = (digits_run1, digits_run2)[abs(trainrun_idx - 1)]

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

            # compute correlation matrix
            corr_mtx = compute_corrmat_digits(test_data=testsub_proj,  trained_srm=srm,
                                              test_digits_arr=test_digits, train_digits_arr=train_digits)
            # compute accuracies
            accuracies = corrmat_accuracies(corr_mtx)
            acc_results[trainrun_idx, testsub_idx] = accuracies  # append result to our results arrays
            elapsed = time.time() - start
            print('this round took: ', elapsed)

        # save results array for this run
        acc_outpath = pjoin(outdir, 'accuracies_k%i.npy' % k)
        with open(acc_outpath, 'wb') as outf:
            np.save(outf, acc_results)

    print('done!')
    return None


def test_different_ks(ks=(3, 5, 10, 20, 50, 100),
                      srm_iter=20):
    """
    Run cross-validated classification to over different numbers of shared responses (k)
    and save the resulting accuracies.
    """
    # load data
    run1_data, run2_data, run1_masks, run2_masks = datagrabber()
    print('loading run 1')
    run1_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=1, force_mask_run1=True)
    print('loading run 2')
    run2_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=2, force_mask_run1=True)
    digits_run1, digits_run2 = get_digit_indices()

    for k in ks:
        print('starting k :', k)
        run_crossval_classification_given_k(run1_arrs=run1_arrs, run2_arrs=run2_arrs, k=k,
                                            digits_run1=digits_run1, digits_run2=digits_run2, niter=srm_iter)

    return None


if __name__ == '__main__':
    test_different_ks()
