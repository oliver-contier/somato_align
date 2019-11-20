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


def run_crossvar_classification_given_k(niter=2, k=3):
    """
    # TODO: z-score bold data before training srm

    # TODO: In order to transform data from run 2 to run 1, the ROI masks from both runs have to have the same
       number of voxels. Maybe we should take the union of both masks for this?

    # TODO: lastly, iterate over subjects and runs and save all the accuracies
    """
    # load data
    run1_data, run2_data, run1_masks, run2_masks = datagrabber()
    print('loading run 1')
    run1_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=1, force_mask_run1=True)
    print('loading run 2')
    run2_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=2, force_mask_run1=True)
    digits_run1, digits_run2 = get_digit_indices()

    # fit srm to data from first run
    testsub_idx = 0  # TODO: iterate over subjects using this index variable
    # training_data = [x for i, x in enumerate(run1_arrs) if i != testsub_idx]
    srm = RSRM(n_iter=niter, features=k)
    srm.fit(run1_arrs)

    # project second run to shared space
    run2_shared, run2_ind = srm.transform(run2_arrs)

    # select test subject
    testsub_projected_data = run2_shared[testsub_idx]

    # compute digit segment correlation matrix
    corrmat = compute_corrmat_digits(test_data=testsub_projected_data,
                                     test_digits_arr=digits_run2,
                                     train_digits_arr=digits_run1,
                                     trained_srm=srm)

    accuracies = corrmat_accuracies(corrmat)

    return None


if __name__ == '__main__':
    run_crossvar_classification_given_k()
