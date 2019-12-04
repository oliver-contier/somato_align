#!/usr/bin/env python3

import os
import runpy
import time
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.rsrm import RSRM
from scipy import signal
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier

file_globals = runpy.run_path('srm_roi.py')
datagrabber = file_globals['datagrabber']
load_data = file_globals['load_data']
grab_subject_ids = file_globals['grab_subject_ids']


def get_digit_indices(n_cycles=20,
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
        # resample to volume space (i.e. take every 100th element)
        # and turn into boolean vector
        digit_bool = all_cycles[::100] > 0.01
        digits_run1.append(digit_bool)
    digits_run1 = np.array(digits_run1)
    digits_run2 = np.flip(digits_run1, axis=0)
    return digits_run1, digits_run2


def digit_indices_to_labels(digits_run1, digits_run2):
    """
    Turn the boolean arrays of digit indices
    into 1d arrays with values 1-6
    for use with SVC.
    """
    labels_run1, labels_run2 = np.zeros(shape=256), np.zeros(shape=256)
    for finger_i in range(1, 6):
        labels_run1[digits_run1[finger_i - 1]] = finger_i
        labels_run2[digits_run2[finger_i - 1]] = finger_i
    return labels_run1, labels_run2


def project_data_crossval(run1_arrs,
                          run2_arrs,
                          nfeatures=10,
                          niter=20,
                          outdir='/data/BnB_USER/oliver/somato/scratch/crossval_projection'):
    """
    Project subjects bold data onto shared response space in a cross-validated way. For every subject, an srm is
    trained on the remaining subjects. this srm is trained on the data from run one, and the data from the test
    subject's other run is than projected onto this template. Data is saved in numpy's npy.
    """
    # prepare results arrays (for projected data and estimated srm time series)
    # both arrays have shape (nsubs, nruns, nfeatures, nvols)
    projected_data = np.zeros(shape=(len(run1_arrs), 2, nfeatures, run1_arrs[0].shape[1]))
    trained_srms = np.zeros(shape=(len(run1_arrs), 2, nfeatures, run1_arrs[0].shape[1]))
    # iterate over runs used for training / testing
    for trainrun_idx in range(2):
        # select run used for training and test
        training_run_arrs = (run1_arrs, run2_arrs)[trainrun_idx]
        test_run_arrs = (run1_arrs, run2_arrs)[abs(trainrun_idx - 1)]
        # iterate over testsubjects
        for testsub_idx in range(len(training_run_arrs)):
            start = time.time()
            print('starting projection of run %i subject %i' % (trainrun_idx, testsub_idx))
            # select training and test subject
            trainsubs_traindata = [x for i, x in enumerate(training_run_arrs) if i != testsub_idx]
            testsub_traindata = training_run_arrs[testsub_idx]
            srm = RSRM(n_iter=niter, features=nfeatures)  # train srm on training subject's training data
            srm.fit(trainsubs_traindata)
            w, s = srm.transform_subject(testsub_traindata)  # estimate test subject's bases
            # reattach weight matrix and individual term to srm instance
            # (to allow transforming test run with builtin brainiak function)
            srm.w_.insert(testsub_idx, w)
            srm.s_.insert(testsub_idx, s)
            allsubs_proj, ind_terms = srm.transform(test_run_arrs)  # project test run into shared space
            testsub_proj = allsubs_proj[testsub_idx]  # select projected data from test subject
            projected_data[testsub_idx, trainrun_idx, :, :] = testsub_proj
            trained_srms[testsub_idx, trainrun_idx, :, :] = srm.r_
            elapsed = time.time() - start
            print('this round took: ', elapsed)

    # save results array to file
    if outdir:
        proj_outpath = pjoin(outdir, 'proj_data_nfeats-%i.npy' % nfeatures)
        with open(proj_outpath, 'wb') as outf:
            np.save(outf, projected_data)
        trained_srms_outpath = pjoin(outdir, 'trainedsrms_nfeats-%i.npy' % nfeatures)
        with open(trained_srms_outpath, 'wb') as outf:
            np.save(outf, trained_srms)
    print('done!')
    return projected_data, trained_srms


def knn_cross_sub_and_run(projected_data,
                          nneighs=5,
                          zscore_over_all=True):
    """
    Train a KNN classifier on all but one subjects data for a given run, and test on the left-out subject for the other
    run. (cross-run cross-subject predictions).
    Returns array with accuracies of shape (nsubs, nruns).
    (remember, input array projected_data has shape (nsubs, nruns, nfeatures, nvols))
    """
    # get digit labels
    digits_run1, digits_run2 = get_digit_indices()
    labels_run1, labels_run2 = digit_indices_to_labels(digits_run1, digits_run2)
    # global zscoring of data if desired
    if zscore_over_all:
        projected_data = zscore(projected_data, axis=3)
    # prepare accuracy array
    nsubs, nruns, nfeatures, nvols = projected_data.shape
    accuracies = np.zeros((nsubs, nruns))
    # iterate over subjects and runs
    for sub_i in range(nsubs):
        for testrun_i in range(2):
            # select training and test data and labels
            test_data = projected_data[sub_i, testrun_i, :, :]
            submask, runmask = np.ones(nsubs, dtype=bool), np.ones(2, dtype=bool)
            submask[sub_i] = False
            runmask[testrun_i] = False
            train_data = projected_data[submask, runmask, :, :]
            train_data = train_data.reshape(nfeatures, nvols * (nsubs-1))
            test_labels = [labels_run1, labels_run2][testrun_i]
            train_labels = np.tile([labels_run1, labels_run2][abs(testrun_i - 1)], (nsubs-1))
            # train classifier and score
            neigh = KNeighborsClassifier(n_neighbors=nneighs)
            neigh.fit(train_data.T, train_labels)
            accuracies[sub_i, testrun_i] = neigh.score(test_data.T, test_labels)
    return accuracies


def knn_within_run(projected_data,
                   nneighs=5,
                   zscore_over_all=True):
    """
    Train KNN on all but one subject and test at left-out subject's data from the same run.
    """
    # get digit labels
    digits_run1, digits_run2 = get_digit_indices()
    labels_run1, labels_run2 = digit_indices_to_labels(digits_run1, digits_run2)
    # global z-scoring
    if zscore_over_all:
        projected_data = zscore(projected_data, axis=3)
    # prepare accuracy array
    nsubs, nruns, nfeatures, nvols = projected_data.shape
    accuracies = np.zeros((nsubs, nruns))
    for testsub_i in range(nsubs):
        for within_run_i in range(2):
            # select training and test data and labels
            test_data = projected_data[testsub_i, within_run_i, :, :]
            submask = np.ones(nsubs, dtype=bool)
            submask[testsub_i] = False
            train_data = projected_data[submask, within_run_i, :, :]
            train_data = train_data.reshape(nfeatures, nvols * (nsubs-1))
            test_labels = [labels_run1, labels_run2][within_run_i]
            train_labels = np.tile(test_labels, (nsubs-1))
            # train classifier and score
            neigh = KNeighborsClassifier(n_neighbors=nneighs)
            neigh.fit(train_data.T, train_labels)
            accuracies[testsub_i, within_run_i] = neigh.score(test_data.T, test_labels)
    return accuracies


def knn_within_sub(projected_data,
                   nneigh=5,
                   zscore_over_all=True):
    """
    Classify digits within subject, across runs.
    """
    digits_run1, digits_run2 = get_digit_indices()
    labels_run1, labels_run2 = digit_indices_to_labels(digits_run1, digits_run2)
    # global z-scoring
    if zscore_over_all:
        projected_data = zscore(projected_data, axis=3)
    # prepare accuracy array
    nsubs, nruns, nfeatures, nvols = projected_data.shape
    results = np.zeros((nsubs, nruns))
    for sub_i in range(nsubs):
        for trainrun_i in range(nruns):
            testrun_i = abs(trainrun_i - 1)
            train_data = projected_data[sub_i, trainrun_i, :, :]
            test_data = projected_data[sub_i, testrun_i, :, :]
            train_labels = [labels_run1, labels_run2][trainrun_i]
            test_labels = [labels_run1, labels_run2][testrun_i]
            neigh = KNeighborsClassifier(n_neighbors=nneigh)
            neigh.fit(train_data.T, train_labels)
            results[sub_i, trainrun_i] = neigh.score(test_data.T, test_labels)
    return results


def classify_over_nfeatures_nneighbors(run1_arrs, run2_arrs,
                                       nfeat_range=(5, 10, 20, 50, 100),
                                       nneigh_range=tuple(range(3, 101, 2)),
                                       proj_outdir='/data/BnB_USER/oliver/somato/scratch/crossval_projection',
                                       knn_outdir='/data/BnB_USER/oliver/somato/scratch/digit_classification_knn'):
    """
    Iterate over different values for the number of features allowed in the SRM
    and number of neighbors the KNN classifier considers.
    Save the whole shebang in npz files in given knn_outdir.
    """
    if not os.path.exists(knn_outdir):
        os.makedirs(knn_outdir)
    for nfeat in nfeat_range:
        proj_outpath = pjoin(proj_outdir, 'proj_data_nfeats-%i.npy' % nfeat)
        if os.path.exists(proj_outpath):
            with open(proj_outpath, 'rb') as f:
                projected_data = np.load(f)
        else:
            print('starting projection with %i features' % nfeat)
            projected_data, trained_srms = project_data_crossval(run1_arrs, run2_arrs, nfeatures=nfeat,
                                                                 outdir=proj_outdir)
        for nneigh in nneigh_range:
            print('and classification with %i neighbors' % nneigh)
            # cross-subject cross-run classification
            crossall_results = knn_cross_sub_and_run(projected_data, nneighs=nneigh)
            out_fname = pjoin(knn_outdir, 'nfeat-%i_nneigh-%i.npz' % (nfeat, nneigh))
            with open(out_fname, 'wb') as f:
                np.save(f, crossall_results)
            # cross-subject within-run classification
            withinrun_results = knn_within_run(projected_data, nneighs=nneigh)
            withinrun_outfname = pjoin(knn_outdir, 'withinrun_nfeat-%i_nneigh-%i.npz' % (nfeat, nneigh))
            with open(withinrun_outfname, 'wb') as f:
                np.save(f, withinrun_results)
            withinsub_results = knn_within_sub(projected_data, nneigh=nneigh)
            withinsub_outfname = pjoin(knn_outdir, 'withinsub_nfeat-%i_nneigh-%i.npz'% (nfeat, nneigh))
            with open(withinsub_outfname, 'wb') as f:
                np.save(f, withinsub_results)
            print('finished nfeats %i nneighs %i' % (nfeat, nneigh))
    return None


if __name__ == '__main__':

    # load input data
    run1_data, run2_data, run1_masks, run2_masks = datagrabber()
    print('loading run 1 data')
    run1_arrays = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=1,
                            force_mask_run1=True, zscore=True, nan2num=True)
    print('loading run 2 data')
    run2_arrays = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=2,
                            force_mask_run1=True, zscore=True, nan2num=True)
    print('starting the long journey of classification ...')
    classify_over_nfeatures_nneighbors(run1_arrays, run2_arrays)
