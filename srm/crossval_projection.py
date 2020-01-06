#!/usr/bin/env python3

import runpy
import time
import os
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.rsrm import RSRM

# import io functions from my other srm script
file_globals = runpy.run_path('srm_roi.py')
datagrabber = file_globals['datagrabber']
load_data = file_globals['load_data']
grab_subject_ids = file_globals['grab_subject_ids']

file_globals2 = runpy.run_path('decode_random_stimulation.py')
load_data_and_labels = file_globals2['load_data_and_labels']


def project_data_crossval_periodiconly(run1_arrs,
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


def project_data_random_to_periodic(ds_dir='/data/project/somato/scratch/dataset',
                                    roiglmdir='/data/project/somato/scratch/roi_glm/work_basedir/',
                                    outdir='/data/project/somato/scratch/project_random_stimulation',
                                    nfeatures=10,
                                    niter=20,
                                    exclude_subjects=('fip66'),
                                    testsubs=False):
    """
    Fit SRM to the periodic data (which has been z-scored and appended) with one subject left out
    and project the test-subject's random runs (blocked_design 1 and 2) to that model.
    Save these results as .npy files.
    """
    # load data
    print('loading data')
    periodic_data, random1_data, random2_data, \
        periodic_labels_concat, random_labels = load_data_and_labels(testsubs_=testsubs,
                                                                     dsdir=ds_dir,
                                                                     roiglm_workdir=roiglmdir,
                                                                     excludesubs=exclude_subjects)
    # create output directory
    if not os.path.exists(outdir):
        print('creating directory : ', outdir)
        os.makedirs(outdir)
    # init results array (nsubs, nruns, nfeatures, nvols_random)
    proj_results = np.zeros(shape=(len(periodic_data), 2, nfeatures, random1_data[0].shape[1]))
    # and one for training models with average responses (nsubs, nfeatures, nvols_random)
    train_results = np.zeros(shape=(len(periodic_data), nfeatures,  512))  # TODO don't hardcode last shape value
    # init srm
    srm = RSRM(n_iter=niter, features=nfeatures)
    # iterate over test subjects
    for testsub_idx in range(len(periodic_data)):
        print('starting subject with index : ', testsub_idx)
        # select training subjects periodic data
        training_data = [entry for idx, entry in enumerate(periodic_data) if idx != testsub_idx]
        # fit srm and estimate mapping/basis for test subject
        srm.fit(training_data)
        w, s = srm.transform_subject(periodic_data[testsub_idx])
        srm.w_.insert(testsub_idx, w)  # insert basis back to our srm object
        srm.s_.insert(testsub_idx, s)
        # project data from random runs
        proj_random1, ind_terms1 = srm.transform(random1_data)
        proj_random2, ind_terms2 = srm.transform(random2_data)
        # append to results
        proj_results[testsub_idx, 0] = proj_random1[testsub_idx]  # first random run
        proj_results[testsub_idx, 1] = proj_random2[testsub_idx]  # second random run
        train_results[testsub_idx] = srm.r_
    # save to npy
    print('saving results')
    proj_fname = pjoin(outdir, 'proj_results.npy')
    train_fname = pjoin(outdir, 'train_results.npy')
    for fname, result in zip([proj_fname, train_fname],
                          [proj_results, train_results]):
        with open(fname, 'wb') as fhandle:
            np.save(fhandle, result)
    return None


if __name__ == '__main__':

    # may this script run for the periodic design only or the random data projected on model trained on periodic data.
    whichdesign = 'random'
    assert whichdesign in (['random', 'periodic'])

    if whichdesign == 'random':
        project_data_random_to_periodic()

    elif whichdesign == 'periodic':
        # todo: make this shorter (not so important now)
        run1_data, run2_data, run3_data, run4_data, run1_masks, run2_masks, run3_masks, run4_masks = datagrabber()
        print('loading run 1 data')
        run1_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=1,
                              force_mask_run1=True, zscore=True, nan2num=True)
        print('loading run 2 data')
        run2_arrs = load_data(run1_data, run2_data, run1_masks, run2_masks, whichrun=2,
                              force_mask_run1=True, zscore=True, nan2num=True)
        project_data_crossval_periodiconly(run1_arrs, run2_arrs)
