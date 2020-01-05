#!/usr/bin/env python3

import csv
import runpy
from os.path import join as pjoin

import numpy as np
from brainiak.funcalign.rsrm import RSRM
from nilearn.image import load_img
from nilearn.masking import apply_mask
from nilearn.masking import intersect_masks
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier

# general io functions
file_globals = runpy.run_path('srm_roi.py')
file_globals2 = runpy.run_path('digit_classification_knn.py')
datagrabber = file_globals['datagrabber']
# load_data = file_globals['load_data']
grab_subject_ids = file_globals['grab_subject_ids']
# functions to get digit indices in periodic runs
get_digit_indices, digit_indices_to_labels = file_globals2['get_digit_indices'], \
                                             file_globals2['digit_indices_to_labels']


def load_and_append_periodic_runs_single_subject(run1_boldfile, run2_boldfile,
                                                 run1_maskfile, run2_maskfile,
                                                 zscorewithinrun=True,
                                                 connected_clusters=False):
    """
    For a given subject, load the data from the first two runs (periodic stimulation),
    mask with a union of the run masks
    and z score if desired.
    """
    # get union mask
    print('trying to intersect masks : ', run1_maskfile, ' and ', run2_maskfile)
    unionmask = intersect_masks([run1_maskfile, run2_maskfile], threshold=0, connected=connected_clusters)
    # load data, apply mask
    run1_arr = apply_mask(load_img(run1_boldfile), mask_img=unionmask).T
    run2_arr = apply_mask(load_img(run2_boldfile), mask_img=unionmask).T
    # zscore within run if desired
    if zscorewithinrun:
        for runimg in [run1_arr, run2_arr]:
            runimg = zscore(np.nan_to_num(runimg), axis=1)
    # concatenate runs
    sub_arr = np.concatenate((run1_arr, run2_arr), axis=1)
    print('finished loading masks: ', run1_maskfile, ' and ', run2_maskfile)
    return sub_arr, unionmask


def load_and_append_periodic_data(run1_data, run2_data,
                                  run1_masks, run2_masks,
                                  zscore_withinrun=True):
    """
    returns periodic_data, which is list of arrays where each element corresponds to the concatenated periodic runs
    for a given subject.
    And union_masks, which is list of arrays.
    """
    periodic_data = []
    union_masks = []
    for sub_idx in range(len(run1_data)):
        sub_arr, unionmask = load_and_append_periodic_runs_single_subject(run1_data[sub_idx], run2_data[sub_idx],
                                                                          run1_masks[sub_idx], run2_masks[sub_idx],
                                                                          zscorewithinrun=zscore_withinrun)
        periodic_data.append(sub_arr)
        union_masks.append(unionmask)
        print('finished loading subject with index : ', sub_idx)
    return periodic_data, union_masks


def load_masked_random_runs(run3_data,
                            run4_data,
                            union_masks):
    # TODO: make z-scoring optional
    random1_data = [
        zscore(np.nan_to_num(apply_mask(load_img(run3_file), mask_img=unionmask).T), axis=1, ddof=1)
        for run3_file, unionmask
        in zip(run3_data, union_masks)
    ]
    random2_data = [
        zscore(np.nan_to_num(apply_mask(load_img(run4_file), mask_img=unionmask).T), axis=1, ddof=1)
        for run4_file, unionmask
        in zip(run4_data, union_masks)]
    return random1_data, random2_data


def fit_srm_and_project_data(periodic_data,
                             random1_data, random2_data,
                             n_responses=5,
                             n_iter=20):
    srm = RSRM(n_iter=n_iter, features=n_responses)
    srm.fit(periodic_data)
    random1_projected, random1_indterms = srm.transform(random1_data)
    random2_projected, random2_indterms = srm.transform(random2_data)
    periodic_data_projected, periodic_indterms = srm.transform(periodic_data)
    # TODO: don't return indterms
    return srm, random1_projected, random2_projected, periodic_data_projected


def get_onsets_randomruns(sub_ids,
                          prepped_ds_dir):
    # array with onsets of shape nsubs, nruns, ndigits
    onsets_array = np.zeros(shape=(len(sub_ids), 2, 5, 10))
    for sub_idx, sub_id in enumerate(sub_ids):
        for run_idx, run_str in enumerate(['blocked_design1', 'blocked_design2']):
            for dig_idx, dig_int in enumerate(range(1, 6)):
                dig_abspath = pjoin(prepped_ds_dir, sub_id, run_str, 'D%i.ons' % dig_int)
                with open(dig_abspath, 'r') as f:
                    csv_reader = csv.reader(f, delimiter='\n')
                    dig_onsets = [float(row[0]) for row in csv_reader]
                    onsets_array[sub_idx, run_idx, dig_idx] = dig_onsets
    return onsets_array


def randomruns_onsets_to_labels(randomruns_onsets_array,
                                stimdur=5.12,
                                tr=2.,
                                nvols_in_random_run=212):
    stimdur_ms = int(stimdur * 100)
    nsubs, nruns, ndigits, nonsets_per_digit = randomruns_onsets_array.shape
    labels_ms = np.zeros(shape=(nsubs, nruns, int(nvols_in_random_run * tr * 100)))
    for sub_idx in range(nsubs):
        for run_idx in range(nruns):
            for digit_idx, digit_id in enumerate(range(1, ndigits + 1)):
                dig_onsets = randomruns_onsets_array[sub_idx, run_idx, digit_idx]
                for ons in dig_onsets:
                    ons_ms = int(ons * 100)
                    labels_ms[sub_idx, run_idx, int(ons_ms):int(ons_ms + stimdur_ms)] = digit_id
    random_labels = labels_ms[:, :, ::int(100 * tr)]
    return random_labels


def make_periodic_labels(stimdur=5.12,
                         tr=2.,
                         nvols_periodic_run=256):
    # TODO: replace old digit_indices functions with this one
    stimdur_ms = int(stimdur * 100)
    nruns, ndigits, nonsets_per_digit = 2, 5, 20
    all_onsets_s = np.arange(0, nvols_periodic_run * tr, stimdur)
    all_onsets_ms = all_onsets_s * 100
    # initiate upscaled array
    periodic_labels_ms_2d = np.zeros(shape=(nruns, nvols_periodic_run * 100 * int(tr)))
    periodic_labels_2d = np.zeros(shape=(nruns, nvols_periodic_run))
    # iterate for first run, create second by flipping
    for dig_idx, dig_id in enumerate(range(1, 6)):
        dig_onsets_ms = all_onsets_ms[dig_idx::5]
        for ons_ms in dig_onsets_ms:
            periodic_labels_ms_2d[0, int(ons_ms):int(ons_ms + stimdur_ms)] = dig_id
    # downscale
    periodic_labels_2d[0] = periodic_labels_ms_2d[0, ::int(tr * 100)]
    # labels for second run by flipping those of first run
    periodic_labels_2d[1] = np.flip(periodic_labels_2d[0])
    # also concatenate for convenience
    periodic_labels_concat = periodic_labels_2d.flatten()
    return periodic_labels_2d, periodic_labels_concat


def load_data_and_labels(testsubs_=False,
                         dsdir='/data/project/somato/scratch/dataset',
                         roiglm_workdir='/data/project/somato/scratch/roi_glm/work_basedir/',
                         excludesubs=()):
    """
    Grab data and labels, fit srm and project data.
    Write as function as prestep for different potential classifiers.
    """
    sub_ids = grab_subject_ids(ds_dir=dsdir, testsubs=testsubs_, exclude_subs=excludesubs)

    print('get labels')
    # get digit labels for periodic data and randomized data
    periodic_labels_2d, periodic_labels_concat = make_periodic_labels()
    # TODO: I think periodic labels / onsets might be wrong.
    random_onsets = get_onsets_randomruns(sub_ids, prepped_ds_dir=dsdir)
    random_labels = randomruns_onsets_to_labels(random_onsets)

    # get relevant file_paths
    print('datagrabber')
    run1_data, run2_data, \
        run3_data, run4_data, \
        run1_masks, run2_masks, \
        run3_masks, run4_masks = datagrabber(roi_glm_workdir=roiglm_workdir,
                                             prepped_dsdir=dsdir,
                                             testsubs=testsubs_)
    print('load_and_append_periodic')
    # load concatenated periodic data
    periodic_data, union_masks = load_and_append_periodic_data(run1_data, run2_data,
                                                               run1_masks, run2_masks,
                                                               zscore_withinrun=True)
    print('load and mask random runs')
    # load data from runs with randomized digit stimulation
    random1_data, random2_data = load_masked_random_runs(run3_data, run4_data, union_masks)
    print('fit srm and project')
    return periodic_data, random1_data, random2_data, periodic_labels_concat, random_labels


def fit_and_project(periodic_data,
                    random1_data, random2_data,
                    n_iters_srm=20,
                    nfeatures=5):
    # fit srm and project randomized data to shared space
    srm, random1_projected, random2_projected, \
        periodic_data_projected_list = fit_srm_and_project_data(periodic_data,
                                                                random1_data,
                                                                random2_data,
                                                                n_responses=nfeatures,
                                                                n_iter=n_iters_srm)

    return srm, random1_projected, random2_projected, periodic_data_projected_list


def knn_randomdata_given_neighs(periodic_data_projected_list,
                                random1_projected, random2_projected,
                                periodic_labels_concat, random_labels,
                                nneighs=5):
    """
    Take data prepared with load_data_and_labels and run a knn with given nneighs.
    returns accuracy_array with shape (nsubs, nruns)
    """
    # initiate accuracy array of shape: nsubs, nruns
    acc_array = np.zeros(shape=(len(random1_projected), 2))
    for testsub_idx in range(len(random1_projected)):
        for testrun_idx, testrun_data in enumerate([random1_projected, random2_projected]):
            training_data_list = [subdata for idx, subdata in enumerate(periodic_data_projected_list)
                                  if idx != testsub_idx]
            training_data_arr = np.concatenate(training_data_list, axis=1)
            training_labels = np.tile(periodic_labels_concat, len(training_data_list))
            neigh = KNeighborsClassifier(n_neighbors=nneighs)
            neigh.fit(training_data_arr.T, training_labels)
            testdata = np.nan_to_num(testrun_data[testsub_idx])
            test_labels = random_labels[testsub_idx, testrun_idx]
            acc_array[testsub_idx, testrun_idx] = neigh.score(testdata.T, test_labels)

    return acc_array


def iterate_knn_over_nneighs_nfeaturs(nfeatures_range=(5, 8, 10, 15, 20, 50, 100),
                                      nneighs_range=(5, 8, 10, 15, 20, 50, 100),
                                      outdir='/data/project/somato/scratch/decode_random_stimulation',
                                      exclude_subs=()):
    # TODO: this is the top level function. also specify all kwargs for lower level functions here.
    # TODO: docstring

    # load data
    print('loading data')
    periodic_data, \
        random1_data, random2_data, \
        periodic_labels_concat, random_labels = load_data_and_labels(excludesubs=exclude_subs)

    # prepare empty results array of shape (nfeatures_range, nneighs_range, nsubs, nruns)
    results = np.zeros(shape=(len(nfeatures_range), len(nneighs_range), len(random1_data), 2))

    for feat_idx, nfeatures_ in enumerate(nfeatures_range):
        # fit srm
        print('fitting srm with ', str(nfeatures_))
        srm, random1_projected, random2_projected, \
            periodic_data_projected_list = fit_and_project(periodic_data,
                                                           random1_data, random2_data, nfeatures=nfeatures_)

        for neighs_idx, nneighs_ in enumerate(nneighs_range):
            print('starting classification with ', str(nneighs_))
            # do classification
            acc_arr_ = knn_randomdata_given_neighs(periodic_data_projected_list,
                                                   random1_projected, random2_projected,
                                                   periodic_labels_concat, random_labels,
                                                   nneighs=nneighs_)
            # append results
            results[feat_idx, neighs_idx] = acc_arr_

    # save result
    with open(pjoin(outdir, 'decode_random_stimulation.npy'), 'wb') as f:
        np.save(f, results)

    return None


# TODO: same vor SVC classifier


if __name__ == '__main__':
    iterate_knn_over_nneighs_nfeaturs(exclude_subs=('fip66'))  # TODO check out what's wrong with fip66 data
