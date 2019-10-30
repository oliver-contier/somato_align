#!/usr/bin/env python3


def grab_boldfiles_subject(sub_id, cond_ids, ds_dir):
    """
    Get file paths for our bold data for a given subject.
    """
    from os.path import join as pjoin
    boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz') for cond_id in cond_ids]
    return boldfiles


def make_bunch_and_contrasts(n_cycles=20,
                             dur_per_digit=5.12,
                             subtractive_contrast=False):
    """
    Produce subject_info as required input of SpecifyModel (Bunch containing conditions, onsets, durations)
    and contrasts as input for modelfit workflow.

    Subtractive contrasts weights regressors of interest with +4 and all others with -1. in this case, we skip the last
    contrast (because it would be a linear combination of the others).
    Non-subtractive contrast (i.e. one-sample t-test) weights regressor of interest with 1 and all others with 0.
    """
    # TODO: design and contrasts don't work for condition 2 atm!!!

    from nipype.interfaces.base import Bunch
    cycle_dur = dur_per_digit * 5

    # onsets are the same for both ocnditions, just the order of regressors is flipped
    onsets = [[0 + (digit_idx * dur_per_digit) + (cycle_idx * cycle_dur)
               for cycle_idx in range(n_cycles)]
              for digit_idx in range(5)]
    durations = [[dur_per_digit] * n_cycles for _ in range(5)]
    run1_conditions = ['D_%i' % i for i in range(1, 6)]
    run2_conditions = ['D_%i' % i for i in range(5, 0, -1)]

    subject_info = [Bunch(conditions=run1_conditions, onsets=onsets, durations=durations),
                    Bunch(conditions=run2_conditions, onsets=onsets, durations=durations)]
    # t-cotrasts
    t_contrasts = []
    for cond in run1_conditions:
        if subtractive_contrast:
            if run1_conditions.index(cond) == len(cond) - 1:
                continue
            else:
                contrast_vector = [-1, -1, -1, -1]
                contrast_vector.insert(run1_conditions.index(cond), 4)
                t_contrasts.append(('tcon_%s' % cond, 'T', run1_conditions, contrast_vector))
        else:
            contrast_vector = [0, 0, 0, 0]
            contrast_vector.insert(run1_conditions.index(cond), 1)
            t_contrasts.append(('tcon_%s' % cond, 'T', run1_conditions, contrast_vector))
    # f-contrast over all t-contrasts
    f_contrast = [('All_Digits', 'F', t_contrasts)]
    contrasts = t_contrasts + f_contrast
    n_copes = len(contrasts)
    return subject_info, contrasts


def flatten_nested_list(nested_list):
    """
    Seems like a bit of a design flaw that this is even necessary,
    but oh, well ...
    """
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def sort_copes(copes, varcopes, contrasts):
    import numpy as np
    if not isinstance(copes, list):
        copes = [copes]
        varcopes = [varcopes]
    num_copes = len(contrasts)
    n_runs = len(copes)
    all_copes = np.array(copes).flatten()
    all_varcopes = np.array(varcopes).flatten()
    # outcopes = all_copes.reshape(len(all_copes) / num_copes, num_copes).T.tolist()
    outcopes = all_copes.reshape(int(len(all_copes) / len(copes[0])), len(copes[0])).T.tolist()
    outvarcopes = all_varcopes.reshape(int(len(all_varcopes) / len(varcopes[0])), len(varcopes[0])).T.tolist()
    return outcopes, outvarcopes, n_runs


def split_zfstats_runs(zfstats_list):
    zfstat_run1 = zfstats_list[0]
    zfstat_run2 = [zfstats_list[1]]  # operand files have to be a list
    return zfstat_run1, zfstat_run2


def pick_first_mask(mask_files):
    first_mask = mask_files[0]
    return first_mask