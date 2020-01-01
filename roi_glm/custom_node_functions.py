#!/usr/bin/env python3


def grab_boldfiles_subject(sub_id, cond_ids, ds_dir):
    """
    Get file paths for our bold data for a given subject.
    """
    from os.path import join as pjoin
    boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz') for cond_id in cond_ids]
    return boldfiles


def grab_blocked_design_onsets_subject(sub_id,
                                       prepped_ds_dir):
    """
    For the given subject, generate a dict with keys blocked_design1 and blocked_design2
    and nested lists as values representing the onsets of the five digits.

    note: durations is always 5.12, so no need to load that.
    """
    import csv
    from os.path import join as pjoin
    blocked_design_onsets_dicts = {}
    cond_ids = ('blocked_design1', 'blocked_design2')
    for condid in cond_ids:
        blocked_design_onsets_dicts[condid] = []
        for dig_int in range(1, 6):
            dig_abspath = pjoin(prepped_ds_dir, sub_id, condid, 'D%i.ons' % dig_int)
            with open(dig_abspath, 'r') as f:
                csv_reader = csv.reader(f, delimiter='\n')
                dig_onsets = [float(row[0]) for row in csv_reader]
                blocked_design_onsets_dicts[condid].append(dig_onsets)
    return blocked_design_onsets_dicts


def make_bunch_and_contrasts(blocked_design_onsets_dicts,
                             n_cycles=20,
                             dur_per_digit=5.12,
                             subtractive_contrast=False):
    """
    Produce subject_info as required input of SpecifyModel (Bunch containing conditions, onsets, durations)
    and contrasts as input for modelfit workflow.

    Subtractive contrasts weights regressors of interest with +4 and all others with -1. in this case, we skip the last
    contrast (because it would be a linear combination of the others).
    Non-subtractive contrast (i.e. one-sample t-test) weights regressor of interest with 1 and all others with 0.
    """

    from nipype.interfaces.base import Bunch
    cycle_dur = dur_per_digit * 5

    # in periodic stimulation runs: onsets are the same for both conditions, just the order of regressors is flipped
    periodic_onsets = [[0 + (digit_idx * dur_per_digit) + (cycle_idx * cycle_dur)
               for cycle_idx in range(n_cycles)]
              for digit_idx in range(5)]
    durations = [[dur_per_digit] * n_cycles for _ in range(5)]
    d1_d5_conditions = ['D_%i' % i for i in range(1, 6)]
    d5_d1_conditions = ['D_%i' % i for i in range(5, 0, -1)]

    # blocked_design conditions and onsets
    blocked1_onsets = blocked_design_onsets_dicts['blocked_design1']
    blocked2_onsets = blocked_design_onsets_dicts['blocked_design2']

    subject_info = [Bunch(conditions=d1_d5_conditions, onsets=periodic_onsets, durations=durations),
                    Bunch(conditions=d5_d1_conditions, onsets=periodic_onsets, durations=durations),
                    Bunch(conditions=d1_d5_conditions, onsets=blocked1_onsets, durations=durations),
                    Bunch(conditions=d1_d5_conditions, onsets=blocked2_onsets, durations=durations)]
    # t-cotrasts
    t_contrasts = []
    for cond_name in d1_d5_conditions:
        if subtractive_contrast:
            if d1_d5_conditions.index(cond_name) == len(cond_name) - 1:
                continue
            else:
                contrast_vector = [-1, -1, -1, -1]
                contrast_vector.insert(d1_d5_conditions.index(cond_name), 4)
                t_contrasts.append(('tcon_%s' % cond_name, 'T', d1_d5_conditions, contrast_vector))
        else:
            contrast_vector = [0, 0, 0, 0]
            contrast_vector.insert(d1_d5_conditions.index(cond_name), 1)
            t_contrasts.append(('tcon_%s' % cond_name, 'T', d1_d5_conditions, contrast_vector))
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