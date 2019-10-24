#!/usr/bin/env python3

"""

"""
import os
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl import BET, SUSAN
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.workflows.fmri.fsl import create_modelfit_workflow
from nipype.interfaces.io import DataSink


def grab_bold_data(cond_id='D1_D5',
                   ds_dir='/data/BnB_USER/oliver/somato/data',
                   testsubs=False):
    """
    Get file paths for our bold data for a given run.
    """
    import os
    import glob
    from os.path import join as pjoin
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_dir + '/*')]
    if testsubs:
        sub_ids = sub_ids[:testsubs]
    boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
                 for sub_id in sub_ids]
    return boldfiles, sub_ids


def make_bunch_and_contrasts(n_cycles=20,
                             dur_per_digit=5.12,
                             n_subs=12,
                             subtractive_contrast=True):
    """
    Produce subject_info as required input of SpecifyModel (Bunch containing conditions, onsets, durations)
    and contrasts as input for modelfit workflow.

    Subtractive contrasts weights regressors of interest with +4 and all others with -1. in this case, we skip the last
    contrast (because it would be a linear combination of the others).
    Non-subtractive contrast (i.e. one-sample t-test) weights regressor of interest with 1 and all others with 0.
    """

    from nipype.interfaces.base import Bunch
    cycle_dur = dur_per_digit * 5
    conditions = ['D_%i' % i for i in range(1, 6)]
    onsets = [[0 + (digit_idx * dur_per_digit) + (cycle_idx * cycle_dur)
               for cycle_idx in range(n_cycles)]
              for digit_idx in range(5)]
    durations = [[dur_per_digit] * n_cycles for _ in range(5)]
    subject_info = [Bunch(conditions=conditions, onsets=onsets, durations=durations)
                    for _ in range(n_subs)]
    # t-cotrasts
    t_contrasts = []
    for cond in conditions:
        if subtractive_contrast:
            if conditions.index(cond) == len(cond)-1:
                continue
            else:
                contrast_vector = [-1, -1, -1, -1]
                contrast_vector.insert(conditions.index(cond), 4)
                t_contrasts.append(('tcon_%s' % cond, 'T', conditions, contrast_vector))
        else:
            contrast_vector = [0, 0, 0, 0]
            contrast_vector.insert(conditions.index(cond), 1)
            t_contrasts.append(('tcon_%s' % cond, 'T', conditions, contrast_vector))
    # f-contrast over all t-contrasts
    f_contrast = [('All_Digits', 'F', t_contrasts)]
    contrasts = t_contrasts + f_contrast
    return subject_info, contrasts


def flatten_nested_list(nested_list):
    """
    Seems like a bit of a design flaw that this is even necessary,
    but oh, well ...
    """
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def create_main_wf(wf_workdir='/data/BnB_USER/oliver/somato/scratch/roi_glm',
                   wf_datasink_dir='/data/BnB_USER/oliver/somato/scratch/roi_glm_results',
                   dsdir='/data/BnB_USER/oliver/somato/data',
                   condid='D1_D5',
                   test_subs=False,
                   tr=2.,
                   bet_fracthr=.2,
                   susan_fwhm=2,
                   susan_brightthresh=1000,
                   hp_vols=30.,
                   lp_vols=2.,
                   film_thresh=0.01,
                   film_model_autocorr=False,
                   use_derivs=True,
                   tcon_subtractive=True):
    """
    Generate analysis pipeline for pre-srm GLM (and possibly later append with real srm)

    # TODO: what's a good film threshold? model autocorrelations?
    - filmgls threshold: nipype default is 1000
    """

    # make work and res dir if necessary
    for target_dir in [wf_datasink_dir, wf_workdir]:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    wf = Workflow(name='somato_melodic_wf')  # TODO: rename
    wf.base_dir = wf_workdir

    # datagrabber node
    datagrabber = Node(Function(input_names=['cond_id', 'ds_dir', 'testsubs'],
                                output_names=['boldfiles', 'sub_ids'],
                                function=grab_bold_data),
                       name='datagrabber')
    datagrabber.inputs.cond_id = condid
    datagrabber.inputs.ds_dir = dsdir
    datagrabber.inputs.testsubs = test_subs

    # BET node
    bet = MapNode(BET(frac=bet_fracthr, functional=True, mask=True),
                  iterfield=['in_file'], name='bet')

    # SUSAN smoothing node
    susan = MapNode(SUSAN(fwhm=susan_fwhm, brightness_threshold=susan_brightthresh),
                    iterfield=['in_file'], name='susan')

    # bandpass filter node
    bpf = MapNode(TemporalFilter(highpass_sigma=hp_vols / 2.3548,
                                 lowpass_sigma=lp_vols / 2.3548),
                  iterfield=['in_file'], name='bpf')

    # produce conditions, onsets, durations (same for all subjects)
    designgen = Node(Function(input_names=['subtracctive_contrast'],
                              output_names=['subject_info', 'contrasts'],
                              function=make_bunch_and_contrasts),
                     name='designgen')
    designgen.inputs.subtractive_contrast = tcon_subtractive

    # creates (intermediate) fsl compatible design file 'session_info' for modelfit
    modelspec = MapNode(SpecifyModel(input_units='secs'), name='modelspec',
                        iterfield=['functional_runs', 'subject_info'])
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = tr
    modelspec.inputs.high_pass_filter_cutoff = hp_vols * tr

    flatten_session_infos = Node(Function(input_names=['nested_list'], output_names=['flat_list'],
                                          function=flatten_nested_list), name='flatten_session_infos')

    # Modelfit is an extra built-in workflow
    # TODO: this modelfit workflow is originally meant to estimate all runs from one subject.
    #  Since our subjects all have same onsets, this works across subjects as well. But keep in mind for later
    modelfit = create_modelfit_workflow(f_contrasts=True)
    modelfit.inputs.inputspec.interscan_interval = tr
    modelfit.inputs.inputspec.film_threshold = film_thresh
    modelfit.inputs.inputspec.model_serial_correlations = film_model_autocorr
    modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivs}}

    # Datasink
    datasink = Node(interface=DataSink(), name="datasink")

    # connect nodes / workflows
    wf.connect(datagrabber, 'boldfiles', bet, 'in_file')
    wf.connect(bet, 'out_file', susan, 'in_file')
    wf.connect(susan, 'smoothed_file', bpf, 'in_file')
    wf.connect(bpf, 'out_file', modelspec, 'functional_runs')
    wf.connect(designgen, 'subject_info', modelspec, 'subject_info')
    wf.connect(modelspec, 'session_info', flatten_session_infos, 'nested_list')
    wf.connect(flatten_session_infos, 'flat_list', modelfit, 'inputspec.session_info')
    # wf.connect(modelspec, 'session_info', modelfit, 'inputspec.session_info')
    wf.connect(designgen, 'contrasts', modelfit, 'inputspec.contrasts')
    wf.connect(bpf, 'out_file', modelfit, 'inputspec.functional_data')

    wf.connect(modelfit, 'modelestimate.zfstats', datasink, 'zfstats')

    return wf


if __name__ == '__main__':
    workflow = create_main_wf(test_subs=False)
    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 8})
    # workflow.run()
