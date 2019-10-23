#!/usr/bin/env python3

"""

"""

from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl import BET, SUSAN
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.workflows.fmri.fsl import create_modelfit_workflow


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
                             dur_per_digit=5.12):
    from nipype.interfaces.base import Bunch
    cycle_dur = dur_per_digit * 5
    conditions = ['D_%i' % i for i in range(1, 6)]
    onsets = [[0 + (digit_idx * dur_per_digit) + (cycle_idx * cycle_dur)
               for cycle_idx in range(n_cycles)]
              for digit_idx in range(5)]
    durations = [[dur_per_digit] * n_cycles for _ in range(5)]
    subject_info = Bunch(conditions=conditions, onsets=onsets, durations=durations)
    # t-cotrasts
    t_contrasts = []
    for cond in conditions:
        contrast_vector = [0, 0, 0, 0]
        contrast_vector.insert(conditions.index(cond), 1)
        t_contrasts.append(['tcon_%s' % cond, 'T', conditions, contrast_vector])
    # f-contrast over all t-contrasts
    f_contrast = [['All_Digits', 'F', t_contrasts]]
    contrasts = t_contrasts + f_contrast
    return subject_info, contrasts


def create_main_wf(wf_workdir='/data/BnB_USER/oliver/somato/scratch/roi_glm',
                   dsdir='/data/BnB_USER/oliver/somato/data',
                   condid='D1_D5',
                   test_subs=False,
                   tr=2.,
                   bet_fracthr=.2,
                   susan_fwhm=2,
                   susan_brightthresh=1000,
                   hp_vols=30.,
                   lp_vols=2.,
                   film_thresh=1000.,  # TODO: what's a good film threshold? model autocorrelations? include derivative?
                   film_model_autocorr=False,
                   use_derivs=False):
    """
    Generate analysis pipeline for pre-srm GLM (and possibly later append with real srm)
    """

    wf = Workflow(name='somato_melodic_wf')
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
    designgen = Node(Function(input_names=[],
                              output_names=['subject_info', 'contrasts'],
                              function=make_bunch_and_contrasts),
                     name='designgen')

    # creates (intermediate) fsl compatible design file 'session_info' for modelfit
    modelspec = MapNode(SpecifyModel(input_units='secs'), name='modelspec', iterfield=['functional_runs'])
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = tr
    modelspec.inputs.high_pass_filter_cutoff = hp_vols * tr

    # Modelfit is an extra built-in workflow
    modelfit = create_modelfit_workflow(f_contrasts=True)
    modelfit.inputs.inputspec.interscan_interval = tr
    modelfit.inputs.inputspec.film_threshold = film_thresh
    modelfit.inputs.inputspec.model_serial_correlations = film_model_autocorr
    modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivs}}

    # connect nodes / workflows
    wf.connect(datagrabber, 'boldfiles', bet, 'in_file')
    wf.connect(bet, 'out_file', susan, 'in_file')
    wf.connect(susan, 'smoothed_file', bpf, 'in_file')
    wf.connect(bpf, 'out_file', modelspec, 'functional_runs')
    wf.connect(designgen, 'subject_info', modelspec, 'subject_info')
    wf.connect(modelspec, 'session_info', modelfit, 'inputspec.session_info')
    wf.connect(designgen, 'contrasts', modelfit, 'inputspec.contrasts')
    wf.connect(bpf, 'out_file', modelfit, 'inputspec.functional_data')

    return wf


if __name__ == '__main__':
    workflow = create_main_wf(test_subs=False)
    # workflow.run(plugin='MultiProc', plugin_args={'n_procs': 4})
    workflow.run()
