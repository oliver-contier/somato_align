#!/usr/bin/env python3

"""

"""
import os

from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl import BET, SUSAN
from nipype.interfaces.fsl.maths import TemporalFilter, MathsCommand
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.workflows.fmri.fsl import create_fixed_effects_flow
from nipype.workflows.fmri.fsl import create_modelfit_workflow


def grab_bold_data(cond_id='D1_D5',
                   ds_dir='/data/BnB_USER/oliver/somato/scratch/dataset',
                   testsubs=False):
    """
    Get file paths for our bold data for a given run.
    """
    import os
    import glob
    from os.path import join as pjoin
    assert os.path.exists(ds_dir)
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_dir + '/*')]
    if testsubs:
        sub_ids = sub_ids[:testsubs]
    if cond_id == 'both':
        boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
                     for sub_id in sub_ids
                     for cond_id in ['D1_D5', 'D5_D1']]
        # yields [sub1-cond1, sub1-cond2, ...]
    else:
        boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
                     for sub_id in sub_ids]

    return boldfiles, sub_ids


def make_bunch_and_contrasts(n_cycles=20,
                             dur_per_digit=5.12,
                             n_subs=12,
                             subtractive_contrast=False,
                             cond_id='D1_D5'):
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
            if conditions.index(cond) == len(cond) - 1:
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


def nest_copes_varcopes_doffiles(copes_flat, varcopes_flat, doffiles_flat):
    copes_nested = [[copes_flat[idx], copes_flat[idx + 1]] for idx in range(0, len(copes_flat), 2)]
    varcopes_nested = [[varcopes_flat[idx], varcopes_flat[idx + 1]] for idx in range(0, len(varcopes_flat), 2)]
    doffiles_nested = [[doffiles_flat[idx], doffiles_flat[idx + 1]] for idx in range(0, len(doffiles_flat), 2)]
    return copes_nested, varcopes_nested, doffiles_nested


def subject_ffx_wfrun(subject_id,
                      copes,
                      varcopes,
                      dof_files):
    fixedfx = create_fixed_effects_flow(name='fixedfx_sub-%i' % subject_id)
    fixedfx.inputspec.copes = copes
    fixedfx.inputspec.varcopes = varcopes
    fixedfx.inputspec.dof_files = dof_files
    fixedfx.run()
    return fixedfx


def create_main_wf(wf_workdir='/data/BnB_USER/oliver/somato/scratch/roi_glm',
                   wf_datasink_dir='/data/BnB_USER/oliver/somato/scratch/roi_glm/results/hemi_onesample_run-1',
                   dsdir='/data/BnB_USER/oliver/somato/scratch/dataset',
                   condid='D1_D5',
                   test_subs=False,
                   tr=2.,
                   bet_fracthr=.2,
                   spatial_fwhm=2,
                   susan_brightthresh=1000,
                   hp_vols=30.,
                   lp_vols=2.,
                   remove_hemi='r',
                   film_thresh=.001,
                   film_model_autocorr=True,
                   use_derivs=True,
                   tcon_subtractive=False):
    """
    Generate analysis pipeline for pre-srm GLM (and possibly later append with real srm)

    # TODO: what's a good film threshold?
    filmgls threshold: nipype default is 1000. However, since this is applied on already heavily filtered data here,
    everything above 0.01 cuts away lots of grey matter voxels.

    # TODO: remove non-stimulated hemisphere
    fslmaths command for cutting away right hemisphere:
    fslmaths data_brain.nii.gz -roi 96 -1 0 -1 0 -1 0 -1 data_brain_roi.nii.gz
    """

    # make work and res dir if necessary
    for target_dir in [wf_workdir, wf_datasink_dir]:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    wf = Workflow(name='somato_roi_glm_wf')
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
    susan = MapNode(SUSAN(fwhm=spatial_fwhm, brightness_threshold=susan_brightthresh),
                    iterfield=['in_file'], name='susan')

    # bandpass filter node
    bpf = MapNode(TemporalFilter(highpass_sigma=hp_vols / 2.3548,
                                 lowpass_sigma=lp_vols / 2.3548),
                  iterfield=['in_file'], name='bpf')

    # cut away hemisphere node
    cut_hemi = MapNode(MathsCommand(), iterfield=['in_file'], name='cut_hemi')
    if remove_hemi == 'r':
        roi_args = '-roi 96 -1 0 -1 0 -1 0 -1'
    elif remove_hemi == 'l':
        roi_args = '-roi 0 96 0 -1 0 -1 0 -1'
    else:
        raise IOError('did not recognite value of remove_hemi %s' % remove_hemi)
    cut_hemi.inputs.args = roi_args

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

    if condid == 'both':
        # make list of result files a nested list where each sublist contains both runs for one subjects
        nest_l1_output = Node(Function(input_names=['copes_flat', 'varcopes_flat', 'doffiles_flat'],
                                       output_names=['copes_nested', 'varcopes_nested', 'doffiles_nested'],
                                       function=nest_copes_varcopes_doffiles),
                              name='nest_l1_output')
        # Set up one ffx node per subject
        ffx_mapnode = MapNode(Function(input_names=['subject_id', 'copes', 'varcopes', 'dof_files'],
                                       output_names=['fixedfx'], function=subject_ffx_wfrun),
                              name='ffx_mapnode',
                              iterfield=['subject_id', 'copes', 'varcopes', 'dof_files'])
        # TODO: generate matching contrasts if both runs

    # TODO: For SRM: node to select above threshold voxels as numpy arrays,
    #  while remembering the voxel indices to be able to reconvert back to original space later on.

    """
    Connect preproc and 1st lvl nodes
    """
    # connect nodes / workflows
    wf.connect(datagrabber, 'boldfiles', bet, 'in_file')
    wf.connect(bet, 'out_file', susan, 'in_file')
    wf.connect(susan, 'smoothed_file', bpf, 'in_file')
    wf.connect(bpf, 'out_file', cut_hemi, 'in_file')
    wf.connect(cut_hemi, 'out_file', modelspec, 'functional_runs')
    wf.connect(designgen, 'subject_info', modelspec, 'subject_info')
    wf.connect(modelspec, 'session_info', flatten_session_infos, 'nested_list')
    wf.connect(flatten_session_infos, 'flat_list', modelfit, 'inputspec.session_info')
    wf.connect(designgen, 'contrasts', modelfit, 'inputspec.contrasts')
    wf.connect(cut_hemi, 'out_file', modelfit, 'inputspec.functional_data')

    """
    connect to datasink
    """
    # Datasink node
    datasink = Node(interface=DataSink(), name="datasink")
    datasink.inputs.base_directory = wf_datasink_dir

    # 1st level output
    wf.connect(modelfit.get_node('modelgen'), 'design_image', datasink, 'design_image')
    wf.connect(modelfit.get_node('modelestimate'), 'zfstats', datasink, 'zfstats')
    wf.connect(modelfit.get_node('modelestimate'), 'thresholdac', datasink, 'thresholdac')
    wf.connect(modelfit, 'outputspec.zfiles', datasink, 'zfiles')
    wf.connect(modelfit, 'outputspec.pfiles', datasink, 'pfiles')
    # TODO: 2nd level output

    return wf


if __name__ == '__main__':
    workflow = create_main_wf()
    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 8})
    # workflow.run()
