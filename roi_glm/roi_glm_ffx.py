#!/usr/bin/env python3

"""
# TODO: description
"""


def grab_subject_ids(ds_dir='/data/BnB_USER/oliver/somato/scratch/dataset',
                     testsubs=False):
    """
    Get list of all subject IDs.
    """
    import os
    import glob
    sub_ids = [os.path.basename(subdir) for subdir in glob.glob(ds_dir + '/*')]
    if testsubs:
        sub_ids = sub_ids[:testsubs]
    return sub_ids


def create_subject_ffx_wf(sub_id, bet_fracthr, spatial_fwhm, susan_brightthresh, hp_vols, lp_vols, remove_hemi,
                          film_thresh, film_model_autocorr, use_derivs, tr, tcon_subtractive, cond_ids, dsdir,
                          meta_wf_workdir):
    """
    Make a workflow including preprocessing, first level, and second level GLM analysis for a given subject.
    This pipeline includes:
    - skull stripping
    - spatial smoothing
    - removing the irrelevant hemisphere
    - temporal band pass filter
    - 1st level GLM
    - 2nd level GLM
    - averaging f-contrasts from 1st level GLM
    """

    from nipype.algorithms.modelgen import SpecifyModel
    from nipype.interfaces.fsl import BET, SUSAN
    from nipype.interfaces.fsl.maths import TemporalFilter, MathsCommand
    from nipype.interfaces.utility import Function
    from nipype.pipeline.engine import Workflow, Node, MapNode
    from nipype.workflows.fmri.fsl import create_modelfit_workflow, create_fixed_effects_flow
    from nipype.interfaces.fsl.maths import MultiImageMaths
    import sys
    from os.path import join as pjoin
    import os
    sys.path.insert(0, "/data/BnB_USER/oliver/somato/raw/code/roi_glm/custom_node_functions.py")
    # TODO: don't hardcode this
    import custom_node_functions

    # set up sub-workflow
    sub_wf = Workflow(name='subject_%s_wf' % sub_id)
    # set up sub-working-directory
    subwf_wd = pjoin(meta_wf_workdir, 'subject_ffx_wfs', 'subject_%s_ffx_workdir' % sub_id)
    if not os.path.exists(subwf_wd):
        os.makedirs(subwf_wd)
    sub_wf.base_dir = subwf_wd

    # Grab both runs of one subject
    grab_boldfiles = Node(Function(function=custom_node_functions.grab_boldfiles_subject,
                                   input_names=['sub_id', 'cond_ids', 'ds_dir'], output_names=['boldfiles']),
                          name='grab_boldfiles')
    grab_boldfiles.inputs.sub_id = sub_id
    grab_boldfiles.inputs.cond_ids = cond_ids
    grab_boldfiles.inputs.ds_dir = dsdir

    # pass bold files through preprocessing pipeline
    bet = MapNode(BET(frac=bet_fracthr, functional=True, mask=True),
                  iterfield=['in_file'], name='bet')

    pick_mask = Node(Function(function=custom_node_functions.pick_first_mask,
                              input_names=['mask_files'], output_names=['first_mask']), name='pick_mask')

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

    # Make Design and Contrasts for that subject
    # subject_info ist a list of two "Bunches", each for one run, containing conditions, onsets, durations
    designgen = Node(Function(input_names=['subtractive_contrast'],
                              output_names=['subject_info', 'contrasts'],
                              function=custom_node_functions.make_bunch_and_contrasts),
                     name='designgen')
    designgen.inputs.subtractive_contrasts = tcon_subtractive

    # create 'session_info' for modelfit
    modelspec = MapNode(SpecifyModel(input_units='secs'), name='modelspec',
                        iterfield=['functional_runs', 'subject_info'])
    modelspec.inputs.high_pass_filter_cutoff = hp_vols * tr
    modelspec.inputs.time_repetition = tr

    flatten_session_infos = Node(Function(input_names=['nested_list'], output_names=['flat_list'],
                                          function=custom_node_functions.flatten_nested_list),
                                 name='flatten_session_infos')

    # Fist-level workflow
    modelfit = create_modelfit_workflow(f_contrasts=True)
    modelfit.inputs.inputspec.interscan_interval = tr
    modelfit.inputs.inputspec.film_threshold = film_thresh
    modelfit.inputs.inputspec.model_serial_correlations = film_model_autocorr
    modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivs}}

    # node that reshapes list of copes returned from modelfit
    cope_sorter = Node(Function(input_names=['copes', 'varcopes', 'contrasts'],
                                output_names=['copes', 'varcopes', 'n_runs'],
                                function=custom_node_functions.sort_copes),
                       name='cope_sorter')

    # Second-level workflow.
    # Problem: Doesn't propagate F-Contrasts from first level ...
    fixedfx = create_fixed_effects_flow()
    fixedfx.get_node('l2model').inputs.num_copes = 2  # TODO: don't hardcode

    # In addition to 2nd level glm, just average zfstats from both runs
    split_zfstats = Node(Function(function=custom_node_functions.split_zfstats_runs,
                                  input_names=['zfstats_list'],
                                  output_names=['zfstat_run1', 'zfstat_run2']),
                         name='split_zfstats')

    average_zfstats = Node(MultiImageMaths(op_string='-add %s -div 2'), name='mean_images')

    # connect preprocessing
    sub_wf.connect(grab_boldfiles, 'boldfiles', bet, 'in_file')
    sub_wf.connect(bet, 'out_file', susan, 'in_file')
    sub_wf.connect(susan, 'smoothed_file', bpf, 'in_file')
    sub_wf.connect(bpf, 'out_file', cut_hemi, 'in_file')
    # connect to 1st level model
    sub_wf.connect(cut_hemi, 'out_file', modelspec, 'functional_runs')
    sub_wf.connect(designgen, 'subject_info', modelspec, 'subject_info')
    sub_wf.connect(modelspec, 'session_info', flatten_session_infos, 'nested_list')
    sub_wf.connect(flatten_session_infos, 'flat_list', modelfit, 'inputspec.session_info')
    sub_wf.connect(designgen, 'contrasts', modelfit, 'inputspec.contrasts')
    sub_wf.connect(cut_hemi, 'out_file', modelfit, 'inputspec.functional_data')
    # connect to averaging f-contrasts
    sub_wf.connect(modelfit.get_node('modelestimate'), 'zfstats', split_zfstats, 'zfstats_list')
    sub_wf.connect(split_zfstats, 'zfstat_run1', average_zfstats, 'in_file')
    sub_wf.connect(split_zfstats, 'zfstat_run2', average_zfstats, 'operand_files')

    # connect l1 to l2
    sub_wf.connect(designgen, 'contrasts', cope_sorter, 'contrasts')
    sub_wf.connect(modelfit, 'outputspec.copes', cope_sorter, 'copes')
    sub_wf.connect(modelfit, 'outputspec.varcopes', cope_sorter, 'varcopes')
    sub_wf.connect(cope_sorter, 'copes', fixedfx, 'inputspec.copes')
    sub_wf.connect(cope_sorter, 'varcopes', fixedfx, 'inputspec.varcopes')
    sub_wf.connect(modelfit, 'outputspec.dof_file', fixedfx, 'inputspec.dof_files')
    sub_wf.connect(cope_sorter, 'n_runs', fixedfx.get_node('l2model'), 'num_copes')
    sub_wf.connect(bet, 'mask_file', pick_mask, 'mask_files')
    sub_wf.connect(pick_mask, 'first_mask', fixedfx.get_node('flameo'), 'mask_file')

    sub_wf.write_graph(graph2use='colored', dotfilename='./sub_graph_colored.dot')
    # sub_wf.run(plugin='MultiProc', plugin_args={'n_procs': 2})
    # sub_wf.run(plugin='SLURM')
    sub_wf.run()
    return sub_wf


def create_group_wf(wf_workdir='/data/BnB_USER/oliver/somato/scratch/roi_glm/workdirs/',
                    wf_datasink_dir='/data/BnB_USER/oliver/somato/scratch/roi_glm/results/hemi_onesample_run-1',
                    dsdir='/data/BnB_USER/oliver/somato/scratch/dataset',
                    test_subs=False,
                    cond_ids=('D1_D5', 'D5_D1'),
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
    Create meta-workflow, which executes a MapNode iterating over each subject-specific analysis pipeline.

    ### Notes ###
    filmgls threshold: nipype default is 1000. However, since this is applied on already heavily filtered data here,
    everything above 0.01 cuts away lots of grey matter voxels.
    """

    import os
    from nipype.interfaces.utility import Function
    from nipype.pipeline.engine import Workflow, Node, MapNode

    # is data dir correct
    assert os.path.exists(dsdir)
    # make work and res dir if necessary
    for target_dir in [wf_workdir, wf_datasink_dir]:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    # initiate meta-workflow
    wf = Workflow(name='somato_roi_glm_wf')
    wf.base_dir = wf_workdir

    # node grabbing list of subject names
    subj_grabber = Node(Function(function=grab_subject_ids,
                                 input_names=['ds_dir', 'testsubs'],
                                 output_names=['subject_ids']),
                        name='subj_grabber')
    subj_grabber.inputs.ds_dir = dsdir
    subj_grabber.inputs.testsubs = test_subs

    # map node performing subject-specific analysis pipelinde
    sub_ffx = MapNode(
        Function(function=create_subject_ffx_wf,
                 inputs=['sub_id', 'bet_fracthr', 'spatial_fwhm', 'susan_brightthresh', 'hp_vols', 'lp_vols',
                         'remove_hemi', 'film_thresh', 'film_model_autocorr', 'use_derivs', 'tr', 'tcon_subtractive',
                         'cond_ids', 'dsdir', 'meta_wf_workdir'],
                 outputs=['sub_wf']),
        iterfield=['sub_id'],
        name='subject_ffx_mapnode')
    sub_ffx.inputs.bet_fracthr = bet_fracthr
    sub_ffx.inputs.spatial_fwhm = spatial_fwhm
    sub_ffx.inputs.susan_brightthresh = susan_brightthresh
    sub_ffx.inputs.hp_vols = hp_vols
    sub_ffx.inputs.lp_vols = lp_vols
    sub_ffx.inputs.remove_hemi = remove_hemi
    sub_ffx.inputs.film_thresh = film_thresh
    sub_ffx.inputs.film_model_autocorr = film_model_autocorr
    sub_ffx.inputs.use_derivs = use_derivs
    sub_ffx.inputs.tr = tr
    sub_ffx.inputs.tcon_subtractive = tcon_subtractive
    sub_ffx.inputs.cond_ids = cond_ids
    sub_ffx.inputs.dsdir = dsdir
    sub_ffx.inputs.meta_wf_workdir = wf_workdir

    wf.connect(subj_grabber, 'subject_ids', sub_ffx, 'sub_id')

    return wf


if __name__ == '__main__':
    workflow = create_group_wf()
    # workflow.write_graph(graph2use='colored', dotfilename='./graph_colored.dot')
    # workflow.run(plugin='MultiProc', plugin_args={'n_procs': 2})
    # workflow.run(plugin='SLURM')
    workflow.run()
