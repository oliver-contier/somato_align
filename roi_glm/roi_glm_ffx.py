#!/usr/bin/env python3

"""
# TODO: description
"""


def grab_subject_ids(ds_dir='/data/project/somato/scratch/dataset',
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
                          film_thresh, film_model_autocorr, use_derivs, tr, tcon_subtractive, cluster_threshold,
                          cluster_thresh_frac, cluster_p, dilate_clusters_voxel, cond_ids, dsdir, work_basedir):
    # todo: new mapnode inputs: cluster_threshold, cluster_p
    """
    Make a workflow including preprocessing, first level, and second level GLM analysis for a given subject.
    This pipeline includes:
    - skull stripping
    - spatial smoothing
    - removing the irrelevant hemisphere
    - temporal band pass filter
    - 1st level GLM
    - averaging f-contrasts from 1st level GLM
    - clustering run-wise f-tests, dilating clusters, and returning binary roi mask
    """

    from nipype.algorithms.modelgen import SpecifyModel
    from nipype.interfaces.fsl import BET, SUSAN, ImageMaths
    from nipype.interfaces.fsl.model import SmoothEstimate, Cluster
    from nipype.interfaces.fsl.maths import TemporalFilter, MathsCommand
    from nipype.interfaces.utility import Function
    from nipype.pipeline.engine import Workflow, Node, MapNode
    from nipype.workflows.fmri.fsl import create_modelfit_workflow
    from nipype.interfaces.fsl.maths import MultiImageMaths
    from nipype.interfaces.utility import IdentityInterface
    import sys
    from os.path import join as pjoin
    import os
    sys.path.insert(0, "/data/project/somato/raw/code/roi_glm/custom_node_functions.py")
    # TODO: don't hardcode this
    import custom_node_functions

    # set up sub-workflow
    sub_wf = Workflow(name='subject_%s_wf' % sub_id)
    # set up sub-working-directory
    subwf_wd = pjoin(work_basedir, 'subject_ffx_wfs', 'subject_%s_ffx_workdir' % sub_id)
    if not os.path.exists(subwf_wd):
        os.makedirs(subwf_wd)
    sub_wf.base_dir = subwf_wd

    # Grab bold files for all four runs of one subject.
    # in the order [d1_d5, d5_d1, blocked_design1, blocked_design2]
    grab_boldfiles = Node(Function(function=custom_node_functions.grab_boldfiles_subject,
                                   input_names=['sub_id', 'cond_ids', 'ds_dir'], output_names=['boldfiles']),
                          name='grab_boldfiles')
    grab_boldfiles.inputs.sub_id = sub_id
    grab_boldfiles.inputs.cond_ids = cond_ids
    grab_boldfiles.inputs.ds_dir = dsdir

    getonsets = Node(Function(function=custom_node_functions.grab_blocked_design_onsets_subject,
                              input_names=['sub_id', 'prepped_ds_dir'],
                              output_names=['blocked_design_onsets_dicts']),
                     name='getonsets')
    getonsets.inputs.sub_id = sub_id
    getonsets.inputs.prepped_ds_dir = dsdir

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
    if remove_hemi == 'r':
        roi_args = '-roi 96 -1 0 -1 0 -1 0 -1'
    elif remove_hemi == 'l':
        roi_args = '-roi 0 96 0 -1 0 -1 0 -1'
    else:
        raise IOError('did not recognite value of remove_hemi %s' % remove_hemi)

    cut_hemi_func = MapNode(MathsCommand(), iterfield=['in_file'], name='cut_hemi_func')
    cut_hemi_func.inputs.args = roi_args

    cut_hemi_mask = MapNode(MathsCommand(), iterfield=['in_file'], name='cut_hemi_mask')
    cut_hemi_mask.inputs.args = roi_args

    # Make Design and Contrasts for that subject
    # subject_info ist a list of two "Bunches", each for one run, containing conditions, onsets, durations
    designgen = Node(Function(input_names=['subtractive_contrast', 'blocked_design_onsets_dicts'],
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

    # average zfstats from both runs
    split_zfstats = Node(Function(function=custom_node_functions.split_zfstats_runs,
                                  input_names=['zfstats_list'],
                                  output_names=['zfstat_run1', 'zfstat_run2']),
                         name='split_zfstats')
    average_zfstats = Node(MultiImageMaths(op_string='-add %s -div 2'), name='mean_images')

    # estimate smoothness of 1st lvl zf-files
    smoothest = MapNode(SmoothEstimate(), name='smoothest', iterfield=['mask_file', 'zstat_file'])

    cluster = MapNode(Cluster(), name='cluster',
                      iterfield=['in_file', 'volume', 'dlh'])
    cluster.inputs.threshold = cluster_threshold
    cluster.inputs.pthreshold = cluster_p
    cluster.inputs.fractional = cluster_thresh_frac
    cluster.inputs.no_table = True
    cluster.inputs.out_threshold_file = True
    cluster.inputs.out_pval_file = True
    cluster.inputs.out_localmax_vol_file = True
    cluster.inputs.out_max_file = True
    cluster.inputs.out_size_file = True

    # dilate clusters
    dilate = MapNode(MathsCommand(args='-kernel sphere %i -dilD' % dilate_clusters_voxel),
                     iterfield=['in_file'], name='dilate')

    # binarize the result to a mask
    binarize_roi = MapNode(ImageMaths(op_string='-nan -thr 0.001 -bin'),
                           iterfield=['in_file'], name='binarize_roi')

    # connect preprocessing
    sub_wf.connect(grab_boldfiles, 'boldfiles', bet, 'in_file')
    sub_wf.connect(bet, 'out_file', susan, 'in_file')
    sub_wf.connect(susan, 'smoothed_file', bpf, 'in_file')
    sub_wf.connect(bpf, 'out_file', cut_hemi_func, 'in_file')
    sub_wf.connect(bet, 'mask_file', cut_hemi_mask, 'in_file')
    # connect to 1st level model
    sub_wf.connect(cut_hemi_func, 'out_file', modelspec, 'functional_runs')
    sub_wf.connect(getonsets, 'blocked_design_onsets_dicts', designgen, 'blocked_design_onsets_dicts')
    sub_wf.connect(designgen, 'subject_info', modelspec, 'subject_info')
    sub_wf.connect(modelspec, 'session_info', flatten_session_infos, 'nested_list')
    sub_wf.connect(flatten_session_infos, 'flat_list', modelfit, 'inputspec.session_info')
    sub_wf.connect(designgen, 'contrasts', modelfit, 'inputspec.contrasts')
    sub_wf.connect(cut_hemi_func, 'out_file', modelfit, 'inputspec.functional_data')
    # connect to cluster thresholding
    sub_wf.connect(cut_hemi_mask, 'out_file', smoothest, 'mask_file')
    sub_wf.connect(modelfit.get_node('modelestimate'), 'zfstats', smoothest, 'zstat_file')
    sub_wf.connect(modelfit.get_node('modelestimate'), 'zfstats', cluster, 'in_file')
    sub_wf.connect(smoothest, 'dlh', cluster, 'dlh')
    sub_wf.connect(smoothest, 'volume', cluster, 'volume')
    sub_wf.connect(cluster, 'threshold_file', dilate, 'in_file')
    sub_wf.connect(dilate, 'out_file', binarize_roi, 'in_file')
    # connect to averaging f-contrasts
    sub_wf.connect(modelfit.get_node('modelestimate'), 'zfstats', split_zfstats, 'zfstats_list')
    sub_wf.connect(split_zfstats, 'zfstat_run1', average_zfstats, 'in_file')
    sub_wf.connect(split_zfstats, 'zfstat_run2', average_zfstats, 'operand_files')
    # redirect to outputspec
    # TODO: redirekt outputspec to datasink in meta-wf
    outputspec = Node(IdentityInterface(
        fields=['threshold_file', 'index_file', 'pval_file', 'localmax_txt_file']), name='outputspec')
    sub_wf.connect(cluster, 'threshold_file', outputspec, 'threshold_file')
    sub_wf.connect(cluster, 'index_file', outputspec, 'index_file')
    sub_wf.connect(cluster, 'pval_file', outputspec, 'pval_file')
    sub_wf.connect(cluster, 'localmax_txt_file', outputspec, 'localmax_txt_file')
    sub_wf.connect(binarize_roi, 'out_file', outputspec, 'roi')

    # run subject-lvl workflow
    sub_wf.write_graph(graph2use='colored', dotfilename='./subwf_graph.dot')
    # sub_wf.run(plugin='MultiProc', plugin_args={'n_procs': 6})
    # sub_wf.run(plugin='CondorDAGMan')
    sub_wf.run()

    return sub_wf


if __name__ == '__main__':
    import sys

    # catch subject id passed in through runscript
    subidx = int(sys.argv[1])
    subids = grab_subject_ids(ds_dir='/data/project/somato/scratch/dataset', testsubs=False)
    subid = subids[subidx]

    # generate workflow
    subwf = create_subject_ffx_wf(sub_id=subid,
                                  bet_fracthr=.2,
                                  spatial_fwhm=2,
                                  susan_brightthresh=1000,
                                  hp_vols=30.,
                                  lp_vols=2.,
                                  remove_hemi='r',
                                  film_thresh=.001,
                                  film_model_autocorr=True,
                                  use_derivs=False,
                                  tr=2.,
                                  tcon_subtractive=False,
                                  cluster_threshold=3.,
                                  cluster_thresh_frac=True,
                                  cluster_p=.001,
                                  dilate_clusters_voxel=2,
                                  cond_ids=('D1_D5', 'D5_D1', 'blocked_design1', 'blocked_design2'),
                                  dsdir='/data/project/somato/scratch/dataset',
                                  work_basedir='/data/project/somato/scratch/roi_glm/work_basedir')

    # run
    subwf.write_graph(graph2use='colored', dotfilename='./groupwf_graph.dot')
    # subwf.run()
    subwf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
