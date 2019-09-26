#!/usr/bin/env python3

from os.path import join as pjoin

from nipype.interfaces.fsl import MELODIC, BET, SUSAN
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode


def grab_somato_data(ds_dir='/data/BnB_USER/oliver/somato',
                     condition_names=('D1_D5', 'D5_D1'),
                     testing=True):
    """
    Grab functional data and attached metainfo (subject and run)
    """
    import glob
    import os
    from os.path import join as pjoin

    subject_dirs = [fpath for fpath in glob.glob(pjoin(ds_dir, '*'))]
    if testing:
        subject_dirs = subject_dirs[:2]
    subject_ids = [os.path.basename(abf) for abf in subject_dirs]
    bold_files = [pjoin(subj_dir, cond, 'data.nii.gz')
                  for subj_dir in subject_dirs
                  for cond in condition_names]
    # make sure all path names are correct
    for bf in bold_files:
        assert os.path.exists(bf)

    return bold_files, subject_ids, condition_names


def create_melodic_wf(wf_basedir='/home/homeGlobal/oli/somato/scratch/ica/MELODIC/melodic_wf_workdir',
                      group_level=False,
                      varnorm=True,
                      seperate_whitening=True,
                      seperate_varnorm=True,
                      tr=2.,
                      allstats=True,
                      outmean=True,
                      report_=True,
                      bet_fracthr=.2,
                      susan_fwhm=1.,
                      susan_brightthresh=1000,
                      melodic_bgthresh=10.):
    wf = Workflow(name='somato_melodic_wf')

    # datagrabber node
    datagrabber = Node(Function(input_names=[],
                                output_names=['bold_files', 'subject_ids', 'condition_names'],
                                function=grab_somato_data),
                       name='datagrabber')

    # construct node or mapnode depending on subject or group level ica
    if group_level:
        workdir = pjoin(wf_basedir, 'group_lvl')
        melodic = Node(MELODIC(sep_whiten=seperate_whitening, sep_vn=seperate_varnorm,
                               var_norm=varnorm, tr_sec=tr, out_stats=allstats, out_pca=True, out_mean=outmean,
                               report=report_, no_bet=True, bg_threshold=melodic_bgthresh),
                       iterfield=['in_files'],
                       name='melodic')
    else:
        workdir = pjoin(wf_basedir, 'subject_lvl')
        melodic = MapNode(MELODIC(var_norm=varnorm, tr_sec=tr, out_stats=allstats, out_pca=True, out_mean=outmean,
                                  report=report_, no_bet=True, bg_threshold=melodic_bgthresh, approach='tica'),
                          iterfield=['in_files'],
                          name='melodic')

    # BET node
    bet = MapNode(BET(frac=bet_fracthr, functional=True, mask=True),
                  iterfield=['in_file'], name='bet')

    # SUSAN smoothing node
    susan = MapNode(SUSAN(fwhm=susan_fwhm, brightness_threshold=susan_brightthresh),
                    iterfield=['in_file'], name='susan')

    wf.connect(datagrabber, 'bold_files', bet, 'in_file')
    wf.connect(bet, 'out_file', susan, 'in_file')
    wf.connect(bet, 'mask_file', melodic, 'mask')
    wf.connect(susan, 'smoothed_file', melodic, 'in_files')

    wf.base_dir = workdir
    return wf


if __name__ == '__main__':
    workflow = create_melodic_wf(group_level=False)
    workflow.run()

    # workflow.run(plugin='MultiProc', plugin_args={'n_procs': 2})
