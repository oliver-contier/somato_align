#!/usr/bin/env python3

#SBATCH --job-name=subject_melodic
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks=4

from os.path import join as pjoin

from nipype.interfaces.fsl import MELODIC, BET, SUSAN
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Workflow, Node, MapNode


def grab_somato_data(ds_dir='/data/BnB_USER/oliver/somato',
                     condition_names=('D1_D5', 'D5_D1'),
                     testing=True):
    """
    Grab functional data and attached metainfo (subject and run).
    """
    import glob
    import os
    from os.path import join as pjoin
    # subject directories (for development, only pick first two subjects)
    subject_dirs = [fpath for fpath in glob.glob(pjoin(ds_dir, '*'))]
    if testing:
        subject_dirs = subject_dirs[:2]
    subject_ids = [os.path.basename(abf) for abf in subject_dirs]
    # if analvl is group or run, return a simple list of all bold files
    bold_files = [pjoin(subj_dir, cond, 'data.nii.gz')
                  for subj_dir in subject_dirs
                  for cond in condition_names]
    return bold_files, subject_ids, condition_names


def reshape_flist(boldlist_flat, masklist_flat, nconds=2):
    """
    turn list of file names into nested list where each sublist contains all conditions for given subjects.
    Also pick only one mask file (from bet) per condition.
    """
    import numpy as np
    assert len(boldlist_flat) % nconds == 0
    nsubs = int(len(boldlist_flat) / nconds)
    # turn list of bold files into nested list where each sublist has the files for both conditions of a subject.
    flatarr = np.array(boldlist_flat)
    nestarr = flatarr.reshape(nsubs, nconds)
    boldlist_nested = nestarr.tolist()
    # pick one mask file per subject
    masklist_picked = masklist_flat[::nconds]
    return boldlist_nested, masklist_picked


def create_melodic_wf(wf_basedir='/home/homeGlobal/oli/somato/scratch/ica/MELODIC/melodic_wf_workdir',
                      ana_lvl='subject',
                      tr=2.,
                      out_report=True,
                      bet_fracthr=.2,
                      susan_fwhm=2.,
                      susan_brightthresh=1000,
                      melodic_bgthresh=10.):
    # TODO: docstring
    wf = Workflow(name='somato_melodic_wf')
    assert ana_lvl in ['run', 'subject']
    melodic, workdir = None, None

    # datagrabber node
    datagrabber = Node(Function(input_names=[],
                                output_names=['bold_files', 'subject_ids', 'condition_names'],
                                function=grab_somato_data),
                       name='datagrabber')

    # BET node
    bet = MapNode(BET(frac=bet_fracthr, functional=True, mask=True),
                  iterfield=['in_file'], name='bet')

    # SUSAN smoothing node
    susan = MapNode(SUSAN(fwhm=susan_fwhm, brightness_threshold=susan_brightthresh),
                    iterfield=['in_file'], name='susan')

    reshapeflist = Node(Function(input_names=['boldlist_flat', 'masklist_flat'],
                                 output_names=['boldlist_nested', 'masklist_picked'],
                                 function=reshape_flist),
                        name='reshapeflist')

    # construct node or mapnode depending on subject, run, or group level ica
    if ana_lvl == 'subject':
        workdir = pjoin(wf_basedir, 'subject_lvl')
        melodic = MapNode(MELODIC(tr_sec=tr, out_all=True, report=out_report, no_bet=True,
                                  bg_threshold=melodic_bgthresh, approach='concat'),
                          iterfield=['in_files', 'mask'],
                          name='melodic')

    elif ana_lvl == 'run':
        workdir = pjoin(wf_basedir, 'run_lvl')
        melodic = MapNode(MELODIC(tr_sec=tr, out_all=True, report=out_report, no_bet=True,
                                  bg_threshold=melodic_bgthresh),
                          iterfield=['in_files', 'mask'],
                          name='melodic')

    wf.connect(datagrabber, 'bold_files', bet, 'in_file')
    wf.connect(bet, 'out_file', susan, 'in_file')
    if ana_lvl == 'subject':
        wf.connect(susan, 'smoothed_file', reshapeflist, 'boldlist_flat')
        wf.connect(bet, 'mask_file', reshapeflist, 'masklist_flat')
        wf.connect(reshapeflist, 'boldlist_nested', melodic, 'in_files')
        wf.connect(reshapeflist, 'masklist_picked', melodic, 'mask')
    else:
        wf.connect(susan, 'smoothed_file', melodic, 'in_files')
        wf.connect(bet, 'mask_file', melodic, 'mask')

    wf.base_dir = workdir
    return wf


if __name__ == '__main__':
    workflow = create_melodic_wf(ana_lvl='run')
    workflow.run()

    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 4})
