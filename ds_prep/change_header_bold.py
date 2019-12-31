#!/usr/bin/env python3


import glob
import os
from os.path import join as pjoin

import nibabel as nib


def get_bold_fnames(ds_dir='/data/project/somato/raw/data/DatenKuehn/',
                    conds=('D1_D5', 'D5_D1', 'blocked_design1', 'blocked_design2')):
    """
    Get file paths for our bold data for a given run.
    returns boldfiles_dict with conditions as keys and filenames per subject as values
    """
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_dir + '/*')
               if '.txt' not in subdir]  # skip readme.txt
    boldfiles_dict = {}
    for cond in conds:
        boldfiles_dict[cond] = [pjoin(ds_dir, sub_id, cond, 'data.nii')
                                for sub_id in sub_ids]
    return boldfiles_dict, sub_ids


def run_change_headers(boldfiles_dict,
                       sub_ids,
                       changes_dict,
                       outdir='/data/project/somato/scratch/dataset'):
    """
    load boldfiles_dict and set some header parameters specified in the changes_dict dict.
    Save these new images in a new dataset under outdir.

    original header entry for pixdim was:
    [-1., 1., 1., 1.0000001, 0., 0., 0., 0.]
    Hence, we should set the fourth entry to 2.0 for TR of 2 secs.
    """
    for subid in sub_ids:
        for cond in boldfiles_dict.keys():
            # if the sub-outdir does already exist, skip
            outsubdir = pjoin(outdir, subid, cond)
            if not os.path.exists(outsubdir):
                print('starting subject ', subid, ' condition ', cond)
                # make specific outdir
                os.makedirs(outsubdir)
                # pick boldfile of given subj and cond
                bold_fname = boldfiles_dict[cond][sub_ids.index(subid)]
                bold_img = nib.load(bold_fname)
                # change header info
                for changes_key in changes_dict.keys():
                    bold_img.header[changes_key] = changes_dict[changes_key]
                # save outfile
                outfile = pjoin(outsubdir, 'data.nii.gz')
                nib.save(bold_img, outfile)
            else:
                print('already exists ', subid, ' ', cond)
    return None


if __name__ == '__main__':
    # only change pixdims
    changesdict = {'pixdim': [-1., 1., 1., 1., 2., 0., 0., 0.]}
    boldfilesdict, sub_ids = get_bold_fnames()
    run_change_headers(boldfilesdict, sub_ids, changesdict)
