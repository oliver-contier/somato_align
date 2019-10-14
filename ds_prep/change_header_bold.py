#!/usr/bin/env python3


import glob
import os
from os.path import join as pjoin

import nibabel as nib


def get_bold_fnames(ds_dir='/data/BnB_USER/oliver/somato'):
    """
    Get file paths for our bold data for a given run.
    """
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_dir + '/*')]
    boldfiles = [pjoin(ds_dir, sub_id, cond_id, 'data.nii.gz')
                 for sub_id in sub_ids
                 for cond_id in ['D1_D5', 'D5_D1']]
    return boldfiles


def change_headers(niftis, changes,
                   outdir='/home/homeGlobal/oli/somato/scratch/dataset'):
    """
    load niftis and set some header parameters specified in the changes dict.
    Save these new images in a new dataset under outdir.
    original header entry for pixdim was:
    [-1., 1., 1., 1.0000001, 0., 0., 0., 0.]
    Hence, we should set the fourth entry to 2000. ms
    """
    for nifti_fname in niftis:
        print('starting with : ', nifti_fname)
        img = nib.load(nifti_fname)
        for changes_key in changes:
            img.header[changes_key] = changes[changes_key]
        outfile = nifti_fname.replace('/data/BnB_USER/oliver/somato', outdir)
        out_subdir = outfile.replace('/data.nii.gz', '')
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir)
        nib.save(img, outfile)
    return None


if __name__ == '__main__':
    # only change pixdims
    changesdict = {'pixdim': [-1., 1., 1., 1., 2., 0., 0., 0.]}
    bold_files = get_bold_fnames()
    change_headers(bold_files, changesdict)
