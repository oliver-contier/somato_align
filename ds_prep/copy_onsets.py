#!/usr/bin/env python3

import glob
from os.path import join as pjoin
import os
from shutil import copyfile


def run_copy_onsets(onsets_orig_dir='/data/project/somato/raw/onsets_from_esther',
                    conds=('blocked_design1', 'blocked_design2'),
                    ds_prepped_dir='/data/project/somato/scratch/dataset'):
    """
    Copy onsets and duration files systematically into the prepped dataset directory.
    """
    sub_ids_nosuff = [fullpath.split('/')[-1]
                      for fullpath in glob.glob(onsets_orig_dir + '/*')]
    sub_ids = [os.path.basename(subdir)
               for subdir in glob.glob(ds_prepped_dir + '/*')
               if '.txt' not in subdir]  # skip readme.txt

    for subid_nosuff, subid in zip(sub_ids_nosuff, sub_ids):
        for cond in conds:
            for digit_str in ['D%i' % i for i in range(1, 6)]:
                for ftype in ['.ons', '.dur']:
                    source = pjoin(onsets_orig_dir, subid_nosuff, cond, digit_str + ftype)
                    dest = pjoin(ds_prepped_dir, subid, cond, digit_str + ftype)
                    copyfile(source, dest)
    return None


if __name__ == '__main__':
    run_copy_onsets()
