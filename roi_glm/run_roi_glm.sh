#!/usr/bin/env bash

source /data/project/somato/raw/venvs/somato_env/bin/activate
source /etc/fsl/fsl.sh

python roi_glm_ffx.py $1
