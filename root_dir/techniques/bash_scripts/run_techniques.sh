#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$1"

python root_dir/techniques/"$2".py "$3" "$4" --dataset_filetype 'jpg' --dataset_newtype 'jpg' 2>&1
