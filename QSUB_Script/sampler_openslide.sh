#! /bin/bash
#$ -q lg-mem
#$ -l h_vmem=50G
#$ -M gu.qiangqiang@mayo.edu
#$ -t 1-5:1
#$ -m abe
#$ -V
#$ -cwd
#$ -j y
#$ -o /path/log
set -x
dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy
INPUT_DIR=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Photos/Benign
cd $dir
lev=0
hed=0.17
PATCH_DIR=$dir/patch_dir_0
TF_DIR=$dir/tfrecord
mkdir -p $PATCH_DIR $TF_DIR
Patch_size=256
threshold_area_percent_patch=0.5
mut="0"

for i in $INPUT_DIR/*tif
do
  /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/tf2/bin/python /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/DP_BACH_CLAM_TF/Data_Preprocessing/CLAM_Data_Prep/tfrecord_from_microscope_img.py -f $PATCH_DIR -o $TF_DIR -s $i
done
