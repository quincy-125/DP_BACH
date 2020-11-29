#! /bin/bash
#$ -q day-rhel7
#$ -l h_vmem=50G
#$ -M gu.qiangqiang@mayo.edu
#$ -t 1-5:1
#$ -m abe
#$ -V
#$ -cwd
#$ -j y
#$ -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/log
set -x
dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/
INPUT_DIR=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/Neg
cd $dir
lev=0
hed=0.17
PATCH_DIR=$dir/patch_dir_0
TF_DIR=$dir/tfrecord
mkdir -p $PATCH_DIR $TF_DIR
Patch_size=512
threshold_area_percent_patch=0.5
mut="0"

for i in $INPUT_DIR/*svs
do
  /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/tf2/bin/python /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/TissueDetector/IMG_FEATUREVEC_TFR/sampler_openslide_quincy.py -i $i -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l $lev -a $threshold_area_percent_patch  -x $hed -c $mut
done
