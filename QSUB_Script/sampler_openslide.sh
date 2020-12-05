#! /bin/bash
#$ -q queue_name
#$ -l h_vmem=50G
#$ -M xxxx@xxx.xxx
#$ -t 1-5:1
#$ -m abe
#$ -V
#$ -cwd
#$ -j y
#$ -o /path/log
set -x
dir=/dir/
INPUT_DIR=/input_dir/
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
  /path/bin/python /path/tfrecird_from_microscope_img.py -i $i -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l $lev -a $threshold_area_percent_patch  -x $hed -c $mut
done
