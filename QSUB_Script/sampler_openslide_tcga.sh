#! /bin/bash
#$ -q lg-mem
#$ -l h_vmem=50G
#$ -M gu.qiangqiang@mayo.edu
#$ -t 1-295:1
#$ -m a
#$ -V
#$ -cwd
#$ -j y
#$ -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/log
set -x
dir=/research/bsi/projects/PI/tertiary/Sun_Zhifu_zxs01/s4331393.GTEX/processing/naresh/Digital_path/BRAF_NEW/TCGA
quincy_dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM
cd $dir
lev=1
#hed=0.1666
hed=-0.41
# SGE_TASK_ID=1
PATCH_DIR=$quincy_dir/patch_level_normal$lev
TF_DIR=$quincy_dir/tfrecord_level_normal$lev
mkdir -p $PATCH_DIR $TF_DIR
Patch_size=256
#for ((SGE_TASK_ID=1;SGE_TASK_ID<=3;SGE_TASK_ID=SGE_TASK_ID+1));
#do
samp=`cat SVS.txt|head -$SGE_TASK_ID|tail -1|cut -f1 `
mut=`cat SVS.txt|head -$SGE_TASK_ID|tail -1|cut -f2 |sed -e 's/BRAF/1/g'|sed -e 's/WILD/0/g'`
/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/tf2/bin/python /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/DigiPath_CLAM_TF/Data_Preprocessing/CLAM_Data_Prep/sampler_openslide.py -i $dir/SVS/$samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l $lev -t 245 -a 0.5 -m 245 -d 0 -x $hed -c $mut -z 0
#done
