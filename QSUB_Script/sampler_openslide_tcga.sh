#! /bin/bash
#$ -q queue_name
#$ -l h_vmem=50G
#$ -M xxxx@xxx.xxx
#$ -t 1-295:1
#$ -m a
#$ -V
#$ -cwd
#$ -j y
#$ -o /path/log
set -x
dir=/dir/
my_dir=/my_dir/
cd $dir
lev=1
#hed=0.1666
hed=-0.41
# SGE_TASK_ID=1
PATCH_DIR=$my_dir/patch_level_normal$lev
TF_DIR=$my_dir/tfrecord_level_normal$lev
mkdir -p $PATCH_DIR $TF_DIR
Patch_size=256
#for ((SGE_TASK_ID=1;SGE_TASK_ID<=3;SGE_TASK_ID=SGE_TASK_ID+1));
#do
samp=`cat SVS.txt|head -$SGE_TASK_ID|tail -1|cut -f1 `
mut=`cat SVS.txt|head -$SGE_TASK_ID|tail -1|cut -f2 |sed -e 's/BRAF/1/g'|sed -e 's/WILD/0/g'`
/path/conda_env/tf2/bin/python /path/tfrecod_from_wsi.py -i $dir/SVS/$samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l $lev -t 245 -a 0.5 -m 245 -d 0 -x $hed -c $mut -z 0
#done
