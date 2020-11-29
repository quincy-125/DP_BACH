#! /bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -l h_vmem=100G
#$ -M gu.qiangqiang@mayo.edu
#$ -m abe
#$ -V
#$ -cwd
#$ -j y
#$ -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/job_out
set -x

train_log='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/'\
'Quincy/Data/CLAM/log/'+current_time+'/train'

val_log='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/'\
'Quincy/Data/CLAM/log/'+current_time+'/val'

train_path='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/'\
'Quincy/Data/CLAM/BACH/Image_Standardization/train/'

val_path='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/'\
'Quincy/Data/CLAM/BACH/Image_Standardization/val/'

test_path='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/'\
'Quincy/Data/CLAM/BACH/Image_Standardization/test/'

result_path='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/'\
'Quincy/Data/CLAM/test_result_file/'

i_model_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/'\
's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/cli/Ins_Classifier'

b_model_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/'\
's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/cli/Bag_Classifier'

c_model_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/'\
's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/cli/CLAM_Model'

result_file_name='bach_clam_all_default_cli.tsv'

dim_features=1024
dim_compress_features=512
n_hidden_units=256
net_size='big'

dropout=True
dropout_rate=0.25

i_optimizer_func=tfa.optimizers.AdamW
b_optimizer_func=tfa.optimizers.AdamW
c_optimizer_func=tfa.optimizers.AdamW
i_loss_func=tf.keras.losses.binary_crossentropy
b_loss_func=tf.keras.losses.binary_crossentropy

c1=0.7
c2=0.3

i_learn_rate=2e-04
b_learn_rate=2e-04
c_learn_rate=2e-04

i_l2_decay=1e-05
b_l2_decay=1e-05
c_l2_decay=1e-05

mut_ex=False
att_only=False
mil_ins=True
att_gate=True
no_warn_op=True
m_clam_op=False

n_class=2
top_k_percent=0.2

batch_size=2000
batch_op=False

epochs=200

is_training=False

/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/\
Anaconda/conda_env/clam/bin/python3 /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/\
s211408.DigitalPathology/Quincy/Code/DigiPath_CLAM_TF/main.py -g $train_log -l $val_log\
-t $train_path -v $val_path -d $test_path -r $result_path -f $result_file_name\
-F $dim_features -A $dim_compress_features -H $n_hidden_units -T $net_size\
-D $dropout -R $dropout_rate -o $i_optimizer_func -p $b_optimizer_func\
-z $c_optimizer_func -y $i_loss_func -b $b_loss_func -u $mut_ex -S $n_class\
-c $c1 -a $c2 -h $i_learn_rate -j $b_learn_rate -k $c_learn_rate\
-n $i_l2_decay -q $b_l2_decay -w $c_l2_decay -K $top_k_percent\
-Z $batch_size -B $batch_op -e $i_model_dir -s $b_model_dir -m $c_model_dir\
-O $att_only -N $mil_ins -x $att_gate -E $epochs -W $no_warn_op -M $m_clam_op\
-i $is_training