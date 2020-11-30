set -x
QSUB_OPTIONS='-q gpu -l gpu=1 -l h_vmem=100G -M Gu.Qiangqiang@mayo.edu -m abe -V -cwd -j y -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/job_out'

dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM
train_log=$dir/log/bach_cli/train/
val_log=$dir/log/bach_cli/val/
train_path=$dir/BACH/Image_Standardization/train/
val_path=$dir/BACH/Image_Standardization/val/
test_path=$dir/BACH/Image_Standardization/test/
result_path=$dir/test_result_file/
i_model_dir=$dir/Saved_Model/cli/Ins_Classifier/
b_model_dir=$dir/Saved_Model/cli/Bag_Classifier/
c_model_dir=$dir/Saved_Model/cli/CLAM_Model/
result_file_name='bach_clam_all_default_cli.tsv'
mkdir -p $train_log $val_log $i_model_dir $b_model_dir $c_model_dir
dim_features=1024
dim_compress_features=512
n_hidden_units=256
net_size='big'
dropout_name='True'
dropout_rate=0.25
i_optimizer_name="AdamW"
b_optimizer_name="AdamW"
c_optimizer_name="AdamW"
i_loss_name="binary_cross_entropy"
b_loss_name="binary_cross_entropy"
c1=0.7
c2=0.3
i_learn_rate=2e-04
b_learn_rate=2e-04
c_learn_rate=2e-04
i_l2_decay=1e-05
b_l2_decay=1e-05
c_l2_decay=1e-05
mut_ex_name='False'
att_only_name='False'
mil_ins_name='True'
att_gate_name='True'
no_warn_op_name='True'
m_clam_op_name='False'
n_class=2
top_k_percent=0.2
batch_size=2000
batch_op_name='False'
epochs=200
is_training_name='True'
qsub $QSUB_OPTIONS -b y /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/clam/bin/python3 /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/DigiPath_CLAM_TF/main.py -g $train_log -l $val_log -t $train_path -v $val_path -d $test_path -r $result_path -f $result_file_name -F $dim_features -A $dim_compress_features -H $n_hidden_units -T $net_size -D $dropout_name -R $dropout_rate -o $i_optimizer_name -p $b_optimizer_name -z $c_optimizer_name -y $i_loss_name -b $b_loss_name -u $mut_ex_name -S $n_class -c $c1 -a $c2 -L $i_learn_rate -j $b_learn_rate -k $c_learn_rate -n $i_l2_decay -q $b_l2_decay -w $c_l2_decay -K $top_k_percent -Z $batch_size -B $batch_op_name -e $i_model_dir -s $b_model_dir -m $c_model_dir -O $att_only_name -N $mil_ins_name -x $att_gate_name -E $epochs -W $no_warn_op_name -M $m_clam_op_name -i $is_training_name