set -x
QSUB_OPTIONS='-q gpu-long -l gpu=1 -l h_vmem=300G -M Gu.Qiangqiang@mayo.edu -m abe -V -cwd -j y -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/LOG'

dir=/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM
folder_name=tcga_mc_ga_ilr_2e-02_blr_2e-03_iwd_1e-03_bwd_1e-04_bs_3000_rest_default_topk_p_0.004
data_name=TCGA
train_log=$dir/log/$data_name/$folder_name/train/
val_log=$dir/log/$data_name/$folder_name/val/
train_path=$dir/$data_name/Image_Standardization/train/
val_path=$dir/$data_name/Image_Standardization/val/
test_path=$dir/$data_name/Image_Standardization/test/
result_path=$dir/test_result_file/$data_name/$folder_name/
c_model_dir=$dir/Saved_Model/$data_name/$folder_name/CLAM_Model/
mkdir -p $result_path
result_file_name='clam_test.tsv'
dim_compress_features=512
net_size='big'
dropout_name='True'
dropout_rate=0.25
i_optimizer_name="AdamW"
b_optimizer_name="AdamW"
a_optimizer_name="AdamW"
i_loss_name="binary_crossentropy"
b_loss_name="binary_crossentropy"
c1=0.7
c2=0.3
i_learn_rate=2e-02
b_learn_rate=2e-03
a_learn_rate=2e-04
i_l2_decay=1e-03
b_l2_decay=1e-04
a_l2_decay=1e-05
imf_norm_op_name='True'
mut_ex_name='False'
att_only_name='False'
mil_ins_name='True'
att_gate_name='True'
no_warn_op_name='True'
i_wd_op_name='True'
b_wd_op_name='True'
a_wd_op_name='True'
m_clam_op_name='True'
n_class=2
top_k_percent=0.004
batch_size=3000
batch_op_name='True'
epochs=200
n_test_steps=10
m_gpu_name='True'
is_training_name='True'
qsub $QSUB_OPTIONS -b y /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/clam/bin/python3 /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/DP_BACH_CLAM_TF/main.py -g $train_log -l $val_log -t $train_path -v $val_path -d $test_path -r $result_path -f $result_file_name -Y $imf_norm_op_name -A $dim_compress_features -T $net_size -D $dropout_name -R $dropout_rate -o $i_optimizer_name -p $b_optimizer_name -z $a_optimizer_name -y $i_loss_name -b $b_loss_name -u $mut_ex_name -S $n_class -c $c1 -a $c2 -L $i_learn_rate -j $b_learn_rate -k $a_learn_rate -n $i_l2_decay -q $b_l2_decay -w $a_l2_decay -K $top_k_percent -Z $batch_size -B $batch_op_name -m $c_model_dir -O $att_only_name -N $mil_ins_name -x $att_gate_name -E $epochs -X $n_test_steps -W $no_warn_op_name -I $i_wd_op_name -J $b_wd_op_name -C $a_wd_op_name -M $m_clam_op_name -G $m_gpu_name -i $is_training_name