import datetime

from UTILITY.model_main import clam_main


train_nis_bach = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                 'Quincy/Data/CLAM/BACH/No_Image_Standardization/train/'

val_nis_bach = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
               'Quincy/Data/CLAM/BACH/No_Image_Standardization/val/'

test_nis_bach = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                'Quincy/Data/CLAM/BACH/No_Image_Standardization/test/'


train_is_bach = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                'Quincy/Data/CLAM/BACH/Image_Standardization/train/'

val_is_bach = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
              'Quincy/Data/CLAM/BACH/Image_Standardization/val/'

test_is_bach = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
               'Quincy/Data/CLAM/BACH/Image_Standardization/test/'


train_nis_tcga = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                 'Quincy/Data/CLAM/TCGA/No_Image_Standardization/train/'

val_nis_tcga = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
               'Quincy/Data/CLAM/TCGA/No_Image_Standardization/val/'

test_nis_tcga = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                'Quincy/Data/CLAM/TCGA/No_Image_Standardization/test/'

extra_nis_tcga = 'research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                 'Quincy/Data/CLAM/TCGA/No_Image_Standardization/extra/'


train_is_tcga = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                'Quincy/Data/CLAM/TCGA/Image_Standardization/train/'

val_is_tcga = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
              'Quincy/Data/CLAM/TCGA/Image_Standardization/val/'

test_is_tcga = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
               'Quincy/Data/CLAM/TCGA/Image_Standardization/test/'

extra_is_tcga = 'research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
                'Quincy/Data/CLAM/TCGA/Image_Standardization/extra/'


clam_result_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM'

i_trained_model_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/' \
                      's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/dynamic_k/Ins_Classifier'

b_trained_model_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/' \
                      's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/dynamic_k/Bag_Classifier'

c_trained_model_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/' \
                      's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/dynamic_k/CLAM_Model'


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
'Quincy/Data/CLAM/log/' + current_time + '/train'

val_log_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
'Quincy/Data/CLAM/log/' + current_time + '/val'

clam_main(train_log=train_log_dir,
          val_log=val_log_dir,
          train_path=train_is_bach,
          val_path=val_is_bach,
          test_path=test_is_bach,
          result_path=clam_result_dir,
          result_file_name='bach_all_default_hyper_parameters.tsv',
          dim_features=1024,
          dim_compress_features=512,
          n_hidden_units=256,
          net_size='big',
          dropout_name='True',
          dropout_rate=0.25,
          i_optimizer_name='AdamW',
          b_optimizer_name='AdamW',
          c_optimizer_name='AdamW',
          i_loss_name='binary_cross_entropy',
          b_loss_name='binary_cross_entropy',
          mut_ex_name='False',
          n_class=2,
          c1=0.7, c2=0.3,
          i_learn_rate=2e-04,
          b_learn_rate=2e-04,
          c_learn_rate=2e-04,
          i_l2_decay=1e-05,
          b_l2_decay=1e-05,
          c_l2_decay=1e-05,
          top_k_percent=0.2,
          batch_size=2000,
          batch_op_name='False',
          i_model_dir=i_trained_model_dir,
          b_model_dir=b_trained_model_dir,
          c_model_dir=c_trained_model_dir,
          att_only_name='False',
          mil_ins_name='True',
          att_gate_name='True',
          epochs=200,
          no_warn_op_name='True',
          m_clam_op_name='False',
          is_training_name='True')