import tensorflow as tf
import tensorflow_addons as tfa
import datetime

from MODEL.model_attention import NG_Att_Net, G_Att_Net
from MODEL.model_bag_classifier import S_Bag, M_Bag
from MODEL.model_clam import S_CLAM, M_CLAM
from MODEL.model_ins_classifier import Ins
from UTILITY.model_main import clam_main


ng_att = NG_Att_Net(dim_features=1024, dim_compress_features=512,
                    n_hidden_units=256, n_class=2,
                    dropout=False, dropout_rate=.25)

g_att = G_Att_Net(dim_features=1024, dim_compress_features=512,
                  n_hidden_units=256, n_class=2,
                  dropout=False, dropout_rate=.25)

ins = Ins(dim_compress_features=512, n_class=2, top_k_percent=0.4, mut_ex=True)

s_bag = S_Bag(dim_compress_features=512, n_class=2)

m_bag = M_Bag(dim_compress_features=512, n_class=2)

s_clam = S_CLAM(att_gate=True, net_size='big', top_k_percent=0.4,
                n_class=2, mut_ex=False,
                dropout=True, drop_rate=.25,
                mil_ins=True, att_only=False)

m_clam = M_CLAM(att_gate=True, net_size='big', top_k_percent=0.4,
                n_class=2, mut_ex=False,
                dropout=True, drop_rate=.25,
                mil_ins=True, att_only=False)

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
                      's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/Ins_Classifier'

b_trained_model_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/' \
                      's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/Bag_Classifier'

c_trained_model_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/' \
                      's211408.DigitalPathology/Quincy/Data/CLAM/Saved_Model/CLAM_Model'


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
          dropout=True,
          dropout_rate=0.25,
          i_optimizer_func=tfa.optimizers.AdamW,
          b_optimizer_func=tfa.optimizers.AdamW,
          c_optimizer_func=tfa.optimizers.AdamW,
          i_loss_func=tf.keras.losses.binary_crossentropy,
          b_loss_func=tf.keras.losses.binary_crossentropy,
          mut_ex=False,
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
          batch_op=False,
          i_model_dir=i_trained_model_dir,
          b_model_dir=b_trained_model_dir,
          c_model_dir=c_trained_model_dir,
          att_only=False,
          mil_ins=True,
          att_gate=True,
          epochs=200,
          no_warn_op=True,
          m_clam_op=False,
          is_training=True)