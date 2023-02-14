python /projects/shart/digital_pathology/scripts/General-OneShot/train.py \
-m DenseNet201 -c 128 \
-o Adam \
-p 256 \
-r 0.00001 -e 8 -b 32 -V DEBUG --filetype images \
#-t /projects/shart/digital_pathology/data/melanoma_braf/tcga_tfrecord_level1_img/train \
#-v /projects/shart/digital_pathology/data/melanoma_braf/tcga_tfrecord_level1_img/val \
#-l /projects/shart/digital_pathology/data/melanoma_braf/LOG_TCGA_BRAF/LOG_oneshot
-t /projects/shart/digital_pathology/data/BACH/final_train_test_val/train \
-v /projects/shart/digital_pathology/data/BACH/final_train_test_val/val \
-l /projects/shart/digital_pathology/data/BACH/log_hardsamp_test  \
#-r 0.0001 \
#-e 5 -b 32 -V DEBUG --filetype images
