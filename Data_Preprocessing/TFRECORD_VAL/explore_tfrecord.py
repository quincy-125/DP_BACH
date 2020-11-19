bach_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
    'Quincy/Data/CLAM/BACH/train/Normal_n100.tif.tfrecords'

tcga_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
    'Quincy/Data/CLAM/TCGA/train/TCGA-Z2-A8RT-01Z-00-DX1.3DE00C7C-0373-425C-A042-A463FD814D50.svs.0.train.tfrecords'

import tensorflow as tf
import io
from PIL import Image
import numpy as np


def _examine_tfrecord(tfrecord):
    for example in tf.compat.v1.python_io.tf_record_iterator(tfrecord):
        result = tf.train.Example.FromString(example)
        for k, v in result.features.feature.items():
            if k == 'image/encoded':
                print(k, "Skipping...")
            elif k == 'image/segmentation/class/encoded':
                stream = io.BytesIO(v.bytes_list.value[0])
                img = Image.open(stream)
                res = np.unique(np.asarray(img), return_counts=True)
                print(k, res)
            else:
                try:
                    print(k, v.bytes_list.value[0])
                except:
                    print(k, v.int64_list.value[0])
        break


print('NOW BACH TEST', '\n')
_examine_tfrecord(bach_dir)
print('NOW TCGA TEST', '\n')
_examine_tfrecord(tcga_dir)