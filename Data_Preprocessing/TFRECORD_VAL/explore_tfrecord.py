bach_dir = '/path/'

tcga_dir = '/path/'

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