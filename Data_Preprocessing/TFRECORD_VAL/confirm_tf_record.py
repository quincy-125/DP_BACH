tfrecord_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/' \
	'Quincy/Data/CLAM/BACH/'
import tensorflow as tf
import io
import sys
from PIL import Image
import numpy as np
import os
tf_files=os.listdir(tfrecord_dir)
for i in tf_files:
	if 'tfrecord' in i:
		tfrecord=tfrecord_dir+'/'+i
		for example in tf.compat.v1.python_io.tf_record_iterator(tfrecord):
			result = tf.train.Example.FromString(example)
			z1=100
			z2="NA"
			for k,v in result.features.feature.items():
				if k == 'image/encoded':
					z=1
				elif k == 'image_feature1':
					#stream=io.BytesIO(v.bytes_list.value[0])
					#img = Image.open(stream)
					#res = np.unique(np.asarray(img), return_counts=True)
					#print(k, res)
					z=1
				else:
					try:
						print(k, v.bytes_list.value[0])
						#if k=="image/name":
							#z1=v.bytes_list.value[0]
							#z1=z1.decode("utf-8")
						#if k=='phenotype/subtype':
							#z2=v.int64_list.value[0]
					except:
						print(k, v.int64_list.value[0])
						#if k=='phenotype/subtype':
							#z2=v.int64_list.value[0]
			#print(str(z2)+"\t"+i+"\t"+str(z1))	
			#sys.exit(0)
