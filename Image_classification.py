#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:52:40 2019

@author: imartinez
"""

# Tensorflow tutorial on image classification and image load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import math
tf.enable_eager_execution()
tf.VERSION

x = tf.random_uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

# Tensor Slices
def function(num):
    print(type(n))
    return tf.square(num) + 1
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
ds_tensors = ds_tensors.map(function).shuffle(2).batch(2)
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

# Gradient Calculation
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#x = tf.ones((2, 2)) 
with tf.GradientTape(persistent=False) as t:
  t.watch(x)
  y = tf.reduce_sum(tf.square(x))
  z = tf.multiply(y, y)

dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    print(dz_dx[i][j].numpy())


# =============================================================================
#  DATASET BUILDING
# =============================================================================
data = pd.read_csv('train.tsv', sep='\t')
label_names = data.label.unique()
label_to_index = dict((name, index) for index, name in enumerate(label_names))
data['label_index'] = [label_to_index[label] for label in data.label]

all_image_paths = "./train/" + data.file.values
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.shuffle(20).take(4)):
    plt.subplot(2, 2, n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(data.label_index, tf.int64))
for label in label_ds.take(10):
    print(label.numpy())
    print(label_names[label.numpy()])






















