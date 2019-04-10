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
tf.enable_eager_execution()
tf.VERSION


x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

x = tf.ones((2, 2)) 
with tf.GradientTape(persistent=False) as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0



import glob
image_paths_train = glob.glob("./train/*")
image_paths_test = glob.glob("./test/*")

import random
random.shuffle(image_paths_train)

image_count = len(image_paths_train)
image_count


import IPython.display as display
for n in range(3):
  image_path = random.choice(image_paths_train)
  display.display(display.Image(image_path))
  print()

# Get labels


train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')

label_to_index = dict((name, index) for index,name in enumerate(train.label.unique()))
train['index'] = [label_to_index[label] for label in train.label]

img_path = image_paths_train[0]
img_path
img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize_images(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

img_path = "./train/" + train.file[0]
label = train.label[0]

fig = plt.figure()
plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.title(label)
print()


# =============================================================================
#  DATASET BUILDING
# =============================================================================
train = pd.read_csv('train.tsv', sep='\t')

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)





















