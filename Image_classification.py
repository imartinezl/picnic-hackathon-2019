#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:52:40 2019

@author: imartinez
"""

# Tensorflow tutorial on image classification and image load
import tensorflow as tf
tf.VERSION


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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
