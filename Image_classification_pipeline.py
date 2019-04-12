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
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# =============================================================================
#  TENSORFLOW PIPELINE
# =============================================================================

data = pd.read_csv('train.tsv', sep='\t')
label_names = data.label.unique()
label_to_index = dict((name, index) for index, name in enumerate(label_names))
data['label_index'] = [label_to_index[label] for label in data.label]

# =============================================================================
# A dataset of (image, label) pairs
# =============================================================================

image_paths = "./train/" + data.file.values
image_labels = data.label_index
image_count = data.shape[0]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
image_label_ds = image_label_ds.map(load_and_preprocess_from_path_label)
image_label_ds

#ds = image_label_ds.shuffle(buffer_size=image_count)
#ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

def change_range(image,label):
  return 2*image-1, label

ds = ds.map(change_range)

model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(label_names))
        ])
    
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = [tf.keras.metrics.categorical_accuracy])

len(model.trainable_variables)
model.summary()

steps_per_epoch = tf.ceil(len(image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds, epochs=1, steps_per_epoch=3)









