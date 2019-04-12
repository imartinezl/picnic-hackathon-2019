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
from sklearn.model_selection import train_test_split

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

data_train, data_val = train_test_split(data, test_size=0.2)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].hist(data.label_index, bins=25)
axarr[1].hist(data_train.label_index, bins=25)
axarr[2].hist(data_val.label_index, bins=25)
f.subplots_adjust(hspace=0)
for ax in axarr:
    ax.label_outer()


# =============================================================================
# A dataset of (image, label) pairs
# =============================================================================

image_paths = "./train/" + data_train.file.values
image_labels = data_train.label_index
image_count = data_train.shape[0]


def preprocess_image(image):
    image = tf.cond(
      tf.image.is_jpeg(image),
      lambda: tf.image.decode_jpeg(image, channels=3),
      lambda: tf.image.decode_png(image, channels=3))
#    image = tf.image.decode_image(image, channels=3)
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
ds = image_label_ds.shuffle(buffer_size=BATCH_SIZE*2)
ds = image_label_ds.repeat()
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

steps_per_epoch = int(tf.ceil(len(image_paths)/BATCH_SIZE).numpy())
steps_per_epoch

#model.fit(ds, epochs=1, steps_per_epoch=3)
model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch)

# Evaluate Validation
image_paths_val = "./train/" + data_val.file.values
image_labels_val = data_val.label_index
ds_val = tf.data.Dataset.from_tensor_slices((image_paths_val, image_labels_val))
ds_val = ds_val.map(load_and_preprocess_from_path_label)
ds_val = ds_val.batch(batch_size=BATCH_SIZE)
ds_val = ds_val.map(change_range)

eval_train = model.evaluate(ds, steps=10)
eval_val = model.evaluate(ds_val, steps=1)

output_train = model.predict(ds, steps=1, verbose=1)
output_train.shape
[np.argmax(x) for x in output_train]


# Predict  samples
data_test = pd.read_csv('test.tsv', sep='\t')
image_paths_test = "./test/" + data_test.file.values

dataset = tf.data.Dataset.from_tensor_slices((image_paths_test))
dataset = dataset.map(load_and_preprocess_image)
dataset = dataset.batch(batch_size=10)
output = model.predict(x=dataset,steps=1,verbose=True)
[np.argmax(x) for x in output]

output = model.predict(x=ds,steps=1,verbose=True)
[np.argmax(x) for x in output]

test_loss, test_acc = model.evaluate(test_images, test_labels)


















