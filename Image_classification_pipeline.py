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
#import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()
tf.VERSION
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 192

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

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

image_paths_train = "./train/" + data_train.file.values
image_labels_train = data_train.label_index
image_paths_val = "./train/" + data_val.file.values
image_labels_val = data_val.label_index

def preprocess_image(image):
    image = tf.cond(
      tf.image.is_jpeg(image),
      lambda: tf.image.decode_jpeg(image, channels=3),
      lambda: tf.image.decode_png(image, channels=3))
#    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize_images(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def change_range(image,label):
  return 2*image-1, label

def generate_dataset(image_paths, image_labels):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    ds = ds.map(load_and_preprocess_from_path_label)
    
    #ds = ds.shuffle(buffer_size=image_count)
    #ds = ds.shuffle(buffer_size=BATCH_SIZE*2)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)    
    ds = ds.map(change_range)
    return ds

# MODEL
mobile_net = tf.keras.applications.Xception(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False)
mobile_net.trainable=False

model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),

#        tf.keras.layers.Conv2D(filters=4, kernel_size=2, padding='same',
#                               activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
#        tf.keras.layers.MaxPooling2D(pool_size=2),
#        
#        tf.keras.layers.Conv2D(filters=8, kernel_size=2, padding='same',activation='relu'),
#        tf.keras.layers.MaxPooling2D(pool_size=2),
#        tf.keras.layers.Dropout(0.1),
#        
#        tf.keras.layers.Conv2D(filters=12, kernel_size=2, padding='same',activation='relu'),
#        tf.keras.layers.MaxPooling2D(pool_size=2),
#        tf.keras.layers.Dropout(0.2),
#        
#        tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same',activation='relu'),
#        tf.keras.layers.MaxPooling2D(pool_size=2),
#        tf.keras.layers.Dropout(0.3),
#        
#        tf.keras.layers.Flatten(),     
#        tf.keras.layers.Dense(256, activation='relu'),     
#        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(len(label_names), activation='softmax')
        ])
    
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])

len(model.trainable_variables)
model.summary()
#model.save('model.h5')
#model = tf.keras.models.load_model('model.h5')


ds_train = generate_dataset(image_paths_train, image_labels_train)
ds_val = generate_dataset(image_paths_val, image_labels_val)

steps_per_epoch_train = int(tf.ceil(len(image_paths_train)/BATCH_SIZE).numpy())
steps_per_epoch_val = int(tf.ceil(len(image_paths_val)/BATCH_SIZE).numpy())

esCallBack = tf.keras.callbacks.EarlyStopping()
#tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', write_graph=True, write_images=True)
model.fit(ds_train, epochs=15, steps_per_epoch=steps_per_epoch_train, 
          validation_data=ds_val, validation_steps=steps_per_epoch_val,
          callbacks = [esCallBack])

eval_train = model.evaluate(ds_train, steps=1)
eval_val = model.evaluate(ds_val, steps=1)

output_train = model.predict(ds_train, steps=1, verbose=1)
output_train.shape
predictions_train = [np.argmax(x) for x in output_train]













# Predict  samples
data_test = pd.read_csv('test.tsv', sep='\t')
image_paths_test = "./test/" + data_test.file.values

ds_test = tf.data.Dataset.from_tensor_slices((image_paths_test))
ds_test = ds_test.map(load_and_preprocess_image)
ds_test = ds_test.batch(batch_size=BATCH_SIZE)

steps_per_epoch_test = int(tf.ceil(len(image_paths_test)/BATCH_SIZE).numpy())

output_test = model.predict(x=ds_test,steps=steps_per_epoch_test,verbose=True)
output_test.shape
data_test['label'] = label_names[[np.argmax(x) for x in output_test]]

data_test.to_csv('submission.tsv', sep = '\t', index=False)




















