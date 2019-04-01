#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:52:40 2019

@author: imartinez
"""

# Tensorflow tutorial on image classification and image load
import tensorflow as tf
tf.VERSION
 
import pathlib

data_root_train = pathlib.Path('./train')
data_root_test = pathlib.Path('./test')

for item in data_root_train.iterdir():
  print(item)


import random
all_image_paths = list(data_root_train.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
image_count


import IPython.display as display

for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print()
