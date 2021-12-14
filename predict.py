#!/usr/bin/env python
# coding: utf-8

# In[53]:


import sys

import argparse
from config import Args

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import matplotlib.pylab as plt
import json
import math
import time
import numpy as np
from PIL import Image


# In[55]:


debug = 0
image_size = 224

if debug == 1:

    image_filename = './test_images/cautleya_spicata.jpg'
    top_k = 5
    category_filename = 'label_map.json'
    saved_keras_model_filepath = 'image_classifier_model_udacity_project_1633673482.h5'
else:

    image_filename = sys.argv[1]
    top_k = Args.top_k
    category_filename = Args.category_names
    saved_keras_model_filepath = sys.argv[2]

    print(f"image_filename      : {image_filename}")
    print(f"top_k      : {top_k}")
    print(f"category_filename : {category_filename}")
    print(f"saved_keras_model_filepath : {saved_keras_model_filepath}")


# In[56]:


with open(category_filename, 'r') as f:
    class_names = json.load(f)


# In[57]:


reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()


# In[58]:


# TODO: Create the process_image function

def process_image(image):
    tf_image = tf.convert_to_tensor(image, np.float32)
    tf_image = tf.image.resize(image, (image_size, image_size))
    tf_image /= 255
    return tf_image.numpy()


# In[62]:


# TODO: Create the predict function
def my_predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)

    processed_test_image = process_image(image)

    ps = model.predict(np.expand_dims(processed_test_image,axis=0))

    # sorting negative ps makes it sorts in decending order
    sorted_index_array = np.argsort(-ps[0])

    # sorted array
    ps_sorted = ps[0, sorted_index_array[:top_k]]

    probs = ps_sorted

    # class_id starts from 1, and sorting index starts from 0, need to +1
    classes = sorted_index_array[:top_k]+1

    return probs, classes


# In[79]:


image = Image.open(image_filename)
image = np.asarray(image)

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

top_k_possibilities, top_k_classes = my_predict(image_filename, reloaded_keras_model, top_k)

print(f'After running the deep learning neural network, we predict the image of {image_filename} that:\n')
for idx, possibility in enumerate(top_k_possibilities):
    class_id = top_k_classes[idx]
    class_name = class_names[str(class_id)]
    if possibility > 0.9:
        print(f'We are highly confident that it\'s {class_name}, and its possibility={possibility*100: 2.5f}%')
    else:
        print(f'The {ordinal(idx+1)} possible is {class_name}, and its possibility={possibility*100: 2.5f}%')


# In[ ]:
