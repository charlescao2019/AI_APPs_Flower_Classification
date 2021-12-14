#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser(description='Image Classifier Using Deep Neural Network')

parser.add_argument('image_filepath',          type=str, help= 'image filepath')
parser.add_argument('model_filepath',          type=str, help= 'deep learning model in .h5 file')
parser.add_argument('--top_k',          type=int, default=1,          help= 'the top k predicted possibilities. default=1')
parser.add_argument('--category_names', type=str, default='map.json', help= 'load category names from a json file. default=map.json')

Args = parser.parse_args()

