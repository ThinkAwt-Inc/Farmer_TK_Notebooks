from __future__ import print_function 
import os
import json
import shutil
import requests
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, save_img, load_img


# Load and process image
img = ".\testimages\cornHealthy.jpg"
img_load = load_img(img, target_size=(224, 224, 3))
img_pic = img_to_array(img_load)
plt.imshow(img_load)


# Classes in the Model
plant_classes = {0:'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot', 
                 1:'Corn_(maize)_Common_rust', 
                 2:'Corn_(maize)_Northern_Leaf_Blight', 
                 3:'Corn_(maize)_healthy'}


# Heroku App
YOUR_APP_NAME = "Scalable-dev"
url = f'https://{YOUR_APP_NAME}.herokuapp.com/v1/models/corn:predict'


# Make predictions
def make_prediction(instances, many=False):
    if not many:
        data = json.dumps({"signature_name": "serving_default", "instances": [instances.tolist()]})
    else:
        data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions


# for p in make_prediction(img_pic):
#     # print(np.argmax(p))

for p in make_prediction(img_pic):
    if np.argmax(p) == 0:
        print("Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot")
    elif np.argmax(p) == 1:
        print("Corn_(maize)_Common_rust")
    elif np.argmax(p) == 2:
        print("Corn_(maize)_Northern_Leaf_Blight")
    else:
        print("Corn_(maize)_healthy")
