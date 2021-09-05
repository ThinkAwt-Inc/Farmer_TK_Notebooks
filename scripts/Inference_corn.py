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


for p in make_prediction(img_pic):
    print(np.argmax(p))