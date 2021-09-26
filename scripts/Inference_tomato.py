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
img = ".\testimages\tomatoBacteria.jpg"
img_load = load_img(img, target_size=(224, 224, 3))
img_pic = img_to_array(img_load)
plt.imshow(img_load)


# Classes in the Model
plant_classes = {0:'Tomato_Bacterial_spot',
                 1:'Tomato_Early_blight', 
                 2:'Tomato_Late_blight', 
                 3:'Tomato_Leaf_Mold',
                 4:'Tomato_Septoria_leaf_spot',
                 5:'Tomato_Spider_mites',
                 6:'Tomato_Target_Spot',
                 7:'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
                 8:'Tomato_Tomato_mosaic_virus',
                 9:'Tomato_healthy'}

# Heroku App
YOUR_APP_NAME = "Scalable-dev"
url = f'https://{YOUR_APP_NAME}.herokuapp.com/v1/models/tomato:predict'


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
#     print(np.argmax(p))
for p in make_prediction(img_pic):
    #     print(np.argmax(p))
    if np.argmax(p) == 0:
        print("Tomato_Bacterial_spot")
    elif np.argmax(p) == 1:
        print("Tomato_Early_Blight")
    elif np.argmax(p) == 2:
        print("Tomato_Late Blight")
    elif np.argmax(p) == 3:
        print("Tomato_Leaf Mold")
    elif np.argmax(p) == 4:
        print("Tomato_Septoria Leaf Mold")
    elif np.argmax(p) == 5:
        print("Tomato_Spider mites")
    elif np.argmax(p) == 6:
        print("Tomato_Target Spot")
    elif np.argmax(p) == 7:
        print("Tomato_Yellow Leaf Curl Virus")
    elif np.argmax(p) == 8:
        print("Tomato_Mosaic Virus")
    else:
        print("Tomato_Healthy")