import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array

img='/content/PotatoHealthy1.JPG'
img=load_img(img,target_size=(128,128,3))
pic=img_to_array(img)

pic = np.transpose(pic)

plt.imshow(img)

YOUR_APP_NAME = "scalable-dev"
url = 'http://scalable-dev.herokuapp.com/v1/models/potato:predict'


def make_prediction(instances, many=False):
    if not many:
        data = json.dumps({"signature_name": "serving_default", "instances": [instances.tolist()]})
    else:
        data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
  
    predictions = json.loads(json_response.text)['predictions']
    return predictions


for p in make_prediction(pic):
    print(np.argmax(p))
