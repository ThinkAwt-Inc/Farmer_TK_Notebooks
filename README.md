# Farmer_TK_Notebooks
Diagnosis using Data Science concepts is one of the interesting usecases Machine learning has been utilized for.

This project looks at the classification of plant images along different disease diagnostic criteria.

Each plant was modeled using deep learning tools; Tensorflow and Pytorch. The notebooks with the particulars of building the models are available in the notebooks folder.

----

All resulting models were served for efficient inferencing using our Scalable serving options; [Cruise](https://github.com/JesuFemi-O/Cruise) and [Scalable-Dev](https://github.com/ThinkAwt-Inc/Scalable-Dev) which utilize Tensorflow Serving to deploy REST APIs to Heroku.

*The process can be automated and scaled for multiple version of the model using Github Actions as is the case in Scalable-Dev*

# Test Inference

Using the potato model as a sample;

```python
# importing required python packages
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array

# path to the test image
img='/content/PotatoHealthy1.JPG'

# resizing the image based on the values the model received
img=load_img(img,target_size=(128,128,3))

# converting the image format and size
pic=img_to_array(img)
pic = np.transpose(pic)

# show image
plt.imshow(img)

# Classes in the Model
plant_classes = {0:'Potato___Early_blight', 
                1:'Potato___Late_blight', 
                2:'Potato___healthy'}

# here we connect to our heroku model server
YOUR_APP_NAME = "scalable-dev"
url = 'http://scalable-dev.herokuapp.com/v1/models/potato:predict'

# this function makes the model queries and reads the outputs
def make_prediction(instances, many=False):
    if not many:
        data = json.dumps({"signature_name": "serving_default", "instances": [instances.tolist()]})
    else:
        data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
  
    predictions = json.loads(json_response.text)['predictions']
    return predictions


# Make predictions
for p in make_prediction(pic):
    print(np.argmax(p))
```

The above code queries our served model server for classification of a test image. Where an output of : 
* **1 = Early Blight Potato**,
* **2 = Late Blight Potato**,
* **3 = Healthy Potato**

The images used for testing the served model is saved in the **testimages** folder, And the Notebooks in the **notebooks** folder.
