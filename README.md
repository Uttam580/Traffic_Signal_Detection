
![GitHub repo size](https://img.shields.io/github/repo-size/Uttam580/Traffic_Signal_Detection?style=plastic)
![GitHub language count](https://img.shields.io/github/languages/count/Uttam580/Traffic_Signal_Detection?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/Uttam580/Traffic_Signal_Detection?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/Uttam580/Traffic_Signal_Detection?color=red&style=plastic)


Medium Blog: 

<a href ="https://medium.com/@uttam94/traffic-signal-detection-system-intrgrated-with-flask-d7c471fd9087"> <img src="https://github.com/Uttam580/Uttam580/blob/master/img/medium.png" width=60 height=30>


# Traffic Signal Detection 

  Traffic Signal Detection UI to detect traffic signal and integrating with Flask.

  Data has downloded form kaggle .Use Below link to download the dataset. 

  <p>Dataset Link: <a href="https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign">GTSRB Data </a></p>

  It conatains three files.


  1. Meta data : Meta data conatins diff- diff type of signal images used in Training the model. It has 43 types of traffic signal data so almot we have 86k images to train.
  2. Train
  3. Test data

  
**quick demo**

![Recordit GIF](http://g.recordit.co/dKSh8n2ARj.gif)

## Technical Aspect

1. Training a deep learning model using tensorflow. I trained model on local system using NVIDIA GEFORCE GTX 1650 for batch size 32 , epoch 20 and I had total 86K images to train .It took about 7 minutes to trian the model.

###### ```To check if model is acelearted by gpu or not```

import tensorflow as tf 

from tensorflow.python.client import device_lib

print(tf.test.is_built_with_cuda())

print(device_lib.list_local_devices())


###### ```To get compatible cuda and cudnn according to your sys config.```

Just create a new environment using ```conda create -n [name] python=[version]``` And then use ```conda install -c conda-forge tensorflow-gpu``` and it will assess which version (CUDA,CUDNN, etc.) you require and download and install it directly ;)

Below is the neural network architect of trained model.

![Network Images](https://github.com/Uttam580/Traffic_Signal_Detection/blob/master/my_model.h5.png)

2. Building and hosting using FLASK .


```
traffic_signal_detection Directory Tree

├─ .git
├─ .gitignore
├─ app.py
├─ gputest.py
├─ model
│  ├─ Traffic_detection.h5
│  ├─ ts_model.py
│  └─ vgg.py
├─ my_model.h5.png
├─ README.md
├─ standalone_predict.py
├─ static
│  └─ main.css
├─ templates
│  ├─ index.html
│  └─ template.html
├─ test_data.zip
└─ uploads
   ├─ 00001.png
   └─ main_temp.jpg

```

# Contents

* ```app.py``` - Front and back end portion of the web application 
* ```gputest.py```-  This script used to check if tensorflow is using GPU acceleration for training the model.
* ```models``` - This contains model training script  saved model file which we will use later of prediction of   uploaded images 
* ```standalone_predict.py```- standalone script for prediction , just need to give path for prediction image
* ```static```- this contains static part of UI.
* ```templates```-this contains templates for home and prediction page
* ```test_data.zip``` - Uploaded some file for testing purpose 
*``` uploads```- this conatins uploaded images  from UI , later this images will used for predcition


# Installation

* Download the entire repository as a folder and open ```app.py``` and run it with IDE .
  Make sure to install  required packeages and librabries. 

   http://127.0.0.1:5000/
```bash
pip install -r requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.tensorflow.org/images/tf_logo_social.png" width=280>](https://www.tensorflow.org)[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) 



## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

