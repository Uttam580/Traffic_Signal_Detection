

# Traffic Signal Detection 

  Traffic Signal Detection UI to detect traffic signal and integrating with Flask.

  Data has downloded form kaggle .Please use Below link to download the dataset. 
  It conatains three files.


  1. Meta data : Meta data conatins diff- diff type of signal images used in Training the model
  2. Train
  3. Test data

  
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


**quick demo**

![Recordit GIF](http://g.recordit.co/oYi4PX5Lq1.gif)


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

* Download the entire repository as a folder and open ```app.py``` and run it with IDE . That's it!
   http://127.0.0.1:5001/
   

