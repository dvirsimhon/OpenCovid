# **OpenCoVid**

<img src="https://in.bgu.ac.il/marketing/graphics/BGU.sig3-he-en-white.png" height="48px" align="right" /> 
<img src="https://res.cloudinary.com/serfati/image/upload/v1605445665/OpenCoVid19/logo_ntvgyv.png" height="90"/> 

<br>
<br>

Detect Faces with or without mask using [YoloV5](https://github.com/ultralytics/yolov5).

## Description  

Automatic systems to detect people wearing masks are becoming more and more important for public health.
Be it for governments who might want to know how many people are actually wearing masks in crowded places like public trains; or businesses who are required by law to enforce the usage of masks within their facilities. 

This projects aims to provide an easy framework to set up such a mask detection system with minimal effort.
We provide a pre-trained model trained for people relatively close to the camera which you can use as a quick start option.

But even if your use case is not covered by the pre-trained model, training your own is also quite easy (also a reasonable recent GPU is highly recommended) and a you should be able to do this by following the short guide provided in this README. 

## ⚠️ Prerequisites  
  
- [`Python >= 3.8`](https://www.python.org/download/releases/3.8/)  
- [`Pytorch >= 1.7`](https://pytorch.org/get-started/locally/)  
- [`Git >= 2.26`](https://git-scm.com/downloads/)  
- [`PyCharm IDEA`](https://www.jetbrains.com/pycharm/) (recommend)  

## 📦 How To Install  
  
You can modify or contribute to this project by following the steps below:  


**0. The pre-trained model can be downloaded from here.**  
 ```bash  
 # pretrained YoloV5 model
 $> wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZxGjMsfogaUGaWc0zuYCbOexJPbFmISv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZxGjMsfogaUGaWc0zuYCbOexJPbFmISv" -O yolomask.pt && rm -rf /tmp/cookies.txt
 ```  
  
**1. Clone the repository**  
  
- Open terminal ( <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd> )  
  
- [Clone](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) to a location on your machine.  
 ```bash  
 # Clone the repository 
 $> git clone https://github.com/dvirsimhon/OpenCovid.git  

 # Navigate to the directory 
 $> cd OpenCovid
  ``` 

**2. Install Dependencies**  

  All the needed python packages can be found in the `requirements.txt` file.

 ```bash  
 # install requirments
 $> pip install -U -r requirements.txt
 ```  

## 💽 Face-Mask Dataset

### 1. Image Sources
- Our photographies
- Images were collected from [Google Images](https://www.google.com/imghp?hl=en), [Bing Images](https://www.bing.com/images/trending?form=Z9LH) and some [Kaggle Datasets](https://www.kaggle.com/vtech6/medical-masks-dataset).
- Chrome Extension used to download images: [link](https://download-all-images.mobilefirst.me/)

### 2. Image Annotation
- Images were annoted using [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark).

### 3. Dataset Description
- Dataset is split into 2 sets:

|_Set_|Number of images|Objects with mask|Objects without mask|
|:--:|:--:|:--:|:--:|
|**Training Set**| 2340 | 9050 | 1586 |
|**Validation Set**| 260 | 1005 | 176 |
|**Total**|2600|10055|1762|

<br>

## 📃 Usage  

### 🔌 Pre-trained model


## ⌨ Scripts:

- `train.pt` - training your object detection model
- `test.py` - evaluate the test dataset
- `detect.py` - runs inference on a variety of sources

## ⚖️ License  
  
This program is free software: you can redistribute it and/or modify it under the terms of the **MIT LICENSE** as published by the Free Software Foundation.  
  
**[⬆ back to top](#description)**

> author Serfati
