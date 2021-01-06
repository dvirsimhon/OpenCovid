# **YoloMask**

<img src="https://in.bgu.ac.il/marketing/graphics/BGU.sig3-he-en-white.png" height="48px" align="right" /> 
<img src="https://res.cloudinary.com/serfati/image/upload/v1609847964/yolomask_logo.png" height="90"/> 

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
detect.py runs inference on a variety of sources, downloading models automatically from the latest YOLOv5 release and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolomask.pt --conf 0.25
```

### 📐 **Own model**

* **Creating Training and Validation sets**

The labeled images have to be split in a training and a validation set. Splitting it 80-20 should be reasonable.

The folder structure is as follows:
```
data  
├── images  
│   ├── train  
│   └── val  
├── labels  
    ├── train  
    └── val  
```
* **Training**

To start the training simply run the `train.py` script.  
Again this script can take a number of arguments, but for a first run you can just start it with the default parameters.
The following options are needed and have default values:

-  `--epochs default=300`: Number of times that the whole dataset is iterated through. 30 was enough for our pre-trained model.
-  `--batch-size default=16`: the number of training examples in one forward/backward pass. The higher the batch size, the more GPU memory is needed.
-  `--cfg default='models/yolov5s.yaml`: general config file used for training
-  `--data' default='data/coco128.yaml`: yaml file containing information about training data
-  `--img-size default=[640, 640]`: image size for training. Images will be resized automatically to this resolution. Higher resolutions might lead to significantly longer training times.
You can find another detailed guide to create your own dataset [here](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data).

## ⌨ Scripts:

- `train.pt` - training your object detection model
- `test.py` - evaluate the test dataset
- `detect.py` - runs inference on a variety of sources

## ⚖️ License  
  
This program is free software: you can redistribute it and/or modify it under the terms of the **MIT LICENSE** as published by the Free Software Foundation.  
  
**[⬆ back to top](#description)**

> author Serfati
