import cv2
import time
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class face_mask_estimator_faster_rcnn:
    def detect(self, frame):
        model = self.load_model()
        output = model(frame.img)
        boxes = output[0]['boxes'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        labels = output[0]['labels'].data.cpu().numpy()
        frame.masks = []
        for i in range(len(labels)):
            label = "mask"
            if labels[i] == "without_mask":
                label = "no_mask"
            frame.masks.append({(boxes[i]), scores[i], label})
    
    def load_model(self):
        #Faster - RCNN Model - pretrained on COCO
        # use GPU if possible, otherwise use CPU
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.empty_cache()

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        #Retriving all trainable parameters from model (for optimizer)
        params = [p for p in model.parameters() if p.requires_grad]

        #Defininig Optimizer
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

        #Load pre-trained model
        checkpoint = torch.load("weights/fastmask.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.to(device)

        return model

