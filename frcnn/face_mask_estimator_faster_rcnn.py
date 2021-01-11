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
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms


class face_mask_estimator_faster_rcnn:
    def detect(self, frame):
        model = self.load_model()
        img = self.transform_image(frame.img)
        output = model([img])
        bbox = output[0]['boxes'].data.cpu().numpy()
        bbox = pd.DataFrame(bbox)
        bbox.columns = ["x1", "y1", "x2", "y2"]
        
        scores = output[0]['scores'].data.cpu().numpy()
        
        labels = output[0]['labels'].data.cpu().numpy()

        frame.masks = []

        for idx, row in bbox.iterrows():
            score = scores[idx]
            if score < 0.5:
                continue
            x1 = row['x1']
            y1 = row['y1']
            x2 = row['x2']
            y2 = row['y2']
            label = labels[idx]
            if label == 1:
                label = "mask"
            elif label == 2:
                label = "no_mask"
            frame.masks.append([[x1, y1, x2, y2], score, label])
        
            
    def load_model(self):
        #Faster RCNN (pretrained on COCO database)
        # use GPU if possible, otherwise use CPU
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.empty_cache()

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        a = list(model.parameters())[0]
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        num_classes = 3 # background class is '0' (FASTER RCNN)

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        checkpoint = torch.load("weights/last_checkpoint.pth", map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        b = list(model.parameters())[0]
        model.to(device)
        model.eval()

        return model


    def transform_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        transform = T.Compose([T.ToTensor()])
        image = transform(image)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.empty_cache()
        image = image.to(device)
        return image



# Plot image
def plot_img(output, image_name):
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 14))
    ax = ax.flatten()
    
    bbox = output[0]['boxes'].data.cpu().numpy()
    labels = output[0]['labels'].data.cpu().numpy()
    bbox = pd.DataFrame(bbox)
    bbox.columns = ["x1", "y1", "x2", "y2"]

    img_path = image_name
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image2 = image
    
    ax[0].set_title('Original Image')
    ax[0].imshow(image)
    
    for idx, row in bbox.iterrows():
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        label = labels[idx]
        if label == 1:
            label = "with_mask"
        else:
            label = "without_mask"
        
        cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image2, label, (int(x1),int(y1-10)), font, 1, (255,0,0), 2)
    
    ax[1].set_title('Image with Bondary Box')
    ax[1].imshow(image2)

    plt.show()


'''img_path = 'ZIAJVXRFDLY2Q1N75WGK.jpg'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0
transform = T.Compose([T.ToTensor()])
image = transform(image)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
image = image.to(device)
temp = face_mask_estimator_faster_rcnn()
temp.detect(image)
'''