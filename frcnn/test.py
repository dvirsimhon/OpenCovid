import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

DIR_INPUT = "C:\\Users\\Liron Simhon\\Desktop\\דביר\\לימודים\\שנה ד\\סמסטר א\\פרויקט גמר\\Data\\"
DIR_IMAGES = DIR_INPUT + 'for testing\\'

df = pd.read_csv("test.csv")

# Total Classes
classes = df["class"].unique()

# adding a background class for Faster R-CNN
# _classes = np.insert(classes, 2, "person", axis=0)
class_to_int = {classes[i]: i for i in range(len(classes))}
int_to_class = {i: classes[i] for i in range(len(classes))}


# Creating Data (Labels & Targets) for Faster R-CNN
class FaceMaskDetectionDataset(Dataset):

    def __init__(self, dataframe, image_dir, mode, transforms=None):

        super().__init__()

        self.image_names = dataframe["name"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, index: int):

        # Retrieve Image name and its records (x1, y1, x2, y2, class) from df
        image_name = self.image_names[index]
        records = self.df[self.df["name"] == image_name]
        # Loading Image
        image = cv2.imread(self.image_dir + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.mode == 'train':

            # Get bounding box co-ordinates for each box
            boxes = records[['x1', 'y1', 'x2', 'y2']].values

            # Getting labels for each box
            temp_labels = records[['class']].values
            labels = []
            for label in temp_labels:
                label = class_to_int[label[0]]
                labels.append(label)

            # Converting boxes & labels into torch tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Creating target
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels

            # Transforms
            if self.transforms:
                image = self.transforms(image)

            return image, target, image_name

        elif self.mode == 'test':

            if self.transforms:
                image = self.transforms(image)

            return image, image_name

    def __len__(self):
        return len(self.image_names)


def get_transform():
    return T.Compose([T.ToTensor()])


# Preparing data for Validation

def collate_fn(batch):
    return tuple(zip(*batch))


# Dataset object
dataset = FaceMaskDetectionDataset(df, DIR_IMAGES, mode='test', transforms=get_transform())

# split the dataset in train and test set - 10% for validation
indices = torch.randperm(len(dataset)).tolist()
# ranges = len(dataset) * 0.1
test_dataset = torch.utils.data.Subset(dataset, indices[:])

test_data_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

# use GPU if possible, otherwise use CPU
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

# load model
# Faster - RCNN Model - pretrained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set trainable parameters from model (for optimizer)
params = [p for p in model.parameters() if p.requires_grad]

# Defininig Optimizer
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

# Load pre-trained model
checkpoint = torch.load("weights/fastmask.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

start_time = time.time()

correct = 0
total = 0
for data in test_data_loader:
    images, image_names = data
    images = list(image.to(device) for image in images)
    recs = test_dataset.data.df[test_dataset.data.df["name"].isin(image_names)]
    targets = recs["class"]
    outputs = model(images)
    labels = outputs[0]['labels'].data.cpu().numpy()
    # int_to_class - labels
    labels_th = []
    for x in range(len(labels)):
        labels_th.append(int_to_class[labels[x]])
    total += len(targets)
    correct += (targets == labels_th).sum()

accuracy = correct / float(total)
print(f'Testing Accuracy: {accuracy:.3f}')
