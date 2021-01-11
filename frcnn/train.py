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

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

DIR_INPUT = "/storage/users/Ise4thYear/OpenCoVid/files/rcnn/data/"
DIR_IMAGES = DIR_INPUT+'train/'

df = pd.read_csv("train.csv")

unq_values = df["name"].unique()

# Total Classes

classes = df["class"].unique()

# adding a background class for Faster R-CNN
_classes = np.insert(classes, 0, "background", axis=0)
class_to_int = {_classes[i]: i for i in range(len(_classes))}
int_to_class = {i: _classes[i] for i in range(len(_classes))}

# Creating Data (Labels & Targets) for Faster R-CNN
class FaceMaskDetectionDataset(Dataset):

    def __init__(self, dataframe, image_dir, mode='train', transforms=None):

        super().__init__()

        self.image_names = dataframe["name"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, index: int):

        # Get Image name and its records (x1, y1, x2, y2, class) from df
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


# Preparing data for Train

def collate_fn(batch):
    return tuple(zip(*batch))


# Dataset object
dataset = FaceMaskDetectionDataset(df, DIR_IMAGES, transforms=get_transform())

# split the dataset in train and test set - using 90% for training, 10% for validation
indices = torch.randperm(len(dataset)).tolist()
# range = len(dataset) * 0.1
train_dataset = torch.utils.data.Subset(dataset, indices[:-195])


# Preparing data loaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)


# use GPU if possible, otherwise use CPU

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()


# Create / load model

# Faster - RCNN Model - pretrained on COCO dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(class_to_int)  # 2: with_mask or without_mask

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Trainable parameters from model (for optimizer)
params = [p for p in model.parameters() if p.requires_grad]

# Optimizer
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

# Learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

model.to(device)

# Number of epochs
epochs = 80

itr = 1
total_train_loss = []

for epoch in np.arange(epochs):

    start_time = time.time()
    train_loss = []

    for images, targets, image_names in train_data_loader:
        
        # Loading images and targets
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        out = model(images, targets)
        losses = sum(loss for loss in out.values())

        optimizer.zero_grad()

        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Avg loss
        loss_value = losses.item()
        train_loss.append(loss_value)

        if itr % 25 == 0:
            print(f"\n Iteration #{itr} loss: {out} \n")

        itr += 1

    lr_scheduler.step()

    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)
    print(f'Epoch train loss is {epoch_train_loss:.4f}')

    time_elapsed = time.time() - start_time
    print("Time elapsed: ", time_elapsed)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_train_loss
    }, "last_checkpoint.pth")

    # torch.save(model, "/")