# importing the required libraries
import os
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import cv2
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import csv

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

class ImageFolderWithPaths(datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

input_path = "FVC-Work/SiameseDatabase/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'Train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'Train': 
    ImageFolderWithPaths(input_path + 'Train', data_transforms['Train'])
}

dataloaders = {
    'Train':
    torch.utils.data.DataLoader(image_datasets['Train'],
                                batch_size=1,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
}


model = models.vgg16(pretrained=True)
    
model.classifier = nn.Sequential(
               nn.Linear(25088, 4096),
               nn.ReLU(inplace=True),
               nn.Dropout(0.5),
               nn.Linear(4096, 4096),
               nn.ReLU(inplace=True),
               nn.Dropout(0.5),
               nn.Linear(4096, 4))
               
model.load_state_dict(torch.load("model/VGG16FVC.h5"))

print(model)
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
        
model.eval()
ftrainArch = open('csv_train3.csv', 'w')
ftrainLabelsArch = open('csv_trainLabels3.csv', 'w')
ftrainPathsArch = open('csv_trainPaths3.csv', 'w')
writertrainArch = csv.writer(ftrainArch)
writertrainLabelsArch = csv.writer(ftrainLabelsArch)
writertrainPathsArch = csv.writer(ftrainPathsArch)

with torch.no_grad():
    for i, (images, labels, path) in enumerate(dataloaders['Train']): 
         #sample_fname, _ = dataloaders.dataset.samples[i] 

        model.classifier[2].register_forward_hook(get_activation('feats'))
        output = model(images)
        Feat=activation['feats']
        Feat=Feat.flatten()
        writertrainArch.writerow(Feat.detach().numpy())
        writertrainLabelsArch.writerow(labels.detach().numpy())
        writertrainPathsArch.writerow(path)
        
ftrainArch.close()
ftrainLabelsArch.close()
ftrainPathsArch.close()
