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

input_path = "Database/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'Test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'Test': 
    ImageFolderWithPaths(input_path + 'Test', data_transforms['Test'])
}

dataloaders = {
    'Test':
    torch.utils.data.DataLoader(image_datasets['Test'],
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
ftestArch = open('csv_test.csv', 'w')
ftestLabelsArch = open('csv_testLabels.csv', 'w')
ftestPathsArch = open('csv_testPaths.csv', 'w')
writertestArch = csv.writer(ftestArch)
writertestLabelsArch = csv.writer(ftestLabelsArch)
writertestPathsArch = csv.writer(ftestPathsArch)

with torch.no_grad():
    for i, (images, labels, path) in enumerate(dataloaders['Test']): 
         #sample_fname, _ = dataloaders.dataset.samples[i] 

        model.classifier[2].register_forward_hook(get_activation('feats'))
        output = model(images)
        Feat=activation['feats']
        Feat=Feat.flatten()
        writertestArch.writerow(Feat.detach().numpy())
        writertestLabelsArch.writerow(labels.detach().numpy())
        writertestPathsArch.writerow(path)
        
ftestArch.close()
ftestLabelsArch.close()
ftestPathsArch.close()
