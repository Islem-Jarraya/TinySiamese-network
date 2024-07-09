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

#input_path = "/home/regimlab/Documents/Databases/FVC2004/DB1_B/"
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
    ImageFolderWithPaths(input_path + 'Test', data_transforms['Train'])
}

dataloaders = {
    'Train':
    torch.utils.data.DataLoader(image_datasets['Train'],
                                batch_size=1,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
}


#model = models.alexnet(pretrained=True)

#model.classifier = nn.Sequential(
##                nn.Dropout(p=0.5, inplace=False),
#                nn.Linear(in_features=9216, out_features=4096, bias=True),
#                nn.ReLU(inplace=True),
##                nn.Dropout(p=0.5, inplace=False),
#                nn.Linear(in_features=4096, out_features=4096, bias=True),
#                nn.ReLU(inplace=True),
#                nn.Linear(in_features=4096, out_features=4, bias=True))

#model.load_state_dict(torch.load("model/alexnetFVC.h5"))

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
#ftrainArch = open('Features/Test/Arch/csv_test2.csv', 'w')
#ftrainLabelsArch = open('Features/Test/Arch/csv_testLabels2.csv', 'w')
#ftrainPathsArch = open('Features/Test/Arch/csv_testPaths2.csv', 'w')
#writertrainArch = csv.writer(ftrainArch)
#writertrainLabelsArch = csv.writer(ftrainLabelsArch)
#writertrainPathsArch = csv.writer(ftrainPathsArch)

ftrainWhorl = open('Features/Test/Whorl/csv_test2.csv', 'w')
ftrainLabelsWhorl = open('Features/Test/Whorl/csv_testLabel2.csv', 'w')
ftrainPathsWhorl = open('Features/Test/Whorl/csv_testPaths2.csv', 'w')
writertrainWhorl = csv.writer(ftrainWhorl)
writertrainLabelsWhorl = csv.writer(ftrainLabelsWhorl)
writertrainPathsWhorl = csv.writer(ftrainPathsWhorl)

#ftrainRightLoop = open('Features/Test/RightLoop/csv_test2.csv', 'w')
#ftrainLabelsRightLoop = open('Features/Test/RightLoop/csv_testLabels2.csv', 'w')
#ftrainPathsRightLoop = open('Features/Test/RightLoop/csv_testPaths2.csv', 'w')
#writertrainRightLoop = csv.writer(ftrainRightLoop)
#writertrainLabelsRightLoop = csv.writer(ftrainLabelsRightLoop)
#writertrainPathsRightLoop = csv.writer(ftrainPathsRightLoop)

#ftrainLeftLoop = open('Features/Test/LeftLoop/csv_test2.csv', 'w')
#ftrainLabelsLeftLoop = open('Features/Test/LeftLoop/csv_testLabels2.csv', 'w')
#ftrainPathsLeftLoop = open('Features/Test/LeftLoop/csv_testPaths2.csv', 'w')
#writertrainLeftLoop = csv.writer(ftrainLeftLoop)
#writertrainLabelsLeftLoop = csv.writer(ftrainLabelsLeftLoop)
#writertrainPathsLeftLoop = csv.writer(ftrainPathsLeftLoop)

with torch.no_grad():
    for i, (images, labels, path) in enumerate(dataloaders['Train']): 
         #sample_fname, _ = dataloaders.dataset.samples[i] 

        model.classifier[2].register_forward_hook(get_activation('feats'))
        output = model(images)
        Feat=activation['feats']
        Feat=Feat.flatten()
        if 'Arch' in path[0]:
          writertrainArch.writerow(Feat.detach().numpy())
          writertrainLabelsArch.writerow(labels.detach().numpy())
          writertrainPathsArch.writerow(path)
        if "Whorl" in path[0]:
          writertrainWhorl.writerow(Feat.detach().numpy())
          writertrainLabelsWhorl.writerow(labels.detach().numpy())
          writertrainPathsWhorl.writerow(path)
        if "RightLoop" in path[0]:
          writertrainRightLoop.writerow(Feat.detach().numpy())
          writertrainLabelsRightLoop.writerow(labels.detach().numpy())
          writertrainPathsRightLoop.writerow(path)
        if "LeftLoop" in path[0]:
          writertrainLeftLoop.writerow(Feat.detach().numpy())
          writertrainLabelsLeftLoop.writerow(labels.detach().numpy())
          writertrainPathsLeftLoop.writerow(path)
ftrainArch.close()
ftrainLabelsArch.close()
ftrainPathsArch.close()

ftrainWhorl.close()
ftrainLabelsWhorl.close()
ftrainPathsWhorl.close()

ftrainRightLoop.close()
ftrainLabelsRightLoop.close()
ftrainPathsRightLoop.close()

ftrainLeftLoop.close()
ftrainLabelsLeftLoop.close()
ftrainPathsLeftLoop.close()
