# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb
import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

# import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


df1 = pd.read_csv('/dockerx/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male'])

# Make 0 (female) & 1 (male) labels instead of -1 & 1
df1.loc[df1['Male'] == -1, 'Male'] = 0

df1.head()


df2 = pd.read_csv('/dockerx/list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
df2.columns = ['Filename', 'Partition']
df2 = df2.set_index('Filename')

df2.head()


df3 = df1.merge(df2, left_index=True, right_index=True)
df3.head()


df3.to_csv('celeba-gender-partitions.csv')
df4 = pd.read_csv('celeba-gender-partitions.csv', index_col=0)
df4.head()

df4.loc[df4['Partition'] == 0].to_csv('celeba-gender-train.csv')
df4.loc[df4['Partition'] == 1].to_csv('celeba-gender-valid.csv')
df4.loc[df4['Partition'] == 2].to_csv('celeba-gender-test.csv')

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
	self.img_dir = img_dir
	self.csv_path = csv_path
	self.img_names = df.index.values
	self.y = df['Male'].values
	self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
	                              self.img_names[index]))
        
        if self.transform is not None:
	            img = self.transform(img)

        label = self.y[index]
	return img, label

    def __len__(self):
        return self.y.shape[0]



# Note that transforms.ToTensor()
# already divides pixels by 255. internally

custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
				       #transforms.Grayscale(),
				       #transforms.Lambda(lambda x: x/255.),
				       transforms.ToTensor()])

train_dataset = CelebaDataset(csv_path='celeba-gender-train.csv',
                              img_dir='/dockerx/img_align_celeba/',
			      transform=custom_transform)

valid_dataset = CelebaDataset(csv_path='celeba-gender-valid.csv',
                              img_dir='/dockerx/img_align_celeba/',
			      transform=custom_transform)

test_dataset = CelebaDataset(csv_path='celeba-gender-test.csv',
                             img_dir='/dockerx/img_align_celeba/',
			     transform=custom_transform)

BATCH_SIZE=64*torch.cuda.device_count()


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
			  shuffle=True,
			  num_workers=0)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
			  shuffle=False,
			  num_workers=0)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
			 shuffle=False,
			 num_workers=0)


device = torch.device("cuda:0")
torch.manual_seed(0)

num_epochs = 2
for epoch in range(num_epochs):

    for batch_idx, (x, y) in enumerate(train_loader):

        print('Epoch:', epoch+1)
	print(' | Batch index:', batch_idx)
	print(' | Batch size:', y.size()[0])

        x = x.to(device)
	y = y.to(device)
	break

##########################
### SETTINGS
##########################

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 3

# Architecture
num_features = 128*128
num_classes = 2


import torchvision.models as models

vgg16 = models.vgg16()

print(vgg16)
