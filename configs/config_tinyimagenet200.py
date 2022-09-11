import math
from continuum import datasets
from pathlib import Path
from continuum import datasets
import time
import torch
import torchvision.transforms as transforms
from models.resnet18 import resnet18

batch_size = 64
lr = 0.1
start_epoch = 1
num_epochs_non_incremental = 120
num_epochs_incremental = 35
patience_non_incremental= 10
patience_incremental = 5

#do not change

dataset = 'TinyImageNet200'
n_classes = 200
optim_type = 'sgd'

first_task_only = True
num_workers = 3
net_type = 'resnet18'

def get_train_test_dataset():
    path_data = Path.cwd() / 'data' / 'TinyImageNet200'
    imagenet100_train = datasets.TinyImageNet200(path_data, download=True, train=True)
    imagenet100_test = datasets.TinyImageNet200(path_data,download=True,train=False)
    return imagenet100_train,imagenet100_test

def get_transformations():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform =  [
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    test_transform = [
            # transforms.Resize(73),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ]
    return train_transform, test_transform

def get_model():
    r = resnet18(pretrained=False, num_classes=200)
    r.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    return r