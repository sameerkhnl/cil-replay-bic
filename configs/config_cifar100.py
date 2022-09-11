import math
from continuum import datasets
import models.resnet32 as resnet32
from pathlib import Path
from continuum import datasets
import time
import torchvision.transforms as transforms
import numpy as np


batch_size = 128
lr = 0.1
start_epoch = 1
num_epochs_non_incremental = 300
num_epochs_incremental = 70
patience_non_incremental= 60
patience_incremental = 15

#do not change
dataset = 'CIFAR100'
n_classes = 100

optim_type = 'sgd'

num_workers = 3

net_type = 'resnet32'


def get_train_test_dataset():
    path_data = Path.cwd() / 'data' / 'CIFAR100'
    cifar100_train = datasets.CIFAR100(path_data, download=True, train=True)
    cifar100_test = datasets.CIFAR100(path_data,download=True,train=False)
    return cifar100_train,cifar100_test


def get_transformations():
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    transform_train = [transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

    transform_test = [transforms.ToTensor(), normalize]
    
    return transform_train, transform_test
    # return transform_train, transform_test

def get_model():
    return resnet32.resnet32()

def get_fine_labels():
    return ['apples', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cups', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges', 'orchids', 'otter', 'palm', 'pears', 'pickup_truck', 'pine', 'plain', 'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflowers', 'sweet_peppers', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm']

def get_coarse_labels():
    dict_labels = {}
    dict_labels['aquatic mammals'] = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
    dict_labels['fish'] = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    dict_labels['flowers'] = ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips']
    dict_labels['food containers'] = ['bottles', 'bowls', 'cans', 'cups', 'plates']
    dict_labels['fruit and vegetables'] = ['apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers']
    dict_labels['household electrical devices'] = ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television']
    dict_labels['household furniture'] = ['bed', 'chair', 'couch', 'table', 'wardrobe']
    dict_labels['insects'] = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
    dict_labels['large carnivores'] = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
    dict_labels['large man-made outdoor things'] = ['bridge', 'castle', 'house', 'road', 'skyscraper']
    dict_labels['large natural outdoor scenes'] = ['cloud', 'forest', 'mountain', 'plain', 'sea']
    dict_labels['large omnivores and herbivores'] = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
    dict_labels['medium-sized mammals'] = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
    dict_labels['non-insect invertebrates'] = ['crab', 'lobster', 'snail', 'spider', 'worm']
    dict_labels['people'] = ['baby', 'boy', 'girl', 'man', 'woman']
    dict_labels['reptiles'] = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
    dict_labels['small mammals'] = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    dict_labels['trees'] = ['maple', 'oak', 'palm', 'pine', 'willow']
    dict_labels['vehicles 1'] = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']
    dict_labels['vehicles 2'] = ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    return dict_labels

#specific ordering, each group of 5 classes comes from the same superclass
def get_classes_specific(seed):
    if seed == 0:
        return np.array([ 4, 30, 55, 72, 95,  1, 32, 67, 73, 91, 54, 62, 70, 82, 92,  9, 10,
       16, 28, 61, 34, 63, 64, 66, 75,  8, 13, 48, 58, 90, 23, 33, 49, 60,
       71, 47, 52, 56, 59, 96,  3, 42, 43, 88, 97,  6,  7, 14, 18, 24, 36,
       50, 65, 74, 80, 27, 29, 44, 78, 93,  5, 20, 25, 84, 94, 41, 69, 81,
       85, 89,  2, 11, 35, 46, 98, 22, 39, 40, 86, 87,  0, 51, 53, 57, 83,
       15, 19, 21, 31, 38, 26, 45, 77, 79, 99, 12, 17, 37, 68, 76])
       
    elif seed == 1:
        return np.array([ 4, 30, 55, 72, 95,  1, 32, 67, 73, 91, 54, 62, 70, 82, 92,  9, 10,
       16, 28, 61,  2, 11, 35, 46, 98,  3, 42, 43, 88, 97, 22, 39, 40, 86,
       87,  6,  7, 14, 18, 24, 36, 50, 65, 74, 80, 26, 45, 77, 79, 99, 12,
       17, 37, 68, 76, 27, 29, 44, 78, 93,  0, 51, 53, 57, 83,  8, 13, 48,
       58, 90, 23, 33, 49, 60, 71, 41, 69, 81, 85, 89, 47, 52, 56, 59, 96,
       34, 63, 64, 66, 75, 15, 19, 21, 31, 38,  5, 20, 25, 84, 94])

    elif seed == 2:
        return np.array([ 4, 30, 55, 72, 95,  1, 32, 67, 73, 91, 54, 62, 70, 82, 92,  9, 10,
       16, 28, 61,  8, 13, 48, 58, 90,  2, 11, 35, 46, 98, 36, 50, 65, 74,
       80, 27, 29, 44, 78, 93, 47, 52, 56, 59, 96,  0, 51, 53, 57, 83, 15,
       19, 21, 31, 38,  6,  7, 14, 18, 24, 23, 33, 49, 60, 71, 12, 17, 37,
       68, 76, 26, 45, 77, 79, 99,  5, 20, 25, 84, 94, 34, 63, 64, 66, 75,
        3, 42, 43, 88, 97, 41, 69, 81, 85, 89, 22, 39, 40, 86, 87])

    #seeds 3,4,5 contain a different set of non-incremental tasks so should be compared separately to seeds 0,1,2
    elif seed == 3:
        return np.array([12, 17, 37, 68, 76, 47, 52, 56, 59, 96, 22, 39, 40, 86, 87, 41, 69,
       81, 85, 89,  3, 42, 43, 88, 97,  4, 30, 55, 72, 95,  5, 20, 25, 84,
       94,  2, 11, 35, 46, 98, 23, 33, 49, 60, 71,  0, 51, 53, 57, 83,  1,
       32, 67, 73, 91,  8, 13, 48, 58, 90, 36, 50, 65, 74, 80, 54, 62, 70,
       82, 92, 26, 45, 77, 79, 99, 15, 19, 21, 31, 38,  9, 10, 16, 28, 61,
       27, 29, 44, 78, 93,  6,  7, 14, 18, 24, 34, 63, 64, 66, 75])

    elif seed == 4:
        return np.array([12, 17, 37, 68, 76, 47, 52, 56, 59, 96, 22, 39, 40, 86, 87, 41, 69,
       81, 85, 89, 34, 63, 64, 66, 75,  6,  7, 14, 18, 24, 23, 33, 49, 60,
       71,  5, 20, 25, 84, 94,  8, 13, 48, 58, 90,  9, 10, 16, 28, 61,  1,
       32, 67, 73, 91,  2, 11, 35, 46, 98, 27, 29, 44, 78, 93, 15, 19, 21,
       31, 38,  0, 51, 53, 57, 83, 54, 62, 70, 82, 92,  4, 30, 55, 72, 95,
        3, 42, 43, 88, 97, 26, 45, 77, 79, 99, 36, 50, 65, 74, 80])

    elif seed == 5:
        return np.array([12, 17, 37, 68, 76, 47, 52, 56, 59, 96, 22, 39, 40, 86, 87, 41, 69,
       81, 85, 89,  0, 51, 53, 57, 83, 36, 50, 65, 74, 80, 27, 29, 44, 78,
       93,  4, 30, 55, 72, 95,  3, 42, 43, 88, 97, 23, 33, 49, 60, 71,  5,
       20, 25, 84, 94,  8, 13, 48, 58, 90, 54, 62, 70, 82, 92,  2, 11, 35,
       46, 98, 26, 45, 77, 79, 99,  1, 32, 67, 73, 91,  9, 10, 16, 28, 61,
       34, 63, 64, 66, 75, 15, 19, 21, 31, 38,  6,  7, 14, 18, 24])


#the following three are methods useful mostly in jupyter notebook for further analysis
def get_indexes_by_task(tid, class_order, init_incr=20, incr=5):
      if tid == 0:
        return class_order[0:init_incr]
      else:
        return class_order[init_incr+(tid-1)*incr: init_incr + tid*incr]
    
#get original permuted version of labels from the transformed labels 
def get_y_original(y, class_order):
    return class_order[y]

#pass a list of labels and get their corresponding class_indexes
def get_class_ids_by_labels(search_labels, fine_labels):
    return np.where(np.isin(fine_labels, search_labels))[0]

#convert list of lists to ndarray
def _convert_to_ndarray(arr):
    d1 = len(arr)
    d2 = len(arr[-1])
    new = np.full((d1,d2),0.0, dtype='float32')
    for i, val in enumerate(arr):
        for j,_ in enumerate(val):
            new[i][j] = val[j]
    return new

#for all seeds, agents convert the pickled list of lists to ndarray
def convert_file_to_numpy(load):
    converted = {}
    for seed in load.keys():
        converted[seed] = {}
        for method in load[seed].keys():
            converted[seed][method] = _convert_to_ndarray(load[seed][method])
    return converted

#get accuracy on previous tasks (average) vs current task
def get_pre_curr_acc(arrs:np.ndarray):
    full = np.full((arrs.shape[0],2), 0, dtype='float32')
    for i,arr in enumerate(arrs):
        prev_task = np.mean(arr[:i])
        cur_task = np.mean(arr[i])
        full[i][0] = prev_task
        full[i][1] = cur_task
    return full

