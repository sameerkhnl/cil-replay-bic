from audioop import avg
from genericpath import exists
from typing import overload
import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
from continuum.metrics.logger import Logger
from continuum.tasks.base import BaseTaskSet, TaskType
from continuum.rehearsal.memory import RehearsalMemory
import time
from Args import Args
import utils
import copy
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from continuum.tasks.task_set import TaskSet, PathTaskSet
from train_eval import eval
from scipy.stats import rankdata
import torchvision.transforms as transforms
from continuum.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset, H5Dataset
)

model_avg = {}
class_avg_exemplars = {}
class_avg = {}
old_class_means = {}


def append_class_avg(logits, args:Args, tid):
    if tid > 1:
        old_classes = np.array(list(class_avg.keys()))
        ocm = np.mean(logits, axis=0)[old_classes]
        for c, m in zip(old_classes, ocm):
            old_class_means[c] = m

    classes_task = args.scenario_train[tid].get_classes()
    means = np.mean(logits, axis=0)[classes_task]
    for c,m in zip(classes_task, means):
        class_avg[c] = m
    

def append_mu_model(logits, args:Args, tid):
    append_class_avg(logits, args, tid)

    seen_classes = list(args.mem.seen_classes)
    m = np.mean(logits[:, seen_classes])
    y_task = args.scenario_train[tid].get_classes()

    for class_id in y_task:
      model_avg[class_id] = m

def append_mu_model2(logits, args:Args, tid):
    append_class_avg(logits, args, tid)
    model_conf = np.mean(np.max(logits,1))
    y_task = args.scenario_train[tid].get_classes()

    for class_id in y_task:
        model_avg[class_id] = model_conf

def append_class_avg_exemplars(x,y, t, logits, tid):
    indexes_new_exemplars = np.where(t == tid)[0]
    _x,_y,_t,_logits = x[indexes_new_exemplars], y[indexes_new_exemplars], t[indexes_new_exemplars], logits[indexes_new_exemplars]
    
    for new_class_id in np.unique(_y):
        class_indexes = np.where(_y == new_class_id)[0]
        l = _logits[class_indexes][:,new_class_id]

        class_avg_exemplars[new_class_id] = np.mean(l)

def get_cfactor_append_exemplar_class_avg(x,y,t,logits, args, tid):

    cfactor = _get_cfactor(x,y,t,logits,tid, args)

    append_class_avg_exemplars(x,y,t, logits, tid)
    return cfactor

def get_cfactor_append_exemplar_class_avg2(x,y,t,logits_mixed, logits_replay, args, tid):
    # cfactor = _get_cfactor(x,y,t,logits_replay,tid, args)
    cfactor = _get_cfactor2(x,y,t,logits_mixed,tid, args)

    append_class_avg_exemplars(x,y,t, logits_replay, tid)
    return cfactor

# def get_cfactor(x,y,t,logits, tid):

def _compute_mu_t(x,y, t, logits, tid):
    indexes_old_exemplars = np.where(t < tid)[0]
    _x,_y,_t,_logits = x[indexes_old_exemplars], y[indexes_old_exemplars], t[indexes_old_exemplars], logits[indexes_old_exemplars]

    mu_t = {}
    
    for old_class_id in np.unique(_y):
        class_indexes = np.where(_y == old_class_id)[0]
        l = _logits[class_indexes][:,old_class_id]
        mu_t[old_class_id] = np.mean(l)
        
    return mu_t

def _compute_mu_t(x,y, t, logits, tid):
    indexes_old_exemplars = np.where(t < tid)[0]
    _x,_y,_t,_logits = x[indexes_old_exemplars], y[indexes_old_exemplars], t[indexes_old_exemplars], logits[indexes_old_exemplars]

    mu_t = {}
    
    for old_class_id in np.unique(_y):
        class_indexes = np.where(_y == old_class_id)[0]
        l = _logits[class_indexes][:,old_class_id]
        mu_t[old_class_id] = np.mean(l)
        
    return mu_t

def get_mean_new_classes(x,y,t,logits,tid):
    indexes_new_exemplars = np.where(t == tid)[0]
    _x,_y,_t,_logits = x[indexes_new_exemplars], y[indexes_new_exemplars], t[indexes_new_exemplars], logits[indexes_new_exemplars]
    l_new_avg = []

    for new_class_id in np.unique(_y):
        class_indexes = np.where(_y == new_class_id)[0]
        l = _logits[class_indexes][:,new_class_id]

        l_new_avg.append(np.mean(l))
    return np.mean(l_new_avg)

def _get_cfactor(x,y,t,logits, tid, args:Args):
    cfactor = np.ones(args.n_classes)
    indexes_old_exemplars = np.where(t < tid)[0]
    indexes_new_exemplars = np.where(t == tid)[0]
    old_classes = np.unique(y[indexes_old_exemplars])

    one_current_class = y[indexes_new_exemplars][0]
    model_avg_current = model_avg[one_current_class]

    mu_t = _compute_mu_t(x,y,t,logits,tid)

    m_n = get_mean_new_classes(x,y,t,logits, tid)

    for o_cid in old_classes:
        mup, mut = class_avg_exemplars[o_cid], mu_t[o_cid]
        mac, mao = model_avg_current, model_avg[o_cid]

        r1 = max(mup / mut,1)
        r2 = mac / mao
        c1 = class_avg[o_cid] / old_class_means[o_cid]

        cf = np.abs(r1 * r2)
        if cf > 1:
            cfactor[o_cid] = cf
    
    return cfactor

def _get_cfactor2(x,y,t,logits, tid, args:Args):
    cfactor = np.ones(args.n_classes)
    old_classes = np.array(list(old_class_means.keys()))
    one_new_class =args.scenario_train[tid].get_classes()[0]
    mac = model_avg[one_new_class]
    for c in old_classes:
        r1 = class_avg[c] / old_class_means[c]
        r2 = model_avg[one_new_class] / model_avg[c]

        cf = r1*r2
        cfactor[c] = cf
    return cfactor




# def _append_cfactor(x,y,t,logits, args:Args, cfactor, tid):
#     logits = _get_softmax(logits)

#     y_new = rankdata(y, method='dense') - 1

#     zp = zip(np.unique(y_new), np.unique(y))

#     for cid1,cid2 in zp:
#         mu_p = args.stats_bias.class_avg_exemplars[cid1]
#         class_indexes = np.where(cid1 == y_new)[0]
#         mu_t = np.mean(logits[class_indexes][:,cid1])

#         # mu_p += np.abs(min(0,mu_t))
#         # mu_t = 1 if mu_t <= 0 else mu_t


#         model_avg_past = args.stats_bias.model_avg[cid1]
#         model_avg_current = args.stats_bias.model_avg[-1]
#         val = (mu_p / mu_t) * (model_avg_current / model_avg_past) 
#         # cfactor[cid2] = val**(1/tid)
#         cfactor[cid2] = val






    # np.savez('exemplars', x=x,y=y,t=t, logits=logits)




    
    






