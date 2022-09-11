from calendar import c
import time
import datetime
from pathlib import Path
import csv
from typing import List
import torch
import argparse
from continuum.scenarios.class_incremental import ClassIncremental
from Args import Args
from replay.cl_margin import CL_MARGIN_CLASS_AVG, CL_MARGIN_H80_L20, CL_MARGIN_HIGH, CL_MARGIN_LOW, CL_MARGIN_HIGH_ALL, CL_MARGIN_LOW_ALL
import numpy as np
from continuum.generators.scenarios_generator import TaskOrderGenerator
import copy
import pandas as pd
import sys
from continuum.tasks.base import BaseTaskSet, TaskType
from continuum.tasks import TaskSet
import torchvision.transforms as transforms
import random
import os
import math


def get_hms(seconds):
    ty_res = time.gmtime(seconds)
    res = time.strftime("%H:%M:%S",ty_res)
    return res

#Path to the three files that are output from the experiment if --output_to_file is provided in args
def get_output_filepath(dataset_results_path, exp_info):
    filepath_main = dataset_results_path / f'{dataset_results_path}' / f'{exp_info}.csv'
    filepath_plot = dataset_results_path / f'{dataset_results_path}' / f'{exp_info}.pdf'
    filepath_acc_per_task = dataset_results_path / f'{dataset_results_path}' / f'{exp_info}.pkl'
    return filepath_main, filepath_plot, filepath_acc_per_task

#Folder to save results from the current experiment: results/{experiment details including timestamp}
def get_dataset_results_path(dataset, perm, extra_info:str=''):
    _extra_info = f'_{extra_info.strip()}' if extra_info != '' else ''

    return Path.cwd() / 'results' / \
    f'{dataset}_{datetime.datetime.now().strftime("%d-%b-%Y_%H_%M_%S")}_perm{perm}_{_extra_info}'

#if --save_memory in args, the path where exemplar memory is stored 
#the parent folder is /store/exemplars
def get_memory_path(args:Args, agent, exp_info):
    memory_dir = Path.cwd() / 'store' / 'exemplars'
    Path.mkdir(memory_dir, parents=True, exist_ok=True)
    _seed = f'_seed{args.seed}' if args.save_memory else '' 
    memory_path = memory_dir / f'{agent}_{exp_info}{_seed}.npz'
    return memory_path

#if --output_to_file is provided in args, initialise the csv file with the appropirate columns
def initialise_file(dataset_results_path, filepath, n_tasks):
    Path.mkdir(dataset_results_path, parents=True, exist_ok=True)
    header = [f'task{i}' for i in range(1,n_tasks)]
    header.append('agent')
    header.append('seed')
    header.append('perm')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

#append saved results into the 'results_for_loading' folder with csv file from earlier experiments to create a single plot that compares results from earlier experiments with results from current experiment
def load_saved_results(args:Args, append_filepath):
    _results_path = Path.cwd() / 'results_for_loading' / f'{args.exp_info_long}.csv'

    if not Path.exists(_results_path):
        return

    df = pd.read_csv(_results_path).round(3)
    df.to_csv(append_filepath, header=False, index=False, mode='a')
    print(f'Successfully loaded previously saved results for the same experiment .... ')


#Get different levels of info about the experiment, used to name model and result files       
def get_exp_info(args:Args):
    _ordering = f'{args.ordering}-order'
    _memory_type = f'{args.memory_type}-memory'

    #e.g. t900-21 means initial task (non-incremental) of 900 classes and then 20 tasks for the remaining 100 classes
    #e100-35 means 100 epochs for the first (non incremental) task and 35 for the rest 

    ds = f'{args.dataset}-subset{args.subset_version}' if args.subset_version is not None else args.dataset
    _distill = f'_distill-temp{args.T}-lamb{args.lamb}' if args.distill else ''
    _oversample = '_oversample' if args.oversample else '' 

    exp_info_short = f'{ds}_t{args.initial_increment}-{args.n_tasks}_{_ordering}' \
    f'_{args.optim_type}_{args.net_type}{_oversample}{_distill}'

    _noise_info = ''

    if args.noise_type == 'gaussian':
        _noise_info = f'_gaussian-noise-mean{args.noise_mean}-var{args.noise_var}'

    _specific_order = ''
    if args.specific_order:
        _specific_order = '_specific-order'

    exp_info_short = f'{exp_info_short}{_noise_info}{_specific_order}_perm{args.perm}'
    exp_info_long = f'{exp_info_short}_m{args.memory_per_class}-pc_lr{args.lr}_batch{args.batch_size}_{_memory_type}'

    exp_info_base = f'{ds}_t{args.initial_increment}_{_ordering}' \
    f'_{args.optim_type}_{args.net_type}_perm{args.perm}'

    return exp_info_long, exp_info_short, exp_info_base


#get model names for each task. The task 0 and 1 models are shared by all methods. 
def get_model_info(args:Args, tid, agent='baseline', best=False):
    post_script = None
    state = 'best' if best else 'latest'

    if args.agent == 'load_from_file':
        agent = f'load_from_file_{args.exemplars_path.name}'

    exp_info = None

    if tid == 0:
        post_script = f'task{tid}_non-incremental_baseline_{state}'
        exp_info = args.exp_info_base
    elif tid == 1:
        post_script = f'seed{args.seed}_task{tid}_incremental_baseline_{state}'
        exp_info = args.exp_info_short
    else:
        post_script = f'seed{args.seed}_task{tid}_incremental_{agent}_{state}'
        mem_info = ''
        if args.agent != 'baseline':
            if args.memory_type == 'flexible':
                mem_info = f'_{args.memory_type}-memory'
                post_script = f'{post_script}{mem_info}'

        exp_info = args.exp_info_short


    return f'{exp_info}_{post_script}'

#simple wrapper function to save model state  
def save_model(state, model_save_path):
    #checking of existing directory is already done in main.py
    torch.save(state, model_save_path)

#when printing training state, get the latest lr information from the optimiser. 
def get_current_lr_from_optimizer(optimizer:torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#The main function that processes that appends metrics into the csv file. 
def process_overall_acc(args:Args, overall_acc_list, agent):
    def _append_metrics_to_file(filepath, overall_acc_list, agent_name, seed, perm):
        agent_name_transform = {'cl_margin_high': 'CORE-high', 'cl_margin_low': 'CORE-low','herding_icarl':'Herding', 'baseline': 'Baseline', 'random':'Random', 'cl_margin_class_avg': 'CORE-avg'}
        an = agent_name
        if agent_name in agent_name_transform.keys():
            an = agent_name_transform[agent_name]

        overall_acc_list.append(an)
        overall_acc_list.append(seed)
        overall_acc_list.append(perm)
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(overall_acc_list)

    overall_acc = np.array(overall_acc_list)
    overall_avg_acc = np.mean(overall_acc)
    print(f'\nAverage incremental accuracy for {agent}: {overall_avg_acc:.3f}')
    overall_acc_list = [round(i,3) for i in overall_acc_list]
    overall_acc_list_copy = copy.deepcopy(overall_acc_list)
    agent = f'{agent}_{args.exemplars_path.name}' if agent == 'load_from_file' else agent
    if args.output_to_file:
        _append_metrics_to_file(args.filepath_main, overall_acc_list_copy, agent, args.seed, args.perm)

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='CIFAR100 (default) | TinyImageNet200')
    parser.add_argument('--verbose', default=False, action='store_true', help='prints loss values after every iteration') 
    parser.add_argument('--save_memory', default=False, action='store_true')
    parser.add_argument('--output_to_file',default=False, action='store_true')

    parser.add_argument('--repeat', type=int, default=1, help='# of experiments')
    parser.add_argument('--memory_type', type=str, default='fixed', help='fixed | flexible')
    # parser.add_argument('--ordering', type=str, default='random', help='sequential (default) | random')
    parser.add_argument('--initial_increment', type=int, default=50, help='# initial increment corresponding non-incremental learning')
    parser.add_argument('--increment', type=int, default=5, help='further increments')
    parser.add_argument('--correct_bias', default=False, action='store_true')
    parser.add_argument('--bic_method', default='bican', help='bias correction method: choice of il2m, bican,nem')

    parser.add_argument('--eval_on_task_zero', default=False, action='store_true', help='by default evaluation is not done on base classes')
    parser.add_argument('--save_yhat_y', default=False, action='store_true', help='save predicted classes vs actual classes information during evaluation')
    parser.add_argument('--save_weights_last_layer', default=False, action='store_true')
    parser.add_argument('--oversample', default=False, action='store_true')
    parser.add_argument('--distill', default=False, action='store_true')
    parser.add_argument('--eval_only', default=False, action='store_true', help='read the accuracy metric from the loaded model only. Assumes the models are fully trained and does not select exemplars. ')
    parser.add_argument('--T', type=int, default=2, help='temperature if using distillation, default is 1')
    parser.add_argument('--lamb', type=float, default=1, help='lambda parameter distillation loss')
    parser.add_argument('--extra_info', type=str, default='', help='extra info for appending results folder')
    parser.add_argument('--exemplars_path', type=str, default='', help='when exemplars are pre computed, path to the exemplars numpy file')
    parser.add_argument('--extra_info_model', type=str, default='', help='when exemplars are loaded from file, append extra info to model name')
    parser.add_argument('--specific_order', default=False, action='store_true', help='use carefully designed order from config file, only for CIFAR100')
    parser.add_argument('--noise_type', type=str, default=None, help='gaussian')
    parser.add_argument('--noise_mean', type=float, default=0.0, help='')
    parser.add_argument('--noise_var', type=float, default=0.5, help='noise variance')
    parser.add_argument('--memory_per_class', type=int, default=20, help='')
    parser.add_argument('--perm', type=int, default=0, help='seed for generating class permutation')
    parser.add_argument('--start_seed', type=int, default=0, help='starting seed. Helpful if running experiments on parallel on different machines. Experiment is performed start_seed+repeat times')

    _args = parser.parse_args(argv)

    args = Args()
    args.dataset = _args.dataset
    args.verbose = _args.verbose
    args.output_to_file = _args.output_to_file
    args.repeat = _args.repeat
    args.memory_type = _args.memory_type
    # args.ordering = _args.ordering
    args.save_memory = _args.save_memory
    args.initial_increment = _args.initial_increment
    args.increment = _args.increment
    args.correct_bias = _args.correct_bias
    args.eval_on_task_zero = _args.eval_on_task_zero
    args.save_yhat_y = _args.save_yhat_y
    args.save_weights_last_layer = _args.save_weights_last_layer
    args.oversample = _args.oversample
    args.distill = _args.distill
    args.T = _args.T
    args.lamb = _args.lamb
    args.extra_info = _args.extra_info
    args.eval_only = _args.eval_only
    args.exemplars_path = _args.exemplars_path
    args.exemplars_path = Path.cwd() / args.exemplars_path
    args.extra_info_model = _args.extra_info_model
    args.noise_mean = _args.noise_mean
    args.noise_var = _args.noise_var
    args.specific_order = _args.specific_order
    args.noise_type = _args.noise_type
    args.memory_per_class = _args.memory_per_class
    args.perm = _args.perm
    args.start_seed = _args.start_seed
    args.bic_method = _args.bic_method
    return args
    
#permute class orders
def get_classes_permuted(args:Args):
    import configs
    # CIFAR100 classes based on similarity. Each set of 5 classes come from the same super class. E.g. classes 36,50,65,74,80) are similar. 
    if args.specific_order and args.dataset == 'CIFAR100':
        return configs.config_cifar100.get_classes_specific(seed=args.perm)
    else:
        all = np.arange(0, args.n_classes)
        all_p = np.random.RandomState(seed=args.perm).permutation(all) #default seed 1993

        p = all_p[:args.initial_increment]
        q = all_p[args.initial_increment:]
        r =  np.concatenate((p,q))
        print(f'class order: {r}')
        return r.tolist()

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#Prepare dataset and the model and log experiment info before training 
def data_setup(args:Args):
    print(f'Using device: {args.device}')      
    print(f'seed: {args.seed}')
                                              
    print('\n[Phase1] : Data Preparation')
    print(f"| | Preparing {args.dataset} dataset...")
  
    print('| Generating random task order...')
    args.scenario_train = ClassIncremental(args.train_d, initial_increment=args.initial_increment, increment=args.increment, transformations=args.transform_train, class_order=args.class_order)

    args.scenario_test = ClassIncremental(args.test_d, initial_increment=args.initial_increment, increment=args.increment, transformations=args.transform_test, class_order=args.class_order)

    print('| Random task order generated...')

    print(f'| | Dataset loaded and tasks successfully generated ')
    print('\n[Phase 2] : Model setup')
    print(f'| | Building net type {args.net_type}...')
    print(f'| | Experiment info {args.exp_info_short}')

#when --save_yhat_y is enabled
def update_yhat_y(args:Args):
    _preds = args.current_task.logger.logger_dict['test']['performance'][0][0]['predictions']
    _targets = args.current_task.logger.logger_dict['test']['performance'][0][0]['targets']
    args.yhat_y['yhat'].append(_preds)
    args.yhat_y['y'].append(_targets)

#when --save_yhat_y is enabled
def init_yhat_y(args:Args):
    args.yhat_y['yhat'] = []
    args.yhat_y['y'] = []

#when --save_yhat_y is enabled
def save_and_reset_yhat_y(args:Args):
    import pickle

    if args.save_yhat_y:
        logger_dicts_dir = Path.cwd() / 'logger_dicts'
        filepath = Path.cwd() / 'logger_dicts' / f'{args.exp_info_long}_{args.agent}.pkl'  

        if not Path.exists(logger_dicts_dir):
            Path.mkdir(logger_dicts_dir)

        with open(filepath, 'wb') as f:
            pickle.dump(args.yhat_y, f, protocol=pickle.HIGHEST_PROTOCOL)

    init_yhat_y(args)

#if save_weights_last_layer is set to True in args
def save_model_weights_last_layer(args:Args, tid):
    if args.save_weights_last_layer:
        saved_weights_path = Path.cwd() / 'saved_weights'

        if not Path.exists(saved_weights_path):
            Path.mkdir(saved_weights_path)

        model_info = get_model_info(args, tid, args.agent)
        
        filepath = saved_weights_path / f'{model_info}'


        # print(args.current_net.module)
        weights = args.current_net.module.linear.weight.detach().cpu().numpy() if args.net_type == 'resnet32' else args.current_net.module.fc.weight.detach().cpu().numpy()

        np.save(filepath, weights)

#since the dataset is imbalanced in each task > 1, a helper function to create a balanced validation set containing equal number of old and new classes. Current implementation does not use a validation set similar to most class-incremental methods, but it is available as an option. 
def split_train_val_balanced(taskset:BaseTaskSet, val_split=0.1, seed=0):
    random_state = np.random.RandomState(seed=seed)

    y = taskset.get_raw_samples()[1]
    
    train_ids= []
    val_ids = []

    for class_id in np.unique(y):
        class_indexes = np.where(class_id == y)[0]

        random_state.shuffle(class_indexes)
        val_indexes = class_indexes[:int(val_split*len(class_indexes))]
        train_indexes = class_indexes[int(val_split*len(class_indexes)):]
        train_ids.append(train_indexes)
        val_ids.append(val_indexes)

    train_ids = np.concatenate(train_ids)
    val_ids = np.concatenate(val_ids)

    x_train, y_train, t_train = taskset.get_raw_samples(train_ids)
    x_val, y_val, t_val = taskset.get_raw_samples(val_ids)
    idx_train, idx_val = None, None
    train_dataset = TaskSet(x_train, y_train, t_train,
                                trsf=taskset.trsf,
                                data_type=taskset.data_type,
                                data_indexes=idx_train)
    val_dataset = TaskSet(x_val, y_val, t_val,
                            trsf=taskset.trsf,
                            data_type=taskset.data_type,
                            data_indexes=idx_val)

    return train_dataset, val_dataset

def get_softmax(arr:np.array):
    t_arr = torch.tensor(arr, device='cpu')
    sm = torch.nn.Softmax(dim=1)
    sm_t_arr = sm(t_arr)
    return sm_t_arr.numpy()

#at each task t, get a list of all seen classes with the option to include base classes and current task classes
def get_seen_classes(args:Args, tid:int, incremental_only=True, include_current_task = False):
    seen_classes = None
    end_index = tid+1 if include_current_task else tid
    
    nb_seen_classes_all = args.scenario_train[0:end_index].nb_classes
    nb_classes_non_incremental = args.scenario_train[0].nb_classes

    start_index = nb_classes_non_incremental if incremental_only else 0

    seen_classes = np.arange(start_index, nb_seen_classes_all)

    return seen_classes

#get original class id. e.g. classes 0,1,2 can correspond to class 18,9 41 etc. 
def get_class_values_original(args:Args, seen_classes):
    class_order = np.array(args.class_order)
    #returns original class labels e.g. first three classes 18, 3, 8 instead of 0,1,2
    return class_order[seen_classes]

def construct_new_taskset(x,y,t,args:Args):
    train_trsf = transforms.Compose(args.transform_train) if args.transform_train is not None else None
    dtype = TaskType.IMAGE_PATH if args.dataset in ['ImageNet100', 'ImageNet1000', 'TinyImageNet200'] else TaskType.IMAGE_ARRAY

    taskset =  TaskSet(x,y,t,trsf=train_trsf, target_trsf=None, data_type=dtype)
    return taskset

#get only old classes in each task
def old_classes_by_task(args, tid):
    seen_classes = set(get_seen_classes(args, tid, True, True))
    new_classes = set(args.scenario_train[tid].get_classes())
    old_classes = np.sort(list(seen_classes - new_classes))
    return old_classes

dict_herding = {'herding_icarl': 'barycenter', 'random': 'random', 'cluster': 'cluster',\
                'cl_margin_high': CL_MARGIN_HIGH(), 'cl_margin_low': CL_MARGIN_LOW(), \
                'cl_margin_h80_l20': CL_MARGIN_H80_L20(),
                'cl_margin_low_all': CL_MARGIN_LOW_ALL(), \
                'cl_margin_high_all': CL_MARGIN_HIGH_ALL(), 
                'cl_margin_class_avg': CL_MARGIN_CLASS_AVG(), 
                } 




    
        

        