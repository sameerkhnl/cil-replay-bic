from pathlib import Path
from typing import List
from xmlrpc.client import boolean
import numpy
import torch
import continuum
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from continuum.metrics.logger import Logger
from continuum.scenarios.class_incremental import ClassIncremental
import time

from CurrentTask import CurrentTask

from stats_bias import StatsBias
from continuum.rehearsal.memory import RehearsalMemory

class Args:
    dataset:str = None
    lr:float = None
    verbose:bool = None
    save_model:bool = None
    output_to_file:bool = None
    repeat:int = None
    memory_type:str = None
    ordering:str = None
    incorrect_ratio:float = None

    device:str = None
    agents:list = None 
    start_epoch:int = None
    batch_size:int = None
    optim_type:str = None
    memory:int= None
    num_workers:int= None
    net_type:str= None
    n_classes:int= None
    cf = None
    initial_increment:int= None
    increment:int= None
    n_tasks:int= None
    num_epochs_non_incremental:int = None
    num_epochs_incremental:int = None
    first_task_only:bool = None

    train_d = None
    test_d = None
    scenario_train:ClassIncremental = None
    scenario_test:ClassIncremental = None
    scenario_generator_train = None
    scenario_generator_test = None

    filepath_main:Path = None
    filepath_acc_per_task:Path
    seed:int = None
    agent:str = None
    memory_path:Path = None

    current_net:torch.nn.DataParallel = None
    model_info: str = None

    model_save_path:Path = None
    store_path:Path = None

    model_from_config = None
    save_memory = None

    memory_per_class:int = None
    filepath_plot:Path = None
    n_classes_incremental:int = None
    start_time:time.time() = None
    dataset_results_path:Path = None 

    exp_info_short:str = None
    exp_info_long:str = None
    exp_info_base:str = None

    patience_non_incremental:int = None
    patience_incremental:int = None
    transform_train:List = None
    transform_test:List = None
    class_order:List[int] = None

    correct_bias:bool = None

    stats_bias:StatsBias = None

    eval_on_task_zero:bool = None

    #if we want to record the yhat and y values at each test step
    yhat_y = {}
    save_yhat_y:bool = None
    save_weights_last_layer: bool = None
    model_subdir: str = None

    #For ImageNet100
    subset_version = None
    oversample: bool = None
    distill: bool = None

    #lambda parameter for distillatin
    lamb:float = None
    old_net: torch.nn.DataParallel
    T: int = None
    extra_info:str = None
    eval_only:bool 
    current_task: CurrentTask
    mem: RehearsalMemory
    exemplars_path:Path #path of the exemplars file relative to current directory. e.g. './joint_training/store/exemplars/CIFAR100/cl_margin_high_m20.npz'

    extra_info_model:str
    noise_mean:float
    noise_var:float
    blur_sigma:float
    specific_order:bool
    noise_type:str
    sp_amount: float
    perm:int #permutation seed
    start_seed: int #starting seed when parallelisation is done
    bic_method:str

    # train_first_task:bool = None
    # train_baselines:bool = None
    # train_replay:bool = None








        
   