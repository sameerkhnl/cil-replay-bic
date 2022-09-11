from continuum.metrics import Logger
from pathlib import Path
from typing import List
from continuum.tasks import TaskSet
import torch
from torch.utils.data import DataLoader

class CurrentTask:
    tid: int
    lr:float
    patience: int
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau

    start_test_id: int
    start_epoch: int
    num_epochs: int

    current_task_latest_path: Path
    previous_task_latest_path:Path

    # current_task_best_path: Path
    # previous_task_best_path: Path


    # current_task_latest_model: torch.nn.DataParallel
    # # current_task_best_model: torch.nn.DataParallel
    # # previous_task_best_model: torch.nn.DataParallel
    # previous_task_latest_model:torch.nn.DataParallel

    

    checkpoint = object
    acc_best_model = float
    logger: Logger
    logger_subset:str
    total_batches: int

    taskset_train:TaskSet
    taskset_train_mixed:TaskSet
    taskset_test:TaskSet
    taskset_test_seen:TaskSet
    
    train_loader: DataLoader
    test_loader: DataLoader
    train_loader_mixed: DataLoader
    train_loader_mixed_eval:DataLoader
    train_loader_eval:DataLoader
    test_loader_seen: DataLoader
    current_epoch: int







