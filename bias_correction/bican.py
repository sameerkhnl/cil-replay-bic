from types import new_class
from Args import Args
import numpy as np
from bias_correction import mlp
from continuum.rehearsal.memory import RehearsalMemory
import torch
from continuum import Logger,datasets
import utils
from continuum.tasks import TaskType, TaskSet, BaseTaskSet
from torch.utils.data import DataLoader
import time
import sys

model_ext=None
batch_size = 8
lr = 0.1
num_workers = 3
num_epochs=5



def init_train_ext_net(args:Args, features_replay, tid):
    _,y,t = args.mem.get()
    ts:BaseTaskSet= datasets.InMemoryDataset(x=features_replay, y=y,  t=np.full(y.shape[0], -1), data_type=TaskType.TENSOR).to_taskset()
    global model_ext

    print('| | Training the external classifier')
    dl_train = DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dl_train_eval = DataLoader(ts, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)  
    model = mlp.MLP_LINEAR(features_replay.shape[1], len(y)).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start_time = time.time()
    logger1 = Logger()
    # mlp.eval(dl_train_eval, model, logger1, 'train', args.device )
    sys.stdout.write('| ')
    
    for epoch in range(1,1+num_epochs):
        running_loss = mlp.train(dl_train, model, optimizer, args.device)
    
        logits = mlp.eval(dl_train_eval,model=model,device=args.device)
        preds = np.argmax(logits, 1)
        logger1.add([preds, y, t], subset='train')
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print(f'| Elapsed time : {utils.get_hms(elapsed_time)} | Epoch[{epoch}/{num_epochs}]:     lr: {lr:.5f}:     Running Loss: {running_loss:.4f}:     train_acc: {logger1.online_accuracy:.3f}')  
        sys.stdout.flush()
    global model_ext
    model_ext = model

def classify(features_eval,args:Args, tid)->Logger:
    _,y,t = args.current_task.taskset_test_seen.get_raw_samples()
    ts:BaseTaskSet = datasets.InMemoryDataset(features_eval, y, np.full(y.shape[0], -1), data_type=TaskType.TENSOR).to_taskset()
    dl =  DataLoader(ts, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    logger=Logger()
    model = model_ext
    logits = mlp.eval(dataloader=dl, model=model, device=args.device)
    preds = np.argmax(logits, 1)
    logger.add([preds, y, t], subset='test')
    return logger

def classify1(features_eval, logits_final, args:Args, tid)->Logger:

    seen_classes = set(seen_classes_by_task(args, tid))
    new_classes = set(args.scenario_train[tid].get_classes())
    old_classes = list(seen_classes - new_classes)
    preds_original = np.argmax(logits_final, 1)
    old_preds = np.where(np.in1d(preds_original,old_classes))[0]
    new_preds = np.where(np.in1d(preds_original, list(new_classes)))[0]

    _,y,t = args.current_task.taskset_test_seen.get_raw_samples()
    ts:BaseTaskSet = datasets.InMemoryDataset(features_eval, y, np.full(y.shape[0], -1), data_type=TaskType.TENSOR).to_taskset()
    dl =  DataLoader(ts, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    logger=Logger()
    model = model_ext
    logits = mlp.eval(dataloader=dl, model=model,device=args.device)
    preds = np.argmax(logits, 1)

    assert preds_original.shape == preds.shape, "original and new preds shape are different"


    preds_final = np.full(preds.shape,-1)
    assert len(old_preds) + len(new_preds) == len(preds_final)

    # preds_final[old_preds] = preds_original[old_preds]
    preds_final[old_preds] = preds[old_preds]
    preds_final[new_preds] = preds[new_preds]
    
    print((preds_original[old_preds] != preds[old_preds]).sum())
    logger.add([preds_final, y, t], subset='test')

    return logger

def seen_classes_by_task(args:Args, tid):
    seen_classes = []
    for i in range(1, tid+1):
        seen_classes.append(args.scenario_train[i].get_classes())
    return np.concatenate(seen_classes)

