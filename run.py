
import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
from continuum.metrics.logger import Logger
from continuum.rehearsal.memory import RehearsalMemory
import time
from Args import Args
import utils
import copy
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from train_eval import train, eval
from continuum.tasks import get_balanced_sampler
from CurrentTask import CurrentTask
import noise_filter as nf
from bias_correction import bican, il2m

def _load_checkpoint(args:Args, tid):
    checkpoint = None

    if Path.exists(args.current_task.current_task_latest_path):
        print(f'| | Checkpoint found for the current task. Loading...')
        checkpoint = torch.load(args.current_task.current_task_latest_path)

    elif Path.exists(args.current_task.previous_task_latest_path):
        print(f'| | Checkpoint not found for the current task. Loading from previous task checkpoint...')
        checkpoint = torch.load(args.current_task.previous_task_latest_path)
        print(f'| | Model successfully loaded from previous task. Resuming training...')

    elif tid > 0:
        print(f'| | No checkpoint found for either the current task or previous task latest. Make sure the model from task0 is available.')
        # print(f'| | Previous task path: {args.params_task.previous_task_best_path}')
        print(f'| Aborting....')
        sys.exit(0)    
    return checkpoint

#load exemplars from the RehearsalMemory or from saved file if agent == 'load_from_file'
def load_exemplars(args:Args, tid):
    mem_x, mem_y, mem_t = None, None, None
    if tid > 1:
        if args.mem is None:
            if args.agent == 'load_from_file':
                loaded = np.load(args.exemplars_path)
                x,y = loaded['x'], loaded['y']
                seen_classes = utils.get_seen_classes(args, tid, True, False)
                seen_classes_original = utils.get_class_values_original(args, seen_classes)

                idxs_seen = np.where(np.isin(y, seen_classes_original))[0]
                y = np.array([list(args.class_order).index(g) for g in y])
                t = np.full(len(y), tid)

                mem_x, mem_y, mem_t = x[idxs_seen], y[idxs_seen], t[idxs_seen]
        else:
            # args.mem.load(args.memory_path)
            mem_x, mem_y, mem_t = args.mem.get()

    return mem_x, mem_y, mem_t

def prepare_task_data_load_checkpoint(args:Args, tid:int):
    args.current_task = CurrentTask()
    args.current_task.tid= tid
    lr = args.lr if tid == 0 else args.lr/tid #args.lr/(tid + 1) For ImageNet100 do not divide by task no
    patience = args.patience_non_incremental if tid == 0 else args.patience_incremental
    optimizer = SGD(args.current_net.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)

    taskset_train = args.scenario_train[tid]       

    start_test_id = 0 if args.eval_on_task_zero else 1
    taskset_test = args.scenario_test[tid]
    taskset_test_seen = args.scenario_test[start_test_id:tid+1] if tid > 0 else args.scenario_test[tid]

    _incr = 'non-incremental' if tid == 0 else 'incremental'

    print(f'\n| Training on task {tid} ({_incr}) # train {len(taskset_train)} # classes {taskset_train.nb_classes}')

    current_task_latest = utils.get_model_info(args, tid, args.agent, best=False)
    previous_task_latest = utils.get_model_info(args, max(0,tid-1), args.agent, best=False)

    current_task_latest_path = args.store_path / f'{current_task_latest}.pt'
    previous_task_latest_path = args.store_path / f'{previous_task_latest}.pt'

    args.current_task.current_task_latest_path= current_task_latest_path
    args.current_task.previous_task_latest_path = previous_task_latest_path

    start_epoch = args.start_epoch
    num_epochs = args.num_epochs_non_incremental if tid == 0 else args.num_epochs_incremental
        
    logger = Logger(list_keywords=["performance"], list_subsets=["train", "test"])

    train_loader = DataLoader(taskset_train, batch_size=args.batch_size, \
                                    shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader_eval = DataLoader(taskset_train, batch_size=args.batch_size, \
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(taskset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader_seen = DataLoader(taskset_test_seen, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    taskset_train_mixed = copy.deepcopy(taskset_train)

    if not args.eval_only:
        mem_x, mem_y, mem_t = load_exemplars(args, tid) #Returns None if using baseline
        if mem_x is not None:
            taskset_train_mixed.add_samples(mem_x, mem_y, mem_t)

    sampler=None
    shuffle=True
    if args.oversample:
        sampler = get_balanced_sampler(taskset_train_mixed)
        shuffle = False

    if sampler is not None:
        print('| | Using oversampler') 

    train_loader_mixed = DataLoader(taskset_train_mixed, batch_size=args.batch_size, \
        shuffle=shuffle, num_workers=args.num_workers, drop_last=True, sampler=sampler)
    train_loader_mixed_eval = DataLoader(taskset_train_mixed, batch_size=args.batch_size, \
        shuffle=False, num_workers=args.num_workers, drop_last=False, sampler=None)

    checkpoint = _load_checkpoint(args, tid)

    if checkpoint is not None:
        args.current_net.load_state_dict(checkpoint['net'])
        if checkpoint['task'] == tid:
            start_epoch = checkpoint['epoch'] + 1
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(f'| | Existing model not found for task 0. Starting training from the beginning...')

    args.current_task.lr = lr
    # args.current_task.acc_best_model = best_accuracy
    # args.params_task.checkpoint = checkpoint
  
    args.current_task.patience = patience
    args.current_task.optimizer = optimizer
    args.current_task.lr_scheduler = lr_scheduler
    args.current_task.start_test_id = start_test_id
    args.current_task.start_epoch = start_epoch
    args.current_task.num_epochs = num_epochs
    args.current_task.logger = logger
    args.current_task.train_loader = train_loader
    args.current_task.train_loader_eval = train_loader_eval
    args.current_task.test_loader = test_loader
    args.current_task.test_loader_seen = test_loader_seen
    args.current_task.train_loader_mixed = train_loader_mixed
    args.current_task.train_loader_mixed_eval = train_loader_mixed_eval

    args.current_task.total_batches = len(taskset_train_mixed) // args.batch_size if taskset_train_mixed is not None else taskset_train
    args.current_task.checkpoint = checkpoint

    args.current_task.taskset_train = taskset_train
    args.current_task.taskset_train_mixed = taskset_train_mixed
    args.current_task.taskset_test = taskset_test
    args.current_task.taskset_test_seen = taskset_test_seen

def _add_exemplars(args:Args, tid:int):
    if args.mem is not None and tid > 0:
        x,y,t = args.current_task.taskset_train.get_raw_samples()

        if args.agent == 'random':
            args.mem.add(x,y,t, None)

        else:
            args.current_task.logger_subset = 'train'
            logits, features, _ = eval(args, args.current_task.train_loader_eval)

            if args.agent in ['herding_icarl', 'cluster']:
                args.mem.add(x,y,t,features)

            else:
                z = lambda:None
                z.logits = logits
                z.features = features
                z.incorrect_ratio = args.incorrect_ratio
                args.mem.add(x,y,t,z)

        # args.mem.save(args.memory_path)

def post_train_bic_method(args:Args, tid):
    def _get_logits_features_replay(args:Args):
        x,y,t = args.mem.get()

        taskset = utils.construct_new_taskset(x,y,t,args)

        dl = DataLoader(taskset, batch_size=args.batch_size, \
                shuffle=False, num_workers=args.num_workers, drop_last=False)
        args.current_task.logger = Logger()
        args.current_task.logger_subset = 'train'    
        logits,features,_ = eval(args, dl)
        return logits, features

    cfactor = None
    if args.correct_bias and tid > 0:   
        if args.bic_method == 'bican':
            _, features_replay = _get_logits_features_replay(args)
            if tid > 1:
                bican.init_train_ext_net(args, features_replay, tid)

        elif args.bic_method == 'il2m':
            logger = Logger()
            args.current_task.logger = logger
            args.current_task.logger_subset = 'train'
            # logits_replay, _= _get_logits_features_replay(args)
            logits_mixed, _, _ = eval(args, args.current_task.train_loader_mixed_eval)
            

            _,y,t = args.current_task.taskset_train_mixed.get_raw_samples()


            il2m.compute_model_conf(logits_mixed, y, args.current_task.taskset_train.get_classes(), tid)  
            il2m.add_init_class_means(logits_mixed, y,args.current_task.taskset_train.get_classes(), tid)

            if tid > 1:
                il2m.add_current_class_means(logits_mixed, y, utils.old_classes_by_task(args, tid), tid) 

def _eval_on_test_seen(args:Args, correction_factor=None):
    args.current_task.logger_subset = 'test'
    eval(args, args.current_task.test_loader_seen, correction_factor)
    avg_accuracy = np.mean(args.current_task.logger.accuracy_per_task)
    # print(f'| Average accuracy on previous tasks using the best model: {avg_accuracy:.3f}')
    print(f'| | Training successfully completed for {args.current_task.num_epochs} epochs. Average accuracy on all seen tasks: {avg_accuracy:.3f}')
    if args.save_yhat_y:
        utils.update_yhat_y(args)

    return avg_accuracy, args.current_task.logger.accuracy_per_task

def train_single_task(args:Args, tid:int):
    start_time = time.time()
    state = None
    for epoch in range(args.current_task.start_epoch, args.current_task.num_epochs + 1):
        if (args.eval_only):
            print('Model not trained fully found in eval_only mode. Re-run with eval_only off,experiment details:')
            print('perm: {}, agent: {}, seed: {}, task_id: {}'.format(args.perm, args.agent, args.seed, tid))
            print('Aborting...')
            sys.exit()

        args.current_task.current_epoch = epoch
        # print(f'mixed_loader: {True if args.current_task.train_loader_mixed is not None else False}')
        train_dataloader = args.current_task.train_loader_mixed
        running_loss = train(args, train_dataloader)
        args.current_task.lr_scheduler.step(running_loss)
        elapsed_time = time.time() - start_time

        args.current_task.logger_subset = 'test'
        
        eval(args, args.current_task.test_loader)
        sys.stdout.write('| ')

        print(f'| Elapsed time : {utils.get_hms(elapsed_time)} | Epoch[{epoch}/{args.current_task.num_epochs}]:     lr: {utils.get_current_lr_from_optimizer(args.current_task.optimizer):.5f}:     Running Loss: {running_loss:.4f}:     test_acc: {args.current_task.logger.accuracy:.3f}')  

        print(f'| | Saving latest model...')

        state = {
            'running_loss': round(running_loss, 3), 'net': args.current_net.state_dict(), 'task': tid, 'optimizer': args.current_task.optimizer.state_dict(), 'net_type': args.net_type, 'lr_scheduler': args.current_task.lr_scheduler.state_dict(), 'epoch': epoch,
            'test_acc': args.current_task.logger.accuracy, 'test_acc_avg_seen': None} 

        model_save_path = args.current_task.current_task_latest_path
        utils.save_model(state, model_save_path)

        sys.stdout.flush()

    if state is not None:
        avg_acc, acc_per_task = _eval_on_test_seen(args)
        state['test_acc_avg_seen'] = round(avg_acc, 3)
        state['test_acc_per_task_seen'] = [round(val, 3) for val in acc_per_task]
        utils.save_model(state, args.current_task.current_task_latest_path)
    else:
        avg_acc = args.current_task.checkpoint['test_acc_avg_seen']
        acc_per_task = args.current_task.checkpoint['test_acc_per_task_seen']
        test_acc = args.current_task.checkpoint['test_acc']
        print(f'| | Accuracy on the current task from the loaded model: {test_acc:.3f}')
        print(f'| | Average accuracy on all seen tasks from the loaded model: {avg_acc:.3f}')

    return avg_acc, acc_per_task

def post_train_processing(args:Args, tid):
    if not args.eval_only:
        _add_exemplars(args, tid=tid)
        if args.distill and tid > 0:
            args.old_net = copy.deepcopy(args.current_net)
            args.old_net.module.eval()

            for param in args.old_net.module.parameters():
                param.requires_grad = False

    if args.save_weights_last_layer:
        utils.save_model_weights_last_layer(args, tid)

def train_first_task(args:Args):
    #initial non-incremental
    args.mem = None
    prepare_task_data_load_checkpoint(args, 0)
    train_single_task(args, 0) 
    post_train_processing(args, 0)

def train_an_agent_full(args:Args):
    all_agents = ['baseline', 'load_from_file', *[k for k in utils.dict_herding.keys()]]
    args.mem = None

    if args.agent not in all_agents:
        print(f'| unknown agent "{args.agent}". Aborting...')
        sys.exit(0)

    if args.agent in utils.dict_herding.keys():
        herding_method = utils.dict_herding[args.agent]
        _fixed_memory = True if args.memory_type == 'fixed' else False
        args.mem = RehearsalMemory(args.memory, herding_method, _fixed_memory, nb_total_classes=args.n_classes_incremental)
        print(f'\n| Using rehearsal memory of {args.memory_per_class} per incremental class, fixed memory: {_fixed_memory}, # classes incremental: {args.n_classes_incremental}')
        print(f'| Sampling method: {args.agent}')        
    
    overall_acc_list = []
    overall_acc_per_task = []
    
    if args.save_yhat_y:
        utils.init_yhat_y(args)

    for tid in range(1,len(args.scenario_train)):
        # args.current_task.logger = logger
        prepare_task_data_load_checkpoint(args, tid)
        avg_accuracy, acc_per_task = train_single_task(args, tid) 
        
        post_train_processing(args, tid)
        post_train_bic_method(args, tid)
        Avg_Acc= avg_accuracy
        Acc_Per_Task = acc_per_task
        if args.correct_bias and tid > 1:
            print("| | Evaluating without bias correction. ")
            args.current_task.logger = Logger()
            args.current_task.logger_subset = 'test'
            logits_final, features_final, _ = eval(args, args.current_task.test_loader_seen)
            Avg_Acc = np.mean(args.current_task.logger.accuracy_per_task)
            Acc_Per_Task = args.current_task.logger.accuracy_per_task
            print('Avg accuracy: {:.3f}'.format(Avg_Acc))
            print(f"| | Evaluating using bias correction")
            logger = None
            if args.bic_method == 'il2m':
                _,y,t = args.current_task.taskset_test_seen.get_raw_samples()
                logger = il2m.classify(logits_final, y, t, utils.old_classes_by_task(args, tid), args, tid)
               
            elif args.bic_method == 'bican':
                logger = bican.classify(features_final,args, tid)
            
            elif args.bic_method == 'nem':
                logger = nem.classify(features_final, args, tid)

            Avg_Acc = np.mean(logger.accuracy_per_task)
            Acc_Per_Task = logger.accuracy_per_task
            print(f'| | Avg accuracy: {Avg_Acc:.3f}')

        overall_acc_list.append(Avg_Acc)
        overall_acc_per_task.append(Acc_Per_Task)

    if args.save_yhat_y:
        utils.save_and_reset_yhat_y(args)

    if args.save_memory:
        if args.mem is not None:
            x,y,t = args.mem.get()
            np.savez_compressed(args.memory_path, x=x, y=y, t=t)
            
    return overall_acc_list, overall_acc_per_task





    





   
            
            