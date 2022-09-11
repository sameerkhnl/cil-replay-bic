from responses import target
from Args import Args
import sys
import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


def train(args:Args, trainloader:DataLoader):
    #todo
    # args.current_task.lr_scheduler.step(train_loss)
    epoch = args.current_task.current_epoch
    net = args.current_net
    total_batches = args.current_task.total_batches
    device = args.device
    optimizer = args.current_task.optimizer

    old_net = args.old_net

    net.train()
    net.training = True
    train_loss = 0

#     print(f'\n=> Training Epoch {epoch}, LR={cf.learning_rate(cf.lr,epoch):.4f}')
    
    for batch_idx, (X,y,t) in enumerate(trainloader):
        X,y = X.to(device),y.to(device)
        optimizer.zero_grad()
        outputs,features = net(X)

        outputs_old = None
        features_old = None
        if old_net is not None:
            outputs_old, features_old = old_net(X)

        loss = get_criterion(args=args, outputs = outputs, targets = y.long(), t = t, outputs_old=outputs_old, features=features, features_old=features_old)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        # q0 = 1
        # q1 = int(total_batches * 0.25)
        # q2 = int(total_batches * 0.5)
        # q3 = int(total_batches * 0.75)
        # q4 = total_batches
        # if args.verbose or (batch_idx + 1 in [q0,q1,q2,q3,q4]):
        if args.verbose:
            sys.stdout.write('\r')
            sys.stdout.write(f'| | Epoch [{epoch}/{args.num_epochs}] Iter[{batch_idx+1}/{total_batches}]' + 
                             f'\t\tLoss: {loss.item():.4f}')
            sys.stdout.flush()
    return train_loss

            
def eval(args:Args, dataloader, correction_factor=None):    
    logger = args.current_task.logger
    logger_subset = args.current_task.logger_subset
    device = args.device

    net = args.current_net
    
    net.training = False
    net.eval()
    _logits = []
    _features = []
    running_loss = 0.0
    with torch.no_grad():
        for i,(X, y,t) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            preds,features= net(X)

            # if correction_factor is not None:
            #     # sm_preds = get_softmax(preds)
            #     sm_preds = preds
            #     cfactor = np.tile(correction_factor, (preds.shape[0],1))

            #     preds = sm_preds * torch.tensor(cfactor).to(device)

            loss = get_ce_loss(preds, y.long())
            running_loss += loss.item()
            y_hat = torch.argmax(preds, dim=1)
            _logits.append(preds.cpu().detach().numpy())
            _features.append(features.cpu().detach().numpy())
            logger.add([y_hat,y.cpu().detach().numpy(),t],subset=logger_subset)
            
    _logits = np.concatenate(_logits)
    _features = np.concatenate(_features)
    return _logits, _features, running_loss

def get_softmax(preds:torch.Tensor)->torch.Tensor:
    sm = torch.nn.Softmax(dim=1)
    sm_preds = sm(preds)
    return sm_preds

def get_criterion(args:Args, outputs, targets, t, outputs_old=None, features=None, features_old=None):
    def multiclass_crossentropy(soft_probs, soft_targets, T):
        log_soft_probs = F.log_softmax(soft_probs/T, dim=1)
        soft_targets = F.softmax(soft_targets/T, dim=1)
        loss = -(soft_targets * log_soft_probs).sum(dim=1).mean()
        return loss

    ce_loss = get_ce_loss(outputs, targets)

    lamb_kd_loss = 0

    if outputs_old is not None:
        kd_loss = multiclass_crossentropy(outputs, outputs_old, args.T)
        
        # current_task = torch.max(t).item()

        # cnt = len(torch.where(t == current_task)[0])
        # cot = max(len(torch.where(t != current_task)[0]), 1)
        
        # args.lamb = np.sqrt(cnt/cot)
        # print(f'| | lamb = {args.lamb}')
        lamb_kd_loss = args.lamb * kd_loss
        # feature_loss = 0

    return ce_loss + lamb_kd_loss

def get_ce_loss(outputs, targets):
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(outputs, targets)
    return ce_loss

    




    

    


