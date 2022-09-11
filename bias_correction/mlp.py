from types import new_class
import torch
from torch import nn
from continuum.metrics.metrics import accuracy
import numpy as np
import torch.nn.functional as F

# Define model
class MLP(nn.Module):
    def __init__(self, in_dim=1000, hidden_dim=800, out_dim=1000):
        def init_weights(m):
          if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

        super(MLP, self).__init__()
        self.in_dim = in_dim
        # self.linear_relu_stack = nn.Sequential(
        #      nn.Linear(self.in_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        # )
        self.last = nn.Linear(in_dim, out_dim)

        # self.linear_relu_stack.apply(init_weights)

        # self.linear_relu_stack.apply(init_weights)
    # def features(self,x):
    #     x = self.linear_relu_stack(x.view(-1, self.in_dim))
    #     return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        # x = self.features(x)
        x = self.logits(x)
        return x

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

def MLP800_CIFAR10():
    return MLP(in_dim=640, out_dim = 10, hidden_dim=800)

def MLP800_CIFAR100():
    return MLP(in_dim=2048, out_dim = 100, hidden_dim=800)

def MLP64_CIFAR100():
    return MLP(in_dim=64, out_dim=100, hidden_dim=64)

def MLP_LINEAR(in_dim, out_dim):
    return MLP(in_dim=in_dim, out_dim=out_dim)

def MLP800_CUB200():
    return MLP(in_dim=2048, out_dim = 200, hidden_dim=800)

def MLP800_ImageNet1000():
    return MLP(in_dim=1000, out_dim = 1000, hidden_dim=800)

def train(dataloader, model, optimizer, device):
    # size = len(dataloader.dataset)
    model.train()
    model.training = True
    train_loss = 0
    for batch_idx, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        preds = model(X)
      
        loss = get_ce_loss(preds, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print(loss.item())
    return train_loss

def get_ce_loss(outputs, targets):
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(outputs, targets)
    return ce_loss

def eval(dataloader, model, device):
    model.eval()
    model.training = False
    _logits = []
    running_loss = 0
    # test_loss, correct = 0, 0
    with torch.no_grad():
        for i,(X, y,t) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = get_ce_loss(logits, y.long())
            _logits.append(logits.cpu().numpy())
            # predictions = logits.argmax(1)
            # if logger_subset is not None:
            # logger.add([predictions, y, t], subset = logger_subset)
            running_loss += loss
            
    _logits = np.concatenate(_logits)
    return _logits
            
            
            # correct /= size

