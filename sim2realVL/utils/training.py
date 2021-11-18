from ..types import *
from ..utils.metrics import *

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader


def train_epoch(model: nn.Module, dl: DataLoader, optim: Optimizer, criterion: nn.Module) -> Metrics:
    model.train()
    
    epoch_loss = 0
    confusion_matrix = torch.zeros(3, 2)
    for batch_idx, (imgs, words, truths, boxes) in enumerate(dl):
        preds = model.forward([imgs, words, boxes])
        loss = criterion(preds, truths.float())

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        # metrics
        epoch_loss += loss.item()
        confusion_matrix += get_confusion_matrix(preds.sigmoid().ge(0.5), truths)

    epoch_loss /= len(dl)
    return {'loss': -round(epoch_loss, 5), **get_metrics_from_matrix(confusion_matrix)}


@torch.no_grad()
def eval_epoch(model: nn.Module, dl: DataLoader, criterion: nn.Module) -> Metrics:
    model.eval()

    confusion_matrix = torch.zeros(3, 2)
    epoch_loss = 0
    for batch_idx, (imgs, words, truths, boxes) in enumerate(dl):
        preds = model.forward([imgs, words, boxes])
        loss = criterion(preds, truths.float())
        epoch_loss += loss.item()
        confusion_matrix += get_confusion_matrix(preds.sigmoid().ge(0.5), truths)

    epoch_loss /= len(dl)
    return {'loss': -round(epoch_loss, 5), **get_metrics_from_matrix(confusion_matrix)}


def train_epoch_classifier(model: nn.Module, dl: DataLoader, optim: Optimizer, criterion: nn.Module) -> Metrics:
    model.train()
    
    epoch_loss = 0
    total_correct = 0
    for batch_idx, (x, y) in enumerate(dl):
        preds = model.forward(x)
        loss = criterion(preds, y)

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        # metrics
        epoch_loss += loss.item()
        total_correct += (preds.argmax(-1) == y).sum().item()
    epoch_loss /= len(dl)
    total_correct /= len(dl.dataset) 
    
    return {'loss': -round(epoch_loss, 5), 'accuracy': round(total_correct, 4)}


def eval_epoch_classifier(model: nn.Module, dl: DataLoader, criterion: nn.Module) -> Metrics:
    model.eval()
    
    epoch_loss = 0
    total_correct = 0
    for batch_idx, (x, y) in enumerate(dl):
        preds = model.forward(x)
        loss = criterion(preds, y)
        epoch_loss += loss.item()
        total_correct += (preds.argmax(-1) == y).sum().item()
    epoch_loss /= len(dl)
    total_correct /= len(dl.dataset) 
    return {'loss': -round(epoch_loss, 5), 'accuracy': round(total_correct, 4)}


class Trainer(ABC):
    def __init__(self, 
            model: nn.Module, 
            dls: Tuple[DataLoader, ...],
            optimizer: Optimizer, 
            criterion: nn.Module,
            target_metric: str,
            train_fn: Map[Any, Metrics] = train_epoch,
            eval_fn: Map[Any, Metrics] = eval_epoch,
            early_stopping: Maybe[int] = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dl, self.dev_dl, self.test_dl = dls
        self.logs = {'train': [], 'dev': [], 'test': []}
        self.target_metric = target_metric
        self.trained_epochs = 0
        self.early_stop_patience = early_stopping
        self.train_fn = train_fn
        self.eval_fn = eval_fn

    def iterate(self, num_epochs: int, print_log: bool = False, with_save: Maybe[str] = None) -> Metrics:
        best = {self.target_metric: -1e2}
        patience = self.early_stop_patience if self.early_stop_patience is not None else num_epochs
        for epoch in range(num_epochs):
            self.step(print_log)

            # update logger for best - save - test - early stopping
            if self.logs['dev'][-1][self.target_metric] > best[self.target_metric]:
                best = self.logs['dev'][-1]
                patience = self.early_stop_patience if self.early_stop_patience is not None else num_epochs

                if with_save is not None:
                    torch.save(self.model.state_dict(), with_save)

                if self.test_dl is not None:
                    self.logs['test'].append({'epoch': epoch+1, **self.eval_epoch(self.model, self.test_dl, self.criterion)})

            else:
                patience -= 1
                if not patience:
                    self.trained_epochs += epoch + 1
                    return best

        self.trained_epochs += num_epochs
        return best

    def step(self, print_log: bool = False):
        current_epoch = len(self.logs['train']) + 1

        # train - eval this epoch
        self.logs['train'].append({'epoch': current_epoch, **self.train_epoch()})
        self.logs['dev'].append({'epoch': current_epoch, **self.eval_epoch()})
        
        # print if wanted
        if print_log:
            print('TRAIN:')
            for k,v in self.logs['train'][-1].items():
                print(f'{k} : {v}')
            print()
            print('DEV:')
            for k,v in self.logs['dev'][-1].items():
                print(f'{k} : {v}')
            print('==' * 72)

    def train_epoch(self):
        return self.train_fn(self.model, self.train_dl, self.optimizer, self.criterion)

    def eval_epoch(self):
        return self.eval_fn(self.model, self.dev_dl, self.criterion)
