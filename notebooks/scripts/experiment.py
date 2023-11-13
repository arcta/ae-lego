#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------
# contarst-VAE with MNIST exercise
#------------------------------------------------

import os
import re
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
from typing import Optional
from sklearn.metrics import silhouette_score

from torch import nn, Tensor
from torchvision import transforms, models
from torchvision.datasets import MNIST
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, AdamW
from torch.cuda.amp import GradScaler
from torchsummary import summary

from .aelego import *
from .utils import *


torch.cuda.empty_cache()
scaler = GradScaler()

trainset = MNIST(root=f'{ROOT}/data', train=True, download=True)
testset  = MNIST(root=f'{ROOT}/data', train=False, download=True)


class AEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, Y = self.data[idx]
        inputs = torch.Tensor(np.array(X)/255.).unsqueeze(0)
        targets = torch.Tensor(np.array(X.resize((REC_SIZE, REC_SIZE))))
        labels = Y
        return inputs, targets, labels


class ReconstructionLoss(nn.Module):
    """
    value depends on batch size and image-size
    """
    def __init__(self, loss: nn.Module, weight: float = 0., trainable: bool = False):
        super().__init__()
        self.loss = loss
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight
        
    def forward(self, x, y):
        return self.loss(x.squeeze(), y.squeeze()) * torch.exp(self.weight)
    
    
class KLDGaussianLoss(nn.Module):
    """
    assuming diagonal Gaussian prior and posterior
    """
    def __init__(self, reduction: str = 'mean', weight: float = 0., trainable: bool = False):
        super().__init__()
        self.reduction = reduction
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight
        
    def forward(self, mean, logvar):
        loss = (mean.pow(2) + logvar.exp() - logvar - 1.) * 0.5
        if self.reduction == 'mean':
            return torch.sum(loss, axis=1).mean() * torch.exp(self.weight)
        return torch.sum(loss) * torch.exp(self.weight)
    
    
class KLDGumbelLoss(nn.Module):
    """
    uniform prior 1/#categories for all categories
    """
    def __init__(self, categorical_dim: int, reduction: str = 'mean', weight: float = 0., trainable: bool = False):
        super().__init__()
        self.dim = categorical_dim
        self.reduction = reduction
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight
        
    def forward(self, proba):
        if self.reduction == 'mean':
            return torch.sum(proba * torch.log(proba * self.dim + 1e-11), dim=1).mean() * torch.exp(self.weight)
        return torch.sum(proba * torch.log(proba * self.dim + 1e-11)) * torch.exp(self.weight)
    
    
class ContrastLoss(nn.Module):
    """
    the input x is a batch of representation vectors --
    focus on difference: make covariance look more like identity
    """
    def __init__(self, weight: float = 0., trainable: bool = False):
        super().__init__()
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight

    def forward(self, x):
        b, d = x.size()
        C = torch.abs(x @ x.T)
        return (1. - torch.trace(C)/torch.sum(C)) * np.log(d + b) * torch.exp(self.weight)
    
    
class AlignLoss(nn.Module):
    """
    cosine similarity between `perceived` and `hinted`
    """
    def __init__(self, weight: float = 0., trainable: bool = False):
        super().__init__()
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight

    def forward(self, x, y):
        xnorm, ynorm = torch.sum(x ** 2, dim=1), torch.sum(y ** 2, dim=1)
        if torch.any(xnorm) and torch.any(ynorm):
            norm = (xnorm * ynorm) ** 0.5
            return 1. - torch.mean(torch.sum(x * y, dim=1)/norm) * torch.exp(self.weight)
        return torch.Tensor([1.])
    
    
class TauLoss(nn.Module):
    """
    thermometer: measures level of the noise in the system
    """
    def __init__(self, weight: float = 0., trainable: bool = False):
        super().__init__()
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight

    def forward(self, x, y=0):
        #sreturn x ** 2 ### distance from the regular softmax
        #return 0. ### do not penalize temperature        
        return (torch.exp(x) - y) ** 4 * torch.exp(self.weight) ### distance from a specific freezing point


def training_step(model, optimizer, criterion, inputs, targets, context):            
    # pass forward
    with torch.cuda.amp.autocast(enabled=False):
        preds = model(inputs, context=context)
        loss = criterion(preds, targets)
    # backp-prop
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())


def train_epoch(model, dataset, context, criterion, optimizer, epoch, batch_size):
    model.train()
    criterion.train()
    history = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for b, (X, Y, C) in enumerate(loader, 1):
        inputs = X.to(DEVICE)
        targets = Y.to(DEVICE)
        history.append(training_step(model, optimizer, criterion, inputs, targets, C.to(DEVICE) if context else None))
        if b % 10 == 0:
            print((f'Epoch {epoch:<4}  training: {b/len(loader):.0%}  '
                   f'loss: {sum(history)/len(history):.4f}            '), end='\r')
    torch.cuda.empty_cache()
    return history


@torch.no_grad()
def validate(model, dataset, context, criterion, epoch, batch_size):
    model.eval()
    criterion.eval()
    history = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (X, Y, C) in enumerate(loader, 1):
        inputs = X.to(DEVICE)
        targets = Y.to(DEVICE)
        preds = model(inputs, context=C.to(DEVICE) if context else None)
        loss = criterion(preds, targets)
        history.append(float(loss.item()))
        if b % 10 == 0:
            print((f'Epoch {epoch:<4}  validation: {b/len(loader):.0%}  '
                   f'loss: {sum(history)/len(history):.4f}              '), end='\r')
    return history


class LossExplorer(nn.Module):
    """
    Trainable mixer with opinionated init and visual evaluation utilities
    """
    # reconstruction keys (static value; required)
    REC = ['rec-AE', 'rec-VAE', 'rec-DVAE']
    # regularizer keys (could be trainable; optional)
    KEYS = REC + ['KLD-Gauss', 'KLD-Gumbel',
                  'Contrast-Gauss', 'Contrast-Gumbel',
                  'Align-Gauss', 'Align-Gumbel',
                  'Temperature']
    
    def __init__(self, keys: list, conf: dict, categorical_dim: int = None, trainable: bool = False):
        super().__init__()
        # outputs ids
        self.keys = keys
        
        # initialize losses all on their "native" scale of magnitude as static
        self.loss = {
            'Reconstruction': ReconstructionLoss(nn.MSELoss(reduction='mean')),
            'KLD-Gauss': KLDGaussianLoss(reduction='mean'),
            'KLD-Gumbel': KLDGumbelLoss(categorical_dim, reduction='mean'),
            'Contrast-Gauss': ContrastLoss(),
            'Contrast-Gumbel': ContrastLoss(),
            'Align-Gauss': AlignLoss(),
            'Align-Gumbel': AlignLoss(),
            # model "termometer": controls level of noise
            'Temperature': TauLoss(),
        }
        # build configuration
        init, order, weight = [],[],{}
        for key in conf:
            if not key in self.KEYS:
                raise Exception(f'Unknown key: {key}')
            if key in self.REC: # first priority non-trainable static weight
                weight[key] = conf[key]
            else: # regularizers with trainable if configured weights
                init.append(conf[key])
                order.append(key)
        self.mixer = torch.Tensor(init)
        if trainable:
            self.mixer = nn.Parameter(self.mixer)
        self.order = order
        self.weight = weight
        self.trainable = trainable
        print(f'Mixer: {self.order} {self.mixer}')        
        # track all components separately
        self.window, self.track = 20, []
        self.history, self.buffer = {'train':[],'test':[]}, {'train':[],'test':[]}
        
    def log(self, *args, finish=False):
        """
        R&D only: windowed training and validation history for each key component
        """
        key = 'train' if self.training else 'test'
        if not finish:
            self.buffer[key].append(tuple(args))
        if len(self.buffer[key]) >= (1 if finish else self.window):
            n = len(self.buffer[key])
            self.history[key].append([sum([x[i] for x in self.buffer[key]])/n for i in range(len(args))])
            self.buffer[key] = []
                
    def forward(self, outputs, targets):
        loss = {}
        # unpack inputs and calculate all losses (even those not in config)
        for i, (k, v) in enumerate(zip(self.keys, outputs)):
            if k in self.REC:
                loss[k] = self.loss['Reconstruction'](v, targets)
            elif k == 'mean':
                loss['KLD-Gauss'] = self.loss['KLD-Gauss'](v, outputs[i + 1])
                loss['Contrast-Gauss'] = self.loss['Contrast-Gauss'](v)
            elif k == 'z': ### do mean instead of z for more stable training
                z = v
            elif k == 'log-variance':
                assert 'KLD-Gauss' in loss
            elif k == 'q':
                loss['KLD-Gumbel'] = self.loss['KLD-Gumbel'](v)
                loss['Contrast-Gumbel'] = self.loss['Contrast-Gumbel'](v)
            elif k == 'p': ### do q instead of p for more stable training
                p = v
            elif k == 'z-context':
                loss['Align-Gauss'] = self.loss['Align-Gauss'](z, v)
            elif k == 'p-context':
                loss['Align-Gumbel'] = self.loss['Align-Gumbel'](p, v)
            elif k == 'tau':
                tau = v.squeeze()
                loss['Temperature'] = self.loss['Temperature'](tau)
                
        # track all variables in their original scale for visual evaluation
        vals = [loss[x].item() for x in self.KEYS if x in loss]
        keys = [k for k in self.KEYS if k in loss]
        vals += [self.mixer[self.order.index(k)].item() if k in self.order else 0 for i,k in enumerate(keys)]
        self.log(*vals)
        self.track = keys
        
        rec = [loss[k] * np.exp(self.weight[k]) for k in self.REC if k in loss]
        # use only those included in conf
        mix = [loss[k] for k in self.order]
        # exp to prevent erasing components with zero factor
        loss = torch.sum(torch.stack(rec + [torch.exp(f) * v for f, v in zip(self.mixer, mix)]))
        # add mixer regularization
        return loss + torch.sum(self.mixer.pow(4))
    
    
def experiment(model: nn.Module,
               tag: str,
               init: dict,
               latent_dim: int,
               categorical_dim: int = None,
               encoder_semantic_dim: int = 0,
               decoder_semantic_dim: int = 0,
               trainable: bool = False,
               tau: float = 0.1,
               dataset: Dataset = AEDataset,
               demo_batch: tuple = None,
               batch_size: int = 16,
               learning_rate: float = 1e-5,
               epochs: int = 3):
    """
    build configuration and run training
    """
    encoder = get_encoder()
    decoder = get_decoder()
    
    context = decoder_semantic_dim > 0 or encoder_semantic_dim > 0
    
    if model == TwinVAE:
        #assert categorical
        model = TwinVAE(encoder, decoder, latent_dim, categorical_dim,
                        encoder_semantic_dim, decoder_semantic_dim, tau).to(DEVICE)
        print('Model: TwinVAE')
    elif model == HydraVAE:
        #dim = CATEGORICAL_DIM if categorical else None
        model = HydraVAE(encoder, decoder, latent_dim, categorical_dim,
                         encoder_semantic_dim, decoder_semantic_dim, tau).to(DEVICE)
        print(f'Model: {"Categorical " if categorical_dim else ""}HydraVAE')
    elif model == DVAE:
        #assert categorical
        model = DVAE(encoder, decoder, latent_dim, categorical_dim,
                     encoder_semantic_dim, decoder_semantic_dim, tau).to(DEVICE)
        print('Model: DVAE')
    else:
        model = VAE(encoder, decoder, latent_dim, 
                    encoder_semantic_dim, decoder_semantic_dim, tau).to(DEVICE)
        print('Model: VAE')

    init['Temperature'] = tau
    criterion = LossExplorer(model.keys, init, categorical_dim, trainable=trainable).to(DEVICE)
    params = [p for p in model.parameters()] + [p for p in criterion.parameters()]
    optimizer = SGD(params, lr=learning_rate, momentum=0.8)

    history, results = [],[]
    for epoch in range(1, epochs + 1):
        
        train_history = train_epoch(model, dataset(trainset), context,
                                    criterion, optimizer, epoch, batch_size=batch_size)
        
        test_history = validate(model, dataset(testset), context,
                                criterion, epoch, batch_size=batch_size)
        
        history.append((np.mean(train_history), np.mean(test_history)))
    criterion.log(None, finish=True)
    k, n = len(criterion.history['test'])//epochs, len(criterion.track)
    for e in range(epochs):
        results.append(np.mean(np.array(criterion.history['test'][k * e:k * (e + 1)])[:,:n], axis=0))
    # show trainig history and return trained model and last test results
    show_progress(model, criterion, history, results, tag, demo_batch)
    return model, {k:v for k,v in zip(criterion.track, results[-1])}

