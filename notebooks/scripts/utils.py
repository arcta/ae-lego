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
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, AdamW
from torch.cuda.amp import GradScaler
from torchsummary import summary

from .experiment import *


# workspace path
ROOT = f'{os.environ["HOME"]}/workspace/ae-lego/notebooks'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = 28
REC_SIZE = 22


def show_inputs(batch, size=(10, 10)):
    X, Y, C = batch
    n = len(X)
    fig, ax = plt.subplots(1, n, figsize=size)
    for i in range(n):
        ax[i].imshow(X[i,:].squeeze().numpy(), 'gray')
        ax[i].axis('off')
    ax[0].set_title(f'Input:    {X.shape[1:]}', fontsize=10, ha='left', x=0)
    plt.show()
    
    
def show_targets(batch, size=(10, 10)):
    X, Y, C = batch
    n = len(Y)
    fig, ax = plt.subplots(1, n, figsize=size)
    for i in range(n):
        ax[i].imshow(Y[i,:].squeeze().numpy(), 'gray')
        ax[i].axis('off')
    ax[0].set_title(f'Target:    {Y.shape[1:]}', fontsize=10, ha='left', x=0)
    plt.show()
    
    
@torch.no_grad()
def show_model_output(model, batch, index: int = None, tag: str = '', size: tuple = (10, 10)):
    model.eval()
    X, Y, C = batch
    n = len(X)
    output = model(X.to(DEVICE), context=C)
    P = output.cpu() if index is None else output[index].cpu()
    fig, ax = plt.subplots(1, n, figsize=size)
    for i in range(n):
        ax[i].imshow(P[i,:].squeeze().numpy(), 'gray')
        ax[i].axis('off')
    tag = '' if tag == '' else f' {tag}'
    ax[0].set_title(f'Model{tag}:    {P.shape[1:]}', fontsize=10, ha='left', x=0)
    plt.show()
    
    
def show_progress(model, criterion, history, results, tag, demo_batch, size=7):
    show_targets(demo_batch)
    for key in criterion.REC:
        if key in criterion.weight:
            show_model_output(model, demo_batch, criterion.keys.index(key), key[4:])
    # show current value
    print(f'Mixer: {criterion.mixer}')
    labels = criterion.track
    n = len(labels)
    fig, ax = plt.subplots(n + 1, 1, figsize=(size, 3 * (n + 1)))
    ax[0].text(0.05, 1.02, 'Composite Loss History',
               ha='left', va='bottom', transform=ax[0].transAxes, fontsize=10)
    history = np.array(history)
    ax[0].plot(history[:,0], color='C0', label='training')
    ax[0].plot(history[:,1], color='#f50', label='validation')
    ax[0].set_xticks([])
    ax[0].tick_params(axis='y', which='major', labelsize=7)
    ax[0].legend(bbox_to_anchor=(1.1, 0.5), loc='center left', frameon=False)    
    for i in range(n):
        x = np.array(criterion.history['train'])[:,i]
        delta = np.max(x) - np.min(x)
        l = len(x)
        ax[i + 1].plot(x, color='C0', alpha=0.75, label='training')
        xt = np.mean(x[len(x)//2:])
        ax[i + 1].axhline(y=xt, color='C0', linestyle=':')        
        x = np.array(results)[:,i]
        ax[i + 1].plot([(j + 0.75) * l//len(results) for j in range(len(results))], x,
                       color='#f50', marker='x', linestyle='', label='validation')
        xv = np.mean(x[len(x)//2:])
        ax[i + 1].axhline(y=xv, color='#f50', linestyle=':')        
        transform=ax[i + 1].transAxes
        title = ' Reconstruction' if labels[i] in criterion.REC else ''
        ax[i + 1].text(0.05, 1.02, f'{labels[i].replace("rec-","")}{title}',
                       ha='left', va='bottom', transform=transform, fontsize=10)
        ax[i + 1].text(0.48, 1.02, f'{xt:.4f}',
                       ha='right', va='bottom', transform=transform, color='C0')
        ax[i + 1].text(0.5, 1.03, '/',
                       ha='center', va='bottom', transform=transform)
        ax[i + 1].text(0.52, 1.02, f'{xv:.4f}',
                       ha='left', va='bottom', transform=transform, color='#f50')
        ax[i + 1].set_xticks([])
        if delta < 1e-4:
            ax[i + 1].set_yticks([np.round(np.min(x), 2)])
            ax[i + 1].ticklabel_format(useOffset=False, style='plain')
        ax[i + 1].tick_params(axis='y', which='major', labelsize=7)
        ax[i + 1].legend(bbox_to_anchor=(1.1, 1), loc='upper left', frameon=False)
        if labels[i] in criterion.order and criterion.trainable:
            axt = ax[i + 1].twinx()
            x = np.array(criterion.history['train'])[:,i + n]
            axt.plot(x, color='gray', linestyle='-.', label='mixer profile')
            xf = np.mean(x[len(x)//2:])
            axt.axhline(y=xf, color='gray', linestyle=':')
            axt.text(0.9, 1.02, f'mixin: {xf:.2f}', ha='right', va='bottom',
                     transform=axt.transAxes, color='gray', fontsize=10)
            axt.set_ylabel('mixin-factor', color='gray', fontsize=8)
            axt.tick_params(axis='y', labelcolor='gray')
            axt.tick_params(axis='y', which='major', labelsize=7)
            axt.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', frameon=False)
    plt.savefig(f'{ROOT}/output/history-{tag}.png')
    plt.show()


@torch.no_grad()
def get_embeddings(encoder, dataset, path, tag='', batch_size=16, num_batches=100):
    encoder.eval()
    vectors, labels = [], []
    for X, _, Y in DataLoader(dataset, batch_size=batch_size):
        vectors.append(encoder(X.to(DEVICE))[0].cpu())
        labels += Y.numpy().tolist()
        if num_batches < 0:
            break
        num_batches -= 1
    vectors = torch.cat(vectors, dim=0)
    writer = SummaryWriter(f'./runs/mnist-{path}')
    writer.add_embedding(vectors, metadata=labels, tag=tag, global_step=None)
    writer.close()
    return vectors.numpy(), np.array(labels)


def show_latent_space(vectors, labels, tag, axes=[(0,1),(0,2),(1,2)], size=7):
    n = len(axes)
    score = silhouette_score(vectors, labels, metric='euclidean')
    fig, ax = plt.subplots(n, 1, figsize=(size, size * n))
    for i, a in enumerate(axes):
        ax[i].set_title(f'VAE: latent space  Z{a}  silhouette-score {score:.4f}', fontsize=10)
        for d in range(10):
            ax[i].axhline(y=0, linestyle=':', color='gray', linewidth=0.5)
            ax[i].axvline(x=0, linestyle=':', color='gray', linewidth=0.5)
            ax[i].scatter(vectors[labels==d,a[0]], vectors[labels==d,a[1]], s=30, marker=f'${d}$', color=f'C{d}')
            ax[i].tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.savefig(f'{ROOT}/output/latent-{tag}.png')
    plt.show()


@torch.no_grad()
def show_reconstruction_map(decoder, tag, zgrid=(-4, 4, 11), axes=[(0,1),(0,2),(1,2)], size=5):
    """
    plot center 2d slice of vae-recontructions manifold
    """
    d = zgrid[-1]
    decoder.eval()
    fig, ax = plt.subplots(len(axes), 1, figsize=(size, size * len(axes)))
    for n, a in enumerate(axes):    
        ax[n].set_title(f'VAE: latent space  Z{a}', fontsize=10)
        manifold = np.zeros((REC_SIZE * d, REC_SIZE * d))
        xgrid, ygrid = np.linspace(*zgrid), np.linspace(*zgrid)
        for i, y in enumerate(ygrid):
            for j, x in enumerate(xgrid):
                z = np.zeros((1, decoder.latent_dim))
                z[0][a[0]] = x
                z[0][a[1]] = y
                z = torch.Tensor(z).to(DEVICE)
                rec = decoder(z)[0].cpu().squeeze().numpy()
                manifold[i * REC_SIZE: (i + 1) * REC_SIZE, j * REC_SIZE: (j + 1) * REC_SIZE] = rec
        pixel_range = np.arange(REC_SIZE//2, (d - 1) * REC_SIZE + REC_SIZE//2 + 1, REC_SIZE)
        ax[n].set_xticks(pixel_range, np.round(xgrid, 1))
        ax[n].set_yticks(pixel_range, np.round(ygrid, 1))
        ax[n].tick_params(axis='both', which='major', labelsize=7)
        ax[n].imshow(manifold, cmap='gray')
    plt.tight_layout()
    plt.savefig(f'{ROOT}/output/recmap-{tag}.png')
    plt.show()
    

@torch.no_grad()
def show_categoric_reconstruction_map(decoder, latent_dim, categorical_dim, tag, num_samples=10, size=5.5):
    decoder.eval()
    manifold = np.zeros((REC_SIZE * categorical_dim, REC_SIZE * num_samples))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_title('DVAE: categorical (codebook) axes', fontsize=10)
    for c in range(categorical_dim):
        samples = np.zeros((num_samples, latent_dim, categorical_dim))
        samples[:,:,c] = (np.random.rand(num_samples, latent_dim) >= 0.5).astype(float)
        samples = torch.Tensor(samples).view(num_samples, latent_dim * categorical_dim)
        output = decoder(samples.to(DEVICE))[0].cpu().squeeze().numpy()
        for i in range(num_samples):
            manifold[c * REC_SIZE:(c + 1) * REC_SIZE, i * REC_SIZE:(i + 1) * REC_SIZE] = output[i,:,:]
    ax.imshow(manifold, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([REC_SIZE * i + REC_SIZE//2 for i in range(categorical_dim)])
    ax.set_yticklabels(range(categorical_dim))
    plt.savefig(f'{ROOT}/output/recmap-{tag}-{categorical_dim}.png')
    plt.show()


@torch.no_grad()    
def show_conditional_reconstruction_map(model, semantic_dim, tag, labels, categorical=None, temperature=1.,
                                        num_samples=10, size=5.5):    
    model.eval()
    fig, ax = plt.subplots(figsize=(size, size))
    prefix = ('DVAE' if categorical else ('VAE' if categorical is not None else ('DVAE' if 'rec-DVAE' in model.keys else 'VAE')))
    ax.set_title(f'{prefix}: conditional (semantic) samples', fontsize=10)
    manifold = np.zeros((REC_SIZE * semantic_dim, REC_SIZE * num_samples))
    for c in range(semantic_dim):
        if categorical is not None:
            samples = model.sample(num_samples, c, temperature, categorical=categorical).cpu().squeeze().numpy()
        else:
            samples = model.sample(num_samples, c, temperature).cpu().squeeze().numpy()
        for i in range(num_samples):
            manifold[c * REC_SIZE:(c + 1) * REC_SIZE, i * REC_SIZE:(i + 1) * REC_SIZE] = samples[i,:,:]
    ax.imshow(manifold, cmap='gray')
    ax.set_yticks([REC_SIZE * i + REC_SIZE//2 for i in range(semantic_dim)])
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    plt.savefig(f'{ROOT}/output/recmap-{tag}-conditional.png')
    plt.show()

