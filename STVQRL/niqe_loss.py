import random
from numpy.random import randint
from scipy.io.matlab.miobase import matdims
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import time
from itertools import chain, combinations
import torch.nn.functional as f
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


######################### Uncomment it to use cosine similarity measure #########################
# def cosine_similarity(feat1, feat2, tau = 0.1):
#     eps = 1e-8
#     B, N, D = feat1.size()
#     feat1_norm = torch.linalg.norm(feat1, dim=-1)
#     feat2_norm = torch.linalg.norm(feat2, dim=-1)
    
#     feat1_norm = torch.max(feat1_norm, eps * torch.ones_like(feat1_norm))
#     feat2_norm = torch.max(feat2_norm, eps * torch.ones_like(feat2_norm))
    
#     feat1 = feat1/feat1_norm.unsqueeze(-1)
#     feat2 = feat2/feat2_norm.unsqueeze(-1)
    
#     mat = torch.bmm(feat1, torch.transpose(feat2, 1, 2))
#     mat = torch.exp(mat / tau)
        
#     pos = torch.diagonal(mat.clone(),dim1=1,dim2=2)
#     neg1 = torch.sum(mat.clone(),dim=1)
#     neg2 = torch.sum(mat.clone(),dim=2)
    
    
#     # mat_same = torch.bmm(feat1, torch.transpose(feat1, 1, 2))
#     # for i in range(B):
#     #     mat_same[i].fill_diagonal_(0)
#     # mat_same = torch.exp(mat_same / tau)
#     # neg_same = torch.sum(mat_same.clone(),dim=1)
    
#     dist1 = -torch.log(pos/(neg1)) 
#     dist2 = -torch.log(pos/(neg2))
#     dist = dist1.mean() + dist2.mean()
#     return(dist)

def param(points):
    B, N, D = points.size()
    mean = points.mean(dim=-2, keepdim=True)
    diffs = (points - mean)
    prods = torch.bmm(diffs.transpose(1,2).conj(), diffs)
    bcov = (prods) / (N - 1)  # Unbiased estimate
    # bcov = bcov+eps
    return mean,bcov

def distance(feat1, feat2):
    
    mu1, cov1 = param(feat1)
    mu2, cov2 = param(feat2)

    device = mu1.get_device()
    eps = 1e-4*torch.eye(cov1.size(-1)).to(device)
    delta = (mu1 - mu2)
    m = torch.bmm(torch.linalg.pinv((cov1 + cov2)/2 +eps), delta.transpose(1,2))    
    dist = torch.bmm(delta, m)

    return torch.sqrt(dist).squeeze(-1)
    
    
def niqe_loss(feat_anc, feat_view, dist, k, tau = 0.1): #tau = 0.01

    neg_dist = 0
    for i in range(k):
        if i == dist:
            pos_dist = torch.exp(-distance(feat_anc, feat_view[:,i,...])*tau)
        else:
            neg_dist = neg_dist + torch.exp(-distance(feat_anc, feat_view[:,i,...])*tau)
    loss = -torch.log(torch.divide(pos_dist, (pos_dist+neg_dist)))
    return (loss)

class contrastive_loss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, feat_p1, feat_p2):
        self.k = feat_p1.shape[1]
        loss1, loss2 = 0, 0
        for i in range(self.k):
            loss1 = loss1 + niqe_loss(feat_p1[:,i,...], feat_p2, i, self.k)
            loss2 = loss2 + niqe_loss(feat_p2[:,i,...], feat_p1, i, self.k)  
        loss = (loss1 + loss2)/self.k
        return(loss.mean())        
