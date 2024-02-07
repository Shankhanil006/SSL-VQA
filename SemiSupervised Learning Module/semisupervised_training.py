from numpy.random import randint
# from loss_modular import NT_XENT
import os, gc
import argparse
import json
from itertools import *
import numpy as np
import time
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms

from torchvision.models import resnet50,resnet34,resnet152, swin_t
from PIL import Image
import random, json
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn
# from torchsummary import summary
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
# from APV_Dataset import VideoDataset
from matplotlib import pyplot as plt
from scipy import io as sio
from scipy.stats import spearmanr as srocc
from scipy.stats import pearsonr as plcc
import math, copy

import decord
from contrastive_loss import *
from reference_loss import *
from itertools import chain, cycle
from collections import OrderedDict
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR, CyclicLR, CosineAnnealingLR

from swin_backbone import SwinTransformer3D as VideoBackbone
from semisupervised_datasets import SemiSupervisedDataset

from fastvqa.models.head import VQAHead
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
device_ids = [2,1,0]

clip = 32               # No of frames per video
bs = 16                 # batchsize 
num_clips = 1           # clips per video
num_label = 500         # number of labelled videos 

class projection(nn.Module):

    def __init__(
        self, in_channels=1, hidden_channels=1, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        # self.fc_hid = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rois=None):
        # x = self.dropout(x)
        qlt_score = self.relu(self.fc_hid(x))
        return qlt_score

class BaseEvaluator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = VideoBackbone()#backbone
        self.vqa_head = VQAHead()

    def forward(self, vclip, get_feat=False, **kwargs):
        # if inference:
        #     self.eval()
        #     with torch.no_grad():
        #         feat = self.backbone(vclip)
        #         score = self.vqa_head(feat)
        #     self.train()
        #     return score
        # else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            if get_feat:
                return score, feat
            else:
                return score
                
def param(points):
    B, N, D = points.size()
    mean = points.mean(dim=-2, keepdim=True)
    diffs = (points - mean)
    prods = torch.bmm(diffs.transpose(1,2).conj(), diffs)
    eps = 1e-3*torch.eye(D).to(device)
    bcov = (prods) / (N - 1)  # Unbiased estimate
    # bcov = bcov+eps
    return mean,bcov
def distance(feat1, feat2):
    
    mu1, cov1 = param(feat1)
    mu2, cov2 = param(feat2)
    eps = 1e-8*torch.eye(cov1.size(-1)).to(device)
    
    delta = (mu1 - mu2)
    m = torch.bmm(torch.inverse((cov1 + cov2)/2 +eps), delta.transpose(1,2))    
    dist = torch.bmm(delta, m)

    return torch.sqrt(dist).squeeze(-1)

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def save_model(primary, auxillary, epoch):
    # savefolder = '/home/mitra/representation_learning/st_ablation/ablation_loss/'

    savefolder = ''
    state = {
        'primary': primary.module.state_dict(),
        'auxillary': auxillary.module.state_dict(),
        'epoch': epoch,
     }
    # torch.save(state, os.path.join(savefolder, 'faster_semi_crop_niqe_contrastive%d.pth'%(epoch)))
    torch.save(state, os.path.join(savefolder, 'sslvqa_%d.pth'%(epoch)))
    return()

########################## Contrastive pretrained #################################

loadfolder = 'path to STVQRL weights/'
cons_state_dict = torch.load(loadfolder+ 'shift_niqe_lsvd_model_30' + '.pth',map_location='cpu')['model']

primary = BaseEvaluator()
primary.backbone.load_state_dict(cons_state_dict, strict=False)

primary= nn.DataParallel(primary,device_ids=device_ids)
primary = primary.to(device=device)

auxillary = VideoBackbone()
auxillary.load_state_dict(cons_state_dict, strict=False)

auxillary= nn.DataParallel(auxillary,device_ids=device_ids)
auxillary = auxillary.to(device=device)

# head = projection().to(device=device)
# mse_loss = nn.MSELoss()

p_opt = optim.AdamW([
                {'params': primary.module.vqa_head.parameters()},
                {'params': primary.module.backbone.parameters(), 'lr': 1e-4}
            ], lr=1e-3, weight_decay= 0.05)
n_opt = optim.AdamW(auxillary.module.parameters(), lr = 1e-4, weight_decay= 0.05)

#########################################################################################

lsvd_file = '/DATA/meta_data_LSVD/'       # path to json file

lbl_dataset = SemiSupervisedDataset(lsvd_file,clip, num_clips, 'sup', num_label)
lbl_dataloader = DataLoader(lbl_dataset, batch_size=bs, shuffle=True, drop_last=True)

unlbl_dataset = SemiSupervisedDataset(lsvd_file,clip, num_clips, 'unsup',num_label)
unlbl_dataloader = DataLoader(unlbl_dataset, batch_size=bs, shuffle=True, drop_last=True)

def cyclic(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

unlbl_dataloader = cyclic(unlbl_dataloader)

num_epochs = 30
warmup_iter = int(2.5 * len(lbl_dataloader))
max_iter = int(num_epochs * len(lbl_dataloader))
p_lambda = (
    lambda cur_iter: cur_iter / warmup_iter
    if cur_iter <= warmup_iter
    else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
)

n_lambda = (
    lambda cur_iter: cur_iter / warmup_iter
    if cur_iter <= warmup_iter
    else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
)

p_scheduler = torch.optim.lr_scheduler.LambdaLR(
    p_opt, lr_lambda=p_lambda,
)
n_scheduler = torch.optim.lr_scheduler.LambdaLR(
    n_opt, lr_lambda=n_lambda,
)

################################### reference feat ######################################3       
pristine_dir = '/data/mitra/DATA/pristine_cubes/'
filenames = os.listdir(pristine_dir)
pris_subset = random.sample(filenames, 48) # Number of pristine video clips == 48

#############################################################################
wt_cons = 1
wt_stab = 1
tau = 0.01

start_time = time.time()
flag = True  # False in case of learning only on labelled data


for epoch in range(num_epochs+1):
    epoch_loss, reg_loss, niqe_loss = 0, 0, 0
    gt_loss, cons_loss, stab_loss = 0, 0, 0
    
    tmp = random.sample(filenames, 24)
    
    for n_count, lbl_batch in enumerate(lbl_dataloader):   
        pris_feat = []
        for name in tmp:
            pris_clip = torch.from_numpy(np.load(pristine_dir + name)).unsqueeze(0).to(device)
            pris_feat.append(auxillary(pris_clip).flatten(-3,-1).swapaxes(-2,-1).detach())
        pris_feat = torch.cat(pris_feat, dim=0).flatten(0,1).unsqueeze(0)

        vclip, aclip, y = lbl_batch[0].to(device=device), lbl_batch[1].to(device=device), lbl_batch[2].detach().to(device=device)
    
        v_mos= primary(vclip).mean((-3,-2,-1))
        a_mos = primary(aclip).mean((-3,-2,-1))
        
        v_feat = auxillary(vclip).flatten(-3,-1).swapaxes(-2,-1)       
        a_feat = auxillary(aclip).flatten(-3,-1).swapaxes(-2,-1)
            
        v_niqe = torch.exp(-distance(pris_feat, v_feat)*tau)
        a_niqe = torch.exp(-distance(pris_feat, a_feat)*tau)
      
        n_loss = plcc_loss(v_niqe, y) + wt_cons*plcc_loss(v_niqe, a_niqe) 
        p_loss = plcc_loss(v_mos, y)  + wt_cons*plcc_loss(v_mos, a_mos) 
        
        gt_loss += (plcc_loss(v_mos, y) + plcc_loss(v_niqe, y)).item() #+ (plcc_loss(a_niqe, y) + plcc_loss(a_mos, y)).item()
        cons_loss += (plcc_loss(v_niqe, a_niqe) + plcc_loss(v_mos, a_mos)).item()
         
        if flag:
            unlbl_batch = next(unlbl_dataloader) 
            
            vclip, aclip = unlbl_batch[0].to(device=device), unlbl_batch[1].to(device=device)

            v_mos= primary(vclip).mean((-3,-2,-1))
            a_mos = primary(aclip).mean((-3,-2,-1))
            
            v_feat = auxillary(vclip).flatten(-3,-1).swapaxes(-2,-1)       
            a_feat = auxillary(aclip).flatten(-3,-1).swapaxes(-2,-1)
            
            v_niqe = torch.exp(-distance(pris_feat, v_feat)*tau)
            a_niqe = torch.exp(-distance(pris_feat, a_feat)*tau)
            
            eps_mos = plcc_loss(v_mos,a_mos).item()
            eps_niqe = plcc_loss(v_niqe,a_niqe).item()
            
            mask = int(eps_mos > eps_niqe)
            
            p_loss += mask*wt_stab*plcc_loss(v_mos, v_niqe.detach()) + wt_cons*plcc_loss(v_mos, a_mos)  
            n_loss += (1-mask)*wt_stab*plcc_loss(v_mos.detach(), v_niqe) + wt_cons*plcc_loss(v_niqe, a_niqe)           
            
            cons_loss += (plcc_loss(v_niqe, a_niqe) + plcc_loss(v_mos, a_mos)).item()
            stab_loss += (mask*(plcc_loss(v_mos, v_niqe.detach())) + (1-mask)*(plcc_loss(v_mos.detach(), v_niqe))).item()      

        p_opt.zero_grad()
        p_loss.backward()
        p_opt.step()
        p_scheduler.step()
        
        n_opt.zero_grad()
        n_loss.backward()
        n_opt.step()
        n_scheduler.step()
        
        del pris_clip, pris_feat, vclip, v_feat, aclip, a_feat, p_loss, n_loss 
        
        gc.collect()

    n_count+=1
    elapsed_time = (time.time() - start_time)/3600
    print('epoch = %4d , gt_loss = %4.4f , cons_loss = %4.4f , stab_loss = %4.4f , time = %4.2f hr' 
                    % (epoch + 1,  gt_loss/n_count, cons_loss/n_count, stab_loss/n_count, elapsed_time))

    if epoch and epoch%10 ==0: 
        save_model(primary, auxillary, epoch)

    torch.cuda.empty_cache()

