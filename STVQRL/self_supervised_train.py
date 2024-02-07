from numpy.random import randint
# from loss_modular import NT_XENT
import os
import argparse
import json
from itertools import *
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torchvision.transforms as transforms
import cv2
from torchvision.models import resnet50,resnet34,resnet152, swin_t
from PIL import Image
import random
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

from synthetic_datasets import SyntheticDataset
# from apv_datasets import APVDataset
# from pristine_datasets import PristineDataset
from synthetic_lsvd_datasets import LSVDDataset

from niqe_loss import *
from reference_loss import *
from itertools import chain, cycle
from collections import OrderedDict
from swin_backbone import SwinTransformer3D as VideoBackbone

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
# writer = SummaryWriter('/home/ece/Shankhanil/VQA/representation_learning/multiview/logs/cnn/')
#export LD_LIBRARY_PATH=/home/souradeepm/shankhanil/anaconda3/envs/neel_torch/lib
writer = SummaryWriter('/logs/')

# print('Starting')
    
lsvd_dir = '/DATA/LSVD_synthetic_video/'

save_freq = 10

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
device_ids = [2,1]

clip = 32                   # No. of frames per video clip 
tao = 0.1                   # temp cofficients
bs = 6                      #nu of scenes per batch
ps = 224                    # spatial fragment resolution 
fps = 1
k= 9                        # No. of synthetic distorted version per scene, max: 12

####################### Popular Train Datasets ##########################

# csiq_dir = '/DATA/synthetic_databases/csiq/csiq_frames_all'
# vqa_dir = '/DATA/synthetic_databases/live_vqa/live_vqa_frames_all'
# mobile_dir = '/DATA/synthetic_databases/live_mobile/live_mobile_frames_all'
# ecvq_dir = '/DATA/synthetic_databases/ecvq/ecvq_frames_all'
# evvq_dir = '/DATA/synthetic_databases/evvq/evvq_frames_all'
# epfl_cif_dir = '/DATA/synthetic_databases/epfl/epfl_cif_frames_all'
# epfl_4cif_dir = '/DATA/synthetic_databases/epfl/epfl_4cif_frames_all'

# csiq_dataset = SyntheticDataset(csiq_dir, ps, clip, k, fps)
# vqa_dataset = SyntheticDataset(vqa_dir, ps, clip, k, fps)
# mobile_dataset = SyntheticDataset(mobile_dir, ps, clip, k, fps)
# ecvq_dataset = SyntheticDataset(ecvq_dir, ps, clip, k, fps)
# evvq_dataset = SyntheticDataset(evvq_dir,ps, clip, k, fps)
# epfl_cif_dataset = SyntheticDataset(epfl_cif_dir, ps, clip, k, fps)
# epfl_4cif_dataset = SyntheticDataset(epfl_4cif_dir, ps, clip, k, fps)

# apv_dataset = APVDataset(apv_dir, ps, k, clip, corpus)
# apv_dataloader = DataLoader(apv_dataset, batch_size=bs, shuffle=True, drop_last=True)

# syn_trainset = ConcatDataset([vqa_dataset, mobile_dataset, csiq_dataset, ecvq_dataset, evvq_dataset, \
#                           epfl_cif_dataset, epfl_4cif_dataset]) 
# train_dataloader = DataLoader(syn_trainset, batch_size=bs, shuffle=True, drop_last=True)

################################### Load LSVQ synthetic data  #####################################################
lsvd_dataset = LSVDDataset(lsvd_dir,ps, clip, k, fps)
train_dataloader = DataLoader(lsvd_dataset, batch_size=bs, shuffle=True, drop_last=True)

class projection(nn.Module):

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc_hid = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(-3,-1)

    def forward(self, x, rois=None):
        # x = self.dropout(x)
        qlt_score = self.relu(self.fc_hid(x))
        return qlt_score

def save_model(model, opt, epoch, scheduler = None):
    # output_dir = '/home/souradeepm/shankhanil/mount/DATA/Shankhanil/VQA/tmp/representation_learning/spatio-temporal/contrastive/'
    output_dir = '/tmp/contrastive/'
    state = {
        'model': model.module.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'epoch': epoch,
     }
     
    torch.save(state, os.path.join(output_dir, 'shift_niqe_lsvd_model_%d.pth' % (epoch)))
    return()
    
model = VideoBackbone()  #window_size=(4,4,4), frag_biases=[0,0,0,0]
head = projection().to(device=device)

########################## Optional loading of pre-trained Video Swin-T weights #######################################
# state_dict =  torch.load('/swin_tiny_patch244_window877_kinetics400_1k.pth',map_location='cpu')
# if "state_dict" in state_dict:
#     ### migrate training weights from mmaction
#     state_dict = state_dict["state_dict"]

#     i_state_dict = OrderedDict()
#     for key in state_dict.keys():
#         if "head" in key:
#             continue
#         if "cls" in key:
#             tkey = key.replace("cls", "vqa")
#         elif "backbone" in key:
#             i_state_dict[key] = state_dict[key]
#             # i_state_dict["fragments_"+key] = state_dict[key]
#             # i_state_dict["resize_"+key] = state_dict[key]
            
#             i_state_dict[key.replace('backbone.', '')] = state_dict[key]
#         else:
#             i_state_dict[key] = state_dict[key]
# t_state_dict = model.state_dict()

# for key, value in t_state_dict.items():
#     if key in i_state_dict and i_state_dict[key].shape != value.shape:
#         print(key)
#         i_state_dict.pop(key)

# model.load_state_dict(i_state_dict, strict=False)
##########################################################################################################################


model= nn.DataParallel(model,device_ids=device_ids)
model = model.to(device=device)
opt = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.05)

num_epochs = 30
warmup_iter = int(2.5 * len(train_dataloader))
max_iter = int(num_epochs * len(train_dataloader))
lr_lambda = (
    lambda cur_iter: cur_iter / warmup_iter
    if cur_iter <= warmup_iter
    else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     opt, lr_lambda=lr_lambda,
# )

scheduler = CosineAnnealingLR(opt,
                              T_max = max_iter, # Maximum number of iterations.
                             eta_min = 1e-5) # Minimum learning rate.

################### Load model #########################################
# savefolder = '/home/souradeepm/shankhanil/representation_learning/tmp/sole_contrastive/'
# checkpoint = torch.load(savefolder+ 'shift_niqe_lsvd_model_'+ str(epoch) + '.pth')
# model.load_state_dict(checkpoint['model'])
# model= nn.DataParallel(model,device_ids=[0,1])
# model = model.to(device=device)
# opt.load_state_dict(checkpoint['optimizer'])
# scheduler.load_state_dict(checkpoint['scheduler'])

#######################################################################################

eps =1e-8
start_time = time.time()
start = 0
total_loss = []

k = k + 1
contrastive_criterion = contrastive_loss()
reference_criterion = reference_loss()

for epoch in range(start,num_epochs+1):
# while(1):
#     epoch = 0

    epoch_loss = 0
    flag = 0
        
    for n_count, batch in enumerate(train_dataloader):
        p1_batch, p2_batch = batch[0], batch[1]
        
        p1_batch = p1_batch.view(bs*k, 3, clip,  ps, ps).to(device)
        p2_batch = p2_batch.view(bs*k, 3, clip//fps,  ps, ps).to(device)
        
        feat_p1   = model(p1_batch).squeeze().flatten(-3,-1) # .mean(-3)
        feat_p2   = model(p2_batch).squeeze().flatten(-3,-1)

        feat_p1   = head(feat_p1).swapaxes(-2,-1)
        feat_p2   = head(feat_p2).swapaxes(-2,-1)
        
        feat_p1   = feat_p1.view(bs, k, -1, 64)
        feat_p2   = feat_p2.view(bs, k, -1, 64)
        
        loss_contrastive = contrastive_criterion( feat_p1, feat_p2)
        
        loss = loss_contrastive 
        
        # total_loss.append(loss.item())
        epoch_loss += loss.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        batchtime = time.time()
        # print(loss.item())
    # break

    total_loss.append(epoch_loss/(n_count + 1))
    np.save( '/tmp/loss.npy', total_loss)
    writer.add_scalar('Loss/training loss',
                            epoch_loss/(n_count + 1),
                            epoch)
    elapsed_time = (time.time() - start_time)/60
    print('epoch = %4d , loss = %4.4f , time = %4.2f m' % (epoch + 1, epoch_loss / (n_count + 1), elapsed_time))
    

    if epoch and epoch % save_freq == 0:
        save_model(model, opt, epoch, scheduler)
        
writer.flush()
torch.cuda.empty_cache()
