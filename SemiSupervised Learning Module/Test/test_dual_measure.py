import torch
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
# from demo_extract_features_resnet_RGB_diff import *
import torch.optim as optim
import glob
import cv2
from scipy import io as sio
import numpy as np
import os
from operator import itemgetter
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from scipy.stats import spearmanr as srocc
from scipy.stats import pearsonr as plcc

from scipy.io import loadmat
import skvideo.io
from scipy import signal
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem

from itertools import combinations,chain
import time
import random
import gc
from matplotlib import pyplot as plt
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from sklearn import preprocessing
from scipy.stats import spearmanr,pearsonr 
from sklearn.decomposition import PCA
from scipy.io import loadmat,savemat
import torchvision.models as models
import decord,math
from numpy import linalg
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance

from .fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
# from swin_backbone import SwinTransformer3D as VideoBackbone
# from supervised_datasets import *
from .fastvqa.models.swin_backbone import SwinTransformer3D as VideoBackbone
from .fastvqa.models.head import VQAHead
from .fastvqa.models import DiViDeAddEvaluator

device = torch.device('cuda')

def param(points):
    B, N, D = points.size()
    mean = points.mean(dim=-2, keepdim=True)
    
    diffs = (points - mean)
    prods = torch.bmm(diffs.transpose(1,2).conj(), diffs)
    eps = 1e-3*torch.eye(D).to(device)
    bcov = (prods) / (N - 1)  # Unbiased estimate
    # bcov = bcov+eps
    return mean, bcov#.squeeze()

def distance(feat1, feat2):
    
    mu1, cov1 = param(feat1)
    mu2, cov2 = param(feat2)
    eps = 1e-8*torch.eye(cov1.size(-1))#.to(device)
    # cov2 = 0
    delta = (mu1 - mu2)
    m = torch.bmm(torch.inverse((cov1 + cov2)/2 +eps), delta.transpose(1,2)) 
    dist = torch.bmm(delta, m)

    return torch.sqrt(dist).squeeze(-1).numpy()

class BaseEvaluator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = VideoBackbone()#window_size=(4,4,4), frag_biases=[0,0,0,0]
        self.vqa_head = VQAHead()

    def forward(self, vclip, inference=False, **kwargs):
        # if inference:
        #     self.eval()
        #     with torch.no_grad():
        #         feat = self.backbone(vclip)
        #         score = self.vqa_head(feat)
        #     self.train()
        #     return score, feat
        # else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score
################################# authentic data ####################################
lsvd_test_dir = '~/LSVD/'
ann_file = '~/LSVQ/labels_test.txt'
lsvd_id = []
lsvd_mos = []
with open(ann_file, "r") as fin:
    count = 0
    for line in fin:
        # if count%10==0:
            line_split = line.strip().split(",")
            # print(line_split[0])
            filename, _, _, label = line_split
            # filename = filename.split('/')[-1]
            label = float(label)
            # filename = os.path.join(data_prefix, filename)
            lsvd_id.append(filename.strip())
            lsvd_mos.append(label)
            count +=1
lsvd_mos = np.array(lsvd_mos)

import csv
flickr_id  = []
konvid_mos = []
konvid_resnet = []
konvid_directory = '~/KoNViD_1k_videos/' 
with open(konvid_directory + 'KoNViD_1k_mos.csv', 'r') as file:
    reader = csv.reader(file) 
    for row in reader:
        flickr_id.append(row[0])
        konvid_mos.append(row[1])
           
flickr_id  = flickr_id[1:]
konvid_mos = konvid_mos[1:]
konvid_mos = np.array([float(i) for i in konvid_mos])
konvid_dmos = np.array([5-float(i) for i in konvid_mos])

vqc_dir = '~/LIVE_VQC/' 
video_list = sio.loadmat(vqc_dir+'data.mat')['video_list']
vqc_mos = sio.loadmat(vqc_dir+'data.mat')['mos'].squeeze()
vqc_dmos = np.array([100-float(i) for i in vqc_mos])
#names = [file for file in glob.glob(directory +'/*.yuv')]

utube_directory = '~/youtube_ugc/'
vid_id = loadmat(utube_directory + 'filename.mat')['name']
utube_mos   = loadmat(utube_directory + 'filename.mat')['mos'].squeeze()
utube_dmos = [5-float(i) for i in utube_mos]
utube_dmos = np.array(utube_dmos)

qcomm_directory = '~/live_qualcomm/'
vid_names = loadmat('/media/ece/Shankhanil1/live_qualcomm/live_qualcommData.mat')['video_names']
qcomm_mos = loadmat('/media/ece/Shankhanil1/live_qualcomm/live_qualcommData.mat')['scores'].squeeze()
qcomm_dmos = np.array([100-float(i) for i in qcomm_mos])

from collections import OrderedDict

model = BaseEvaluator().to(device)
path = '/'
state_dict = torch.load(path + 'SSL_VQA_500_30.pth', map_location=device)['primary']
model.load_state_dict(state_dict, strict=True)
model.eval()
        
backbone = VideoBackbone().to(device)
state_dict = torch.load(path + 'SSL_VQA_500_30.pth', map_location=device)['auxillary']
backbone.load_state_dict(state_dict, strict=True)
backbone.eval()
################################### reference feat ######################################3
pristine_dir = '~/pristine_cubes/'
filenames = os.listdir(pristine_dir)
pris_feat = []
for name in filenames:
    clip = torch.from_numpy(np.load(pristine_dir + name)).unsqueeze(0)
    # _,tmp = model(clip.to(device),inference=True)
    with torch.no_grad():
        tmp = backbone(clip.to(device))
    pris_feat.append(tmp.flatten(-3,-1).swapaxes(-2,-1).cpu())
pris_feat = torch.cat(pris_feat, dim=0).flatten(0,1).unsqueeze(0)

################################# authentic niqe feat #######################################################
num_clips = 2
t_frag = 8
depth = 64
# sampler =  SampleFrames(clip_len = 32,frame_interval = 2, num_clips=num_clips)
sampler = FragmentSampleFrames(depth//t_frag,t_frag,2,num_clips)
mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])     


test_db = 'konvid'
#data_feat = []
vid_score = []

reg_score = []
niqe_score = []


tau = 0.1
start = time.time()
for i in range(len(flickr_id)): #  vid_names # video_list  # vid_id # flickr_id test_lsvd_names

    if test_db == 'lsvq_test':
        name = lsvd_id[i]
        video = decord.VideoReader(lsvd_test_dir + str(name).strip())
    
    elif test_db ==  'konvid':
        name = flickr_id[i]
        video = decord.VideoReader(konvid_directory + name + '.mp4')
    
    elif test_db == 'CLIVE':
        name = video_list[i][0][0]
        video = decord.VideoReader(vqc_dir + 'Video/'+  name) 
    
    elif test_db == 'youtube-UGC':
        name = str(vid_id[i]).strip()
        video = decord.VideoReader(utube_directory + 'new_ugc_videos/' + name + '.mp4' )           
    
    elif test_db == 'live qualcomm':
        name = str(vid_names[i][0][0])                                              # liveQualcomm
        get_video = skvideo.io.vread(qcomm_directory + 'Videos/'+ name, 1080, 1920, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
 
    
    # video = torch.tensor(get_video)
    frames = sampler(len(video))
    frame_dict = {idx: video[idx] for idx in np.unique(frames)}
    imgs = [frame_dict[idx] for idx in frames]
    video = torch.stack(imgs, 0)
    
    video = video.permute(3, 0, 1, 2)
    sampled_video = get_spatial_fragments(video)
    
    sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
    sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
    
    
    with torch.no_grad():
        pred = model(sampled_video.to(device))
        feat = backbone(sampled_video.to(device))
    dist_feat = feat.flatten(-3,-1).swapaxes(-2,-1).cpu()#.squeeze()
    
    reg_score.append(pred.mean().item())
    niqe_score.append(np.exp(-distance(pris_feat, dist_feat)*tau).mean())
    
    duration = (time.time() - start)/60.0
    print(vid_score[-1], niqe_score[-1],  duration, i)
    
    gc.collect()
    torch.cuda.empty_cache()

reg_score= np.array(reg_score)
niqe_score= np.array(niqe_score)

# vid_score = (reg_score + niqe_score)/2

print('srocc: %4.4f, plcc: %4.4f' % (srocc(vid_score, konvid_mos)[0],plcc(vid_score, konvid_mos)[0]))
