import torch
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

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

from scipy.io import loadmat
import skvideo.io, decord

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem
from itertools import combinations
import time
import random
import gc
from matplotlib import pyplot as plt
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from sklearn import preprocessing
from scipy.stats import spearmanr,pearsonr 
from scipy.io import loadmat,savemat
from tqdm import tqdm
import torchvision.models as models
# from demo_extract_features_resnet_RGB_diff import *
# from sharp_patch_generator_old import *

from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda:0')

strred_directory = '/media/ece/DATA/Shankhanil/VQA/strred_files/'
##################################  LIVE VQA  #####################################

live_directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
with open(live_directory + 'live_video_quality_seqs.txt') as f:
    video_names = f.readlines()
live_video_list = [x.strip('.yuv\n') for x in video_names] 

with open(live_directory + 'live_video_quality_data.txt') as f:
    video_dmos = f.readlines()
live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
live_dmos = np.array(live_dmos_list)

#feat = np.zeros([len(video_list),2048*2])

live_seq = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
rate = [25,25,25,25,25,25,25,50,50,50]

seq_id = {0:'pa', 1:'rb', 2:'rh', 3:'tr', 4:'st', 5:'sf', 6:'bs', 7:'sh',  8:'mc', 9:'pr'}

########################## EPFL-POLIMI CIF #####################################

epfl_cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
with open(epfl_cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_cif_dmos = 5-np.array(epfl_cif_dmos_list)

epfl_cif_seq = np.array(['foreman','hall','mobile','mother','news','paris'])

########################## EPFL-POLIMI 4CIF #####################################

epfl_4cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
with open(epfl_4cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_4cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_4cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_4cif_dmos = 5-np.array(epfl_4cif_dmos_list)

epfl_4cif_seq = np.array(['CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'])

############################### Mobile LIVE ###################################
mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
mobile_dmos = sio.loadmat(strred_directory + 'strred_mobile.mat')['dmos'].squeeze()

mobile_video_list = sio.loadmat(strred_directory + 'strred_mobile.mat')['names'].squeeze()
#mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'}
mobile_seq = np.array(['bf', 'dv', 'fc', 'hc', 'la', 'po', 'rb', 'sd', 'ss', 'tk'])
mobile_dist = np.array(['r1','r2','r3','r4','s14','s24','s34','t14','t124','t421','t134','t431','w1','w2','w3','w4'])

###############################  CSIQ  #############################################
csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
with open(csiq_directory + 'video_subj_ratings.txt') as f:
    file_names = f.readlines()
csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
csiq_dmos = np.array(csiq_dmos_list)

csiq_seq = np.array(['BQMall', 'BQTerrace','BasketballDrive','Cactus', \
            'Carving','Chipmunks','Flowervase','Keiba','Kimono', \
            'ParkScene','PartyScene','Timelapse'])

################## ECVQ ###########################################################
ecvq_directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/'
with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
    file_names = f.readlines()
ecvq_video_list = [x.split()[0] for x in file_names] 
ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
ecvq_dmos = np.array(ecvq_dmos_list)

ecvq_seq = np.array(['container_ship', 'flower_garden', 'football', 'foreman', 'hall', 'mobile', 'news', 'silent'])

################## EVVQ ###########################################################
evvq_directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/'
with open(evvq_directory + 'subjective_scores_vga.txt') as f:
    file_names = f.readlines()
evvq_video_list = [x.split()[0] for x in file_names] 
evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
evvq_dmos = np.array(evvq_dmos_list)

evvq_seq = np.array(['cartoon', 'cheerleaders', 'discussion', 'flower_garden', 'football', 'mobile', 'town_plan', 'weather'])

live_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/live_vqa/liveVideo/reference/'
epfl_cif_vid_dir  = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/originals/CIF/'
epfl_4cif_vid_dir = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/originals/4CIF/'
mobile_vid_dir    = '/media/ece/DATA/Shankhanil/VQA/live_mobile/LIVE_VQA_mobile/'
csiq_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
ecvq_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/reference/'
evvq_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/reference/'

bvi_vid_dir = '/media/ece/DATA/Shankhanil/VQA/BVI/ORIG/'

apv_vid_dir = '/media/ece/vip_vol1/LIVE_APV/videos/'

from natsort import natsort_keygen,natsorted, ns
natsort_key = natsort_keygen(alg=ns.IGNORECASE)

live_filenames = natsorted([file for file in glob.glob(live_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

epfl_cif_filenames = natsorted([file  for file in glob.glob(epfl_cif_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

epfl_4cif_filenames = natsorted([file  for file in glob.glob(epfl_4cif_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

mobile_filenames = natsorted([file for file in glob.glob(mobile_vid_dir +'*org.yuv')], alg=ns.IGNORECASE)

csiq_filenames = natsorted([file for file in glob.glob(csiq_vid_dir +'*ref.yuv')], alg=ns.IGNORECASE)

ecvq_filenames = natsorted([file for file in glob.glob(ecvq_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

evvq_filenames = natsorted([file for file in glob.glob(evvq_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

bvi_filenames = natsorted([file for file in glob.glob(bvi_vid_dir +'*.yuv')], alg=ns.IGNORECASE)
apv_filenames = natsorted([file for file in glob.glob(apv_vid_dir +'*o.yuv')], alg=ns.IGNORECASE)
###################################3 Network ###########################################

def random_crop(img, random_crop_size = (112,112)):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 1
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
#    return img[y:(y+dy), x:(x+dx), :]
    return (x,y)

def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def pyramid(image, scale, res=0):
    G = image.copy()
    gpA = [G]
    for i in range(scale):
        G = cv2.pyrDown(G)
        gpA.append(G)
        
    lpA = [np.expand_dims(gpA[scale], -1)]
    for i in range(scale,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(np.expand_dims(L,-1))
        
#    return(lpA[res-1])
    return(lpA)
    
#############################################################
databases = {  'live'      :{'num':150, 'ref':10, 'dist': 15, 'height': 432, 'width' : 768, 'seq' : live_seq,   'files' : live_filenames,      'dmos' : live_dmos}, \
               'mobile'    :{'num':160, 'ref':10, 'dist': 16, 'height':720, 'width' :1280,  'seq' : mobile_seq,   'files' : mobile_filenames,    'dmos' : mobile_dmos}, \
               'epfl_cif'  :{'num':72,  'ref':6,  'dist': 12, 'height':288, 'width' :352, 'seq' : epfl_cif_seq,   'files' : epfl_cif_filenames,  'dmos' : epfl_cif_dmos},   \
               'epfl_4cif' :{'num':72,  'ref':6,  'dist': 12, 'height':576, 'width' : 704, 'seq' : epfl_4cif_seq,  'files' : epfl_4cif_filenames,  'dmos' : epfl_4cif_dmos},   \
               'csiq'      :{'num':216, 'ref':12, 'dist': 18, 'height':480, 'width' : 832, 'seq' : csiq_seq,       'files' : csiq_filenames,  'dmos' : csiq_dmos}, \
               'ecvq'      :{'num':90,  'ref':8,  'dist': 10, 'height':288, 'width' :  352,'seq' : ecvq_seq,       'files' : ecvq_filenames,  'dmos' : ecvq_dmos},   \
               'evvq'      :{'num':90,  'ref':8,  'dist': 9, 'height':480, 'width' : 640, 'seq' : evvq_seq,       'files' : evvq_filenames,  'dmos' : evvq_dmos}, \
               'bvi'       :{'height':1080, 'width' : 1920, 'files' : bvi_filenames}, \
               'apv'       :{'height':2160, 'width' : 3840, 'files' : apv_filenames}}

ecvq_num_seq = [12, 11, 10, 11, 12, 10, 12, 12]
evvq_num_seq = [12, 9, 12, 11, 11, 11, 12, 12]

################################# authentic data ####################################
import csv
flickr_id  = []
konvid_mos = []
konvid_resnet = []
konvid_directory = '/media/ece/DATA/Shankhanil/VQA/konvid/KoNViD_1k_videos/' 
with open(konvid_directory + 'KoNViD_1k_mos.csv', 'r') as file:
    reader = csv.reader(file) 
    for row in reader:
        flickr_id.append(row[0])
        konvid_mos.append(row[1])
           
flickr_id  = flickr_id[1:]
konvid_mos = konvid_mos[1:]
konvid_dmos = np.array([5-float(i) for i in konvid_mos])

liveVQC_dir = '/media/ece/DATA/Shankhanil/VQA/LIVE_VQC/' 
video_list = sio.loadmat(liveVQC_dir+'data.mat')['video_list']
vqc_mos = sio.loadmat(liveVQC_dir+'data.mat')['mos'].squeeze()
vqc_dmos = np.array([100-float(i) for i in vqc_mos]  )
#names = [file for file in glob.glob(directory +'/*.yuv')]

utube_directory = '/media/ece/DATA/Shankhanil/VQA/youtube_ugc/'
vid_id = loadmat(utube_directory + 'filename.mat')['name']
utube_mos   = loadmat(utube_directory + 'filename.mat')['mos'].squeeze()
utube_dmos = [5-float(i) for i in utube_mos]
utube_dmos = np.array(utube_dmos)
########################################################################################
all_train_dbs = ['live','mobile',  'epfl_4cif', 'epfl_cif', 'csiq', 'ecvq', 'evvq']#

output_dir = '/media/ece/DATA/Shankhanil/VQA/representation_features/'

num_clips =4
sampler =  SampleFrames(clip_len = 32,frame_interval = 2, num_clips=num_clips)
mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375]) 

frame_feat, diff_feat, flow_feat = [], [] , []
start = time.time()
clip = 1
for train_db in all_train_dbs:
#    for ref_id in range(databases[train_db]['ref']):
    filenames = databases[train_db]['files']
    height, width = databases[train_db]['height'], databases[train_db]['width']
        
    for name in tqdm(filenames):
        vid_name = name.split('/')[-1]
        video = skvideo.io.vread(name,height, width, as_grey = False, inputdict={'-pix_fmt': 'yuvj420p'})
        
        frames = sampler(len(video))
        frame_dict = {idx: video[idx] for idx in np.unique(frames)}
        imgs = [torch.from_numpy(frame_dict[idx]) for idx in frames]
        video = torch.stack(imgs, 0).permute(3, 0, 1, 2)
        
        sampled_video = get_spatial_fragments(video, aligned = 32)
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        # tmp = torch.chunk(sampled_video, chunks=num_clips, dim = -3)
        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
    
        for i in range(num_clips):
            np.save('/media/ece/SSD/pristine_cubes/'+ str(clip),sampled_video[i].numpy())
            clip+=1
            
    #     break
    # break
    gc.collect()
    end = time.time()
    duration = (end - start)/60
    print(train_db, duration)
    # break

