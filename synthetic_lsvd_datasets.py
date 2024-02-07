from numpy.random import randint
import os,glob
import argparse
import json
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from torchvision.models import resnet50,resnet34,resnet152
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
import cv2
from itertools import cycle
from random import shuffle
# import skvideo.io
import decord

optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()  
# flow_scale = 1/8
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class LSVDDataset(Dataset):
    def __init__(self, lsvd_dir, ps = 224, depth = 32,  k = 9, fps = 2, transform=transforms.ToTensor()):
        
        self.lsvd_dir = lsvd_dir
        self.pathnames = os.listdir(lsvd_dir)
        ref_filenames = os.listdir(lsvd_dir +'0/')
        self.ref_pathnames = [os.path.join(lsvd_dir +'0', f) for f in ref_filenames]
        # self.dic = json.load(open(lsvd_dir))
        
        self.transform = transform
        self.k = k
        self.fps = fps
        self.depth = depth
        
        self.ps = ps
        
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        
        self.sampler =  SampleFrames(clip_len = depth,frame_interval = 2, num_clips=1)

        shuffle(self.ref_pathnames)
        # self.id = np.linspace(0, len(self.dic) - 1, int(0.1*len(self.dic)), dtype = int)
        # self.id = random.sample(range(len(self.dic)), 200)
        
    def __len__(self):
        return len(self.ref_pathnames)//10
        # return(len(self.id))

    def __getitem__(self, idx):
        ref_path = self.ref_pathnames[idx]
        # ref_name = glob.glob(lsvd_folder+'/*o')[0].rsplit('/',1)[-1]
        vid_name = ref_path.rsplit('/',1)[-1]
        
        # dist_paths = np.arange(1,self.k+1)#random.sample(self.pathnames, self.k)
        dist_paths = np.random.choice(np.arange(1,13), size = self.k, replace = False)
        dist_names = [os.path.join(self.lsvd_dir, str(p), vid_name) for p in dist_paths]
        
        ref_name = ref_path
        lsvd_names = [ref_name] + dist_names

        # vid_ref = skvideo.io.vread(ref_name)
        vid_ref = decord.VideoReader(ref_name)
        frame_ids = self.sampler(len(vid_ref) - 5,True)
        # height,width = vid_ref.shape[1], vid_ref.shape[2]
        
        # st_frame_id = np.random.randint(1,vid_len-2*self.depth-1)
        # frame_ids = np.arange(st_frame_id, st_frame_id + 2*depth,2)
        
        all_clips = []
        for name in lsvd_names:
            video_reader = decord.VideoReader(name)
            frame_dict = {idx: torch.from_numpy(video_reader[idx].asnumpy()) for idx in np.unique(frame_ids)}
            imgs = [frame_dict[idx] for idx in frame_ids]
            clip = torch.stack(imgs, 0)

            # vid = torch.tensor(skvideo.io.vread(name))
            # clip = vid[frame_ids, ...]
            
            norm_clip = ((clip - self.mean) / self.std)
            all_clips.append(norm_clip.permute(3,0,1,2))
            
        all_clips = torch.stack(all_clips)
        # sampled_clip = get_spatial_fragments_v2(all_clips, aligned = self.depth)
        clip_view1 =  get_spatial_fragments_v2(all_clips, aligned = self.depth)
        clip_view2 =  get_spatial_fragments_v2(all_clips, aligned = self.depth, top= False)

        return clip_view1, clip_view2

def get_spatial_fragments_v2(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    top=True,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_upsample:

        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)



    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if top:
        start_h = 0
        start_w = 0
    else:
        start_h = res_h // (2*fragments_h)
        start_w = res_w // (2*fragments_w)
        
    if hlength//2 > fsize_h:
        rnd_h = torch.randint(start_h,
            start_h + hlength//2 - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
        )
    else:
        rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    if wlength//2 > fsize_w:
        rnd_w = torch.randint(start_w,
           start_w + wlength//2 - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
        )
    else:
        rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[..., t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    ..., t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video

class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.
        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.
        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=True, start_index=0):
        """Perform the SampleFrames loading.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)
