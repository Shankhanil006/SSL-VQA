import numpy as np
import os
import glob
import skvideo.io
#import matplotlib.pyplot as plt
import time
import cv2
from natsort import natsort_keygen,natsorted, ns
natsort_key = natsort_keygen(alg=ns.IGNORECASE)

import argparse
parser = argparse.ArgumentParser(description='Nothing')
parser.add_argument('--batch',  type=str,   help='folder number')
parser.add_argument('--len',  type=float,   help='folder number')
args = parser.parse_args()

#args.batch = '1'
#args.len = 0.1

path_to_dir = '/media/ece/'
out_dir = '/media/ece/vip_vol2/'

lsvd_dir = path_to_dir + 'Shankhanil1/LSVD/'
outpath = path_to_dir + 'Shankhanil1/tmp/inter3.mp4'

optical_flow = cv2.optflow.DualTVL1OpticalFlow_create() 

dist = '10'


#global scale
scale = 0.25
def resize(inp, scale):
    out = cv2.resize(inp, None, fx=scale, fy=scale)
#    out = cv2.resize(out, None, fx= 1/scale, fy = 1/scale) *(1/scale)
    return(out)
    
start = time.time()
#for i in range(1,2):
folder = 'yfcc-batch' + args.batch + '/'
folder_dir = lsvd_dir + folder

names = natsorted([file for file in glob.glob(folder_dir +'*.mp4')], alg=ns.IGNORECASE)
num = np.linspace(0,len(names)-1, int(args.len*len(names)), dtype = int)

for j in range(len(num)):
    # print('yo')
    name = names[num[j]]
    filename = name.split('/')[-1]
    
    videometadata = skvideo.io.ffprobe(name) 
    frame_rate = videometadata['video']['@avg_frame_rate']
    frame_rate = int(np.round(int(frame_rate.split('/')[0])/int(frame_rate.split('/')[1])))
    new_rate = frame_rate//2
#    os.system('ffmpeg -i ' + name  + ' -qscale:v 1  -vcodec mpeg2video -acodec copy -y ' + outpath) #   -b:v 10000k
    
    curr_path = out_dir + 'LSVD_synthetic_video/' + folder + dist + '/' 
    filename = curr_path + str(j+1) + '.mp4'
    
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)
        
    inp_vid = skvideo.io.vread(name, as_grey = False, outputdict={'-r':str(new_rate)})
    writer = skvideo.io.vwrite(filename, inp_vid, inputdict={'-r': str(new_rate)},outputdict={
          '-filter:v': 'minterpolate', '-r': str(frame_rate)
        })
    
end =time.time() 
# print((end-start)/3600)
#for name in names:
#    filename = name.split('/')[-1]#.rstrip('.mp4')
#    os.system('ffmpeg -i ' + name  + ' -b:v 100k -vcodec libx264 -acodec copy -y ' + outpath + filename)
#    
#    
##    vid = skvideo.io.vread(name, as_grey = True)
##    for i in range(len(vid)):
##        np.save(outpath + filename + '_0_' +   str(i),vid[i])   
#    videometadata = skvideo.io.ffprobe(outpath + filename) 
#    frame_rate = videometadata['video']['@avg_frame_rate']
#    frame_rate = int(np.round(int(frame_rate.split('/')[0])/int(frame_rate.split('/')[1])))
#    break
