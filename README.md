# SSL-VQA
This is the official repository for the paper titled " Knowledge Guided Semi-Supervised Learning for Quality Assessment of User Generated Videos" accepted in Proceedings of AAAI conf. on AI (AAAI 24) by Shankhanil Mitra and Rajiv Soundararajan. 

In this work, we address the challenge of requirement of large scale human annotated videos for training by first designing a self- supervised Spatio-Temporal Visual Quality Representation Learning (ST-VQRL) framework to generate robust quality aware features for videos. Then, we propose a dual-model based Semi Supervised Learning (SSL) method specifically designed for the Video Quality Assessment (SSL-VQA) task, through a novel knowledge transfer of quality predictions between the two models.

![SSL-VQA](https://github.com/Shankhanil006/SSL-VQA/blob/main/sslvqa.png?raw=true)

## Installation 
>conda env create -f environment.yml

## Generating Syntheitc distortion for UGC Videos
Download LSVQ database from [LSVQ](https://github.com/baidut/PatchVQ?raw=true) . Randomly choose 200 or more videos and generate 12 distorted version of each scene by running the file LSVQ Synthetic/script.py. On the other hand user can also download synthetically distorted databases like LIVE-VQA, EPFL-PoLiMI, LIVE Mobile, CSIQ VQD, ECVQ and EVVQ for training ST_VQRL.

To train self-supervised video feature model (ST-VQRL) using LSVQ videos run following:
>python3 STVQRL/self_supervised_train.py

We provide Pre-trained ST-VQRL models weights on 200x12 synthetically distorted LSVQ videos.

Google Drive: [pretrained-stvqrl](https://drive.google.com/file/d/1uE0QgCZAsjXrvRHP_bdC8xVu5xb4eZUa/view?usp=drive_link)
