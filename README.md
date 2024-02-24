# SSL-VQA
This is the official repository for the paper titled " Knowledge Guided Semi-Supervised Learning for Quality Assessment of User Generated Videos" accepted in **Proceedings of AAAI conf. on AI (AAAI 24)** by Shankhanil Mitra and Rajiv Soundararajan. 

In this work, we address the challenge of requirement of large scale human annotated videos for training by first designing a self- supervised Spatio-Temporal Visual Quality Representation Learning (ST-VQRL) framework to generate robust quality aware features for videos. Then, we propose a dual-model based Semi Supervised Learning (SSL) method specifically designed for the Video Quality Assessment (SSL-VQA) task, through a novel knowledge transfer of quality predictions between the two models.

![SSL-VQA](https://github.com/Shankhanil006/SSL-VQA/blob/main/sslvqa.png?raw=true)

## Installation 
>conda env create -f environment.yml

## Generating Syntheitc distortion for UGC Videos
Download LSVQ database from [LSVQ](https://github.com/baidut/PatchVQ?raw=true) . Randomly choose 200 or more videos and generate 12 distorted version of each scene by running the file LSVQ Synthetic/script.py. On the other hand user can also download synthetically distorted databases like LIVE-VQA, EPFL-PoLiMI, LIVE Mobile, CSIQ VQD, ECVQ and EVVQ for training ST_VQRL.

## Pristine Clip generation
Download pristine fragmented video clip from:
[pristine_clips](https://drive.google.com/file/d/1pCABOnY2K5STtGW3XXQsFEmJIYZElWtT/view?usp=drive_link)
Alternatively, you can run pristine_clip_generator.py on any pristine videos to generate pristine fragmented video clips.
## Training Self-supervised Video Quality Representation Learning (ST-VQRL) Model
To train self-supervised video feature model (ST-VQRL) using LSVQ synthetically generated videos run following:
>python3 STVQRL/self_supervised_train.py

We provide Pre-trained ST-VQRL models weights on 200x12 synthetically distorted LSVQ videos.

Google Drive: [pretrained-stvqrl](https://drive.google.com/file/d/1uE0QgCZAsjXrvRHP_bdC8xVu5xb4eZUa/view?usp=drive_link)

## Training Semi-supervised Video Quality Assessment (SSL-VQA) Model

To train SSL-VQA with 500 labelled and 1500 unlabelled samples from LSVQ official train set run the following script:

> python3 SemiSupervised Learning Module/semisupervised_training.py

We have provided 2000 video names in semisupervised.json files and randomly chosen 500 labelled samples from this 2000. User can choose any other set of labelled and unlabelled videos from entire LSVQ train set of 28053 in LSVQ_train.json file. 

Pre-trained weights of SSL-VQA trained on 1 random split of 500 labelled and 1500 unlabelled video in semisupervised.json:

[pretrained SSL-VQA](https://drive.google.com/file/d/1EHtMEXPpQZAu2GRxG8jgVKxeGrv9JrII/view?usp=drive_link)

## Acknowledgement 
Video fragment generation code is taken from FAST-VQA [link](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/tree/dev?tab=readme-ov-file)

## Citation
If you find this work useful for your research, please cite our paper:
>@misc{mitra2023knowledge,

      title={Knowledge Guided Semi-Supervised Learning for Quality Assessment of User Generated Videos}, 
      
      author={Shankhanil Mitra and Rajiv Soundararajan},
      
      year={2023},
      
      eprint={2312.15425},
      
      archivePrefix={arXiv},
      
      primaryClass={cs.CV}
      
}
