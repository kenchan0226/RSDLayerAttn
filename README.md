# Grounding Commands for Autonomous Vehicles via Region-specific Dynamic Layer Attention

This repository contains the source code for our IROS 2022 paper "[Grounding Commands for Autonomous Vehicles via Region-specific Dynamic Layer Attention](https://arxiv.org/abs/2203.06822)". 

Our code is built on the excellent repository of [VOLTA](https://github.com/e-bug/volta).

We also upload a demo video of our model in [here](article_and_demo_video/demo_video.mp4).

If you use this code, please cite our paper:
```
@inproceedings{conf/acl/chan19keyphraseRL,
  title={Grounding Commands for Autonomous Vehicles via Layer Fusion with Region-specific Dynamic Layer Attention},
  author={Hou Pong Chan and Mingxi Guo and Cheng-Zhong Xu},
  booktitle={Proceedings of IROS},
  year={2022}
}
```

## Repository Setup

1\. Create a fresh conda environment, and install all dependencies.
```text
conda create -n volta python=3.6
conda activate volta
pip install -r requirements.txt
```

2\. Install PyTorch
```text
conda install pytorch=1.4.0 torchvision=0.5 cudatoolkit=10.1 -c pytorch
```

3\. Install [apex](https://github.com/NVIDIA/apex).
If you use a cluster, you may want to first run commands like the following:
```text
module load cuda/10.1.105
module load gcc/8.3.0-cuda
```

4\. Setup the `refer` submodule for Referring Expression Comprehension:
```
cd tools/refer; make
```

5\. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Data
We conduct experiments on the [Talk2Car](https://github.com/talk2car/Talk2Car) dataset. If you use this dataset, please cite their paper. 
```
Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, Marie-Francine Moens:
Talk2Car: Taking Control of Your Self-Driving Car. EMNLP 2019
```

The followings are our preprocessed data.

Download the regions extracted by centernet (we only keep the top 36 regions) from [here](https://www.dropbox.com/s/zlc8cv53lzqwjc0/talk2car_centernet_dets_36.json?dl=0) and move this file to `./data/talk2car/talk2car_centernet_dets_36.json`

Download the `instances.json` from [here](https://www.dropbox.com/s/neb3fhggqujs8rp/instances.json?dl=0) and move it to `./data/talk2car/annotations/talk2car/instances.json`

Download the `refs_spacy.json` from [here](https://www.dropbox.com/s/ad6okorhusz4t9c/refs_spacy.p?dl=0) and move it to `./data/talk2car/annotations/talk2car/refs_spacy.json`

Download the region features extracted by Faster R-CNN from [here](https://pan.baidu.com/s/10Jo2KQlUAV1MKRA_a22_dg), the password is `RSDL`, unzip data.zip.001 and data.zip.002, then move the files to `./data/talk2car/resnet101_faster_rcnn_genome_imgfeats_centernet/volta/refcoco+_unc_dets36_feat.lmdb/lock.mdb` and `./data/talk2car/resnet101_faster_rcnn_genome_imgfeats_centernet/volta/refcoco+_unc_dets36_feat.lmdb/data.mdb`. 

## Pre-trained Models

Download the pre-trained UNITER and LXMERT checkpoints provided by VOLTA
```
wget https://sid.erda.dk/share_redirect/FeYIWpMSFg
mv FeYIWpMSFg checkpoints/conceptual_captions/ctrl_uniter/ctrl_uniter_base/pytorch_model_9.bin
wget https://sid.erda.dk/share_redirect/Dp1g16DIA5
mv Dp1g16DIA5 checkpoints/conceptual_captions/ctrl_lxmert/ctrl_lxmert/pytorch_model_9.bin
```

## Training

We provide sample scripts to train our RSD-UNITER and RSD-LXMERT models:
[examples/ctrl_uniter/talk2car/train_RSD_uniter.sh](examples/ctrl_uniter/talk2car/train_RSD_uniter.sh) and [examples/ctrl_lxmert/talk2car/train_RSD_lxmert.sh](examples/ctrl_uniter/talk2car/train_RSD_uniter.sh)

## Evaluate
Run the following script to construct a mapping between the id of the sample and corresponding token in the leaderboard of talk2car
```
python3 generate_token.py
```
Run inference on the validation and test sets (the computed AP50 score on the test set is always 0 since we do not have the ground-truth):
[examples/ctrl_lxmert/talk2car/val_RSD_uniter.sh](examples/ctrl_uniter/talk2car/val_RSD_uniter.sh) and [examples/ctrl_lxmert/talk2car/test_RSD_uniter.sh](examples/ctrl_uniter/talk2car/test_RSD_uniter.sh)

Export the predictions to a json file
```
python generate_prediction.py --result_path ./results/talk2car/ctrl_uniter/pytorch_model_best.bin-
```

Submit the json file `./results/talk2car/ctrl_uniter/pytorch_model_best.bin/predictions_for_leaderboard.json` to the leaderboard of Talk2Car [here](https://www.aicrowd.com/challenges/eccv-2020-commands-4-autonomous-vehicles) (create submission button). 
