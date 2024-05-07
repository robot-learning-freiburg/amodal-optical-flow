# Amodal Optical Flow

[**arXiv**](https://arxiv.org/abs/2311.07761) |
[**Website**](http://amodal-flow.cs.uni-freiburg.de/) |
[**Video**](https://www.youtube.com/watch?v=tzeQ0h9ttYM)

This repository contains the official implementation of the paper:

> **[Amodal Optical Flow](https://arxiv.org/abs/2311.07761)**
>
> [Maximilian Luz](https://mxnluz.io/)<sup>\*</sup>,
> [Rohit Mohan](https://rl.uni-freiburg.de/people/mohan)<sup>\*</sup>,
> Ahmed Rida Sekkat,
> Oliver Sawade,
> Elmar Matthes,
> [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox/), and
> [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
> <br>
> *Equal contribution.
>
> _IEEE International Conference on Robotics and Automation (ICRA) 2024_ <br/>

If you find our work useful, please consider citing our paper [via BibTeX](CITATIONS.bib).


## 📔 Abstract

Optical flow estimation is very challenging in situations with transparent or occluded objects. In this work, we address these challenges at the task level by introducing Amodal Optical Flow, which integrates optical flow with amodal perception. Instead of only representing the visible regions, we define amodal optical flow as a multi-layered pixel-level motion field that encompasses both visible and occluded regions of the scene. To facilitate research on this new task, we extend the AmodalSynthDrive dataset to include pixel-level labels for amodal optical flow estimation. We present several strong baselines, along with the Amodal Flow Quality metric to quantify the performance in an interpretable manner. Furthermore, we propose the novel AmodalFlowNet as an initial step toward addressing this task. AmodalFlowNet consists of a transformer-based cost-volume encoder paired with a recurrent transformer decoder which facilitates recurrent hierarchical feature propagation and amodal semantic grounding. We demonstrate the tractability of amodal optical flow in extensive experiments and show its utility for downstream tasks such as panoptic tracking.


## ⚙️ Installation and Requirements

- Create conda environment: `conda create --name amodal-flow python=3.11`
- Activate conda environment: `conda activate amodal-flow`
- Install PyTorch: `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia`
- Install OpenCV: `conda install opencv`
- Install remaining dependencies: `pip install -r requirements.txt`

This code has been developed and tested with PyTorch version 2.0.1 and CUDA version 11.7.


## 💾 Data Preparation

We use pre-trained [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) weights to initialize the modal base network and use the [AmodalSynthDrive dataset](http://amodalsynthdrive.cs.uni-freiburg.de/) for amodal training.
Therefore
- Download the FlowFormer++ [`sintel.pth` checkpoint](https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI) and place it at `./checkpoints/sintel.pth`.
- Download the [AmodalSynthDrive dataset](http://amodalsynthdrive.cs.uni-freiburg.de/) and place it at `./datasets/AmSynthDrive`.

The final folder structure should look like this:
```
.
├── checkpoints
│  └── sintel.pth
├── datasets
│  ├── AmSynthDrive
│  │  ├── empty
│  │  ├── full
│  │  └...
│  └── amsynthdrive.json
└...
```

## Training
The script will load the config according to the training stage. The trained model will be saved in a directory in `logs`. For example, the following script will load the config `configs/pretrain_config.py`. The trained model will be saved as `logs/xxxx/final`.
```shell
python -u pretrain_FlowFormer_maemask.py --stage youtube
```
To finish the entire finetuning schedule, you can run:
```shell
./run_train.sh
```

## Models
We provide [models](https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI?usp=sharing) trained in the four stages. The default path of the models for evaluation is:
```Shell
├── checkpoints
    ├── chairs.pth
    ├── things.pth
    ├── sintel.pth
    ├── kitti.pth
    ├── things_288960.pth
```

## Evaluation
The model to be evaluated is assigned by the `_CN.model` in the config file.

Evaluating the model on the Sintel training set and the KITTI training set. The corresponding config file is `configs/submissions.py`.
```Shell
# with tiling technique
python evaluate_FlowFormer_tile.py --eval sintel_validation
python evaluate_FlowFormer_tile.py --eval kitti_validation --model checkpoints/things_kitti.pth
```

Generating the submission for the Sintel and KITTI benchmarks. The corresponding config file is `configs/submissions.py`.
```Shell
python evaluate_FlowFormer_tile.py --eval sintel_submission
python evaluate_FlowFormer_tile.py --eval kitti_submission
```
Visualizing the sintel dataset:
```Shell
python visualize_flow.py --eval_type sintel --keep_size
```
Visualizing an image sequence extracted from a video:
```Shell
python visualize_flow.py --eval_type seq
```
The default image sequence format is:
```Shell
├── demo_data
    ├── mihoyo
        ├── 000001.png
        ├── 000002.png
        ├── 000003.png
            .
            .
            .
        ├── 001000.png
```


## 👩‍⚖️ License

For academic use, code for AmodalFlowNet is released under the [Apache License](LICENSE), following FlowFormer++.
For any commercial usage, please contact the authors.


## 🙏 Acknowledgment

The code of this project is based on [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus).
Subsequently, we use parts of:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)

In addition, this work was funded by the German Research Foundation (DFG) Emmy Noether Program grant No 468878300 and an academic grant from NVIDIA.
