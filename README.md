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


## 🏃 Training and Evaluation

The configuration used for both evaluation and training can be found at [`./configs/amsynthdrive.py`](configs/amsynthdrive.py).
Training output will be saved in a subdirectory of the `logs` directory. 

### Training

Our model can be trained by running
```shell
python ./train_FlowFormer.py --name amsynthdrive --stage amsynthdrive --validation amsynthdrive
```
We train our model using 6 GPUs with 48 GB VRAM each.

### Evaluation

For a given checkpoint `logs/some_run/model.pth`, the model can be evaluated using
```shell
python ./evaluate_FlowFormer_tile.py --eval amsynthdrive_validation --model logs/some_run/model.pth
```
For evaluation, a single 12GB GPU is enough.


## 🤖 Models

We will provide a pre-trained checkpoint soon.


## 📒 Notes

- The initial training, resuming from a pre-trained FlowFormer++ checkpoint, will generate a warning that the base checkpoint could not be loaded in strict mode. This is expected. The missing keys are the additional amodal decoder strands added on top of FlowFormer++.

- While AmodalSynthDrive uses `.png` files for storing optical flow similar to KITTI, the scaling differs. Please refer to `readFlowKITTI()` in [`core/utils/frame_utils.py`](core/utils/frame_utils.py#L106) on how to load AmodalSynthDrive flow files.


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
