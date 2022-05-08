
<div id="top"></div>






<!-- ABOUT THE PROJECT -->

# CARTI_Dataset
CARLA-based 3D object detection and tracking dataset generator using KITTI-format




### Built With

This repo is built based on the following outstanding works, which are greatly appreciated by the authors.

* [CARLA](https://github.com/carla-simulator/carla)
* [MMDetection3D](https://github.com/open-mmlab/mmdetection)
* [KITTI_vis_kit](https://github.com/zwbai/kitti_object_vis)


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Please install CARLA==0.9.13 and MMDetection3D==v1.0.0rc2

## Prerequisites

- Linux (Ubuntu 18.04)
- Python 3.7
- PyTorch 1.3+
- CUDA 11.1
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation for CARLA
Download the CARLA 0.9.10 at [Here](https://github.com/carla-simulator/carla/releases/tag/0.9.13)
```shell
git clone https://github.com/carla-simulator/carla.git
git checkout tags/0.9.13
```


## Installation for MMdetection3D

The required versions of MMCV, MMDetection and MMSegmentation for different versions of MMDetection3D are as below. Please install the correct version of MMCV, MMDetection and MMSegmentation to avoid installation issues.

| MMDetection3D version |   MMDetection version   | MMSegmentation version |        MMCV version        |
| :-------------------: | :---------------------: | :--------------------: | :------------------------: |
|       v1.0.0rc2       | mmdet>=2.19.0, <=3.0.0  | mmseg>=0.20.0, <=1.0.0 | mmcv-full>=1.4.8, <=1.7.0  |


### Quick installation instructions script

Assuming that you already have CUDA 11.0 installed, here is a full script for quick installation of MMDetection3D with conda.
Otherwise, you should refer to the step-by-step installation instructions in the next section.

```shell
conda create -n open-mmlab-rc2 python=3.7 -y
conda activate open-mmlab-rc2
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
git checkout tags/v1.0.0rc2
cd mmdetection3d
pip install -e .
pip install open3d
```

