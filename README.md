
<div id="top"></div>






<!-- ABOUT THE PROJECT -->

# CARTI_Dataset
CARLA-based 3D object detection and tracking dataset generator using KITTI-format


<!-- GETTING STARTED -->
## Getting Started

Please install `CARLA==0.9.13` and `MMDetection3D==0.18.0`

## Prerequisites

- Linux (Ubuntu 18.04)
- Python 3.7
- PyTorch 1.3+
- CUDA 11.1
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation for CARLA
Download the CARLA 0.9.13 at [Here](https://github.com/carla-simulator/carla/releases/tag/0.9.13)
```shell
git clone https://github.com/zwbai/CARTI_Dataset.git
```
Run CARLA server
```shell
python CARTI_Dataset_V1.0.py
```


## Installation for MMdetection3D

The required versions of MMCV, MMDetection and MMSegmentation for different versions of MMDetection3D are as below. Please install the correct version of MMCV, MMDetection and MMSegmentation to avoid installation issues.

| MMDetection3D version |   MMDetection version   | MMSegmentation version |        MMCV version        |
| :-------------------: | :---------------------: | :--------------------: | :------------------------: |
|       v1.0.0rc2       | mmdet>=2.19.0, <=3.0.0  | mmseg>=0.20.0, <=1.0.0 | mmcv-full>=1.4.8, <=1.7.0  |
|        0.18.0         | mmdet>=2.19.0, <=3.0.0  | mmseg>=0.20.0, <=1.0.0 | mmcv-full>=1.3.17, <=1.5.0 |

### Quick installation instructions script

Assuming that you already have CUDA 11.0 installed, here is a full script for quick installation of MMDetection3D with conda.
Otherwise, you should refer to the step-by-step installation instructions in the next section.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.19.0
pip install mmsegmentation==0.20.0

git clone https://github.com/open-mmlab/mmdetection3d.git
git checkout tags/0.18.0
cd mmdetection3d
pip install -v -e .
pip install open3d
```
### Validation
Do not forget to add `./checkpoints/{}.pth` and `./demo/data/kitti/{}.bin`
```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --show
```
## Acknowledgement

This repo is built based on the following outstanding works, which are greatly appreciated by the authors.

* [CARLA](https://github.com/carla-simulator/carla)
* [MMDetection3D](https://github.com/open-mmlab/mmdetection)
* [KITTI_vis_kit](https://github.com/zwbai/kitti_object_vis)


<p align="right">(<a href="#top">back to top</a>)</p>
