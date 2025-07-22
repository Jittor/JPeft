# JPeft
## Introduction
JPeft is an parameter-efficient fine-tuning toolkit based on [Jittor](https://github.com/Jittor/jittor) and [JDet](https://github.com/Jittor/JDet). 

<!-- **Features**
- Automatic compilation. Our framwork is based on Jittor, which means we don't need to Manual compilation for these code with CUDA and C++.
-  -->

<!-- Framework details are avaliable in the [framework.md](docs/framework.md) -->
## Install
JDet environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

**Step 1: Install the requirements**
```shell
git clone https://github.com/Jittor/JDet
cd JDet
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JDet**
 
```shell
cd JDet
# suggest this 
python setup.py develop
# or
python setup.py install
```
If you don't have permission for install,please add ```--user```.

Or use ```PYTHONPATH```: 
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path}/JDet/python``` into ```.bashrc```, and run
```shell
source .bashrc
```

## Getting Started

### Datasets
The following datasets are supported in JDet, please check the corresponding document before use. 

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

FAIR Dataset: [fair.md](docs/fair.md)

SSDD/SSDD+: [ssdd.md](docs/ssdd.md)

You can also build your own dataset by convert your datas to DOTA format.
### Config
JDet defines the used model, dataset and training/testing method by `config-file`, please check the [config.md](docs/config.md) to learn how it works.
### Train
```shell
python tools/run_net.py --config-file=configs/peft_resnet/s2anet_r50_fpn_1x_dota_full.py --task=train
```

### Test
If you want to test the downloaded trained models, please set ```resume_path={you_checkpointspath}``` in the last line of the config file.
```shell
python tools/run_net.py --config-file=configs/peft_resnet/s2anet_r50_fpn_1x_dota_full.py --task=test
```
### Test on images / Visualization
You can test and visualize results on your own image sets by:
```shell
python tools/run_net.py --config-file=configs/peft_resnet/s2anet_r50_fpn_1x_dota_full.py --task=vis_test
```
You can choose the visualization style you prefer, for more details about visualization, please refer to [visualization.md](docs/visualization.md).

### Build a New Project
In this section, we will introduce how to build a new project(model) with JDet.
We need to install JDet first, and build a new project by:
```sh
mkdir $PROJECT_PATH$
cd $PROJECT_PATH$
cp $JDet_PATH$/tools/run_net.py ./
mkdir configs
```
Then we can build and edit `configs/base.py` like `$JDet_PATH$/configs/retinanet.py`.
If we need to use a new layer, we can define this layer at `$PROJECT_PATH$/layers.py` and import `layers.py` in `$PROJECT_PATH$/run_net.py`, then we can use this layer in config files.
Then we can train/test this model by:
```sh
python run_net.py --config-file=configs/base.py --task=train
python run_net.py --config-file=configs/base.py --task=test
```

## Methods of PEFT

All experiments are conducted on S2ANet-R50-FPN.

|   Models    | Dataset | Sub_Image_Size/Overlap | Lr schd | mAP  |                                                                                   Paper                                                                                   |                               Config                                |
|:-----------:|:-------:|:----------------------:|:-------:|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|    Full     | DOTA1.0 |        1024/200        |   1x    | 59.1 |                                                                                     -                                                                                     |    [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_full.py)     
|   BitFit    | DOTA1.0 |        1024/200        |   1x    | 59.8 |                                                                [ACL'22](https://arxiv.org/abs/2106.10199)                                                                 |   [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_bitfit.py)    
|    Fixed    | DOTA1.0 |        1024/200        |   1x    | 64.2 |                                                                                     -                                                                                     |    [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_fixed.py)    
| ConvAdapter | DOTA1.0 |        1024/200        |   1x    | 66.2 | [CVPRW'24](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Chen_Conv-Adapter_Exploring_Parameter_Efficient_Transfer_Learning_for_ConvNets_CVPRW_2024_paper.pdf) | [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_convadapter.py) 
| AdaptFormer | DOTA1.0 |        1024/200        |   1x    | 68.9 |                      [NeurIPS'22](https://proceedings.neurips.cc/paper_files/paper/2022/file/69e2f49ab0837b71b0e0cb7c555990f8-Paper-Conference.pdf)                       | [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_adaptformer.py) |
|    LoRA     | DOTA1.0 |        1024/200        |   1x    | 69.7 |                                                            [ICLR'21](https://arxiv.org/pdf/2106.09685v1/1000)                                                             |    [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_lora.py)     |
|  Partial-1  | DOTA1.0 |        1024/200        |   1x    | 70.6 |                            [NeurIPS'14](https://proceedings.neurips.cc/paper_files/paper/2014/file/532a2f85b6977104bc93f8580abbb330-Paper.pdf)                            |  [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_partial1.py)   |
|    Mona     | DOTA1.0 |        1024/200        |   1x    | 70.8 |   [CVPR'25](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_5100_Breaking_Performance_Shackles_of_Full_Fine-Tuning_on_Visual_Recognition_CVPR_2025_paper.pdf)   |    [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_mona.py)     |
|   Adapter   | DOTA1.0 |        1024/200        |   1x    | 73.8 |                                                   [ICML'19](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)                                                   |   [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_adapter.py)   |
|   LoRand    | DOTA1.0 |        1024/200        |   1x    | 60.2 |                                                                                [CVPR'23](https://openaccess.thecvf.com/content/CVPR2023/html/Yin_1_VS_100_Parameter-Efficient_Low_Rank_Adapter_for_Dense_Predictions_CVPR_2023_paper.html) |   [config](configs/peft_resnet/s2anet_r50_fpn_1x_dota_LoRand.py)    |                                  |  |                       



**Notice**:

1. 1x : 12 epochs
2. mAP: mean Average Precision on DOTA1.0 test set



## Contact Us


Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083


<img src="https://cg.cs.tsinghua.edu.cn/jittor/images/news/2020-12-8-21-19-1_2_2/fig4.png" width="200"/>

## The Team


JDet is currently maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in JDet and want to improve it, Please join us!


## Citation


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

## Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [mmrotate](https://github.com/open-mmlab/mmrotate)
3. [Detectron2](https://github.com/facebookresearch/detectron2)
4. [mmdetection](https://github.com/open-mmlab/mmdetection)
5. [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
6. [RotationDetection](https://github.com/yangxue0827/RotationDetection)
7. [s2anet](https://github.com/csuhan/s2anet)
8. [gliding_vertex](https://github.com/MingtaoFu/gliding_vertex)
9. [oriented_rcnn](https://github.com/jbwang1997/OBBDetection/tree/master/configs/obb/oriented_rcnn)
10. [r3det](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection)
11. [AerialDetection](https://github.com/dingjiansw101/AerialDetection)
12. [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
13. [OBBDetection](https://github.com/jbwang1997/OBBDetection)
14. [nk-remote](https://github.com/NK-JittorCV/nk-remote)


