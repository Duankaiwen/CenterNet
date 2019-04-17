# CenterNet: Keypoint Triplets for Object Detection
by Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang and Qi Tian

**The code to train and eval our CenterNet is available here. Thanks [Princeton Vision & Learning Lab](https://github.com/princeton-vl)!**

**Our method is an one-stage detector and learns from scratch. On the MS-COCO dataset, CenterNet achieves an AP of 47.0%！**

## Introduction

CenterNet is a framework for object detection with deep ConvNets. You can use the code to train/evaluate a network for object detection task.

* It achieves state-of-the-art performance on one of the most challenging dataset: MS-COCO.
* Our code is written by Python, based on [CornerNet](https://github.com/princeton-vl/CornerNet).

## Architecture

![Network_Structure](https://github.com/Duankaiwen/CenterNet/blob/master/Network_Structure.jpg)

## Comparison with other methods

![Tabl](https://github.com/Duankaiwen/CenterNet/blob/master/Table1.png)

![Tabl](https://github.com/Duankaiwen/CenterNet/blob/master/Table2.png)

![Tabl](https://github.com/Duankaiwen/CenterNet/blob/master/Table3.png)

In terms of speed, we test the inference speed of both CornerNet and CenterNet on a NVIDIA Tesla P100 GPU. We obtain that the average inference time of CornerNet511-104 (means that the resolution of input images is 511X511 and the backbone is Hourglass-104) is 300ms per image and that of CenterNet511-104 is 340ms. Meanwhile, using the Hourglass-52 backbone can speed up the inference speed. Our CenterNet511-52 takes an average of 270ms to process per image, which is faster and more accurate than CornerNet511-104.

## Preparetion
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CenterNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CenterNet
```

## Compiling Corner Pooling Layers
```
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS
```
cd <CenterNet dir>/external
make
```

## Installing MS COCO APIs
```
cd <CenterNet dir>/data/coco/PythonAPI
make
```

## Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CenterNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CenterNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation

To train CenterNet-104:
```
python train.py CenterNet-104
```
We provide the configuration file (`CenterNet-104.json`) and the model file (`CenterNet-104.py`) for CenterNet in this repo. 

We also provide a trained model for `CenterNet-104`, which is trained for 480k iterations using 8 Tesla V100 (32GB) GPUs. You can download it from [here](https://pan.baidu.com/s/17RvbWaxrvW1kXRuk7XmIfw  code:2clj) and put it under `<CenterNet dir>/cache/nnet` (You may need to create this directory by yourself if it does not exist). If you want to train you own CenterNet, please adjust the batch size in `CenterNet-104.json` to accommodate the number of GPUs that are available to you.

To use the trained model:
```
python test.py CenterNet-104 --testiter 480000 --split <split>
```

To train CenterNet-52:
```
python train.py CenterNet-52
```
We provide the configuration file (`CenterNet-52.json`) and the model file (`CenterNet-52.py`) for CenterNet in this repo. 

We also provide a trained model for `CenterNet-52`, which is trained for 480k iterations using 8 Tesla V100 (32GB) GPUs. You can download it from [here](https://pan.baidu.com/s/1Ltig0csUPp4T5HA4BjHikA  code:ed0y) and put it under `<CenterNet dir>/cache/nnet` (You may need to create this directory by yourself if it does not exist). If you want to train you own CenterNet, please adjust the batch size in `CenterNet-52.json` to accommodate the number of GPUs that are available to you.

To use the trained model:
```
python test.py CenterNet-52 --testiter 480000 --split <split>
```

We also include a configuration file for multi-scale evaluation, which is `CenterNet-104-multi_scale.json` and `CenterNet-52-multi_scale.json` in this repo, respectively. 

To use the multi-scale configuration file:
```
python test.py CenterNet-52 --testiter <iter> --split <split> --suffix multi_scale
```
or
```
python test.py CenterNet-104 --testiter <iter> --split <split> --suffix multi_scale
```
