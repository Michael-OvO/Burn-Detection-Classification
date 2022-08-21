<H1 align="center">
皮肤烧伤检测 </H1>
<h4 align = "center">
基于Yolov7的皮肤烧伤检测器</h4>
<p align = "center">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Michael-OvO/Burn-Detection-Classification?label=Please%20Support%20by%20Giving%20a%20Star&logoColor=blue&style=social">
  <img alt="GitHub followers" src="https://img.shields.io/github/followers/Michael-OvO?logoColor=blue&style=social">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Michael-OvO/Burn-Detection-Classification?logoColor=blue&style=social">
  <img alt="GitHub forks" src="https://img.shields.io/github/forks/Michael-OvO/Burn-Detection-Classification?logoColor=blue&style=social">
</p>
<p align = "center"> <b> 作者: Micahel.W </b> </p>
<p align = "center" style="bold">适配环境:</p> 
<div align = "center">
<a href="https://colab.research.google.com/">
  <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252">
  </a>
  <a href="https://github.com/">
  <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href = "https://www.kaggle.com/"> 
<img src = "https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white">
    </a>
      <br>
</div>

<div align="center">
    <a href="../README.md">English</a> | 简体中文
</div>


​																																											


## Project Overview:

这个项目的目标是开发一个基于深度学习的烧伤检测模型，将烧伤检测问题转换为物体检测问题。然后利用深度学习算法快速定位图像中的烧伤位置，并根据图像的视觉特征对烧伤深度进行判断。

![](../figures/final_precision.png)

烧伤检测精度达到88%，mAP_0.5达到72%，能够在日常医疗情况下进行出色的烧伤深度判断和识别。这些改进是通过修改最新的Yolov7模型的网络结构和使用各种广泛使用的目标检测框架而实现的。这是诊断不需要专家协助的烧伤的最简单和最昂贵的方法。它将在救灾和医疗资源不足的偏远山区县镇发挥重要作用。

## 使用说明:

### 快速开始:

两个最简单的方法是直接运行Kaggle和Google Colab上创建的笔记本。这些环境已经设置好了，你可以直接从头开始训练。(一套完整的训练大概需要4-5个小时）你也可以直接在笔记本文件夹中找到这些笔记本。**但是，请注意，这些笔记本不能达到论文中或这个repo中提到的准确度，因为数据集是在一个公共数据集上训练的，而不是我自己用于这个项目的数据集。(由于患者隐私问题，我无法公布我自己的数据集，但我将公布我自己数据集上的训练权重）**。

**kaggle笔记本:**

<a href="https://www.kaggle.com/code/michaelwovo/skin-burn/notebook?scriptVersionId=103621232" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a><br>

**Colab笔记本:**

<a href="https://colab.research.google.com/github/Michael-OvO/Burn-Detection-Classification/blob/main/notebooks/colab_skin_burn(demo).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



### 在本地运行(使用最新的模型):

####  安装:

``` shell
# or just download this entire repo
git clone https://github.com/Michael-OvO/Burn-Detection-Classification.git
```

#### 安装依赖包(推荐使用虚拟环境):

``` shell
cd Burn-Detection-Classification/
pip install -r requirements.txt
```

#### 开始检测:

下载最新的预训练权重文件并放与本项目放在同一个文件夹下: [Skin_burn_2022_8_21.pt](https://github.com/Michael-OvO/Burn-Detection-Classification/releases/download/v1.0.0/skin_burn_2022_8_21.pt)

样本图像可以在inference文件夹中找到，每张图像的名称对应于每张图像的真实值（模型在每次运行后应预测这些值）。

下面是文件`1st_degree_2.jpg`（这是晒伤，所以模型应该输出一度烧伤）。

<div align="center">
    <a href="../">
        <img src="../inference/images/1st_degree_2.jpg" width="59%"/>
    </a>
</div>



视频检测:

``` shell
python detect.py --weights Skin_burn_2022_8_21.pt  --source yourvideo.mp4
```

图像检测:

``` shell
python detect.py --weights Skin_burn_2022_8_21.pt --source inference/images/first_degree_2.jpg
```

<div align="center">
    <a href="../">
        <img src="../inference/results/1st_degree_2.jpg" width="59%"/>
    </a>
</div>



## Export (Same  as  Yolov7: )

**Pytorch to CoreML (and inference on MacOS/iOS)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7CoreML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

</details>

Tested with: Python 3.7.13, Pytorch 1.12.0+cu113

## Todos:

- [x] Finish Colab Notebook [2022.8.18]
- [x] Set up the rest of the code space
- [x] Add Chinese Markdown
- [ ] Flask Environment for the trained model (or other kinds of web demo using the finally trained classifier)
- [ ] Finish Paper

## Resources:

Please refer to the resources folder

- [ ] Pretrained Models and weights
- [ ] Datasets
- [ ] Burn Guidelines
- [x] Appendix 1
- [ ] Appendix 2
- [ ] Appendix 3
- [ ] Appendix 4
- [ ] Appendix 5
- [ ] Appendix 6

## 最新更新:

<p align="center">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Michael-OvO/Burn-Detection-Classification?style=for-the-badge">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/Michael-OvO/Burn-Detection-Classification?style=for-the-badge">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/Michael-OvO/Burn-Detection-Classification?color=orange&style=for-the-badge">
</p>


[![Star History Chart](https://api.star-history.com/svg?repos=Michael-OvO/Burn-Detection-Classification&type=Date)]()


## 鸣谢:



## Acknowledgments: 

<details><summary> <b>Expand</b> </summary>


* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>