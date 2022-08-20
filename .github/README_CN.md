

<H1 align="center">
皮肤烧伤检测 </H1>
<h4 align = "center">
Official implementation of the paper Deep learning based burn detection model</h4>
<p align = "center">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Michael-OvO/Burn-Detection-Classification?label=Please%20Support%20by%20Giving%20a%20Star&logoColor=blue&style=social">
  <img alt="GitHub followers" src="https://img.shields.io/github/followers/Michael-OvO?logoColor=blue&style=social">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Michael-OvO/Burn-Detection-Classification?logoColor=blue&style=social">
  <img alt="GitHub forks" src="https://img.shields.io/github/forks/Michael-OvO/Burn-Detection-Classification?logoColor=blue&style=social">
</p>
<p align = "center"> <b> 作者: Michael Wang</b> </p>
<p align = "center" style="bold">支持环境:</p> 
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
    <a href=../README.md>English</a> | 简体中文
</div>

​																																											


## 项目概览:

这个项目的目标是开发一个基于深度学习的烧伤检测模型，将烧伤检测问题转换为物体检测问题。然后利用深度学习算法快速定位图像中的烧伤位置，并根据图像的视觉特征对烧伤深度进行排序。

烧伤检测准确率达到84%，mAP达到70%，能够在日常医疗情况下进行出色的烧伤深度判断和识别。这些改进是通过修改最新的Yolov7模型的网络结构和使用各种广泛使用的目标检测框架而实现的。这是诊断不需要专家协助的烧伤的最简单和最昂贵的方法。它将在救灾和医疗资源不足的偏远山区县镇发挥重要作用。

## 使用说明:

### 快速开始:

两个最简单的方法是直接运行Kaggle和Google Colab上创建的笔记本。这些环境已经设置好了，你可以直接从头开始训练。(一套完整的训练大概需要4-5个小时）你也可以直接在笔记本文件夹中找到这些笔记本。**但是，请注意，这些笔记本不能达到论文中或这个repo中提到的准确度，因为数据集是在一个公共数据集上训练的，而不是我自己用于这个项目的数据集。(不幸的是，由于隐私问题，我无法公布我自己的数据集，但我将公布我自己数据集上的训练权重）**。

**The Kaggle Notebook:**

<a href="https://www.kaggle.com/code/michaelwovo/skin-burn/notebook?scriptVersionId=103621232" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a><br>

**The Colab Notebook:**

<a href="https://colab.research.google.com/github/Michael-OvO/Burn-Detection-Classification/blob/main/notebooks/colab_skin_burn(demo).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 代办:

- [x] 完成Colab Notebook [2022.8.18] 
- [ ] 上传其余训练代码
- [ ] 项目汉化
- [ ] 为项目配好Flask环境
- [ ] 完成论文

## Resources:

- [ ] Pretrained Models and weights
- [ ] Datasets
- [ ] Burn Guidelines

## 最新更新:

<p align="center">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Michael-OvO/Burn-Detection-Classification?style=for-the-badge">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/Michael-OvO/Burn-Detection-Classification?style=for-the-badge">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/Michael-OvO/Burn-Detection-Classification?color=orange&style=for-the-badge">
</p>

[![Star History Chart](https://api.star-history.com/svg?repos=Michael-OvO/Burn-Detection-Classification&type=Date)]()


## 鸣谢:



## 项目参考

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