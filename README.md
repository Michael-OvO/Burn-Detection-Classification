

<H1 align="center">
Skin Burn Detection </H1>
<h4 align = "center">
Official implementation of the paper Deep learning based burn detection model</h4>
<p align = "center">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Michael-OvO/Burn-Detection-Classification?label=Please%20Support%20by%20Giving%20a%20Star&logoColor=blue&style=social">
  <img alt="GitHub followers" src="https://img.shields.io/github/followers/Michael-OvO?logoColor=blue&style=social">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Michael-OvO/Burn-Detection-Classification?logoColor=blue&style=social">
  <img alt="GitHub forks" src="https://img.shields.io/github/forks/Michael-OvO/Burn-Detection-Classification?logoColor=blue&style=social">
</p>
<p align = "center"> <b> Author: Michael Wang</b> </p>
<p align = "center" style="bold">Supported Environment:</p> 
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

<p align="center">
    English | [简体中文](.github/README_cn.md) 	
</p>

​																																											


## Project Overview:

The goal of this project is to develop a deep learning-based burn detection model that converts the burn detection problem into an object detection problem. Deep learning algorithms are then used to quickly locate burn locations in images and rank burn depths according to visual features of the images.

The burn detection accuracy has reached 84% and the mAP has reached 70%, which is able to perform excellent burn depth determination and recognition in everyday medical situations. These improvements were made by modifying the network structure of the most recent Yolov7 model and using a variety of widely used target detection frameworks. This is the simplest and least expensive method of diagnosing burns that do not need expert assistance. It will play an important role in disaster relief and remote mountainous counties and towns with insufficient medical resources.

## Documentation:

### Quick Start:

The two easiest ways to get your feet wet is by directly running the notebooks created on Kaggle and Google Colab. The environments are already set up and you can directly train from scratch. (Aprox. 4-5 hours for one complete set of training) You may also find the notebooks directly in the notebooks folder. **However, note that these notebook cannot achieve the accuracy as mentioned in the paper or in this repo, because the dataset was trained on a public dataset rather than my own dataset used for this project. (Unfortunately I am unable to release my own dataset used due to privacy issues, but I will release the trained weights on my own dataset)**

**The Kaggle Notebook:**

<a href="https://www.kaggle.com/code/michaelwovo/skin-burn/notebook?scriptVersionId=103621232" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a><br>

**The Colab Notebook:**

<a href="https://colab.research.google.com/github/Michael-OvO/Burn-Detection-Classification/blob/main/notebooks/colab_skin_burn(demo).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Todos:

- [x] Finish Colab Notebook [2022.8.18]
- [ ] Set up the rest of the code space
- [ ] Add Chinese Markdown
- [ ] Flask Environment for the trained model
- [ ] Finish Paper

## Resources:

- [ ] Pretrained Models and weights
- [ ] Datasets
- [ ] Burn Guidelines

## Latest Updates:

<p align="center">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Michael-OvO/Burn-Detection-Classification?style=for-the-badge">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/Michael-OvO/Burn-Detection-Classification?style=for-the-badge">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/Michael-OvO/Burn-Detection-Classification?color=orange&style=for-the-badge">
</p>

[![Star History Chart](https://api.star-history.com/svg?repos=Michael-OvO/Burn-Detection-Classification&type=Date)]()


## Special Thanks:



## Acknowledgements

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
