# 🥈 NTIRE 2026 CD-FSOD Challenge @ CVPR Workshop

We are the **CDiscover team** of the **NTIRE 2026 Cross-Domain Few-Shot Object Detection (CD-FSOD) Challenge** at the **CVPR Workshop**.

- 🏆 **Track**: `open-source track`
- 🎖️ **Award**: **2nd Place**
- 🧰 **Method**: *GiPL-Grounding: Generative augmented iterative Pseudo-Labeling for Grounding*

🔗 [NTIRE 2026 Official Website](https://cvlai.net/ntire/2026/)  
🔗 [NTIRE 2026 Challenge Website](https://codabench.org/competitions/12873/)  
🔗 [CD-FSOD Challenge Repository](https://github.com/ohMargin/NTIRE2026_CDFSOD)

<p align="center">
    <img src="https://upload-images.jianshu.io/upload_images/9933353-3d7be0d924bd4270.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" alt="Image" width="500">
</p>
<!-- 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhance-then-search-an-augmentation-search/cross-domain-few-shot-object-detection-on)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on?p=enhance-then-search-an-augmentation-search)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhance-then-search-an-augmentation-search/cross-domain-few-shot-object-detection-on-1)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-1?p=enhance-then-search-an-augmentation-search)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhance-then-search-an-augmentation-search/cross-domain-few-shot-object-detection-on-3)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-3?p=enhance-then-search-an-augmentation-search)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhance-then-search-an-augmentation-search/cross-domain-few-shot-object-detection-on-2)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-2?p=enhance-then-search-an-augmentation-search)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhance-then-search-an-augmentation-search/cross-domain-few-shot-object-detection-on-neu)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-neu?p=enhance-then-search-an-augmentation-search)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhance-then-search-an-augmentation-search/cross-domain-few-shot-object-detection-on-4)](https://paperswithcode.com/sota/cross-domain-few-shot-object-detection-on-4?p=enhance-then-search-an-augmentation-search) -->

---

## 📰 News
<!-- - [2025.4] 🎉 Update the leaderboards on Paper With Code: [Cross-Domain Few-Shot Object Detection](https://paperswithcode.com/task/cross-domain-few-shot-object-detection/latest) based on open-source settings.
- [2025.4] 🎉 Release the paper "Enhance Then Search: An Augmentation-Search Strategy with Foundation Models for Cross-Domain Few-Shot Object Detection" in [arXiv](https://arxiv.org/abs/2504.04517).
- [2025.4] 🎉 Release the **ETS** code based on GroundingDINO Swin-B. -->
- [2026.3] 🎉 Win the **2nd Place** in the NTIRE 2026 CD-FSOD Challenge, CVPR2026.

## 🧠 Overview

This repository contains our solution for the `open-source track` of the NTIRE 2026 CD-FSOD Challenge.  
We propose a method that integrates **Qwen-based generative augmentation for GroundingDINO and GLIP-driven iterative pseudo-labeling**, which achieves strong performance on the challenge. 

<!-- <p align="center">
    <img src="assets/ets.png" alt="Image" width="300">
</p> -->


<p align="center">
    <img src="assets/framework.png" alt="Image" width="500">
</p>

<!-- <p align="center">
    <img src="assets/ets-pipeline.png" alt="Image" width="500">
</p> -->
---



Our method addresses the CDFSOD challenge through a domain-adaptive hybrid strategy. 

For Dataset 2 (a multi-object natural domain with vehicles), we identify that only a single instance is labeled per image despite the presence of multiple targets. To prevent the model from treating unlabeled vehicles as background, we employ a GLIP-based pseudo-labeling pipeline to recover missing annotations, followed by iterative self-training to achieve dense and robust object localization. Refer to [GLIP-Repository](https://github.com/Jywsuperman/NTIRE_2026_CDFSOD).


For Dataset 1\&3, we leverage the Qwen to synthesize generative data, enriching the feature space for GroundingDINO.  Implementation details are given as follows.







## 🛠️ Environment Setup

The experimental environment is based on [mmdetection](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md), the installation environment reference mmdetection's [installation guide](https://github.com/open-mmlab/mmdetection/blob/main/docs/zh_cn/get_started.md).

```bash
conda create --name ets python=3.8 -y
conda activate ets
cd ./mmdetection
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Develop and run directly mmdet
pip install -v -e .
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
Then download the BERT weights `bert-base-uncased` into the weights directory,
```bash
cd CDiscover/
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir weights/bert-base-uncased
```


## 📂 Dataset Preparation
Please follow the instructions in the [official CD-FSOD repo](https://github.com/ohMargin/NTIRE2026_CDFSOD) to download and prepare the dataset.

```bash
.
├── configs
├── data
├── LICENSE
├── mmdetection
├── pkl2coco.py
├── pkls
├── README.md
├── submit
├── submit_codalab
└── weights
```

## 🏋️ Training

Mix Image Augmentation Config

```python
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0, prob=0.6),
    # dict(type='CopyPaste', max_num_pasted=5, paste_by_box=True),  # 添加 CopyPaste 数据增强
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114),
        prob = 0.3),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    # dict(type='RandomErasing', n_patches=(0,2), ratio=0.3, img_border_value=128, bbox_erased_thr=0.9),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

```


To train the model: 

Experiments were carried out on on a single NVIDIA A6000 GPU for Datasets 1 and 3. 


<!-- , a total of 50 x 8 groups of experiments.  -->

<!-- 50 groups of experiments were carried out on the 8 x A100, a total of 50 x 8 groups of experiments.  -->

```bash
cd ./mmdetection

./tools/dist_train_muti.sh configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py "0" 1
```
use `sampling4val.py` for sampling test set for validation set.

use `sata_logs` for search to get best model parameter from train logs.

pretrained model: 

Download the checkpoint files to dir `./weights`.
> Baidu Disk: [[link]](https://pan.baidu.com/s/1r5H8TsV1qgwuWPuvc1EJZQ?pwd=hg8e)
or
> 通过网盘分享的文件：weights
链接: https://pan.baidu.com/s/1r5H8TsV1qgwuWPuvc1EJZQ?pwd=hg8e 提取码: hg8e 
--来自百度网盘超级会员v5的分享


## 🔍 Inference & Evaluation

Run evaluation:

```bash
cd ./mmdetection

bash tools/dist_test.sh configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py /path/to/model/ 4
```

Run inference:

Save to `*.pkl` file and convert to submit `.json` format.
```bash
cd ./mmdetection

## 1-shot-dataset1
bash tools/dist_test_out.sh ../configs/1-shot-dataset1.py ../weights/1-shot-dataset1-db4c5ebf.pth 1 ../pkls/dataset1_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ../pkls/dataset1_1shot.pkl --output_json ../pkls/dataset1_1shot_coco.json --annotations_json ../submit/dataset1_1shot.json

## 1-shot-dataset2
bash tools/dist_test_out.sh ../configs/1-shot-dataset2.py ../weights/1-shot-dataset2-0bd5d280.pth 1 ../pkls/dataset2_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset2/annotations/test.json --pkl_file ../pkls/dataset2_1shot.pkl --output_json ../pkls/dataset2_1shot_coco.json --annotations_json ../submit/dataset2_1shot.json

## 1-shot-dataset3
bash tools/dist_test_out.sh ../configs/1-shot-dataset3.py ../weights/1-shot-dataset3-433149f8.pth 1 ../pkls/dataset3_1shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset3/annotations/test.json --pkl_file ../pkls/dataset3_1shot.pkl --output_json ../pkls/dataset3_1shot_coco.json --annotations_json ../submit/dataset3_1shot.json

## 5-shot-dataset1
bash tools/dist_test_out.sh ../configs/5-shot-dataset1.py ../weights/5-shot-dataset1-ad2ac5f0.pth 1 ../pkls/dataset1_5shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ../pkls/dataset1_5shot.pkl --output_json ../pkls/dataset1_5shot_coco.json --annotations_json ../submit/dataset1_5shot.json

## 5-shot-dataset2
bash tools/dist_test_out.sh ../configs/5-shot-dataset2.py ../weights/5-shot-dataset2-0bfccba8.pth 1 ../pkls/dataset2_5shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset2/annotations/test.json --pkl_file ../pkls/dataset2_5shot.pkl --output_json ../pkls/dataset2_5shot_coco.json --annotations_json ../submit/dataset2_5shot.json

## 5-shot-dataset3
bash tools/dist_test_out.sh ../configs/5-shot-dataset3.py ../weights/5-shot-dataset3-0011f4b1.pth 1 ../pkls/dataset3_5shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset3/annotations/test.json --pkl_file ../pkls/dataset3_5shot.pkl --output_json ../pkls/dataset3_5shot_coco.json --annotations_json ../submit/dataset3_5shot.json

## 10-shot-dataset1
bash tools/dist_test_out.sh ../configs/10-shot-dataset1.py ../weights/10-shot-dataset1-33caf03b.pth 1 ../pkls/dataset1_10shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset1/annotations/test.json --pkl_file ../pkls/dataset1_10shot.pkl --output_json ../pkls/dataset1_10shot_coco.json --annotations_json ../submit/dataset1_10shot.json

## 10-shot-dataset2
bash tools/dist_test_out.sh ../configs/10-shot-dataset2.py ../weights/10-shot-dataset2-46b5584c.pth 1 ../pkls/dataset2_10shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset2/annotations/test.json --pkl_file ../pkls/dataset2_10shot.pkl --output_json ../pkls/dataset2_10shot_coco.json --annotations_json ../submit/dataset2_10shot.json

## 10-shot-dataset3
bash tools/dist_test_out.sh ../configs/10-shot-dataset3.py ../weights/10-shot-dataset3-7325994e.pth 1 ../pkls/dataset3_10shot.pkl

python ../pkl2coco.py --coco_file ../data/dataset3/annotations/test.json --pkl_file ../pkls/dataset3_10shot.pkl --output_json ../pkls/dataset3_10shot_coco.json --annotations_json ../submit/dataset3_10shot.json
```




<!-- 
## 📄 Citation
If you use our method or codes in your research, please cite:
```
@inproceedings{fu2025ntire, 
  title={NTIRE 2025 challenge on cross-domain few-shot object detection: methods and results},
  author={Fu, Yuqian and Qiu, Xingyu and Ren, Bin and Fu, Yanwei and Timofte, Radu and Sebe, Nicu and Yang, Ming-Hsuan and Van Gool, Luc and others},
  booktitle={CVPRW},
  year={2025}
}
```

```
@inproceedings{pan2025enhance, 
  title={Enhance Then Search: An Augmentation-Search Strategy with Foundation Models for Cross-Domain Few-Shot Object Detection},
  author={Pan, Jiancheng and Liu, Yanxing and He, Xiao and Peng, Long and Li, Jiahao and Sun, Yuze and Huang, Xiaomeng},
  booktitle={CVPRW},
  year={2025}
}
```




 -->


 # Acknowledgements
Provided codes were adapted from:

- [jaychempan/ETS](https://github.com/jaychempan/ETS)
- [microsoft/GLIP](https://github.com/microsoft/GLIP)
- [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO/)
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [longzw1997/Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino.git)

