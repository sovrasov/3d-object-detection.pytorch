# 3d-object-detection.pytorch

This project provides code to train a two stage 3d object detection models on the [Objectron](https://github.com/google-research-datasets/Objectron) dataset.

Training includes the following stages:
- Converting the original Objectron annotation to COCO-like format
- Training a 2d bounding box detection model
- Training a 3d bounding box regression model

Trained models can be deployed on CPU using [OpenVINO](https://docs.openvinotoolkit.org) framework and then run in [live demo](demo/demo.py).

## Installation guide
```bash
git clone https://github.com/sovrasov/3d-object-detection.pytorch.git --recursive
python setup.py develop
```
All the mentioned below scripts should be launched from the repo root folder because they depend on the Objectron python package which is distributed in source codes only. This issue is planned to be addressed.

## Converting data

Download raw Objectron data and extract it preserving the following layout:
```
Objectron_root
├── annotation
│   ├── bike
|   |   └── <batch-* folders with .pbdata files>
│   ├── book
|   |   └── <batch-* folders with .pbdata files>
│   |── ....
│   └── shoe
|       └── <batch-* folders with .pbdata files>
│
└── videos
    ├── bike
    |   └── <batch-* folders with videos>
    ├── book
    |   └── <batch-* folders with videos>
    └── ....
    └── shoe
        └── <batch-* folders with videos>
```

Then run the converter:

```bash
python annotation_converters/objectron_2_coco.py --data_root <Objectron_root> --output_folder <output_dir> --fps_divisor 5 --res_divisor 2 --obj_classes all
```

Adjacent frames on the 30 FPS videos are close to each other, so we can take only each `fps_divisor` frame and downscale them from the FullHD resolution by a factor `res_divisor` without significant loss of information.

## Train 2d detector

Detector trained on this step is supposed to retrieve 2d bounding boxes that enclose 3d boxes from the original Objectron annotation.
At the next stage a multi-class regression model is launched on detected regions to finally obtain 2d coordinates of projected 3d bounding box vertexes.

To launch training refer to the instructions from the modified [mmdetection](https://github.com/openvinotoolkit/mmdetection/) repo.
Config for detector is stored in `configs/detection/mnv2_ssd_300_2_heads.py`, COCO-formatted data required for training is obtained at the previous step.
