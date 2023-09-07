# DentexSegAndDet

This repository contains our algorithm for the [MICCAI  2023 Dentex challange](https://dentex.grand-challenge.org/).

+ Method paper: [ Intergrated Segmentation and Detection Models for Dentex Challenge 2023 (arxiv.org)](https://arxiv.org/abs/2308.14161)
+ Dataset structure
  Datasets are organized as:

  ```
  dentex_dataset
  ├── coco
  │   ├── disease
  │   │   ├── annotations
  │   │   ├── train2017
  │   │   └── val2017
  │   ├── disease_all
  │   │   ├── annotations
  │   │   ├── train2017
  │   │   └── val2017
  │   ├── enumeration32
  │   │   ├── annotations
  │   │   ├── train2017
  │   │   └── val2017
  │   └── quadrant
  │       ├── annotations
  │       ├── train2017
  │       └── val2017
  ├── origin
  │   ├── quadrant
  │   ├── quadrant_enumeration
  │   ├── quadrant_enumeration_disease
  │   └── unlabelled
  ├── segmentation
  │   ├── enumeration32
  │   │   ├── masks
  │   │   └── xrays
  │   └── enumeration9
  │       ├── masks
  │       └── xrays
  └── yolo
      ├── disease
      │   ├── images
      │   │   ├── train2017
      │   │   └── val2017
      │   └── labels
      │       ├── train2017
      │       └── val2017
      └── disease_all
          ├── images
          │   ├── train2017
          │   └── val2017
          └── labels
              ├── train2017
              └── val2017
  ```
+ Process:

  + prepare detection dataset  
  
    Run each `process...` function in `process_dataset.py` to convert the dataset to expected format (coco or yolo). The processes are intended to be executed sequentially.
  + train detection models  
  
    Download pretrained weights from each offical repos(swin-transformer, dino, yolo, etc.) and refer to those offical repos and `command_snippets.sh` to train detection models.
  + prepare segmentaion dataset  
  
    32-class segmentaion dataset can be generated from the origin dataset. 9-class segmentation dataset depends on the prediction result by a quadrant detection model. See `results/enumeration_dataset_quadrant_predictions.json` for example.
  + train segmentaion models  
  
    Refer to the `command_snippets.sh`
  + run prediction  
  
    Choose best checkpoints for each model, rename them or modify the paths in the `predict.py`, and run `predict.py`.  `results/abnormal-teeth-detection.json` is an example output.
