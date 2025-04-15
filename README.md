# NYCU VRDL spring 2025 HW2

Student ID: 110550138

Name: 鄭博文

## Introduction

A fine-tuned `fasterrcnn_mobilenet_v3_large_fpn` model for digit recognition, with training and inference code.

## How to Run

### Install requirements:

  ```bash
  pip install -r requirements.txt
  ```

### Configuration:

  Check the comments in `config-example.yml`.

### Training:

  ```bash
  python train.py [--config <config_file_path>]
  ```
  The default value of `config` argument is `./config.yml`.

### Inference:

  ```
  python inference.py --checkpoint <checkpoint_name> [--config <config_file_path>]
  ```
  The checkpoint name should be the stem of `.pth` file in `MODEL_DIR`, with formate `{date}-{time}_epoch_{epoch}`. For example, `20250414-222522_epoch_4` (without .pth).

## Performance:

  Task-1 mAP: 0.38, task-2 accuracy: 0.75.