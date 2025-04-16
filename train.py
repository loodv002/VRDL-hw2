import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

import argparse
import yaml
import os

from detector import Detector, Trainer
from detector import train_transform_wrapper, val_transform_wrapper
from detector import COCO_to_fasterRCNN_target

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
args = parser.parse_args()

config_path = args.config
if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

print(f'Use config file "{config_path}"')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['path']['DATA_DIR']
MODEL_DIR = config['path']['MODEL_DIR']
OUTPUT_DIR = config['path']['OUTPUT_DIR']

N_CLASSES = config['global']['N_CLASSES']

BATCH_SIZE = config['train']['BATCH_SIZE']
LEARNING_RATE = config['train']['LEARNING_RATE']
MAX_EPOCHES = config['train']['MAX_EPOCHES']
EARLY_STOP = config['train']['EARLY_STOP']

train_dataset = CocoDetection(root=f'{DATA_DIR}/train',
                              annFile=f'{DATA_DIR}/train.json',
                              transforms=train_transform_wrapper)
train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True, 
                          collate_fn=COCO_to_fasterRCNN_target)

val_dataset = CocoDetection(root=f'{DATA_DIR}/valid',
                              annFile=f'{DATA_DIR}/valid.json',
                              transforms=val_transform_wrapper)
val_loader = DataLoader(val_dataset, 
                        batch_size=BATCH_SIZE, 
                        collate_fn=COCO_to_fasterRCNN_target)

model = Detector(N_CLASSES)
print(f'Model name: {model.model_name}')

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
loss_weights = {
    'loss_classifier': 5.0,
    'loss_box_reg': 1.0,
    'loss_objectness': 1.0,
    'loss_rpn_box_reg': 1.0,
}

trainer = Trainer()
trainer.train(
    model,
    train_loader,
    val_loader,
    MODEL_DIR,
    MAX_EPOCHES,
    optimizer,
    scheduler,
    loss_weights,
    EARLY_STOP,
)