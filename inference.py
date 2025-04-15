import torch
from torch.utils.data import DataLoader

import argparse
import sys
import os
import yaml
from PIL import Image
from tqdm import tqdm
import json
import zipfile

from detector import Detector, TestSet, val_transform
from detector import predict_from_model_outputs

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
parser.add_argument('--checkpoint', required=True, help='checkpoint name')
args = parser.parse_args()

config_path = args.config
checkpoint = args.checkpoint

print(f'Config file: {config_path}')
print(f'Checkpoint: {checkpoint}')

if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['path']['DATA_DIR']
MODEL_DIR = config['path']['MODEL_DIR']
OUTPUT_DIR = config['path']['OUTPUT_DIR']

N_CLASSES = config['global']['N_CLASSES']
BATCH_SIZE = config['train']['BATCH_SIZE']

test_dataset = TestSet(root=f'{DATA_DIR}/test', transforms=val_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda batch: batch)
n_images = len(test_dataset)

device = torch.device('cuda' 
                      if torch.cuda.is_available()
                      else 'cpu')

model = Detector(N_CLASSES)
model.load_state_dict(torch.load(f'{MODEL_DIR}/{checkpoint}.pth'))
model.to(device)
model.eval()

json_outputs = []
csv_outputs = {}
for images in tqdm(test_loader):
    image_ids = [image_id for image_id, _ in images]
    images = [image.to(device) for _, image in images]

    outputs = model(images)
    prediction = predict_from_model_outputs(outputs)

    for image_id, output, pred in zip(image_ids, outputs, prediction):
        image_id = int(image_id.split('.')[0])
        
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if label > N_CLASSES: continue

            box = box.detach().cpu().numpy().tolist()
            score = score.detach().cpu().item()
            label = label.detach().cpu().item()
            box[2] -= box[0]
            box[3] -= box[1]

            json_outputs.append({
                'image_id': image_id,
                'bbox': box,
                'score': score,
                'category_id': label
            })

        csv_outputs[image_id] = pred

output_zip_path = f'{OUTPUT_DIR}/{checkpoint}.zip'
json_path = f'{OUTPUT_DIR}/{checkpoint}.json'
csv_path = f'{OUTPUT_DIR}/{checkpoint}_v2.csv'

with open(json_path, 'w') as f:
    json.dump(json_outputs, f, indent=4)

with open(csv_path, 'w') as f:
    f.write('image_id,pred_label\n')
    for image_id in range(1, n_images+1):
        f.write(f'{image_id},{csv_outputs[image_id]}\n')

with zipfile.ZipFile(output_zip_path, mode='w') as f:
    f.write(json_path, 'pred.json')
    f.write(csv_path, 'pred.csv')