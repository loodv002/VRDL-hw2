import json
import yaml
import os
import sys
import torch
from torchvision.ops import nms
from collections import defaultdict

from typing import List, Dict, Any

def _order_by_bbox_center(bboxes, labels, scores) -> Dict[float, int]:
    order = {}
    for bbox, label, score in zip(bboxes, labels, scores):
        if label > 10: continue
        if score < 0.5: continue
        
        x, _, w, _ = bbox
        center = x + w / 2
        
        if isinstance(label, torch.Tensor): label = label.detach().item()

        order[center] = round(label - 1)
    
    return order

def _order_by_nms(bboxes_tensor, labels_tensor, scores_tensor) -> Dict[float, int]:
    if len(bboxes_tensor) == 0: return {}
    
    # Drop too wide or too height
    min_w = bboxes_tensor[:, 2].min()
    min_h = bboxes_tensor[:, 3].min()
    dropped_idx = torch.logical_or(
        bboxes_tensor[:, 2] < min_w*2, 
        bboxes_tensor[:, 3] < min_h*2
    )
    bboxes_tensor = bboxes_tensor[dropped_idx]
    labels_tensor = labels_tensor[dropped_idx]
    scores_tensor = scores_tensor[dropped_idx]

    bboxes_tensor[:, 2] += bboxes_tensor[:, 0]
    bboxes_tensor[:, 3] += bboxes_tensor[:, 1]

    keep_idx = nms(bboxes_tensor, 
                   scores_tensor, 
                   0.2)
    
    order = {}
    for i in keep_idx:
        bbox = bboxes_tensor[i]
        label = labels_tensor[i]
        score = scores_tensor[i]

        if score < 0.2: continue
        if label > 10: continue

        center = (bbox[0] + bbox[2]) / 2
        order[center] = round((label - 1).detach().item())

    return order

def predict_image(bboxes, labels, scores) -> str:
    '''bbox: [x, y, w, h]'''
    order = (_order_by_bbox_center(bboxes, labels, scores)
             or _order_by_nms(bboxes, labels, scores))
    
    if not order: return '-1'
    
    digits = ''.join(
        str(order[k])
        for k in sorted(order.keys())
    )
    return digits

def predict_from_model_outputs(outputs: List[Dict[str, Any]]) -> List[str]:
    prediction = []

    for output in outputs:
        bboxes = output['boxes'].cpu().detach()
        labels = output['labels'].cpu().detach()
        scores = output['scores'].cpu().detach()

        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
        
        prediction.append(predict_image(bboxes, labels, scores))
    
    return prediction

def predict_from_json_file(file_path: str, n_images=13068) -> List[str]:
    with open(file_path, 'r') as f:
        outputs = json.load(f)

    bboxes = defaultdict(list)
    scores = defaultdict(list)
    labels = defaultdict(list)

    for output in outputs:
        image_id = output['image_id']
        bboxes[image_id].append(output['bbox'])
        scores[image_id].append(output['score'])
        labels[image_id].append(output['category_id'])

    return [
        predict_image(torch.Tensor(bboxes[image_id]),
                      torch.Tensor(labels[image_id]),
                      torch.Tensor(scores[image_id]))
        for image_id in range(1, n_images+1)
    ]

if __name__ == '__main__':
    checkpoint = sys.argv[1]
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open('./config.yml', 'r') as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR = config['path']['OUTPUT_DIR']

    prediction = predict_from_json_file(f'{OUTPUT_DIR}/{checkpoint}.json')

    with open(f'{OUTPUT_DIR}/{checkpoint}_v2.csv', 'w') as f:
        f.write('image_id,pred_label\n')
        for i, p in enumerate(prediction, 1):
            f.write(f'{i},{p}\n')
