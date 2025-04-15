import torch

from .transform import train_transform, val_transform

train_transform_wrapper = lambda image, target: (train_transform(image), target)
val_transform_wrapper = lambda image, target: (val_transform(image), target)

def COCO_to_fasterRCNN_target(data):
    images = []
    targets = []

    for image, COCO_targets in data:
        boxes = []
        labels = []
        image_ids = []

        for COCO_target in COCO_targets:
            x, y, w, h = COCO_target['bbox']
            boxes.append([x, y, x+w, y+h]),
            labels.append(COCO_target['category_id']),
            image_ids.append(COCO_target['image_id'])

        target = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels),
            'image_id': torch.LongTensor(image_ids),
        }

        images.append(image)
        targets.append(target)

    return images, targets

def COCO_to_answer(targets):
    ret = []

    for target in targets:
        order = {}
        for bbox, label in zip(target['boxes'].cpu().detach(), target['labels'].cpu().detach()):
            x, _, w, _ = bbox
            center = x + w / 2
            order[center] = round(label.item()) - 1
        
        ret.append(
            ''.join(str(order[k])
                    for k in sorted(order.keys()))
        )

    return ret