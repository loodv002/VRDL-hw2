from torch import nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

from datetime import datetime

class Detector(nn.Module):
    def __init__(self, n_classes: int):
        super(Detector, self).__init__()

        self.backbone = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        
        self.backbone.rpn.pre_nms_top_n_train = 500
        self.backbone.rpn.post_nms_top_n_train = 500
        self.backbone.rpn.pre_nms_top_n_test = 300
        self.backbone.rpn.post_nms_top_n_test = 300

        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes + 1)

        self.model_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    def forward(self, *args):
        return self.backbone(*args)