import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision


from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any

from .model import Detector
from .format import COCO_to_answer
from .task2 import predict_from_model_outputs

class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' 
                                   if torch.cuda.is_available()
                                   else 'cpu')
        
    def _train_epoch(self,
                     model: Detector,
                     train_loader: DataLoader,
                     optimizer: optim.Optimizer,
                     loss_weights: Dict[str, float]) -> Tuple[float, float]:
        
        loss_sum = 0
        n_correct = 0
        n_data = 0

        for images, targets in tqdm(train_loader, ncols=100):
            images = [image.to(self.device) for image in images]
            targets = [
                {
                    key: value.to(self.device)
                    for key, value in target.items()
                }
                for target in targets
            ]

            # Training
            model.train()
            losses = model(images, targets)
            loss_weighted_sum = sum(
                sub_loss * loss_weights.get(loss_name, 1.0)
                for loss_name, sub_loss in losses.items()
            )

            optimizer.zero_grad()
            loss_weighted_sum.backward()
            optimizer.step()

            loss_sum += loss_weighted_sum.detach().item()

            # Compute training accuracy
            with torch.no_grad():
                model.eval()
                outputs = model(images)
                gt_answer = COCO_to_answer(targets)
                pred_answer = predict_from_model_outputs(outputs)
                n_correct += sum(gt == pred for gt, pred in zip(gt_answer, pred_answer))
                n_data += len(gt_answer)

            break # DEBUG

        loss_sum /= len(train_loader)
        accuracy = n_correct / n_data
        return loss_sum, accuracy

    def _val_epoch(self,
                   model: Detector,
                   val_loader: DataLoader,
                   loss_weights: Dict[str, float]) -> Tuple[float, float]:
        
        loss_sum = 0
        mAP = MeanAveragePrecision(iou_type='bbox')
        n_correct = 0
        n_data = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, ncols=100):
                images = [image.to(self.device) for image in images]
                targets = [
                    {
                        key: value.to(self.device)
                        for key, value in target.items()
                    }
                    for target in targets
                ]

                # Compute vaildation loss
                model.train()
                losses = model(images, targets)
                loss_weighted_sum = sum(
                    sub_loss * loss_weights.get(loss_name, 1.0)
                    for loss_name, sub_loss in losses.items()
                )
                loss_sum += loss_weighted_sum

                # Compute validation accuracy
                model.eval()
                outputs = model(images)
                mAP.update(outputs, targets)

                gt_answer = COCO_to_answer(targets)
                pred_answer = predict_from_model_outputs(outputs)
                n_correct += sum(gt == pred for gt, pred in zip(gt_answer, pred_answer))
                n_data += len(gt_answer)

                break # DEBUG

        accuracy = n_correct / n_data
        return loss_sum, mAP.compute(), accuracy

    def train(self,
              model: Detector,
              train_loader: DataLoader,
              val_loader: DataLoader,
              checkpoint_dir: str,
              max_epoches: int,
              optimizer: optim.Optimizer,
              scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
              loss_weights: Dict[str, float] = None,
              early_stop: bool = True,
              ):
        
        print(f'Train model by {self.device}')
        
        loss_weights = loss_weights or {'loss_classifier': 1.0,
                                        'loss_box_reg': 1.0,
                                        'loss_objectness': 1.0,
                                        'loss_rpn_box_reg': 1.0}

        model = model.to(self.device)

        min_val_loss = float('inf')
        val_loss_increase_count = 0
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(max_epoches):
            train_loss, train_accuracy = self._train_epoch(
                model,
                train_loader,
                optimizer,
                loss_weights,
            )

            val_loss, val_map, val_accuracy = self._val_epoch(
                model,
                val_loader,
                loss_weights,
            )

            print(f'Epoch {epoch} train loss: {train_loss:.3f}')
            print(f'Epoch {epoch} val loss: {val_loss:.3f}')
            print(f'Epoch {epoch} val accuracy: {val_accuracy * 100:.3f}%')
            print(f'Epoch {epoch} val mAP: {val_map}')

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if scheduler: scheduler.step()

            model_path = f'{checkpoint_dir}/{model.model_name}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_path)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                val_loss_increase_count = 0
            else:
                val_loss_increase_count += 1

            if val_loss_increase_count >= 2 and early_stop:
                print('Loss increased, training stopped.')
                break

        else:
            print('Max epoches reached.')

        print(f'Train losses: {train_losses}')
        print(f'Val losses: {val_losses}')
        print(f'Train accuracies: {train_accuracies}')
        print(f'Val accuracies: {val_accuracies}')