from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm
import torch
import copy
import os

class Learner:
    def __init__(self, model, criterion, optimizer, device, n_epoch, checkpoint_dir='checkpoints'):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.n_epoch = n_epoch

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True) # Tạo thư mục nếu chưa có

        # model = nn.DataParallel(model)

    def train_epoch(self, train_dataloader, num_classes):
        self.model.train()
        running_loss = 0.0
        accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(self.device)
        pbar = tqdm(train_dataloader, desc='Training', unit='batch')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)    # loss từng ảnh * batch_size
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, labels)
            iou_metric.update(preds, labels)
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
            })
        epoch_loss = running_loss / len(train_dataloader.dataset)
        mean_accuracy = accuracy_metric.compute().cpu().numpy()
        mean_iou = iou_metric.compute().cpu().numpy()

        return epoch_loss, mean_accuracy, mean_iou

    def evaluate(self, val_test_dataloader, num_classes):
        self.model.eval()
        running_loss = 0.0
        accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(self.device)
        pbar = tqdm(val_test_dataloader, desc='Evaluating', unit='batch')
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                # Update metrics
                accuracy_metric.update(preds, labels)
                iou_metric.update(preds, labels)
                # Update tqdm description with metrics
                pbar.set_postfix({
                    'Batch Loss': f'{loss.item():.4f}',
                    'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                    'Mean IoU': f'{iou_metric.compute():.4f}',
                })

        epoch_loss = running_loss / len(val_test_dataloader.dataset)
        mean_accuracy = accuracy_metric.compute().cpu().numpy()
        mean_iou = iou_metric.compute().cpu().numpy()

        return epoch_loss, mean_accuracy, mean_iou

    def train(self, train_dataloader, val_dataloader, num_classes):
        epoch_saved = 0
        best_val_mAcc = 0.0
        best_model_state = None

        for epoch in range(self.n_epoch):
            epoch_loss_train, mAcc_train, mIoU_train = self.train_epoch(train_dataloader, num_classes)
            epoch_loss_val, mAcc_val, mIoU_val = self.evaluate(val_dataloader, num_classes)

            print(f"Epoch {epoch + 1}/{self.n_epoch}")
            print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}")
            print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")

            # save last model
            checkpoint = {
                "epoch": epoch + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(checkpoint, f"{self.checkpoint_dir}/22139078_VuongTran_last_model.pt")

            # save best model
            if mAcc_val >= best_val_mAcc:
                best_val_mAcc = mAcc_val
                checkpoint = {
                    "epoch": epoch + 1,
                    "best_acc": best_val_mAcc,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                torch.save(checkpoint, f"{self.checkpoint_dir}/22139078_VuongTran_best_model.pt")

