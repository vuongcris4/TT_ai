from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset, MyDataset
from torchvision import transforms
import torch
from torch.optim import Adam
from learner import Learner, evaluate
from model import myModel
import torch.nn as nn
from utils import count_parameters, train_val_split, calculate_mean_standard

dataset = SemanticSegmentationDataset(
    image_dir='kaggle/input',
    label_dir='kaggle/label',
)

train_subset, val_subset = train_val_split(dataset)

mean, std = calculate_mean_standard(train_subset)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)

])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = MyDataset(train_subset, transform=train_transform)
val_dataset = MyDataset(val_subset, transform=val_transform)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.class_colors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel(num_classes).to(device)

count_parameters(model) # Đếm số parameters của model

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 5

learner = Learner(model=model, criterion=criterion, optimizer=optimizer, device=device, n_epoch=num_epochs)
learner.fit(train_dataloader, val_dataloader, 3) # train và lưu best model

# epoch_loss_val, mAcc_val, mIoU_val = evaluate(learner.model, val_dataloader, criterion, device, num_classes)
# print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")

# Load best model
# checkpoint = torch.load("checkpoints/22139078_22139044_best_model.pt", weights_only=False)
# print("===================")
# print(f"Best Model at epoch : {checkpoint["epoch"]}")
#
# model2 = myModel(num_classes).to(device)
# model2.load_state_dict(checkpoint["model"])
# epoch_loss_val, mAcc_val, mIoU_val = evaluate(model2, val_dataloader, criterion, device, num_classes)
# print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")

# LOAD BEST MODEL
model2 = torch.jit.load("checkpoints/22139078_22139044_best_model.pt")
epoch_loss_val, mAcc_val, mIoU_val = evaluate(model2, val_dataloader, criterion, device, num_classes)
print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")

