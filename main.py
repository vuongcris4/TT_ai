from torch.utils.data import Dataset, DataLoader
from dataset import SemanticSegmentationDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.optim import Adam
from torchvision import models
from learner import Learner
from model import myModel
import torch.nn as nn
from utils import count_parameters, train_val_split

train_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

dataset = SemanticSegmentationDataset(
    image_dir='kaggle/input',
    label_dir='kaggle/label',
    transform=train_val_transform
)

train_dataset, val_dataset = train_val_split(dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.class_colors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel(num_classes).to(device)

count_parameters(model) # Đếm số parameters của model

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 5

learner = Learner(model, criterion, optimizer, device, num_epochs)
# learner.train(train_dataloader, val_dataloader, 3) # train và lưu best model

checkpoint = torch.load("checkpoints/22139078_VuongTran_best_model.pt", weights_only=False)
print("===================")
print(f"Best Model at epoch : {checkpoint["epoch"]}")

model2 = myModel(num_classes).to(device)
model2.load_state_dict(checkpoint["model"])
learner.evaluate(model2, val_dataloader, 3)
