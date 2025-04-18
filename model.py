import torch.nn as nn
import torch.nn.functional as F

class myModel(nn.Module):
    def __init__(self, n_classes):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.upSampling = nn.Upsample(scale_factor=4, mode="bilinear")
        self.batchnorm = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.upSampling(x)
        x = self.batchnorm(x)
        x = self.conv_out(x)

        return F.softmax(x, dim=1)

