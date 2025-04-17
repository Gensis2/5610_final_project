import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, img_size=28, num_classes=10):
        super(CNN, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size // 2) * (self.img_size // 2) * 64, 512, bias=False)
        self.bn_fc = nn.BatchNorm1d(512, eps=1e-4, momentum=0.1, affine=True)
        self.fc2 = nn.Linear(512, self.num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):
        x = F.relu(self.bn1(self.conv1(inp)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)

        return x