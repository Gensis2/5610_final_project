import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, img_size=28, num_classes=10, save_flops=False):
        # initialize CNN model
        super(CNN, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # layer initization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size // 2) * (self.img_size // 2) * 64, 512, bias=False)
        self.bn_fc = nn.BatchNorm1d(512, eps=1e-4, momentum=0.1, affine=True)
        self.fc2 = nn.Linear(512, self.num_classes, bias=False)

        # initialize flops computation
        self.save_flops = save_flops
        if save_flops:
            self.total_flops = 0

    def forward(self, inp):

        # first conv / bn layer with relu output
        x = F.relu(self.bn1(self.conv1(inp)))

        # add cnn and bn 1 flops count
        if self.save_flops:
            batch_size, out_channels, out_height, out_width = x.shape
            _, in_channels, kernel_height, kernel_width = self.conv1.weight.shape
            cnn_flops = 2 * batch_size * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width * (10**-12)
            cnn_bn_flops = batch_size * out_channels * out_height * out_width * 7 * (10**-12)
        
        # second conv / bn layer with relu output
        x = F.relu(self.bn2(self.conv2(x)))
        
        # add cnn and bn 2 flops count
        if self.save_flops:
            batch_size, out_channels, out_height, out_width = x.shape
            _, in_channels, kernel_height, kernel_width = self.conv2.weight.shape
            cnn_flops = 2 * batch_size * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width * (10**-12)
            cnn_bn_flops = batch_size * out_channels * out_height * out_width * 7 * (10**-12)
        
        # average pool layer
        x = self.pool(x)
        
        # add pool flops count
        if self.save_flops:
            pkernel_size = self.pool.kernel_size
            pool_flops = batch_size * out_channels * (out_height // pkernel_size) * (out_width // pkernel_size) * pkernel_size * pkernel_size * (10**-12)
        
        # flatten the output and pass through fc layer with relu
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        
        # add fc and bn flops count
        if self.save_flops:
            m, k = x.shape
            k, n = self.fc1.weight.shape
            fc_flops = 2 * m * k * n * (10**-12)
            fc_bn_flops = m * k * 7 * (10**-12)
        
        # final output layer
        x = self.fc2(x)
        
        # add final fc flops count
        if self.save_flops:
            m, k = x.shape
            k, n = self.fc2.weight.shape
            fc_flops += 2 * m * k * n * (10**-12)
            fc_bn_flops = m * k * 7 * (10**-12)
            self.total_flops += cnn_flops + cnn_bn_flops + pool_flops + fc_flops + fc_bn_flops

        return x