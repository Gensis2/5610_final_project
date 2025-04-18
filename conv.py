import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, img_size=28, num_classes=10, path=None):
        super(CNN, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.path = path

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

        with open(self.path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 1:
                    cnn_flops = float(line.split(':')[1].strip())
                elif i == 2:
                    cnn_bn_flops = float(line.split(':')[1].strip())
                elif i == 3:
                    pool_flops = float(line.split(':')[1].strip())
                elif i == 4:
                    fc_flops = float(line.split(':')[1].strip())
                elif i == 5:
                    fc_bn_flops = float(line.split(':')[1].strip())
                elif i == 6:
                    total_flops = float(line.split(':')[1].strip())

        x = F.relu(self.bn1(self.conv1(inp)))

        batch_size, out_channels, out_height, out_width = x.shape
        _, in_channels, kernel_height, kernel_width = self.conv1.weight.shape
        cnn_flops += 2 * batch_size * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width * (10**-12)
        cnn_bn_flops += batch_size * out_channels * out_height * out_width * 7 * (10**-12)
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        batch_size, out_channels, out_height, out_width = x.shape
        _, in_channels, kernel_height, kernel_width = self.conv2.weight.shape
        cnn_flops += 2 * batch_size * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width * (10**-12)
        cnn_bn_flops += batch_size * out_channels * out_height * out_width * 7 * (10**-12)
        
        x = self.pool(x)
        
        pkernel_size = self.pool.kernel_size
        pool_flops += batch_size * out_channels * (out_height // pkernel_size) * (out_width // pkernel_size) * pkernel_size * pkernel_size * (10**-12)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc(self.fc1(x)))
        
        m, k = x.shape
        k, n = self.fc1.weight.shape
        fc_flops += 2 * m * k * n * (10**-12)
        fc_bn_flops += m * k * 7 * (10**-12)
        
        x = self.fc2(x)
        
        m, k = x.shape
        k, n = self.fc2.weight.shape
        fc_flops += 2 * m * k * n * (10**-12)
        fc_bn_flops += m * k * 7 * (10**-12)
        total_flops = cnn_flops + cnn_bn_flops + pool_flops + fc_flops + fc_bn_flops

        with open(self.path, 'w') as f:
            f.write("SNN FLOPS\n")
            f.write(f"Conv Flops: {cnn_flops}\n")
            f.write(f"Conv BN Flops: {cnn_bn_flops}\n")
            f.write(f"Pool Flops: {pool_flops}\n")
            f.write(f"FC Flops: {fc_flops}\n")
            f.write(f"FC BN Flops: {fc_bn_flops}\n")
            f.write(f"Total Flops: {total_flops}\n")

        return x