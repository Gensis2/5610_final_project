import torch
import torch.nn as nn
import torch.nn.functional as F

# poisson generation of input images 
# converts to rate coded spikes for SNN input
def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

# SNN function
class snn_bp(torch.autograd.Function):
    # forward pass spiking neuron
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    # backward pass gradient update given voltage
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad

class SNN_Conv(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=28, num_classes=10, save_flops=False):
        super(SNN_Conv, self).__init__()

        # initialize SNN model
        self.num_steps = num_steps
        self.leak_mem = leak_mem
        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_num = self.num_steps
        # apply custom snn function
        self.spike_fn = snn_bp.apply

        # initialize flops computation
        self.save_flops = save_flops
        if self.save_flops:
            self.total_flops = 0

        # layer initialization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bsnn1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bsnn2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True) for i in range(self.batch_num)])
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size // 2) * (self.img_size // 2) * 64, 512, bias=False)
        self.bsnn_fc = nn.ModuleList([nn.BatchNorm1d(512, eps=1e-4, momentum=0.1, affine=True) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(512, self.num_classes, bias=False)

        # initialize lists for conv, fc, and pool layers
        self.conv_list = [self.conv1, self.conv2]
        self.bsnn_list = [self.bsnn1, self.bsnn2, self.bsnn_fc]
        self.pool_list = [False, self.pool]
        self.fc_list = [self.fc1, self.fc2]

        # initialize weights
        for bn_list in self.bsnn_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        # initialize threshold voltage for convolutional and fully connected layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        # initialize spiking memory and reset memory for snn activity
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv_list = [mem_conv1, mem_conv2]

        mem_fc1 = torch.zeros(batch_size, 512).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_classes).cuda()

        rst_conv1 = torch.zeros_like(mem_conv1).cuda()
        rst_conv2 = torch.zeros_like(mem_conv2).cuda()
        rst_conv_list = [rst_conv1, rst_conv2]

        rst_fc1 = torch.zeros_like(mem_fc1).cuda()

        # initialize flops computation
        if self.save_flops:
            cnn_neuron_flops = 0
            cnn_flops = 0
            bsnn_cnn_flops = 0
            bsnn_fc_flops = 0
            fc_neuron_flops = 0
            fc_flops = 0
            pool_flops = 0

        # loop through time steps
        for t in range(self.num_steps):
            
            # generate poisson spikes for input
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            # loop through conv layers
            for i in range(len(self.conv_list)):
                # apply batch norm and conv layer, add to voltage memory and allow for leakage
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bsnn_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)

                # set cnn shapes for flops computation, and sparsity of output. When output is 0, there is no addition so computation is skipped
                if self.save_flops:
                    out_sparsity = (torch.sum(out_prev) / out_prev.numel()).item()
                    batch_size, out_channels, out_height, out_width = out_prev.shape
                    _, in_channels, kernel_height, kernel_width = self.conv_list[i].weight.shape

                # set reset memory for where voltage meets threshold
                rst_conv_list[i].zero_()
                rst_conv_list[i][mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] -= rst_conv_list[i]
                out_prev = out.clone()

                # apply average pool after last conv layer
                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

                # compute pooling layer sparsity and compute cnn flops
                if self.save_flops:
                    pool_out_sparsity = torch.sum(out_prev) / out_prev.numel()
                    cnn_neuron_flops += 2 * batch_size * out_channels * out_height * out_width * (10**-12)
                    cnn_flops += batch_size * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width * out_sparsity * (10**-12)
                    bsnn_cnn_flops += batch_size * out_channels * out_height * out_width * 7 * (10**-12)

            out_prev = out_prev.reshape(batch_size, -1)

            # FC layer 1 with batch norm, added to voltage memory and allow for leakage
            mem_fc1 = self.leak_mem * mem_fc1 + self.bsnn_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)

            # compute fc and bsnn flops for fc1
            if self.save_flops:
                m, k = out_prev.shape
                k, n = self.fc1.weight.shape
                out_sparsity = torch.sum(out_prev) / out_prev.numel()
                fc_neuron_flops += 2 * m * k * (10**-12)
                fc_flops += m * k * n * out_sparsity * (10**-12)
                bsnn_fc_flops += m * n * 7 * (10**-12)

            # set reset memory for where voltage meets threshold
            rst_fc1.zero_()
            rst_fc1[mem_thr > 0] = self.fc1.threshold
            mem_fc1 -= rst_fc1
            out_prev = out.clone()

            # sparsity for fc2
            if self.save_flops:
                out_sparsity = torch.sum(out_prev) / out_prev.numel()

            mem_fc2 = mem_fc2 + self.fc2(out_prev)

            # compute fc2 flops
            if self.save_flops:
                m, k = out_prev.shape
                k, n = self.fc2.weight.shape
                fc_flops += m * k * n * out_sparsity * (10**-12)
                pkernel_size = self.pool.kernel_size
                pool_flops += batch_size * out_channels * (out_height // pkernel_size) * (out_width // pkernel_size) * pkernel_size * pkernel_size * pool_out_sparsity * (10**-12)

        # accumulate flops for all layers
        if self.save_flops:
            self.total_flops += (cnn_neuron_flops + cnn_flops + bsnn_cnn_flops + fc_neuron_flops + fc_flops + bsnn_fc_flops + pool_flops).item()

        # return output voltage of neurons
        out_voltage = mem_fc2 / self.num_steps
        return out_voltage