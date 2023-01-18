import torch.nn as nn
import torch.nn.functional as f
import torch
import netron     ###################
import torch.onnx ################
from utils import Config as cfg
class BasicBlock(nn.Module):
    def __init__(self, input, output, stride, dropRate=0.0) -> None:
        super(BasicBlock, self).__init__()
        # self.input = in.size(1)
        self.is_channel_equal = (input == output)
        self.bn1 = nn.BatchNorm2d(input)
        self.relu1 = nn.ReLU() 
        self.conv1 = nn.Conv2d(input, output, kernel_size=3, stride=stride, 
                                padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn2 = nn.BatchNorm2d(output)
        self.relu2 = nn.ReLU() 
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=1, 
                                padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.droprate = dropRate
        self.convShortcut = (not self.is_channel_equal) and nn.Conv2d(input, output, kernel_size=1, stride=stride, padding=0, bias=False) or None
    def forward(self, x):
        bn1 = self.bn1(x)
        bn1 = self.relu1(bn1)
        out = self.conv1(bn1)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.droprate > 0:
            out = f.dropout(out,p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(out, bn1 if self.is_channel_equal else self.convShortcut(bn1))
        return out

class Conv_Group(nn.Module):
    def __init__(self, block, input, output, n, strides, dropoutRate=0.0):
        super(Conv_Group, self).__init__()
        self.layer = self.make_layer(block, input, output, n, strides, dropoutRate)
    def make_layer(self, block, input, output, n, strides, dropoutRate=0.0):
        layers = []
        for i in range(int(n)):
            layers.append(block(i==0 and input or output, output, i==0 and strides or 1, dropoutRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth-4)%6 == 0), 'depth should be 6n+4'
        n = (depth - 4) / 6
        self.h = int(cfg.INPUT_H / 2 / 2 / 8)
        self.w = int(cfg.INPUT_W / 2 / 2 / 8)
        self.nchannels = nChannels[3]
        block = BasicBlock
        self.conv1 = nn.Conv2d(cfg.INPUT_CH, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)  # 1st layer
        nn.init.kaiming_normal_(self.conv1.weight)
        # 1st Block
        self.block1 = Conv_Group(block, nChannels[0], nChannels[1], n, 1, dropRate)
        # 2st Block 
        self.block2 = Conv_Group(block, nChannels[1], nChannels[2], n, 2, dropRate)
        # 3st Block 
        self.block3 = Conv_Group(block, nChannels[2], nChannels[3], n, 2, dropRate)
        # global average pooling
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nChannels[3]*self.h*self.w, num_classes)
        nn.init.kaiming_uniform_(self.fc.weight)
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = f.avg_pool2d(out, 8)
        out = out.view(-1, self.nchannels * self.w * self.h)
        out = self.fc(out)

        return out

# x = torch.randn(32,1,64,64)
# block = BasicBlock
# net = WideResNet(16, 72, 10, 0)
# # l = nn.Conv2d(32,3,3,stride=2,padding=1)#Conv2d(1, 1, kernel_size=4,stride=1,padding=2)
# # y = l(x) # y.shape:[1,1,5,5]
# y = net(x)
# print(y.data.shape)
# onnx_path = "onnx_model_name.onnx"
# torch.onnx.export(net, x, onnx_path)
# netron.start(onnx_path)


