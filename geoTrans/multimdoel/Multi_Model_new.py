import torch.nn as nn
import torch.nn.functional as f
import torch
import netron     ###################
import torch.onnx ################

import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')
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


class ResBlock(nn.Module):
    def __init__(self, input, output):
        super(ResBlock, self).__init__()
        self.is_channel_equal = (input == output)
        self.bn1 = nn.BatchNorm1d(input)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(input, output)
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.bn2 = nn.BatchNorm1d(output)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(output, output)
        nn.init.kaiming_uniform_(self.fc2.weight)
        self.shortcut = (not self.is_channel_equal) and nn.Linear(input, output) or None

    def forward(self, x):
        bn1 = self.bn1(x)
        bn1 = self.relu1(bn1)
        out = self.fc1(bn1)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc2(out)
        out = torch.add(out, bn1 if self.is_channel_equal else self.shortcut(bn1))
        return out


class MlpBlock(nn.Module):
    def __init__(self, input, output):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(input, output)
        nn.init.kaiming_uniform_(self.fc1.weight)
        # self.bn1 = nn.BatchNorm1d(output)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output, output)
        nn.init.kaiming_uniform_(self.fc2.weight)
        # self.bn2 = nn.BatchNorm1d(output)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu2(out)

        return out

class MLP_Group(nn.Module):
    def __init__(self, block, input, output, n):
        super(MLP_Group, self).__init__()
        self.layer = self.make_layer(block, input, output, n)
    def make_layer(self, block, input, output, n,):
        layers = []
        for i in range(int(n)):
            layers.append(block(i==0 and input or output, output))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


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
    def __init__(self, depth, num_classes, widen_factor, res_factor=5, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        fcnChannels = [3*res_factor, 2*res_factor, 120]
        assert ((depth-4)%6 == 0), 'depth should be 6n+4'
        n = (depth - 4) / 6
        self.h = int(cfg.INPUT_H / 2 / 2 / 8)
        self.w = int(cfg.INPUT_W / 2 / 2 / 8)
        self.nchannels = nChannels[3]
        block = BasicBlock
        MLPblock = MlpBlock
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
        self.cnnoutfc = nn.Linear(nChannels[3]*self.h*self.w, fcnChannels[2])
        nn.init.kaiming_uniform_(self.cnnoutfc.weight)
        # self.cnnoutbn = nn.BatchNorm1d(fcnChannels[0])
        self.cnnoutrelu = nn.ReLU()
        #Branch for fcn
        self.inputLayer = nn.Linear(cfg.INPUT_MULTI, fcnChannels[0])
        nn.init.kaiming_uniform_(self.inputLayer.weight)
        # self.inputbn = nn.BatchNorm1d(fcnChannels[0])
        self.inputrelu = nn.ReLU()
        #fcn resblock1
        # self.fcnblock1 = MlpBlock(fcnChannels[0], fcnChannels[1], 1)
        self.fcnblock1 = MLP_Group(MLPblock, fcnChannels[0], fcnChannels[1], 1)
        # fcn resblock2
        self.fcnblock2 = MLP_Group(MLPblock, fcnChannels[2] + fcnChannels[1], fcnChannels[1], 1)
        # fcv resblock3
        self.fcnblock3 = MLP_Group(MLPblock, fcnChannels[2] + 2 * fcnChannels[1], fcnChannels[1], 1)
        # outputlayer
        # self.outbn = nn.BatchNorm1d(fcnChannels[1])
        # self.outrelu = nn.ReLU()
        self.outLayer = nn.Linear(fcnChannels[1], num_classes)
    def forward(self, x1, x2):
        #Branch for Image
        out1 = self.conv1(x1)
        out1 = self.block1(out1)
        out1 = self.block2(out1)
        out1 = self.block3(out1)
        out1 = self.bn(out1)
        out1 = self.relu(out1)
        out1 = f.avg_pool2d(out1, 8)
        out1 = out1.view(-1, self.nchannels * self.w * self.h)
        out1 = self.cnnoutfc(out1)
        # out1 = self.cnnoutbn(out1)
        out1 = self.cnnoutrelu(out1)

        #Branch for Prozessparameter
        out2 = self.inputLayer(x2)
        # out2 = self.inputbn(out2)
        out2 = self.inputrelu(out2)
        out2 = self.fcnblock1(out2)
        # concate 1
        out3 = torch.cat((out1, out2), dim=1)
        out3 = self.fcnblock2(out3)
        # concate 2
        out4 = torch.cat((out1, out2, out3), dim=1)
        out4 = self.fcnblock3(out4)
        # out4 = self.outbn(out4)
        # out4 = self.outrelu(out4)
        out4 = self.outLayer(out4)

        return out4

# device = torch.device('cuda')
# criterion = nn.CrossEntropyLoss()
#     # viz = visdom.Visdom()

# model = WideResNet(16, cfg.NUM_TRANS, 8)
# print(model)
# x1 = torch.randn(64, 1, 64, 64)
# x2 = torch.randn(64, 2)
# print(x2.size())

# logits = model(x1, x2)
# print(logits.size())
