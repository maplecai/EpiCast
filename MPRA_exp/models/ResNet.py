import torch.nn as nn
import torch
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# UTR detector 专用 
# 分类模型
class ResNet(nn.Module):
    def __init__(self,
                 n_targets=1,
                 blocks_num=[1,1,1,1,1,1,1,1],
                 conv = [32,64,96,128,160,196],
                 block=BasicBlock,
                 include_top=True,
                 groups=1,
                 fc=[100,50],
                 width_per_group=64,
                 drop = 0.2
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = conv[0]
        self.conv_layer_num = len(conv)
        self.final_output_size = [250,125,63,32,16,8,4,2]

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv1d(4, self.in_channel, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, conv[0], blocks_num[0])
        self.layer2 = self._make_layer(block, conv[1], blocks_num[1], stride=2)

        if self.conv_layer_num > 2:
            self.layer3 = self._make_layer(block, conv[2], blocks_num[2], stride=2)
        if self.conv_layer_num > 3:
            self.layer4 = self._make_layer(block, conv[3], blocks_num[3], stride=2)
        if self.conv_layer_num > 4:
            self.layer5 = self._make_layer(block, conv[4], blocks_num[4], stride=2)
        if self.conv_layer_num > 5:
            self.layer6 = self._make_layer(block, conv[5], blocks_num[5], stride=2)
        if self.conv_layer_num > 6:
            self.layer7 = self._make_layer(block, conv[6], blocks_num[6], stride=2)
        if self.conv_layer_num > 7:
            self.layer8 = self._make_layer(block, conv[7], blocks_num[7], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(self.final_output_size[self.conv_layer_num-1]//7)  # output size = (1, 1)

            self.classifier = nn.Sequential(
                nn.Linear(conv[self.conv_layer_num-1] * block.expansion * (self.final_output_size[self.conv_layer_num-1]//7), fc[0]),
                nn.BatchNorm1d(fc[0]),
                nn.ReLU(inplace=True),
                nn.Linear(fc[0], fc[1]),
                nn.BatchNorm1d(fc[1]),
                nn.ReLU(inplace=True),
                nn.Linear(fc[1], n_targets),)
                #nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)###output_size:1000
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#output_size:500


        x = self.layer1(x)#output_size:500
        if self.conv_layer_num > 1:
            x = self.layer2(x)#output_size:250
        if self.conv_layer_num > 2:
            x = self.layer3(x)#output_size:125
        if self.conv_layer_num > 3:
            x = self.layer4(x)#output_size:63
        if self.conv_layer_num > 4:
            x = self.layer5(x)#output_size:32
        if self.conv_layer_num > 5:
            x = self.layer6(x)#output_size:16
        if self.conv_layer_num > 6:
            x = self.layer7(x)#output_size:8
        if self.conv_layer_num > 7:
            x = self.layer8(x)
        if self.conv_layer_num > 8:
            x = self.layer9(x)

        # x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.layer7(x)
        # x = self.layer8(x)
        # x = self.layer9(x)

        # if self.include_top:
        x = self.avgpool(x)
        reshape_x = x.view(x.size(0), x.size(1) * x.size(2))
        prediction = self.classifier(reshape_x)
        prediction = prediction.squeeze(-1)

        return prediction

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conv = [5,6,7,8]
    blocks_num = [1,1,1,1]
    input = torch.rand([2,4,80])
    target = torch.tensor([1, 1]).to(device)
    input = input.to(device)
    model = ResNet(conv=conv, blocks_num=blocks_num)
    model.to(device)
    output = model(input)
    criteron = nn.MSELoss()
    criteron(output, target)

