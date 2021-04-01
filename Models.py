import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups = groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias = False)

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes*(base_width / 64.))*groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes* self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)

        return out

class AutoNet(nn.Module):

    def __init__(self, block, layers, in_channels = 3, out_channels = 3, auto = True, num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64, norm_layer = None):
        super(AutoNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.auto = auto

        self.conv = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        #Decoder
        if auto:
            self.deconv = nn.ConvTranspose2d(self.inplanes, out_channels, kernel_size=8, stride=2, padding=3, bias=False)
            self.unpool = nn.MaxUnpool2d(kernel_size=2)
            self.layer1_inverse = self._make_layer_inverse(block, 64, layers[0])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups = self.groups, base_width = self.base_width, dilation = self.dilation, norm_layer = norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_inverse(self, block, planes, blocks, stride = 1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        layers = []
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, groups = self.groups, base_width = self.base_width, dilation = self.dilation, norm_layer = norm_layer))

        layers.append(conv1x1(inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        rec = None

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, idx = self.maxpool(x)

        x = self.layer1(x)


        if self.auto:
            rec = self.layer1_inverse(x)
            rec = self.unpool(rec, idx)
            rec = self.deconv(rec)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, rec

    def forward(self, x):
        return self._forward_impl(x)
