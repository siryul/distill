import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes))

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet32(nn.Module):

  def __init__(self, num_classes=10):
    super(ResNet32, self).__init__()
    self.in_planes = 16

    # 初始卷积层
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)

    # 3个stage，每个stage包含5个残差块
    self.layer1 = self._make_layer(16, 5, stride=1) # 32x32
    self.layer2 = self._make_layer(32, 5, stride=2) # 16x16
    self.layer3 = self._make_layer(64, 5, stride=2) # 8x8

    # 全连接层
    self.linear = nn.Linear(64, num_classes)

  def _make_layer(self, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(BasicBlock(self.in_planes, planes, stride))
      self.in_planes = planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.avg_pool2d(out, out.size()[3])
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


def resnet32(num_classes=10):
  return ResNet32(num_classes)
