import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import autoaugment


class BalancedSelfDistillation(nn.Module):

  def __init__(self, backbone, num_classes, alpha=1.0, lambda_bsd=1.0):
    super().__init__()
    self.backbone = backbone
    self.num_classes = num_classes
    self.alpha = alpha
    self.lambda_bsd = lambda_bsd

    # 定义弱数据增强和强数据增强
    self.weak_aug = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    self.strong_aug = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      autoaugment.RandAugment(num_ops=2, magnitude=10),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

  def forward(self, x):
    # 应用弱增强和强增强到每个图像
    batch_size = x.size(0)
    x_weak = torch.stack([self.weak_aug(x[i]) for i in range(batch_size)])
    x_strong = torch.stack([self.strong_aug(x[i]) for i in range(batch_size)])

    # 获取模型输出
    z_weak = self.backbone(x_weak)
    z_strong = self.backbone(x_strong)

    return z_weak, z_strong

  def compute_class_weights(self, class_counts):
    # 计算类别权重
    max_count = max(class_counts)
    weights = [(max_count / count)**self.alpha for count in class_counts]
    return torch.tensor(weights)

  def compute_bsd_loss(self, p_weak, p_strong, class_weights=None, labels=None):
    # 计算BSD损失
    kl_div = F.kl_div(F.log_softmax(p_strong, dim=1), F.softmax(p_weak, dim=1),
                      reduction='none').sum(dim=1)

    if class_weights is not None and labels is not None:
      # 应用类别权重
      sample_weights = class_weights[labels]
      kl_div = kl_div * sample_weights

    return kl_div.mean()

  def compute_total_loss(self, z_weak, z_strong, labels, class_counts):
    # 计算交叉熵损失
    ce_loss = 0.5 * (F.cross_entropy(z_weak, labels) + F.cross_entropy(z_strong, labels))

    # 计算类别权重
    class_weights = self.compute_class_weights(class_counts)

    # 计算BSD损失
    bsd_loss = self.compute_bsd_loss(z_weak, z_strong, class_weights, labels)

    # 计算总损失
    total_loss = ce_loss + self.lambda_bsd * bsd_loss

    return total_loss, ce_loss, bsd_loss
