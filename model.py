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
    device = x.device
    x_weak = torch.stack([self.weak_aug(x[i]) for i in range(batch_size)]).to(device)
    x_strong = torch.stack([self.strong_aug(x[i]) for i in range(batch_size)]).to(device)

    # 获取模型输出
    z_weak = self.backbone(x_weak)
    z_strong = self.backbone(x_strong)

    return z_weak, z_strong

  def compute_class_weights(self, class_counts):
    # 计算类别权重，使用对数平滑
    max_count = max(class_counts)
    total_samples = sum(class_counts)
    weights = [((max_count/count) * (total_samples/max_count))**self.alpha
               for count in class_counts]
    weights = torch.tensor(weights)
    # 归一化权重
    weights = weights / weights.sum() * len(class_counts)
    # 将权重移动到与模型相同的设备上
    weights = weights.to(self.backbone.linear.weight.device)
    return weights

  def compute_bsd_loss(self, p_weak, p_strong, class_weights=None, labels=None, temperature=2.0):
    # 使用温度参数软化概率分布
    p_weak_soft = F.softmax(p_weak / temperature, dim=1)
    p_strong_soft = F.softmax(p_strong / temperature, dim=1)

    # 计算双向KL散度
    kl_div_strong_to_weak = F.kl_div(F.log_softmax(p_strong / temperature, dim=1),
                                     p_weak_soft,
                                     reduction='none').sum(dim=1)
    kl_div_weak_to_strong = F.kl_div(F.log_softmax(p_weak / temperature, dim=1),
                                     p_strong_soft,
                                     reduction='none').sum(dim=1)

    # 综合双向KL散度
    kl_div = (kl_div_strong_to_weak+kl_div_weak_to_strong) * (temperature**2) / 2

    if class_weights is not None and labels is not None:
      # 应用动态类别权重
      sample_weights = class_weights[labels]
      kl_div = kl_div * sample_weights

    return kl_div.mean()

  def compute_total_loss(self, z_weak, z_strong, labels, class_counts):
    # 计算带权重的交叉熵损失
    class_weights = self.compute_class_weights(class_counts)
    ce_loss_weak = F.cross_entropy(z_weak, labels, weight=class_weights)
    ce_loss_strong = F.cross_entropy(z_strong, labels, weight=class_weights)
    ce_loss = 0.5 * (ce_loss_weak+ce_loss_strong)

    # 计算改进的BSD损失
    bsd_loss = self.compute_bsd_loss(z_weak, z_strong, class_weights, labels)

    # 使用动态权重组合损失
    lambda_t = self.lambda_bsd * (2.0 / (1.0 + torch.exp(-0.1 * torch.tensor(sum(class_counts)))))
    total_loss = ce_loss + lambda_t*bsd_loss

    return total_loss, ce_loss, bsd_loss
