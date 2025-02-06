import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import BalancedSelfDistillation
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
from resnet import resnet32
from sampler import LongTailDistributionSampler


def get_class_counts(dataset):
  # 统计每个类别的样本数量
  labels = [y for _, y in dataset]
  return [labels.count(i) for i in range(max(labels) + 1)]


def evaluate(model, test_loader, device):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Evaluating'):
      data, target = data.to(device), target.to(device)

      # 前向传播（只使用弱增强的输出进行评估）
      z_weak, _ = model(data)

      # 获取预测结果
      _, predicted = torch.max(z_weak.data, 1)

      total += target.size(0)
      correct += (predicted == target).sum().item()

  accuracy = 100 * correct / total
  return accuracy


def train(model, train_loader, optimizer, device, class_counts):
  model.train()
  total_loss = 0
  total_ce_loss = 0
  total_bsd_loss = 0

  for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # 前向传播
    z_weak, z_strong = model(data)

    # 计算损失
    loss, ce_loss, bsd_loss = model.compute_total_loss(z_weak, z_strong, target, class_counts)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 累计损失
    total_loss += loss.item()
    total_ce_loss += ce_loss.item()
    total_bsd_loss += bsd_loss.item()

  # 计算平均损失
  avg_loss = total_loss / len(train_loader)
  avg_ce_loss = total_ce_loss / len(train_loader)
  avg_bsd_loss = total_bsd_loss / len(train_loader)

  return avg_loss, avg_ce_loss, avg_bsd_loss


def main():
  # 设置设备
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # 加载数据集
  transform = transforms.Compose([transforms.ToTensor()])

  train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  # 创建长尾分布采样器
  sampler = LongTailDistributionSampler(train_dataset, num_classes=10, imbalance_factor=100)
  train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler, num_workers=2)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

  # 获取类别样本数量
  class_counts = sampler.target_counts

  # 打印每个类别的样本数量
  print("\n类别样本数量统计:")
  for i, count in enumerate(class_counts):
    print(f"类别 {i}: {count} 个样本")
  print()

  # 初始化模型
  backbone = resnet32(num_classes=10)
  model = BalancedSelfDistillation(backbone=backbone, num_classes=10, alpha=1.0,
                                   lambda_bsd=1.0).to(device)

  # 初始化训练参数
  num_epochs = 200

  # 设置优化器和学习率调度器
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

  # 训练循环
  best_accuracy = 0.0

  for epoch in range(num_epochs):
    avg_loss, avg_ce_loss, avg_bsd_loss = train(model, train_loader, optimizer, device,
                                                class_counts)

    # 更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # 评估模型
    accuracy = evaluate(model, test_loader, device)

    # 更新最佳准确率
    best_accuracy = max(best_accuracy, accuracy)

    print(f'Epoch: {epoch+1}')
    print(f'Learning Rate: {current_lr:.6f}')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average CE Loss: {avg_ce_loss:.4f}')
    print(f'Average BSD Loss: {avg_bsd_loss:.4f}')
    print(f'Top-1 Accuracy: {accuracy:.2f}%')
    print(f'Best Top-1 Accuracy: {best_accuracy:.2f}%\n')


if __name__ == '__main__':
  main()
