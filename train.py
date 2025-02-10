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
import os
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import argparse


def load_config(args):
  # 设置默认配置文件名
  cfg = args.cfg if args.cfg else f'cifar10_imb{args.imbalance_factor}.yaml'
  config_path = os.path.join('config', cfg)

  # 如果指定的配置文件不存在，使用默认配置
  if not os.path.exists(config_path):
    config_path = 'config/default.yaml'
    print(f'警告：配置文件 {cfg} 不存在，使用默认配置文件')

  # 加载配置
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

  # 使用命令行参数覆盖配置
  for key, value in vars(args).items():
    if value is not None:
      config[key] = value

  return config


def get_class_counts(dataset):
  # 统计每个类别的样本数量
  labels = [y for _, y in dataset]
  return [labels.count(i) for i in range(max(labels) + 1)]


def evaluate(model, test_loader, device):
  model.eval()
  correct = 0
  total = 0
  all_preds = []
  all_targets = []
  class_correct = {}
  class_total = {}

  with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Evaluating'):
      data, target = data.to(device), target.to(device)

      # 前向传播（只使用弱增强的输出进行评估）
      z_weak, _ = model(data)

      # 获取预测结果
      _, predicted = torch.max(z_weak.data, 1)

      total += target.size(0)
      correct += (predicted == target).sum().item()

      # 收集预测结果和真实标签
      all_preds.extend(predicted.cpu().numpy())
      all_targets.extend(target.cpu().numpy())

      # 统计每个类别的正确预测数和总样本数
      for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
        if t not in class_correct:
          class_correct[t] = 0
          class_total[t] = 0
        class_total[t] += 1
        if t == p:
          class_correct[t] += 1

  # 计算整体准确率
  accuracy = 100 * correct / total

  # 计算每个类别的准确率
  class_accuracies = {}
  for class_idx in class_total.keys():
    class_accuracies[class_idx] = 100 * class_correct[class_idx] / class_total[class_idx]

  # 根据测试集中的样本数量将类别分为many-shot（>100）、medium-shot（20-100）和few-shot（<20）
  many_shot_classes = []
  medium_shot_classes = []
  few_shot_classes = []

  for class_idx, count in class_total.items():
    if count > 100:
      many_shot_classes.append(class_idx)
    elif count >= 20:
      medium_shot_classes.append(class_idx)
    else:
      few_shot_classes.append(class_idx)

  # 计算many-shot、medium-shot和few-shot类别的平均准确率
  many_shot_acc = np.mean([class_accuracies[c] for c in many_shot_classes])
  medium_shot_acc = np.mean([class_accuracies[c] for c in medium_shot_classes])
  few_shot_acc = np.mean([class_accuracies[c] for c in few_shot_classes])

  # 计算每个类别的性能指标
  conf_matrix = confusion_matrix(all_targets, all_preds)
  class_report = classification_report(all_targets, all_preds, output_dict=True)

  return accuracy, conf_matrix, class_report, many_shot_acc, medium_shot_acc, few_shot_acc


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
  # 解析命令行参数
  parser = argparse.ArgumentParser(description='Training script for Balanced Self-Distillation')
  parser.add_argument('--cfg', type=str, help='配置文件名称，例如：cifar10_imb100.yaml')
  parser.add_argument('--device', type=str, help='Training device (cuda/cpu)')
  parser.add_argument('--num_classes', type=int, help='Number of classes')
  parser.add_argument('--imbalance_factor', type=int, help='Imbalance factor')
  parser.add_argument('--batch_size', type=int, help='Batch size')
  parser.add_argument('--num_workers', type=int, help='Number of workers')
  parser.add_argument('--num_epochs', type=int, help='Number of epochs')
  parser.add_argument('--initial_lr', type=float, help='Initial learning rate')
  parser.add_argument('--momentum', type=float, help='Momentum')
  parser.add_argument('--weight_decay', type=float, help='Weight decay')
  parser.add_argument('--alpha', type=float, help='Alpha parameter')
  parser.add_argument('--lambda_bsd', type=float, help='Lambda BSD parameter')
  parser.add_argument('--log_dir', type=str, help='Log directory')

  args = parser.parse_args()

  # 加载配置
  config = load_config(args)

  # 创建日志目录
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  log_dir = os.path.join(config['log_dir'], f'train_log_{timestamp}')
  os.makedirs(log_dir, exist_ok=True)

  # 设置设备
  device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

  # 加载数据集
  transform = transforms.Compose([transforms.ToTensor()])

  # 根据num_classes选择加载CIFAR10或CIFAR100数据集
  dataset_class = datasets.CIFAR100 if config['num_classes'] == 100 else datasets.CIFAR10
  train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
  test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)

  # 创建长尾分布采样器
  sampler = LongTailDistributionSampler(train_dataset,
                                        num_classes=config['num_classes'],
                                        imbalance_factor=config['imbalance_factor'])
  train_loader = DataLoader(train_dataset,
                            batch_size=config['batch_size'],
                            sampler=sampler,
                            num_workers=config['num_workers'])
  test_loader = DataLoader(test_dataset,
                           batch_size=config['batch_size'],
                           shuffle=False,
                           num_workers=config['num_workers'])

  # 获取类别样本数量
  class_counts = sampler.target_counts

  # 保存配置到JSON文件
  with open(os.path.join(log_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

  # 打印每个类别的样本数量
  print("\n类别样本数量统计:")
  for i, count in enumerate(class_counts):
    print(f"类别 {i}: {count} 个样本")
  print()

  # 初始化模型
  backbone = resnet32(num_classes=config['num_classes'])
  model = BalancedSelfDistillation(backbone=backbone,
                                   num_classes=config['num_classes'],
                                   alpha=config['alpha'],
                                   lambda_bsd=config['lambda_bsd']).to(device)

  # 设置优化器和学习率调度器
  optimizer = optim.SGD(model.parameters(),
                        lr=config['initial_lr'],
                        momentum=config['momentum'],
                        weight_decay=config['weight_decay'])
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 180], gamma=0.1)

  # 训练循环
  best_accuracy = 0.0
  training_log = []

  for epoch in range(config['num_epochs']):
    avg_loss, avg_ce_loss, avg_bsd_loss = train(model, train_loader, optimizer, device,
                                                class_counts)

    # 更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # 评估模型
    accuracy, conf_matrix, class_report, many_shot_acc, medium_shot_acc, few_shot_acc = evaluate(
      model, test_loader, device)

    # 更新最佳准确率
    best_accuracy = max(best_accuracy, accuracy)

    # 记录本轮训练信息
    epoch_info = {
      'epoch': epoch + 1,
      'learning_rate': current_lr,
      'avg_loss': avg_loss,
      'avg_ce_loss': avg_ce_loss,
      'avg_bsd_loss': avg_bsd_loss,
      'accuracy': accuracy,
      'best_accuracy': best_accuracy,
      'many_shot_accuracy': many_shot_acc,
      'medium_shot_accuracy': medium_shot_acc,
      'few_shot_accuracy': few_shot_acc,
      'confusion_matrix': conf_matrix.tolist(),
      'class_report': class_report
    }
    training_log.append(epoch_info)

    # 保存训练日志
    with open(os.path.join(log_dir, 'training_log.json'), 'w') as f:
      json.dump(training_log, f, indent=2)

    print(f'Epoch: {epoch+1}')
    print(f'Learning Rate: {current_lr:.6f}')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average CE Loss: {avg_ce_loss:.4f}')
    print(f'Average BSD Loss: {avg_bsd_loss:.4f}')
    print(f'Top-1 Accuracy: {accuracy:.2f}%')
    print(f'Many-shot Accuracy: {many_shot_acc:.2f}%')
    print(f'Medium-shot Accuracy: {medium_shot_acc:.2f}%')
    print(f'Few-shot Accuracy: {few_shot_acc:.2f}%')
    print(f'Best Top-1 Accuracy: {best_accuracy:.2f}%\n')

    # 每个epoch结束后保存当前混淆矩阵
    np.save(os.path.join(log_dir, f'confusion_matrix_epoch_{epoch+1}.npy'), conf_matrix)

  # 保存最终模型
  torch.save(model.state_dict(), os.path.join(log_dir, 'final_model.pth'))


if __name__ == '__main__':
  main()
