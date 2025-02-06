import numpy as np
from torch.utils.data import Sampler


class LongTailDistributionSampler(Sampler):

  def __init__(self, dataset, num_classes, imbalance_factor=100):
    self.dataset = dataset
    self.num_classes = num_classes
    self.imbalance_factor = imbalance_factor

    # 获取数据集中的所有标签
    self.labels = [label for _, label in dataset]

    # 计算每个类别的目标样本数量
    # 头部类别（类别0）设置为5000个样本
    max_samples = 5000
    # 根据不平衡率计算指数衰减率
    decay_rate = np.power(1 / imbalance_factor, 1 / (num_classes-1))
    # 计算每个类别的目标样本数量
    self.target_counts = [int(max_samples * np.power(decay_rate, i)) for i in range(num_classes)]

    # 为每个类别创建索引列表
    self.class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(self.labels):
      self.class_indices[label].append(idx)

    # 计算采样权重
    self.weights = []
    for i in range(num_classes):
      # 对每个类别的样本进行权重计算
      if len(self.class_indices[i]) > 0:
        class_weight = self.target_counts[i] / len(self.class_indices[i])
      else:
        class_weight = 0
      self.weights.extend([class_weight] * len(self.class_indices[i]))

    self.weights = np.array(self.weights)
    if np.sum(self.weights) > 0:
      self.weights = self.weights / np.sum(self.weights)

  def __iter__(self):
    # 根据权重进行采样
    indices = np.random.choice(len(self.dataset),
                               size=sum(self.target_counts),
                               replace=True,
                               p=self.weights)
    return iter(indices.tolist())

  def __len__(self):
    return sum(self.target_counts)
