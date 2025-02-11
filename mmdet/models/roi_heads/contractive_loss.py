"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, stage=None, layer=None, penalty=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.stage = stage
        self.layer = layer
        self.penalty = penalty
    def weight_matrix_stage0(self, labels, weight_matrix, penalty):
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
        label_diff_abs = torch.abs(label_diff)

        weight_matrix[(label_diff_abs == 2) & (labels.unsqueeze(1) != 4) & (labels.unsqueeze(0) != 4)] = penalty

        return weight_matrix

    def weight_matrix_stage2(self, labels, weight_matrix, penalty):
        # 创建类别张量
        classes = (labels <= 10).to(torch.bool)
        # 同一个single或者multi类为2
        weight_matrix[classes[:, None] == classes] = penalty
        # 自己或者同一细分类为1
        weight_matrix[labels.unsqueeze(1) == labels] = 1

        return weight_matrix

    def forward(self, features, labels=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (features.device
                  if features.is_cuda
                  else torch.device('cpu'))
        lenght = labels[0].size(0) if len(labels) == 2 else labels.size(0)
        weight_matrix = torch.ones((lenght, lenght)).to(device)
        if self.stage == 0:
            if self.layer == 0:
                nonzero_mask = torch.where(labels[0] != 2, torch.tensor(1.), torch.tensor(0.)).unsqueeze(1)  # 背景为2，将2的置为0
            else:
                nonzero_mask = torch.where(labels[1] != 4, torch.tensor(1.), torch.tensor(0.)).unsqueeze(1)  # 背景为4，将4的置为0
                # 根据labels[1]更新权重矩阵
            weight_matrix = self.weight_matrix_stage0(labels[1], weight_matrix, self.namuda)

        else:
            if self.layer == 0:
                nonzero_mask = torch.where(labels[0] != 11, torch.tensor(1.), torch.tensor(0.)).unsqueeze(1)  # 背景为11，将11的置为0
            else:
                nonzero_mask = torch.where(labels[1] != 22, torch.tensor(1.), torch.tensor(0.)).unsqueeze(1)  # 背景为22，将22的置为0
                # 根据labels[1]更新权重矩阵
            weight_matrix = self.weight_matrix_stage2(labels[1], weight_matrix, self.namuda)

        labels = labels[self.layer]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()   # 和自己的转置做比较，对角线为1，而且相同值的位置相同为1,背景置为0
        mask = mask * nonzero_mask
        mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  #
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # 小trick, 减每行的最大值，一般在对角线上，因为自己与自己的相似度最大，
        # 既能将本身排除，也能防止数值过大，增加训练的稳定性

        logits_mask = (torch.ones(features.shape[0], features.shape[0]) - torch.eye(features.shape[0])).to(device)
        # 将自己与自己点乘的位置设为0，其余位置设为1
        mask = mask * logits_mask  # 与原先的mask相乘，将将自己与自己点乘的位置乘为0，其余位置不变（除自己和背景以外的正样本）
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 将所有值乘e^x，还需乘logits_mask，将自己排除，相当于负样本的损失
        exp_logits = exp_logits * weight_matrix   #
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # exp_logits：除自己以外的样本，通过sum将每一行值相加融合； logits：没有用任何mask
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # 将小于0的数置为1，分母不为0
        mean_log_prob_temp1 = (mask * log_prob)  # mask 为除了自己之外的所有正样本
        mean_log_prob_temp = mean_log_prob_temp1.sum(1)
        mean_log_prob_pos = mean_log_prob_temp / mask_pos_pairs  # 将损失之和除正对数量

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
#
