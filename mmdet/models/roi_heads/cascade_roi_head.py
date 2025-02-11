# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
import copy
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)
from ..utils.misc import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from .feature_attention import attention
from .contractive_loss import SupConLoss


class Proposal_Sampler:
    def __init__(self, bbox_assigner, bbox_sampler, batch_gt_instances):
        # 初始化类，传入区分正负样本的方法，以及所有的gt（包括正常和异常细胞）
        self.bbox_assigner = bbox_assigner
        self.bbox_sampler = bbox_sampler
        self.batch_gt_instances_abnormal = batch_gt_instances["batch_gt_instances_abnormal"]
        self.batch_gt_instances_ignore_abnormal = batch_gt_instances["batch_gt_instances_ignore_abnormal"]
        self.batch_gt_instances_normal = batch_gt_instances["batch_gt_instances_normal"]
        self.batch_gt_instances_ignore_normal = batch_gt_instances["batch_gt_instances_ignore_normal"]
        self.batch_gt_instances_all = batch_gt_instances["batch_gt_instances_all"]
        self.batch_gt_instances_ignore_all = batch_gt_instances["batch_gt_instances_ignore_all"]

    def process_stages(self, results_list, num_imgs, x, choice="abnormal", stage=None):
        # assign gts and sample proposals
        bbox_assigner = self.bbox_assigner
        bbox_sampler = self.bbox_sampler
        if choice == "abnormal":
            batch_gt_instances = self.batch_gt_instances_abnormal
            batch_gt_instances_ignore = self.batch_gt_instances_ignore_abnormal
        elif choice == "normal":
            batch_gt_instances = self.batch_gt_instances_normal
            batch_gt_instances_ignore = self.batch_gt_instances_ignore_normal
        elif choice == "all":
            batch_gt_instances = self.batch_gt_instances_all
            batch_gt_instances_ignore = self.batch_gt_instances_ignore_all

        sampling_results = []
        for i in range(num_imgs):
            results = copy.deepcopy(results_list[i])
            results.priors = results.bboxes
            del results.bboxes
            assign_result = bbox_assigner.assign(
                results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])  # 计算每个prior的IOU，并将每个prior划分正负样本
            # gt_ind代表bbox是第几个gt（从1开始，0代表不是gt），labels代表bbox的正确label（-1代表不是gt）
            sampling_result = bbox_sampler.sample(
                assign_result,
                results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])  # 从划分的正负样本中取出特定数量的priors
            if choice == 'all' and stage == 0:
                sampling_result.pos_gt_single_multi_labels = torch.gather(batch_gt_instances[i].single_multi_labels, 0,
                                                                          sampling_result.pos_assigned_gt_inds)
            if choice == 'abnormal' and stage == 2:
                sampling_result.pos_gt_single_multi_labels = torch.gather(batch_gt_instances[i].labels_temp, 0,
                                                                          sampling_result.pos_assigned_gt_inds)
            sampling_results.append(sampling_result)
        return sampling_results


@MODELS.register_module()
class CascadeRoIHead(BaseRoIHead):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: Union[List[float], Tuple[float]],
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head_abnormal: OptMultiConfig = None,
                 bbox_head_normal: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head_abnormal is not None
        assert bbox_head_normal is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head_abnormal=bbox_head_abnormal,
            bbox_head_normal=bbox_head_normal,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ICL_head = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )
        self.attention = attention()

    def init_bbox_head(self, bbox_roi_extractor: MultiConfig,
                       bbox_head_abnormal: MultiConfig, bbox_head_normal: MultiConfig, ) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of box roi extractor.
            bbox_head (:obj:`ConfigDict`, dict or list): Config
                of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head_abnormal = ModuleList()
        self.bbox_head_normal = ModuleList()

        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head_abnormal, list):
            bbox_head_abnormal = [bbox_head_abnormal for _ in range(self.num_stages)]

        if not isinstance(bbox_head_normal, list):
            bbox_head_normal = [bbox_head_normal for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head_abnormal) == self.num_stages

        for roi_extractor, head_abnormal in zip(bbox_roi_extractor, bbox_head_abnormal):
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.bbox_head_abnormal.append(MODELS.build(head_abnormal))

        for head_normal in bbox_head_normal:
            self.bbox_head_normal.append(MODELS.build(head_normal))

    def init_mask_head(self, mask_roi_extractor: MultiConfig,
                       mask_head: MultiConfig) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_head (dict): Config of mask in mask head.
            mask_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of mask roi extractor.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(MODELS.build(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(MODELS.build(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    TASK_UTILS.build(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    TASK_UTILS.build(
                        rcnn_train_cfg.sampler,
                        default_args=dict(context=self)))

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor, abnormal: bool = True) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head_list = self.bbox_head_abnormal if abnormal else self.bbox_head_normal
        bbox_head = bbox_head_list[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)  # ROL feature(num_img * 512, 256, 7, 7), 在5个feature map中取前4个，rpn取全部
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)  # TWO MLP head + (cls and bbox reg)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results

    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], abnormal: bool = True) -> [dict]:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
        """
        bbox_head_list = self.bbox_head_abnormal if abnormal else self.bbox_head_normal
        bbox_head = bbox_head_list[stage]

        rois = bbox2roi(
            [res.priors for res in sampling_results])  # 将image_id添加到bbox4列之前形成5列，并且batch进行拼接

        bbox_results = self._bbox_forward(stage, x, rois, abnormal=abnormal)  # 得到7*7的feature，以及每个分类回归参数
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])

        bbox_results.update(bbox_loss_and_target)
        return bbox_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
        """
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_preds = mask_head(mask_feats)

        mask_results = dict(mask_preds=mask_preds)
        return mask_results

    def mask_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_head = self.mask_head[stage]

        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage])
        mask_results.update(mask_loss_and_target)

        return mask_results

    def pre_opt_cl(self, stage, x, sampling_results):

        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        features = self.avgpool(bbox_feats).squeeze(-1).squeeze(-1)
        features = self.ICL_head(features)
        features = F.normalize(features, dim=1)

        pos_gt_cls_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_single_multi_labels_list = [res.pos_gt_single_multi_labels for res in sampling_results]
        if stage == 0:
            combined_list = []
            for a, b in zip(pos_gt_cls_labels_list, pos_gt_single_multi_labels_list):
                combined = a * 2 + b
                combined_list.append(combined)

            cls_labels = []
            sub_cls_labels = []
            for i in range(len(sampling_results)):
                num_pos = sampling_results[i].pos_priors.size(0)
                num_neg = sampling_results[i].neg_priors.size(0)
                num_samples = num_pos + num_neg
                label_value = 2 if stage == 0 else 11
                cls_label = rois.new_full((num_samples,), label_value,
                                          dtype=torch.long)  # 生成一个全2/11，长度为num_samples的tensor
                cls_label[:num_pos] = pos_gt_cls_labels_list[i]
                cls_labels.append(cls_label)

                sub_cls_label = rois.new_full((num_samples,), 4,
                                              dtype=torch.long)  # 生成一个全4，(a_s, a_m, n_s, n_m 四个类)，长度为num_samples的tensor
                sub_cls_label[:num_pos] = combined_list[i]
                sub_cls_labels.append(sub_cls_label)

            labels = [torch.cat(cls_labels, dim=0), torch.cat(sub_cls_labels, dim=0)]

        else:
            cls_labels = []
            sub_cls_labels = []
            for i in range(len(sampling_results)):
                num_pos = sampling_results[i].pos_priors.size(0)
                num_neg = sampling_results[i].neg_priors.size(0)
                num_samples = num_pos + num_neg
                label_value = 2 if stage == 0 else 11
                cls_label = rois.new_full((num_samples,), label_value,
                                          dtype=torch.long)  # 生成一个全2/11，长度为num_samples的tensor
                cls_label[:num_pos] = pos_gt_single_multi_labels_list[i]
                cls_labels.append(cls_label)

                sub_cls_label = rois.new_full((num_samples,), 22,
                                              dtype=torch.long)  # 生成一个全4/22，(a_s, a_m, n_s, n_m 四个类或a_s, a_m)，长度为num_samples的tensor
                sub_cls_label[:num_pos] = pos_gt_cls_labels_list[i]
                sub_cls_labels.append(sub_cls_label)

            labels = [torch.cat(cls_labels, dim=0), torch.cat(sub_cls_labels, dim=0)]
        return features, labels

    def contrastive(self, stage: int, namuda: List[Tensor], x: Tuple[Tensor], sampling_results: List[SamplingResult], ):
        features, labels = self.pre_opt_cl(stage, x, sampling_results)
        cumulative_loss = torch.tensor(0.0).to(features.device)
        for l in range(0, len(labels)):
            criterion = SupConLoss(stage=stage, layer=l, penalty=(l+1))
            layer_loss = criterion(features, labels)
            cumulative_loss += namuda[0] * layer_loss if l==1 else layer_loss
        return cumulative_loss * 0.1

    def process_batch_data(self, batch_data_samples: SampleList):
        """

        Args:
            batch_data_samples: 真实的label标签，但是正常细胞和异常细胞混合再一起，每个细胞还带有单个和细胞团分类

        Returns:
            batch_data_sample_normal代表正常细胞数据集，第一个阶段标签为单细胞和细胞团，第二个阶段才详细分病变类别
            batch_data_sample_abnormal代表异常细胞数据集，只分为单细胞和细胞团
        """

        device = batch_data_samples[0].gt_instances.bboxes.device
        batch_data_sample_normal = []
        batch_data_sample_abnormal = []
        batch_data_sample_all = []

        for det_data in batch_data_samples:
            # 先复制 DetDataSample 实例
            data_sample_normal = copy.deepcopy(det_data)
            data_sample_abnormal = copy.deepcopy(det_data)
            data_sample_all = copy.deepcopy(det_data)

            # 先删除，再填充
            del data_sample_normal.gt_instances.bboxes, data_sample_normal.gt_instances.labels, data_sample_normal.gt_instances.single_or_mul
            del data_sample_abnormal.gt_instances.bboxes, data_sample_abnormal.gt_instances.labels, data_sample_abnormal.gt_instances.single_or_mul
            del data_sample_all.gt_instances.labels, data_sample_all.gt_instances.single_or_mul

            # 创建布尔掩码，找到 label != 11 的位置
            label_mask = det_data.gt_instances.labels != 11

            # 更新正常细胞 (label == 11)
            labels_normal = det_data.gt_instances.single_or_mul[~label_mask].to(device=device)
            bboxes_normal = det_data.gt_instances.bboxes[~label_mask].to(device=device)

            # 更新异常细胞 (label != 11)
            labels_abnormal_single_multi = det_data.gt_instances.single_or_mul[label_mask].to(device=device)
            labels_abnormal_temp = det_data.gt_instances.labels[label_mask].to(device=device)
            bboxes_abnormal = det_data.gt_instances.bboxes[label_mask].to(device=device)

            # 更新全部细胞 (label != 11)
            labels_all = torch.zeros_like(det_data.gt_instances.labels).to(device=device)
            labels_all[det_data.gt_instances.labels == 11] = 1

            if len(labels_normal) == 0:
                labels_normal = torch.empty(0, 1, device=device)
                bboxes_normal = torch.empty(0, 4, device=device)
            else:
                pass
            # 将字典添加到gt_instances中
            data_sample_normal.gt_instances.labels = labels_normal
            data_sample_normal.gt_instances.bboxes = bboxes_normal

            data_sample_abnormal.gt_instances.labels = labels_abnormal_single_multi
            data_sample_abnormal.gt_instances.labels_temp = labels_abnormal_temp
            data_sample_abnormal.gt_instances.bboxes = bboxes_abnormal

            data_sample_all.gt_instances.labels = labels_all
            data_sample_all.gt_instances.single_multi_labels = det_data.gt_instances.single_or_mul

            # 将创建的实例添加到列表中
            batch_data_sample_normal.append(data_sample_normal)
            batch_data_sample_abnormal.append(data_sample_abnormal)
            batch_data_sample_all.append(data_sample_all)
        return batch_data_sample_abnormal, batch_data_sample_normal, batch_data_sample_all

    def tok_num_matching(self, top_num, stage0_abnormal_results, stage0_normal_results):
        """
        Args:
            top_num:
            stage1_abnormal_results:
            stage1_normal_results:

        Returns:
            根据pred的值大小进行匹配，找到每类top_num个feature 的索引，为交叉注意力做准备，注意是在图片维度寻找top_num，不是在batch维度
        """
        abnormal_pred_cls = [stage0_abnormal_results[i]['scores'] for i in range(len(stage0_abnormal_results))]
        normal_pred_cls = [stage0_normal_results[i]['scores'] for i in range(len(stage0_normal_results))]
        abnormal_max_values, abnormal_max_indices = map(list, zip(*[torch.max(abnormal_pred_cls[i], dim=1) for i in
                                                                    range(len(abnormal_pred_cls))]))  # 找到每行最大值以及最大值所在位置
        normal_max_values, normal_max_indices = map(list, zip(*[torch.max(normal_pred_cls[i], dim=1) for i in
                                                                range(len(normal_pred_cls))]))
        # 初始化输出tensor为-1
        # abnormal_init_tensor = [[-1] * len(tensor) for tensor in abnormal_max_values]
        # normal_init_tensor = torch.full_like(normal_max_values[0], -1)
        # 遍历每一类(0, 1),背景类除外
        abnormal_topk_tensor = []
        normal_topk_tensor = []
        for i in range(len(abnormal_max_values)):
            abnormal_init_tensor = torch.full_like(abnormal_max_values[i], -1)
            normal_init_tensor = torch.full_like(normal_max_values[i], -1)
            for class_idx in range(abnormal_pred_cls[0].shape[1]-1):
                # 找到每行最大值的相同类别
                abnormal_class_indices = torch.nonzero(abnormal_max_indices[i] == class_idx,
                                                       as_tuple=False)  # 找到abnormal_max_indices中相同类别的位置
                normal_class_indices = torch.nonzero(normal_max_indices[i] == class_idx, as_tuple=False)
                # 如果当前类别的数量大于top_num，保留最大的两个值对应的索引，其余置为-1
                if len(abnormal_class_indices) > top_num:
                    abnormal_temp = abnormal_max_values[i][abnormal_class_indices]  # 找到相同类别的值大小
                    _, topk_indices = torch.topk(abnormal_temp.view(-1), k=top_num)  # 找到top_k个值所在位置
                    abnormal_temp2 = abnormal_class_indices[topk_indices]
                    abnormal_init_tensor[abnormal_temp2] = class_idx
                else:
                    abnormal_init_tensor[abnormal_class_indices] = class_idx

                if len(normal_class_indices) > top_num:
                    normal_temp = normal_max_values[i][normal_class_indices]
                    _, topk_indices = torch.topk(normal_temp.view(-1), k=top_num)
                    normal_temp2 = normal_class_indices[topk_indices]
                    normal_init_tensor[normal_temp2] = class_idx
                else:
                    normal_init_tensor[normal_class_indices] = class_idx
            abnormal_topk_tensor.append(abnormal_init_tensor)
            normal_topk_tensor.append(normal_init_tensor)

        for i in range(len(stage0_abnormal_results)):
            stage0_abnormal_results[i].top_ind = abnormal_topk_tensor[i]
            stage0_normal_results[i].top_ind = normal_topk_tensor[i]

        return stage0_abnormal_results, stage0_normal_results

    def change_order(self, stage0_results):
        predict_labels = [temp.scores for temp in stage0_results]
        refine_bboxes = [temp.priors for temp in stage0_results]
        topk = [temp.top_ind for temp in stage0_results]
        # prop_nums = [len(stage0_results[i]) for i in range(len(stage0_results))]
        return dict(predict_labels=predict_labels,
                    stage0_refine_bboxes=refine_bboxes,
                    top_ind=topk)

    def stage1_cls_reg(self, abnormal_bbox_head, normal_bbox_head, sampling_results_abnormal, sampling_results_normal,
                       abnormal_feature, normal_feature, n_a_feature):
        stage = 1
        abnormal_bbox_head = abnormal_bbox_head[stage]
        normal_bbox_head = normal_bbox_head[stage]

        abnormal_cls_score, abnormal_bbox_pred = abnormal_bbox_head(abnormal_feature)  # TWO MLP head + (cls and bbox reg)
        normal_cls_score, normal_bbox_pred = normal_bbox_head(normal_feature)
        n_a_cls_score, n_a_bbox_pred = abnormal_bbox_head(n_a_feature)  # n_a_feature相当于abnormal（k,v）加上normal(q) 的噪声

        abnormal_bbox_loss_targets = abnormal_bbox_head.loss_and_target(
            cls_score=abnormal_cls_score,
            bbox_pred=abnormal_bbox_pred,
            rois=None,
            sampling_results=sampling_results_abnormal,
            rcnn_train_cfg=self.train_cfg[stage])

        normal_bbox_loss_targets = normal_bbox_head.loss_and_target(
            cls_score=normal_cls_score,
            bbox_pred=normal_bbox_pred,
            rois=None,
            sampling_results=sampling_results_normal,
            rcnn_train_cfg=self.train_cfg[stage])

        n_a_bbox_loss_targets = abnormal_bbox_head.loss_and_target(
            cls_score=n_a_cls_score,
            bbox_pred=n_a_bbox_pred,
            rois=None,
            sampling_results=sampling_results_abnormal,
            rcnn_train_cfg=self.train_cfg[stage])

        stage1_abnormal_results = dict(cls_score=abnormal_cls_score,
                                       bbox_pred=abnormal_bbox_pred, bbox_feats=abnormal_feature,
                                       loss_bbox=abnormal_bbox_loss_targets['loss_bbox'],
                                       bbox_targets=abnormal_bbox_loss_targets['bbox_targets'])

        stage1_losses = dict(abnormal=abnormal_bbox_loss_targets['loss_bbox'],
                             normal=normal_bbox_loss_targets['loss_bbox'],
                             n_a=n_a_bbox_loss_targets['loss_bbox'])

        return stage1_abnormal_results, stage1_losses

    def stage0_result(self, batch_gt_instances, rpn_stage1, num_imgs, x, batch_img_metas, namuda):
        stage = 0
        bbox_assigner = self.bbox_assigner[stage]
        bbox_sampler = self.bbox_sampler[stage]

        proposal_sampler = Proposal_Sampler(bbox_assigner=bbox_assigner,
                                            bbox_sampler=bbox_sampler,
                                            batch_gt_instances=batch_gt_instances,
                                            )
        # Call the process_stages method with your specific parameters

        sampling_results_abnormal = proposal_sampler.process_stages(
            results_list=rpn_stage1,
            num_imgs=num_imgs,
            x=x,
            choice="abnormal",
            stage=stage
        )

        sampling_results_normal = proposal_sampler.process_stages(
            results_list=rpn_stage1,
            num_imgs=num_imgs,
            x=x,
            choice="normal",
            stage=stage
        )

        sampling_results_all = proposal_sampler.process_stages(
            results_list=rpn_stage1,
            num_imgs=num_imgs,
            x=x,
            choice="all",
            stage=stage
        )

        bbox_results_abnormal = self.bbox_loss(stage, x, sampling_results_abnormal, abnormal=True)
        bbox_results_normal = self.bbox_loss(stage, x, sampling_results_normal, abnormal=False)

        stage0_contractive_loss = self.contrastive(stage, namuda, x, sampling_results_all)

        with torch.no_grad():
            stage0_refine_abnormal = self.bbox_head_abnormal[stage].refine_bboxes(
                sampling_results_abnormal, bbox_results_abnormal,
                batch_img_metas)  # 将gt去除，并将bbox根据预测值进行回归, 进入下一个阶段只有bbox坐标，没有标签
            stage0_refine_normal = self.bbox_head_normal[stage].refine_bboxes(
                sampling_results_normal, bbox_results_normal,
                batch_img_metas)  # 将gt去除，并将bbox根据预测值进行回归, 进入下一个阶段只有bbox坐标，没有标签

        stage0_losses = dict(contractive_loss=stage0_contractive_loss, abnormal=bbox_results_abnormal['loss_bbox'],
                             normal=bbox_results_normal['loss_bbox'])
        return stage0_refine_abnormal, stage0_refine_normal, stage0_losses

    def stage1_result(self, batch_gt_instances, stage0_abnormal_results, stage0_normal_results, num_imgs, x,
                      batch_img_metas):
        stage = 1
        top_num = 3  # 在正常/异常细胞中找到最高的3个个进行交叉

        bbox_assigner = self.bbox_assigner[stage]
        bbox_sampler = self.bbox_sampler[stage]

        for i in range(len(batch_gt_instances['batch_gt_instances_abnormal'])):
            # 获取当前 InstanceData 对象的 labels_stage2 和 labels
            labels_temp = batch_gt_instances['batch_gt_instances_abnormal'][i].labels_temp
            labels_single_multi = batch_gt_instances['batch_gt_instances_abnormal'][i].labels
            labels = torch.where(labels_single_multi == 1, labels_temp + 11, labels_temp)
            # 更新 labels_temp 和 labels
            batch_gt_instances['batch_gt_instances_abnormal'][i].labels_temp = labels_temp
            batch_gt_instances['batch_gt_instances_abnormal'][i].labels = labels

        proposal_sampler = Proposal_Sampler(bbox_assigner=bbox_assigner,
                                            bbox_sampler=bbox_sampler,
                                            batch_gt_instances=batch_gt_instances,
                                            )
        # Call the process_stages method with your specific parameters
        sampling_results_abnormal = proposal_sampler.process_stages(
            results_list=stage0_abnormal_results,
            num_imgs=num_imgs,
            x=x,
            choice="abnormal",
            stage=stage
        )

        sampling_results_normal = proposal_sampler.process_stages(
            results_list=stage0_normal_results,
            num_imgs=num_imgs,
            x=x,
            choice="normal",
            stage=stage
        )

        stage_temp_abnormal_results = []
        stage_temp_normal_results = []

        for i in range(num_imgs):  # 由于选正样本的过程会使样本减少，会打乱priors的顺序, 故还需根据筛选结果选stage0中的scores
            abnormal_score_ind = torch.cat((sampling_results_abnormal[i].pos_inds, sampling_results_abnormal[i].neg_inds), dim=0)
            normal_score_ind = torch.cat((sampling_results_normal[i].pos_inds, sampling_results_normal[i].neg_inds),dim=0)

            abnoraml_scores_temp = torch.index_select(stage0_abnormal_results[i].scores, dim=0, index=abnormal_score_ind)
            noraml_scores_temp = torch.index_select(stage0_normal_results[i].scores, dim=0, index=normal_score_ind)

            stage_temp_abnormal_result = InstanceData(priors=sampling_results_abnormal[i].priors, scores=abnoraml_scores_temp)
            stage_temp_normal_result = InstanceData(priors=sampling_results_normal[i].priors, scores=noraml_scores_temp)
            stage_temp_abnormal_results.append(stage_temp_abnormal_result)
            stage_temp_normal_results.append(stage_temp_normal_result)

        stage0_abnormal_results, stage0_normal_results = \
            self.tok_num_matching(top_num, stage_temp_abnormal_results, stage_temp_normal_results)

        stage0_abnormal_results = self.change_order(stage0_abnormal_results)
        stage0_normal_results = self.change_order(stage0_normal_results)

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        abnormal_rois = bbox2roi(stage0_abnormal_results["stage0_refine_bboxes"])  # 根据roi坐标形成feature map
        normal_rois = bbox2roi(stage0_normal_results["stage0_refine_bboxes"])
        abnormal_bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], abnormal_rois)
        normal_bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], normal_rois)

        stage0_abnormal_results["roi_feature"] = abnormal_bbox_feats
        stage0_normal_results["roi_feature"] = normal_bbox_feats

        abnormal_self_attn_feature, normal_self_attn_feature, n_a_cross_feature \
            = self.attention(stage0_abnormal_results, stage0_normal_results)  # 将feature进行交叉注意力和自注意力

        stage1_abnormal_results, stage1_losses = \
            self.stage1_cls_reg(self.bbox_head_abnormal, self.bbox_head_normal,
                                sampling_results_abnormal, sampling_results_normal,
                                abnormal_self_attn_feature, normal_self_attn_feature,
                                n_a_cross_feature)  # 将feature map通过预测头进行分类回归

        stage1_abnormal_results.update(rois=abnormal_rois)
        with torch.no_grad():
            stage1_abnormal_results = self.bbox_head_abnormal[stage].refine_bboxes(
                sampling_results_abnormal, stage1_abnormal_results,
                batch_img_metas)

        return stage1_abnormal_results, stage1_losses

    def stage2_result(self, batch_gt_instances, stage1_abnormal_result, num_imgs, x, namuda):
        stage = 2
        bbox_assigner = self.bbox_assigner[stage]
        bbox_sampler = self.bbox_sampler[stage]

        proposal_sampler = Proposal_Sampler(bbox_assigner=bbox_assigner,
                                            bbox_sampler=bbox_sampler,
                                            batch_gt_instances=batch_gt_instances,
                                            )
        # Call the process_stages method with your specific parameters
        sampling_results_abnormal = proposal_sampler.process_stages(
            results_list=stage1_abnormal_result,
            num_imgs=num_imgs,
            x=x,
            choice="abnormal",
            stage=stage
        )

        bbox_results_abnormal = self.bbox_loss(stage, x, sampling_results_abnormal, abnormal=True)
        stage2_contractive_loss = self.contrastive(stage, namuda, x, sampling_results_abnormal)

        return dict(contractive_loss=stage2_contractive_loss, abnormal=bbox_results_abnormal['loss_bbox'])

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data_true samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        batch_data_samples_abnormal, batch_data_samples_normal, batch_data_samples_all = self.process_batch_data(
            batch_data_samples)
        # TODO: May add a new function in baseroihead
        assert len(rpn_results_list) == len(batch_data_samples_abnormal)

        batch_gt_instances_abnormal, batch_gt_instances_ignore_abnormal, batch_img_metas_abnormal = unpack_gt_instances(
            batch_data_samples_abnormal)
        batch_gt_instances_normal, batch_gt_instances_ignore_normal, batch_img_metas_normal = unpack_gt_instances(
            batch_data_samples_normal)
        batch_gt_instances_all, batch_gt_instances_ignore_all, batch_img_metas_all = unpack_gt_instances(
            batch_data_samples_all)

        batch_gt_instances = {
            'batch_gt_instances_abnormal': batch_gt_instances_abnormal,
            'batch_gt_instances_ignore_abnormal': batch_gt_instances_ignore_abnormal,
            'batch_gt_instances_normal': batch_gt_instances_normal,
            'batch_gt_instances_ignore_normal': batch_gt_instances_ignore_normal,
            "batch_gt_instances_all": batch_gt_instances_all,
            'batch_gt_instances_ignore_all': batch_gt_instances_ignore_all,
        }
        num_imgs = len(batch_data_samples)
        losses = dict()
        namuda = torch.tensor([3], dtype=torch.float32).to(rpn_results_list[0].bboxes.device)

        rpn_stage1 = copy.deepcopy(rpn_results_list)
        try:
            if rpn_stage1[0].bboxes.shape[0] == 0:
                img_path = batch_img_metas_abnormal[0]['img_path']
                raise ValueError(f"Shape of tensor is [0, 4]. The img id is :{img_path}")
        except ValueError as e:
            print("Error:", e)
        # print("namuda :", namuda)
        # stage 0 :区分出ab_s;ab_m; n_s; n_m
        stage0_abnormal_result, stage0_normal_result, stage0_losses = \
            self.stage0_result(batch_gt_instances, rpn_stage1, num_imgs, x, batch_img_metas_abnormal, namuda)

        stage1_abnormal_result, stage1_losses = \
            self.stage1_result(batch_gt_instances, stage0_abnormal_result, stage0_normal_result, num_imgs, x, batch_img_metas_abnormal)

        stage2_losses = self.stage2_result(batch_gt_instances, stage1_abnormal_result, num_imgs, x, namuda)

        # 遍历 stage0_losses，stage1_losses，stage2_losses 同时将每个loss放入losses字典中
        for stage, loss in enumerate([stage0_losses, stage1_losses, stage2_losses]):
            for key, value in loss.items():
                if key != "contractive_loss":
                    for sub_key, sub_value in value.items():
                        losses[f's{stage}_{key}_{sub_key}'] = sub_value * self.stage_loss_weights[stage]
                else:
                    losses[f's{stage}_{key}'] = value * self.stage_loss_weights[stage]
        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     **kwargs) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)  # 在roi框前面加上图片序列

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head[-1].predict_box_type,
                num_classes=self.bbox_head[-1].num_classes,
                score_per_cls=rcnn_test_cfg is None)

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img,
            **kwargs)

        results_list = self.bbox_head_abnormal[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=rcnn_test_cfg)
        return results_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: List[InstanceData],
                     rescale: bool = False) -> List[InstanceData]:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, mask_rois)
            mask_preds = mask_results['mask_preds']
            # split batch mask prediction back to each image
            mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
            aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        merged_masks = []
        for i in range(len(batch_img_metas)):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
            merged_masks.append(merged_mask)
        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)
        return results_list

    def _refine_roi(self, x: Tuple[Tensor], rois: Tensor,
                    batch_img_metas: List[dict],
                    num_proposals_per_img: Sequence[int], **kwargs) -> tuple:
        """Multi-stage refinement of RoI.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): shape (n, 5), [batch_ind, x1, y1, x2, y2]
            batch_img_metas (list[dict]): List of image information.
            num_proposals_per_img (sequence[int]): number of proposals
                in each image.

        Returns:
            tuple:

               - rois (Tensor): Refined RoI.
               - cls_scores (list[Tensor]): Average predicted
                   cls score per image.
               - bbox_preds (list[Tensor]): Bbox branch predictions
                   for the last stage of per image.
        """
        # "ms" in variable names means multi-stage

        ms_scores = []
        for stage in range(self.num_stages):
            if stage == 1:
                bbox_roi_extractor = self.bbox_roi_extractor[stage]
                bbox_head = self.bbox_head_abnormal[stage]
                bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
                bbox_feats = self.attention(bbox_feats)
                cls_score, bbox_pred = bbox_head(bbox_feats)
                bbox_results = dict(
                    cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
            else:
                bbox_results = self._bbox_forward(stage=stage, x=x, rois=rois, **kwargs)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']

            rois = rois.split(num_proposals_per_img, 0)  # 将每张图片的roi（1000，5）分为一个元组
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            ms_scores.append(cls_scores)

            # some detector with_reg is False, bbox_preds will be None
            if bbox_preds is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head_abnormal[stage].bbox_pred_split(
                        bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None,) * len(batch_img_metas)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head_abnormal[stage]
                if bbox_head.custom_activation:
                    cls_scores = [
                        bbox_head.loss_cls.get_activation(s)
                        for s in cls_scores
                    ]
                refine_rois_list = []
                for i in range(len(batch_img_metas)):
                    if rois[i].shape[0] > 0:
                        bbox_label = cls_scores[i][:, :-1].argmax(dim=1)  # 排除背景类，得到某个proposal的类别标签
                        # Refactor `bbox_head.regress_by_class` to only accept
                        # box tensor without img_idx concatenated.
                        refined_bboxes = bbox_head.regress_by_class(
                            rois[i][:, 1:], bbox_label, bbox_preds[i],
                            batch_img_metas[i])
                        refined_bboxes = get_box_tensor(refined_bboxes)  # 确认refined_bboxes是否为tensor
                        refined_rois = torch.cat(
                            [rois[i][:, [0]], refined_bboxes], dim=1)  # 将图片序列加入refined_bboxes的第一列
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by last two stages
        cls_scores = [(ms_scores[1][0] + ms_scores[2][0]) / float(2)]
        # cls_scores = [(ms_scores[2][0])]
        return rois, cls_scores, bbox_preds

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x, rois, batch_img_metas, num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)
        # mask head
        if self.with_mask:
            aug_masks = []
            rois = torch.cat(rois)
            for stage in range(self.num_stages):
                mask_results = self._mask_forward(stage, x, rois)
                mask_preds = mask_results['mask_preds']
                mask_preds = mask_preds.split(num_proposals_per_img, 0)
                aug_masks.append([m.sigmoid().detach() for m in mask_preds])

            merged_masks = []
            for i in range(len(batch_img_metas)):
                aug_mask = [mask[i] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
                merged_masks.append(merged_mask)
            results = results + (merged_masks,)
        return results
