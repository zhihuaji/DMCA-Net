# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms.loading import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

from mmdet.datasets.coco_panoptic import CocoPanopticDataset
from mmdet.datasets.samplers.batch_sampler import AspectRatioBatchSampler
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadPanopticAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Resize
from mmdet.evaluation.metrics.coco_panoptic_metric import CocoPanopticMetric

# dataset settings
dataset_type = 'CocoPanopticDataset'
data_root = 'data_true/coco/'

# Example to use different file client
# Method 1: simply set the data_true root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data_true/': 's3://openmmlab/datasets/detection/',
#         'data_true/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadPanopticAnnotations, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=LoadPanopticAnnotations, backend_args=backend_args),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=CocoPanopticDataset,
        data_root=data_root,
        ann_file='annotations/panoptic_train2017.json',
        data_prefix=dict(
            img='train2017/', seg='annotations/panoptic_train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CocoPanopticDataset,
        data_root=data_root,
        ann_file='annotations/panoptic_val2017.json',
        data_prefix=dict(img='val2017/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoPanopticMetric,
    ann_file=data_root + 'annotations/panoptic_val2017.json',
    seg_prefix=data_root + 'annotations/panoptic_val2017/',
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(
#         type=CocoPanopticDataset,
#         data_root=data_root,
#         ann_file='annotations/panoptic_image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type=CocoPanopticMetric,
#     format_only=True,
#     ann_file=data_root + 'annotations/panoptic_image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_panoptic/test')
