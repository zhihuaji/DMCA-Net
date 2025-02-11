# DMCA-Net: Dual-Branch Multi-granularity Hierarchical Contrast and Cross-Attention Network for Cervical Abnormal Cell Detection
Our approach uses mmdetection, some modules and code refer to mmdetection(https://github.com/open-mmlab/mmdetection)

## Datasets
The additional annotation data will be fully open-sourced after the publication of our paper.

## Method

![2](https://github.com/user-attachments/assets/59a4f578-e86a-4f86-8df6-3713d5e81248)

Our overall framework is implemented in [mmdet/models/roi_heads/cascade_roi_head.py]. The implementation of the IPCA and HMWC modules are in [mmdet/models/roi_heads/feature_attention.py] and [mmdet/models/roi_heads/contractive_loss.py] respectively.


