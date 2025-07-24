# DMCA-Net: Dual-Branch Multi-granularity Hierarchical Contrast and Cross-Attention Network for Cervical Abnormal Cell Detection
Our approach uses mmdetection, some modules and code refer to mmdetection(https://github.com/open-mmlab/mmdetection)

## Datasets
The additional annotation JSON file for the training set based on the Comparison Detector Datasets can be accessed via the following link:
Link: https://pan.baidu.com/s/1DecibQnxkZ38kFxTDboF3A
Extraction code: 7et4

## Method
<img width="1469" alt="image" src="https://github.com/user-attachments/assets/a09e35cf-35e8-467b-9a67-c227657c5f07" />

Our overall framework is implemented in [mmdet/models/roi_heads/cascade_roi_head.py]. The implementation of the IPCA and MHCL modules are in [mmdet/models/roi_heads/feature_attention.py] and [mmdet/models/roi_heads/contractive_loss.py] respectively.


