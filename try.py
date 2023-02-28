import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import torch
#
print(torch.cuda.device_count())
print(torch.cuda.is_available())   # 检查cuda是否可用
print(torch.version.cuda)          # 查看cuda版本
print(torch.backends.cudnn.is_available())# 检查cudnn是否可用
print(torch.backends.cudnn.version())  # 查看cudnn版本

import tensorflow as tf
print(tf.test.is_gpu_available())