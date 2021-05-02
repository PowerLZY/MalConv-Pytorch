import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
print(torch.__version__)
print(torch.cuda.is_available())

print(1)
# Decide which device we want to run on

