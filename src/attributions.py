# Integrated gradients applied to malware programs
import os
import time
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.util import *
import pandas as pd
from src.model import MalConv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.model import MalConv

from captum.attr import (
    IntegratedGradients,# 积分梯度
    LayerIntegratedGradients # embedding层积分梯度
)
label_path = rootPath + "/data/"
train_data_path =rootPath + "/data/one/"  # Training data
train_label_path = rootPath +'/data/example-train-label.csv' # Training label
valid_label_path = rootPath +'/data/example-valid-label.csv' # Validation Label

exp_name = 'IG'

### Parameter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = True              #
use_cpu = 1                # Number of cores to use for data loader
batch_size = 1          #
first_n_byte = 2000000     # First N bytes of a PE file as the input of MalConv (defualt: 2 million)

### output path
log_dir = curPath + '/log/'
pred_dir = curPath + '/pred/'
checkpoint_dir = curPath +'/checkpoint/'
log_file_path = log_dir+exp_name+'.log'
chkpt_acc_path = checkpoint_dir+exp_name+'.pt'
pred_path = pred_dir+exp_name+'.pred'
"""
df1 = pd.read_csv(label_path+'white.csv')
df2 = pd.read_csv(label_path+'black.csv')
dataset = merge(df1, df2, "dataset", label_path)

train, valid, train_label, valid_label = train_test_split(
    dataset['id'],
    dataset['labels'],
    test_size=0.2,
    stratify= dataset['labels'],
    random_state=100)

trainset = pd.DataFrame({'id':train, 'labels':train_label})
validset = pd.DataFrame({'id':valid, 'labels':valid_label})

trainset.to_csv(label_path + "example-train-label.csv", index=False, header= False, encoding="utf-8")
validset.to_csv(label_path + "example-valid-label.csv", index=False, header= False, encoding="utf-8")
"""
model = MalConv()
model.load_state_dict(torch.load('/home/lizy/ml/MalConv-Pytorch/checkpoint/malconv.pt'))
model.eval()

torch.manual_seed(123)
np.random.seed(123)

validloader = DataLoader(ExeDataset(['0a1ba0fea8bab5cabe474df91b7e44cd'], train_data_path, [1],first_n_byte),
                        batch_size=batch_size, shuffle=False, num_workers=use_cpu)
if use_gpu:
    model = model.cuda()

for _, val_batch_data in enumerate(validloader):
    cur_batch_size = val_batch_data[0].size(0)

    exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
    exe_input = Variable(exe_input.long(), requires_grad=False)
    baseline = torch.zeros(1, 2000000).cuda() if use_gpu else torch.zeros(1, 2000000)
    baseline = Variable(baseline.long(), requires_grad=False)

    lig = LayerIntegratedGradients(model, model.embed)
    attributions_ig = lig.attribute(exe_input, baseline, target=0)

    print('IG Attributions:', attributions_ig)
    break;