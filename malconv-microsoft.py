import sys
import os
import time
from src.model import *
from src.util import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

label_path = "/public/malware_dataset/kaggle_microsoft_9_10000/"
train_data_path = label_path + "bytes/"  # Training data
train_label_path = label_path + "kaggle_microsoft_trainlabels.csv"  # Training label
#valid_label_path = label_path + "example-valid-label.csv"  # Validation Label

#name
exp_name = "malconv-classification"

# Parameter
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # single-GPU

use_gpu = True  #
use_cpu = 32  # Number of cores to use for data loader
display_step = 5  # Std output update rate during training 和 保存训练结果步长
test_step = 50  # Test per n step
learning_rate = 0.0001  #
max_step = 1000  # Number of steps to train
batch_size = 768  #
first_n_byte = (
    100000  # First N bytes of a PE file as the input of MalConv (defualt: 2 million)
)
window_size = 512  # Kernel size & stride for Malconv (defualt : 500)

# output path
log_dir = "/log/"
pred_dir = "/pred/"
checkpoint_dir = "/checkpoint/"
log_file_path = log_dir + exp_name + ".log"
chkpt_acc_path = checkpoint_dir + exp_name + "1000.pt"
pred_path = pred_dir + exp_name + ".pred"

df = pd.read_csv(train_label_path)
train, valid, train_label, valid_label = train_test_split(
    df["Id"],
    df["Class"],
    test_size=0.2,
    stratify=df["Class"],
    random_state=100,
)
"""
# Dataset preparation
class ExeDataset(Dataset):
	def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000):
		self.fp_list = fp_list
		self.data_path = data_path
		self.label_list = label_list
		self.first_n_byte = first_n_byte

	def __len__(self):
		return len(self.fp_list)

	def __getitem__(self, idx):
		try:
			with open(self.data_path + self.fp_list[idx],'rb') as f:
				tmp = [i+1 for i in f.read()[:self.first_n_byte]] # index 0 will be special padding index 每个值加一
				tmp = tmp+[0]*(self.first_n_byte-len(tmp))
		except:
			with open(self.data_path + self.fp_list[idx].lower(),'rb') as f:
				tmp = [i+1 for i in f.read()[:self.first_n_byte]]
				tmp = tmp+[0]*(self.first_n_byte-len(tmp))

		return np.array(tmp), np.array([self.label_list[idx]])
"""

trainset = pd.DataFrame({"id": train, "labels": train_label})
validset = pd.DataFrame({"id": valid, "labels": valid_label})
trainloader = DataLoader(
    ExeDataset(
        list(trainset["id"]), train_data_path, list(trainset["labels"]), first_n_byte
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=use_cpu,
    pin_memory=True,
)
validloader = DataLoader(
    ExeDataset(
        list(validset["id"]), train_data_path, list(validset["labels"]), first_n_byte
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=use_cpu,
    pin_memory=True,
)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:1" if USE_CUDA else "cpu")

malconv = MalConv(input_length=first_n_byte, window_size=window_size)
malconv = nn.DataParallel(malconv, device_ids=[1,2,3]) # multi-GPU

#malconv = MalConvBase(8, 4096, 128, 32)
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()
adam_optim = optim.Adam([{"params": malconv.parameters()}], lr=learning_rate)
sigmoid = nn.Sigmoid()

if use_gpu:
    malconv = malconv.to(device)
    bce_loss = bce_loss.to(device)
    sigmoid = sigmoid.to(device)

step_msg = "step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}s"
valid_msg = "step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}"
log_msg = "{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}"
history = {}
history["tr_loss"] = []
history["tr_acc"] = []
train_acc = []  # 保存训练结果

valid_best_acc = 0.0
total_step = 0
step_cost_time = 0
valid_idx = list(validset["id"])

while total_step < max_step:

    # Training
    for step, batch_data in enumerate(trainloader):
        start = time.time()

        adam_optim.zero_grad()

        cur_batch_size = batch_data[0].size(0)

        exe_input = batch_data[0].to(device) if use_gpu else batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        label = batch_data[1].to(device) if use_gpu else batch_data[1]
        label = Variable(label, requires_grad=False)
        label = label.squeeze() - 1

        pred = malconv(exe_input)
        loss = ce_loss(pred, label)
        loss.backward()
        adam_optim.step()

        _, predicted = torch.max(pred.data, 1)
        train_Macc = (label.cpu().data.numpy().astype(int) == (predicted.cpu().data.numpy()).astype(int)).sum().item()
        train_Macc = train_Macc / cur_batch_size

        if (step + 1) % display_step == 0:
            print("train：{}".format(train_Macc))

        total_step += 1
        # Interupt for validation
        if total_step % test_step == 0:
            break

    for step, val_batch_data in enumerate(validloader):

        cur_batch_size = val_batch_data[0].size(0)

        exe_input = val_batch_data[0].to(device) if use_gpu else val_batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        label = val_batch_data[1].to(device) if use_gpu else val_batch_data[1]
        label = Variable(label, requires_grad=False)
        label = label.squeeze() - 1

        pred = malconv(exe_input)
        loss = ce_loss(pred, label)
        # loss.backward()
        # adam_optim.step()

        _, predicted = torch.max(pred.data, 1)
        val_Macc = (label.cpu().data.numpy().astype(int) == (predicted.cpu().data.numpy()).astype(int)).sum().item()
        val_Macc = val_Macc / cur_batch_size

        if (step + 1) % display_step == 0:
            print("test：{}".format(val_Macc))



