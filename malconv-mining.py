# coding: utf-8
import os
import time
import sys

curPath = os.path.abspath(os.path.dirname("__file__"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.util import *
import numpy as np
import pandas as pd
from src.model import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# 读取数
label_path = "/public/malware_dataset/mining-2-6000/data/"
train_data_path = label_path + "mining/"  # Training data
train_label_path = label_path + "example-train-label.csv"  # Training label
valid_label_path = label_path + "example-valid-label.csv"  # Validation Label

"""
服务器执行语法：
nohup /usr/local/anaconda3/bin/python3.7 /home/lizy/ml/MalConv-Pytorch/malconv-mining.py >> /home/lizy/ml/MalConv-Pytorch/log/malconv-mining-1000.log 2>&1 
cd 到MalConv-Pytorch下
"""
exp_name = "malconv"

### Parameter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_gpu = True  #
use_cpu = 32  # Number of cores to use for data loader
display_step = 10  # Std output update rate during training 和 保存训练结果步长
test_step = 100  # Test per n step
learning_rate = 0.0001  #
max_step = 500  # Number of steps to train
batch_size = 128  #
first_n_byte = (
    100000  # First N bytes of a PE file as the input of MalConv (defualt: 2 million)
)
window_size = 500  # Kernel size & stride for Malconv (defualt : 500)
### output path
log_dir = "/log/"
pred_dir = "/pred/"
checkpoint_dir = "/checkpoint/"
log_file_path = log_dir + exp_name + ".log"
chkpt_acc_path = checkpoint_dir + exp_name + "1000.pt"
pred_path = pred_dir + exp_name + ".pred"

# get_data_label("/Users/apple/Desktop/机器学习/DataCon_2020/恶意代码检测/gray/1_2000_black/*", label_path, "/black.csv")
# get_data_label("/Users/apple/Desktop/机器学习/DataCon_2020/恶意代码检测/gray/1_4000_white/*", label_path, "/white.csv")

df1 = pd.read_csv(label_path + "white.csv")
df2 = pd.read_csv(label_path + "black.csv")
dataset = merge(df1, df2, "dataset", label_path)

train, valid, train_label, valid_label = train_test_split(
    dataset["id"],
    dataset["labels"],
    test_size=0.2,
    stratify=dataset["labels"],
    random_state=100,
)

trainset = pd.DataFrame({"id": train, "labels": train_label})
validset = pd.DataFrame({"id": valid, "labels": valid_label})

trainset.to_csv(
    label_path + "example-train-label.csv", index=False, header=False, encoding="utf-8"
)
validset.to_csv(
    label_path + "example-valid-label.csv", index=False, header=False, encoding="utf-8"
)


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

# 加载模型

malconv = MalConv(input_length=first_n_byte, window_size=window_size)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
#malconv = MalConvBase(8, 4096, 128, 32)
bce_loss = nn.BCEWithLogitsLoss()
adam_optim = optim.Adam([{"params": malconv.parameters()}], lr=learning_rate)
sigmoid = nn.Sigmoid()

if use_gpu:
    malconv = malconv.cuda()
    bce_loss = bce_loss.cuda()
    sigmoid = sigmoid.cuda()

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

        exe_input = batch_data[0].cuda() if use_gpu else batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        label = batch_data[1].cuda() if use_gpu else batch_data[1]
        label = Variable(label.float(), requires_grad=False)

        pred = malconv(exe_input)
        loss = bce_loss(pred, label)
        loss.backward()
        adam_optim.step()

        history["tr_loss"].append(loss.cpu().data.numpy())
        history["tr_acc"].extend(
            list(
                label.cpu().data.numpy().astype(int)
                == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)
            )
        )
        acc = list(
            label.cpu().data.numpy().astype(int)
            == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)
        )
        # print(np.mean(acc))
        step_cost_time = time.time() - start

        if (step + 1) % display_step == 0:
            print(
                step_msg.format(
                    total_step,
                    np.mean(history["tr_loss"]),
                    np.mean(history["tr_acc"]),
                    step_cost_time,
                )
            )
            train_acc.append(np.mean(history["tr_acc"]))  # 保存训练结果
            # print(np.mean(history['tr_acc']))
        total_step += 1

        # Interupt for validation
        if total_step % test_step == 0:
            break

    # Testing
    history["val_loss"] = []
    history["val_acc"] = []
    history["val_pred"] = []

    for _, val_batch_data in enumerate(validloader):

        cur_batch_size = val_batch_data[0].size(0)

        exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
        label = Variable(label.float(), requires_grad=False)

        pred = malconv(exe_input)
        loss = bce_loss(pred, label)

        history["val_loss"].append(loss.cpu().data.numpy())
        history["val_acc"].extend(
            list(
                label.cpu().data.numpy().astype(int)
                == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)
            )
        )
        history["val_pred"].append(list(sigmoid(pred).cpu().data.numpy()))

    print(
        log_msg.format(
            total_step,
            np.mean(history["tr_loss"]),
            np.mean(history["tr_acc"]),
            np.mean(history["val_loss"]),
            np.mean(history["val_acc"]),
            step_cost_time,
        )
    )

    print(
        valid_msg.format(
            total_step,
            np.mean(history["tr_loss"]),
            np.mean(history["tr_acc"]),
            np.mean(history["val_loss"]),
            np.mean(history["val_acc"]),
        )
    )

    history["tr_loss"] = []
    history["tr_acc"] = []


train_acc_list = np.array(train_acc)
np.save(
    "saves/{0}_{2}_{1}_train_acc_list.npy".format(exp_name, first_n_byte, batch_size),
    train_acc_list,
)  # 保存为.npy格式

# 训练对别图
# Plot TPR
"""
plt.figure()
plt.plot(range(len(Train_TPR)), Train_TPR, c='r', label='Training Set', linewidth=2)
plt.plot(range(len(Test_TPR)), Test_TPR, c='g', linestyle='--', label='Validation Set', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('TPR')
plt.legend()
plt.savefig('saves/Epoch_TPR({0}, {1}).png'.format(self.blackbox, flag[self.same_train_data]))
plt.show()
"""
