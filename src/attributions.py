import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch
from torch.autograd import Variable
from src.model import MalConv
from src.util import *
from torch.utils.data import DataLoader

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
# 文件名列表、数据文件夹、文件列表标签、前n位字节
validloader = DataLoader(ExeDataset(['Backdoor.Win32.Agent.bflv_ce22.exe'], train_data_path, [1],first_n_byte),
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

    attributions = attributions_ig.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    print('IG Attributions:', attributions)
    malware = exe_input.squeeze(0).cpu().detach().numpy()
    conf = model(exe_input)
    title = 'Confidence: {0:.4f}%\nDOS + COFF + OPT + SECT Headers\nBaseline : empty file'.format(conf.item() * 100)
    plot_code_segment(malware, 0, 512, attributions, title, force_plot=True, show_positives=True, show_negatives=True)
    plot_header_contribution_histogram(malware.tobytes(), attributions, force_plot=True)
    break;