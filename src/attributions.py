import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch
from torch.autograd import Variable
from src.model import PreMalConv, MalConv
#from secml_malware.models.malconv import MalConv
#from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel

from src.util import *
from torch.utils.data import DataLoader

from captum.attr import (
    IntegratedGradients,# 积分梯度
    LayerIntegratedGradients # embedding层积分梯度
)


def load_binaries(data_path, first_n_byte):
    '''
    Import the sample and convert it into a fixed length byte sequence for MalConv
    :param data_path: Analysis sample path
    :param first_n_byte: fixed length
    :return numpy: byte sequence
    '''
    with open(data_path, 'rb') as f:
        tmp = [i + 1 for i in f.read()[:first_n_byte]]  # index 0 will be special padding index 每个值加一
        tmp = tmp + [0] * (first_n_byte - len(tmp))

    return np.array(tmp)


train_data_path =rootPath + "/data/one/"  # Training data
model_path = rootPath + '/checkpoint/pretrained_malconv.pth'
save_path = rootPath + "/picture"
filename = get_filename(train_data_path+"*") # 样本列表
labels = [0 for _ in filename] # filename 标签列表
# Parameter
use_gpu = True              #
use_cpu = 1                # Number of cores to use for data loader
batch_size = 1          #
first_n_byte = 2 ** 20     # First N bytes of a PE file as the input of MalConv (defualt: 2 million)

# 加载模型
model = PreMalConv()
#model = CClassifierEnd2EndMalware(model)
#model.load_pretrained_model()
model.load_state_dict(torch.load(model_path))
model.eval()
if use_gpu:
    model = model.cuda()


sample = load_binaries(train_data_path+"Backdoor.Win32.Agent.bflv_ce22.exe", first_n_byte)
exe_input = torch.from_numpy(sample).unsqueeze(0)

exe_input = exe_input.cuda() if use_gpu else exe_input
exe_input = Variable(exe_input.long(), requires_grad=False)

conf = model(exe_input)

pred = conf.detach().cpu().numpy()[0,0]


# 文件名列表、数据文件夹、文件列表标签、前n位字节
validloader = DataLoader(ExeDataset(filename, train_data_path, labels, first_n_byte),
                        batch_size=batch_size, shuffle=False, num_workers=use_cpu)

# 显示 train_data_path 下的样本的贡献度热力图和直方图
for x, val_batch_data in enumerate(validloader):
    cur_batch_size = val_batch_data[0].size(0)

    exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
    exe_input = Variable(exe_input.long(), requires_grad=False)
    baseline = torch.zeros(1, first_n_byte).cuda() if use_gpu else torch.zeros(1, first_n_byte)
    baseline = Variable(baseline.long(), requires_grad=False)

    lig = LayerIntegratedGradients(model, model.embedding_1) # MalConv：model.embed; PreMalConv: model.embedding_1
    attributions_ig = lig.attribute(exe_input, baseline, target=0)

    attributions = attributions_ig.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    print('IG Attributions:', attributions)
    malware = exe_input.squeeze(0).cpu().detach().numpy()
    conf = model(exe_input)
    title = 'Confidence: {0:.4f}%\nDOS + COFF + OPT + SECT Headers\nBaseline : empty file'.format(conf.item()*100)
    plot_code_segment(malware, 0, 512, attributions, title, force_plot=True,
                      show_positives=True, show_negatives=True, save_path=save_path, filename = filename[x])
    # 字节流转化
    malware = [i-1 for i in malware.tolist() if i > 0]
    plot_header_contribution_histogram(bytearray(malware), attributions, force_plot=True,
                                       save_path = save_path, filename = filename[x])
    break;