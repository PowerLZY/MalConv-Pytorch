# MalConv-Pytorch
A Pytorch implementation of MalConv

---
## Desciprtion

This is the implementation of MalConv proposed in [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435).

## Dependency

Please make sure each of them is installed with the correct version

- numpy
- pytorch (0.3.0.post4)
- pandas (0.20.3)


## Setup

#### Preparing data

For the training data, please place PE files under [`data/train/`](`data/train`) and build [the label table](data/example-train-label.csv) for training set with each row being

        <File Name>, <Label>

where label = 1 refers to malware. Validation set should be handled in the same way.

#### Training

Run the following command for training progress

        python3 train.py <config_file_path> <random_seed>
        Example : python3 train.py config/example.yaml 123

#### Training Log & Checkpoint

Log file, prediction on validation set & Model checkpoint will be stored at the path specified in config file.

## Parameters & Model Options

For parameters and options availible, please refer to [`config/example.yaml`](config/example.yaml).


## 基于深度学习的恶意软件检测

​		与特性空间不同的是，即使稍加修改，也**不能简单地更改原始二进制数据，否则会损坏其功能**。此外，二进制数据的大小差异很大，这进一步增加了攻击困难。我们同时发现在保存生成的对抗样本时，将连续空间中的对抗有效载荷转换回离散二进制时，会忽略细微的扰动，从而影响对抗攻击的有效性。因此，如何在保护原有功能的同时，对基于恶意软件二进制文件的深度学习模型进行有效而实用的黑盒攻击仍然是一个巨大的挑战。==**原始二进制文件具有可变的输入大小**==。

+ **深度学习恶意软件检测模型：**深度神经网络可以有效地挖掘原始数据中的潜在特征，而无需大量数据预处理和先验经验。

  + **Malware Detection by Eating a Whole EXE（2018 AAAIW)**

  + **==Malware detection using 1-dimensional convolutional neural networks== （ 2019 EuroS&PW）**
  + **==Activation analysis of a byte based deep neural network for malware classification==** **(2019 S&PW)**
  + **Adversarial Malware Binaries: Evading Deep Learning for Malware Detection in Executables**
    + **Malconv优化**
    + 论文 https://arxiv.org/abs/2012.09390
    + 开发了一种新的时间最大池方法，使得所需的内存对序列长度T保持不变。这使得MalConv的内存效率提高了116倍，在原始数据集上训练的速度提高了25.8倍，同时**消除了MalConv的输入长度限制**
    + 复现 https://github.com/NeuromorphicComputationResearchProgram/MalConv2

+ **基于问题空间的对抗样本生成方法**

  + **Adversarial Malware Binaries: Evading Deep Learning for Malware Detection in Executables**
    + 白盒
    + **第一篇攻击基于字节序列**
    + 在文件末尾增加字节来产生对抗样本
  + **Deceiving end-to-end deep learning malware detectors using adversarial examples (2018)**
    + 白盒
    + 修改了FGSM的损失函数，使其能更好应用于恶意软件数据的离散性
  + ==Adversarial examples for cnn-based malware detectors （2019 IEEE Access）==【随机产生扰动】
  + **Adversarial EXEmples: A Survey and Experimental Evaluation of Practical Attacks on Machine Learning for Windows Malware Detection.  (2020 abs)**