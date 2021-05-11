# MalConv-Pytorch
A Pytorch implementation of MalConv

- [x] 改写数据读取为 black/white 文件夹
- [x] 服务器保存模型
- [x] [Integrated gradients applied to malware programs](https://captum.ai/tutorials/IMDB_TorchText_Interpret)
- [ ] [复现 Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection](https://github.com/PowerLZY/Malconv-Pytorch)
- [ ] 添加更多Malware样本
---
## Desciprtion

+ This is the implementation of MalConv proposed in [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435).
+ This is the implementation of visualizing the integrated gradients applied to malware programs

## Setup

#### Preparing data

For the training data, please place PE files under [`data/train/`]() and build [the label table](data/example-train-label.csv) for training set with each row being

        <File Name>, <Label>

where label = 1 refers to malware. Validation set should be handled in the same way.

#### Training Log & Checkpoint

Log file, prediction on validation set & Model checkpoint will be stored at the path specified in config file.

#### Attributions

Show ``train_ data_path`` histogram graph and histogram of sample contribution.

#### Parameters & Model Options

For parameters and options availible, please refer to [`config/example.yaml`](config/example.yaml).

## Tips

#### TFRecord reader and writer

为了高效地读取数据，比较有帮助的一种做法是对数据进行序列化并将其存储在一组可线性读取的文件（每个文件 100-200MB）中。
这尤其适用于通过网络进行流式传输的数据。这种做法对缓冲任何数据预处理也十分有用。
TFRecord 格式是一种用于存储二进制记录序列的简单格式。
https://github.com/vahidk/tfrecord

