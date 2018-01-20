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
