# coding: utf-8
import glob
import numpy as np
import pandas as pd

def get_data_label(data_path, label_values, label_path):
    names = []
    df = pd.DataFrame()
    for path in glob.glob(data_path):
        names.append(path.split('/')[-1])

    df["id"] = names
    df["labels"] = label_values
    df.to_csv(label_path + "/black.csv", index=False, encoding="utf-8")

get_data_label("/Users/apple/Desktop/机器学习/DataCon_2020/恶意代码检测/gray/1_2000_black/*", 1, "/Users/apple/Documents/GitHub/Deep learning for malware detection/MalConv-Pytorch/data")