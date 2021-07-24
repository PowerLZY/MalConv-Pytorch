import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pefile


def walk():
    len_count = []
    for directory in ("benign", "malware"):
        data_dir = os.path.join("raw", directory)
        for file_name in os.listdir(data_dir):
            try:
                file = pefile.PE(os.path.join(data_dir, file_name))
                header = list(file.header)
                len_count.append(len(header))
            except pefile.PEFormatError:
                print(f"Skipping {file_name}")
    return Counter(len_count)


def plot(len_count):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(len_count.keys(), len_count.values())
    plt.show()


def write(len_count):
    sorted_count = {
        len_: count for len_, count in sorted(len_count.items(), key=lambda x: -x[1])
    }
    with open("len_count.json", "w+") as outfile:
        json.dumps(sorted_count, outfile, indent=4)


if __name__ == "__main__":
    len_count = walk()
    plot(len_count)
    write(len_count)
