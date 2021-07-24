import os

import pandas as pd


def main():
    cache = []
    for file_name in os.listdir(os.getcwd()):
        _, extension = os.path.splitext(file_name)
        if extension.lower() == ".csv":
            df = pd.read_csv(file_name, header=None)
            cache.append(df)
    out = pd.concat(cache)
    out.to_csv("combined_labels.csv", index=False, header=False)


if __name__ == "__main__":
    main()
