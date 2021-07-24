import json
import os
import pickle
import time

import pefile


def main():
    failed_files = []
    for directory in ("benign", "malware"):
        data_dir = os.path.join("raw", directory)
        for file_name in os.listdir(data_dir):
            output_dir = os.path.join(directory, file_name)
            try:
                file = pefile.PE(os.path.join(data_dir, file_name))
                header = list(file.header)
                with open(f"{output_dir}.pickle", "wb") as f:
                    pickle.dump(header, f)
            except pefile.PEFormatError:
                print(f"Skipping {file_name}")
                failed_files.append(output_dir)
    with open("log.json", "w") as outfile:
        json.dump(failed_files, outfile, indent=4)


if __name__ == "__main__":
    print("Pickling files...")
    start = time.time()
    main()
    end = time.time()
    print(f"Process completed in {int(end - start)} seconds")
