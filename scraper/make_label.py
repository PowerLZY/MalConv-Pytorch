import csv
import os


def main():
    dir = "raw"
    with open(os.path.join(dir, "labels.csv"), mode="w") as f:
        writer = csv.writer(f, delimiter=",")
        for file_name in os.listdir(dir):
            is_malware = len(file_name) > 31
            writer.writerow([file_name, is_malware])


if __name__ == "__main__":
    main()
