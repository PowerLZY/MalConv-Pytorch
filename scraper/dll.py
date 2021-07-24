import argparse
import os
import random
from io import BytesIO
from zipfile import BadZipFile, ZipFile

import requests
from bs4 import BeautifulSoup


def get_href(index):
    link = f"https://wikidll.com/download/{index}"
    html = requests.get(link, allow_redirects=True).text
    soup = BeautifulSoup(html, "html.parser")
    href = soup.find("a", {"class": "download__link"})["href"]
    return href


def download(index, save_dir):
    href = get_href(index)
    source = requests.get(href, allow_redirects=True)
    try:
        with ZipFile(BytesIO(source.content)) as f:
            f.extractall(path=save_dir)
    except BadZipFile:
        with open(os.path.join(save_dir, f"{index}.dll"), "w+b") as f:
            f.write(source.content)


def main(args):
    try:
        assert os.path.isdir(args.save_dir)
    except AssertionError:
        os.mkdir(args.save_dir)
    indices = random.sample(range(1, 27786), args.num_files)
    _ = [download(index, args.save_dir) for index in indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download .dll files")
    parser.add_argument(
        "--num_files", type=int, default=100, help="number of files to download"
    )
    parser.add_argument(
        "--save_dir", type=str, default="raw/dll", help="directory to save files"
    )
    args = parser.parse_args()
    main(args)
