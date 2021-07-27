import argparse
import os
import random
from zipfile import *
from io import BytesIO
import requests
from bs4 import BeautifulSoup

# shell
# 查看文件数量 ls -l | grep "^-" | wc -l
# 将指定类型文件带目录结构打包 find test/ -name '*.txt' |xargs tar czf test3.tgz
# 解包到目录test3 tar zxvf test3.tgz -C test3
def construct_href(link):
    root, rest = link.split("sample")
    action, hash_ = rest.split("detail")
    return f"{root}sampleshare{action}getfile{hash_}"


def download(session, href, save_dir, index):
    source = session.get(href, allow_redirects=True)
    try:
        with ZipFile(BytesIO(source.content)) as f:
            f.extractall(path=save_dir, pwd=b"infected")
    except BadZipFile:
        with open(os.path.join(save_dir, f"{index}"), "w+b") as f:
            f.write(source.content)


def main(args):
    try:
        assert os.path.isdir(args.save_dir)
    except AssertionError:
        os.mkdir(args.save_dir)
    with requests.Session() as session:
        credentials = {"api_key": args.api_key or os.getenv("api_key")}
        response = session.post("https://malshare.com", credentials)
        assert response.status_code == 302
        html = session.get("https://malshare.com/search.php?query=YRP/IsPE32").text
        soup = BeautifulSoup(html, "html.parser")
        tds = soup.find_all("td", {"class": "hash_font sorting_1"})
        indices = random.sample(range(len(tds)), args.num_files)
        for index in indices:
            link = tds[index].find("a")["href"]
            href = construct_href(link)
            download(session, href, args.save_dir, index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download malware")
    parser.add_argument(
        "--num_files", type=int, default=1000, help="number of malware to download"
    )
    parser.add_argument(
        "--save_dir", type=str, default="raw/malshare", help="directory to save malware"
    )
    parser.add_argument("--api_key", type=str, default="", help="malshare api key")
    args = parser.parse_args()
    main(args)
