import os
from io import BytesIO
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup


def get_hrefs():
    html = requests.get("https://das-malwerk.herokuapp.com").text
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")[1:]
    malware2href = {}
    for row in rows:
        a_tags = row.find_all("a")
        file_hash = a_tags[1].text
        href = a_tags[0]["href"]
        malware2href[file_hash] = href
    return malware2href


def download(malware2href, save_dir="raw/dasmalwerk"):
    try:
        assert os.path.isdir(save_dir)
    except AssertionError:
        os.mkdir(save_dir)
    for file_hash, href in malware2href.items():
        source = requests.get(href, allow_redirects=True)
        with ZipFile(BytesIO(source.content)) as f:
            f.extractall(path=save_dir, pwd=b"infected")


def main():
    malware2href = get_hrefs()
    download(malware2href)


if __name__ == "__main__":
    main()
