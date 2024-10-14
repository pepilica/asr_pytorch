import gzip
import os
import shutil

import gdown
import wget


def download():
    gdown.download(id="1whPIPL81eS6VJLlqgKYyzG5BwP8KvvVV")

    if not os.path.exists("saved/4-gram.arpa"):
        if not os.path.exists("saved/"):
            os.mkdir("saved")
        lm_url = "https://openslr.elda.org/resources/11/4-gram.arpa.gz"
        lm_gzip_path = wget.download(lm_url)
        with gzip.open(lm_gzip_path, "rb") as f_zipped:
            with open("saved/4-gram.arpa", "wb") as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)

    os.rename("model_best.pth", "saved/model_best.pth")


if __name__ == "__main__":
    download()
