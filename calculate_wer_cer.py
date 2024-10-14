import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import read_json


def main(dir_path):
    wer, cer, count = 0, 0, 0
    for path in Path(dir_path).iterdir():
        if path.suffix == ".json":
            text = read_json(path)
            target_text = CTCTextEncoder.normalize_text(text["text"])
            predicted_text = text["pred_text"]
            cer += calc_cer(target_text, predicted_text)
            wer += calc_wer(target_text, predicted_text)
            count += 1
    print("WER: {:.4f}".format(wer / count))
    print("CER: {:.4f}".format(cer / count))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dir_path", default="data/saved/predict", type=str)
    args = args.parse_args()
    main(args.dir_path)
