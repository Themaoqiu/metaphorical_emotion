"""Convert raw jsonl files into a single train split for EasyR1's RLHFDataset.

Each input row's `image_path` is used as-is (assumed to already be an absolute
path). Output schema:
    prompt: str   (with leading "<image>" token)
    images: list[str]
    answer: str
    source: str   (input jsonl stem)

Usage:
    python examples/prepare_emotion.py \
        --inputs /path/imagemet.jsonl /path/memecap.jsonl /path/metmeme.jsonl /path/vflute.jsonl \
        --out_dir data/emotion_grpo
"""

import argparse
import json
import random
from pathlib import Path


EMOTION_QUESTION = "What emotion is expressed in this image?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="raw jsonl files")
    parser.add_argument("--out_dir", default="data/emotion_grpo")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = []
    counts = {}
    for path in args.inputs:
        stem = Path(path).stem
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({
                    "prompt": f"<image>{EMOTION_QUESTION}",
                    "images": [obj["image_path"]],
                    "answer": obj["emotion_type"],
                    "source": stem,
                })
                n += 1
        counts[stem] = n

    random.Random(args.seed).shuffle(rows)
    train_rows = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("per-source rows:", counts)
    print(f"train: {len(train_rows)}  -> {out_dir}")


if __name__ == "__main__":
    main()
