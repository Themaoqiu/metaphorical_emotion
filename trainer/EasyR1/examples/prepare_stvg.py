"""
Convert CompSTVG / DORO-STVG sharegpt-format JSONL to EasyR1 training format.

Input JSONL schema (one line per sample):
  {
    "messages": [
      {"role": "user",      "content": "<video>Where does ..."},
      {"role": "assistant", "content": "{\"obj\": {\"0\": [...]}}"}
    ],
    "videos": ["MOSE_1_fps/abc.mp4"],
    "difficulty_bucket": "easy",
    ...
  }

Output JSONL schema expected by EasyR1 RLHFDataset:
  {
    "prompt":  "<video>Where does ...",
    "answer":  "{\"obj\": {\"0\": [...]}}",
    "videos":  ["MOSE_1_fps/abc.mp4"]
  }

Usage (standard, shuffled):
  python prepare_stvg.py \\
      --input  /home/wangxingjian/data/compstvg/compstvg_rl_2of7.sharegpt.with_difficulty.jsonl \\
      --output ./data/stvg/train.jsonl \\
      --val_output ./data/stvg/val.jsonl \\
      --val_ratio 0.02

Usage (curriculum — sorted easy→hard, set shuffle:false in yaml):
  python prepare_stvg.py \\
      --input  /home/wangxingjian/data/compstvg/compstvg_rl_2of7.sharegpt.with_difficulty.jsonl \\
      --output ./data/stvg/train_curriculum.jsonl \\
      --val_output ./data/stvg/val.jsonl \\
      --val_ratio 0.02 \\
      --curriculum

  Within each difficulty bucket the samples are shuffled randomly so the model
  doesn't overfit to a fixed ordering inside a bucket.
"""

import argparse
import json
import os
import random
from pathlib import Path


def convert_sample(raw: dict) -> dict | None:
    messages = raw.get("messages", [])
    if len(messages) < 2:
        return None

    user_msg = next((m for m in messages if m["role"] == "user"), None)
    asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
    if user_msg is None or asst_msg is None:
        return None

    prompt = user_msg["content"]
    answer = asst_msg["content"]

    if not prompt or not answer:
        return None

    out: dict = {"prompt": prompt, "answer": answer}

    # carry over video paths
    if "videos" in raw and raw["videos"]:
        out["videos"] = raw["videos"]

    # carry over optional metadata (not used by EasyR1 but useful for debugging)
    for key in ("query_id", "difficulty_bucket", "source"):
        if key in raw:
            out[key] = raw[key]

    return out


DIFFICULTY_ORDER = ["very_easy", "easy", "medium", "hard", "very_hard"]


def sort_curriculum(samples: list[dict]) -> list[dict]:
    """Sort samples easy→hard; within each bucket shuffle randomly."""
    buckets: dict[str, list[dict]] = {b: [] for b in DIFFICULTY_ORDER}
    unknown: list[dict] = []
    for s in samples:
        bucket = s.get("difficulty_bucket", "")
        if bucket in buckets:
            buckets[bucket].append(s)
        else:
            unknown.append(s)

    ordered = []
    for bucket in DIFFICULTY_ORDER:
        group = buckets[bucket]
        random.shuffle(group)
        ordered.extend(group)
        print(f"  {bucket}: {len(group)} samples")
    if unknown:
        random.shuffle(unknown)
        ordered.extend(unknown)
        print(f"  unknown difficulty: {len(unknown)} samples")
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input sharegpt JSONL file")
    parser.add_argument("--output", required=True, help="Output train JSONL file")
    parser.add_argument("--val_output", default=None, help="Output val JSONL file (optional)")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Fraction held out for val")
    parser.add_argument("--curriculum", action="store_true",
                        help="Sort train set easy→hard (set shuffle:false in yaml)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    samples = []
    skipped = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            converted = convert_sample(raw)
            if converted is None:
                skipped += 1
                continue
            samples.append(converted)

    print(f"Loaded {len(samples)} samples, skipped {skipped}")

    # Val split: always stratified by difficulty so every bucket is represented.
    # Pull val samples before curriculum sorting so the val set stays balanced.
    val_output = args.val_output
    if val_output is not None:
        from collections import defaultdict
        by_bucket: dict[str, list] = defaultdict(list)
        for s in samples:
            by_bucket[s.get("difficulty_bucket", "unknown")].append(s)

        val_samples, train_samples = [], []
        for bucket, group in by_bucket.items():
            random.shuffle(group)
            n_val = max(1, int(len(group) * args.val_ratio))
            val_samples.extend(group[:n_val])
            train_samples.extend(group[n_val:])
        random.shuffle(val_samples)
    else:
        train_samples = samples
        val_samples = []

    if args.curriculum:
        print("Curriculum order (easy → hard):")
        train_samples = sort_curriculum(train_samples)
    else:
        random.shuffle(train_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_samples)} train samples → {args.output}")

    if val_output and val_samples:
        Path(val_output).parent.mkdir(parents=True, exist_ok=True)
        with open(val_output, "w") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_samples)} val samples → {val_output}")


if __name__ == "__main__":
    main()
