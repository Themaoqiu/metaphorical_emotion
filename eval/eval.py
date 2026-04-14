import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from qwen_family import (
    Qwen3VL,
    _METAPHOR_LABELS,
    _SENTIMENT_LABELS,
    _SENTIMENT_QUESTIONS,
    _compute_metrics,
    _parse_label,
)


def _fix_threads_var(name: str) -> None:
    val = os.environ.get(name, "")
    try:
        if int(val) < 1:
            raise ValueError
    except Exception:
        os.environ[name] = "1"


_fix_threads_var("OMP_NUM_THREADS")
_fix_threads_var("MKL_NUM_THREADS")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")


def _load_fixed_test_df(data_root: str, language: str, fixed_dir_name: str = "fixed_test"):
    lang = language.upper()
    root = Path(data_root)
    direct_metadata = root / lang / "metadata.csv"
    nested_metadata = root / fixed_dir_name / lang / "metadata.csv"

    if direct_metadata.exists():
        metadata_path = direct_metadata
    elif nested_metadata.exists():
        metadata_path = nested_metadata
    else:
        raise FileNotFoundError(f"Missing fixed test metadata at {root}")

    df = pd.read_csv(metadata_path)
    required_cols = ["image", "text", "metaphor", "sentiment"]
    out = df[required_cols].copy()
    out["image"] = out["image"].astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    out["metaphor"] = out["metaphor"].astype(int)
    out["sentiment"] = out["sentiment"].astype(int)
    out["language"] = lang
    return out


def _batched(items: List, batch_size: int) -> Iterable[List]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _build_unified_prompt(text: str, idx: int) -> Tuple[str, str]:
    system_prompt = (
        """You are an expert visual emotion analyst tasked with understanding emotions based on visual content.
  When analyzing a image's emotion, you should carefully observe and think the metaphor meaning from the image to answer.
  For the image you notice, first observe the key elements that convey the meaning of the image, then analyze which metaphor understanding pathway will be activated, using the following format:
  describe the key visual clues with <caption>Description of key visual clues</caption>,
  based on visual information, judge whether there is a metaphor in the image. If a metaphor exists, further analyze which of the following comprehension pathways it will activate within <metaphor></metaphor>.
  Metaphor comprehension pathways which you can choose:
  - direct: The image employs common idioms or fixed expressions, or its metaphor can be recognized at a glance without additional interpretation.
  - sequential: When reading the text sequentially and viewing the image, one first perceives the literal meaning of the picture. However, upon integrating the context and the content of the image, this literal meaning is revealed to be incorrect, and a cognitive shift is required to truly grasp the metaphorical meaning it conveys.
  - parallel: After examining the entire image, both its metaphorical and literal meanings are quite common, with roughly equal weight in comprehension. Unlike direct expressions, which only evoke one meaning (for instance, one does not think of the literal meaning when using an idiom).
  and provide your analysis about how to understand the emotions based on the above comprehension pathway with <think>Your analysis and thoughts about this segment</think>.
  Throughout your analysis, think about the question as if you were a human pondering deeply,
  engaging in an internal dialogue using natural thought expressions such as such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions.
  After examining the key visual clues, continue with deeper reasoning that connects your observations and metaphor comprehension pathway to the answer.
  Self-reflection or verification in your reasoning process is encouraged when necessary,
  though if the answer is straightforward, you may proceed directly to the conclusion.
  Finally, conclude by placing your final answer in <answer> </answer> tags. """
    )
    matten = "Please analyze the image emotion carefully by identifying key elements and metaphor in the image within `<caption> </caption>`, `<metaphor> </metaphor>`, `<think> </think>` tags then conduct deep analysis and reasoning to arrive at your answer to the question, finally provide only the single emotion(among postive, negative and neutral) within the `<answer> </answer>` tags. Follow the format specified in the instructions."
    sentiment_q = _SENTIMENT_QUESTIONS[idx % len(_SENTIMENT_QUESTIONS)]
    query = f"<image>{sentiment_q}\n\n" + matten + "\nText: {text}"
    return system_prompt, query


def _parse_unified_response(response: str) -> Tuple[int, int]:
    match = re.search(r"<answers>(.*?)</answers>", response, flags=re.IGNORECASE | re.DOTALL)
    content = match.group(1) if match else response

    m_match = re.search(r"Metaphor:\s*(\d)", content, flags=re.IGNORECASE)
    if m_match:
        m_label = int(m_match.group(1))
    else:
        m_label = 0 if "no metaphor" in response.lower() else 1

    s_match = re.search(r"Sentiment:\s*(\w+)", content, flags=re.IGNORECASE)
    if s_match:
        s_text = s_match.group(1).lower()
        s_label = _parse_label(s_text, task="sentiment", default_label=0)
    else:
        s_label = _parse_label(response, task="sentiment", default_label=0)

    return m_label, s_label


def _predict_unified(model: Qwen3VL, test_df) -> Dict[str, List]:
    texts = test_df["text"].tolist()
    media_paths = test_df["image"].tolist()

    queries = []
    for idx, text in enumerate(texts):
        _, query = _build_unified_prompt(text, idx)
        queries.append(query)

    system_prompt, _ = _build_unified_prompt("", 0)
    responses = []

    for q_batch, m_batch in zip(
        _batched(queries, model.batch_size),
        _batched(media_paths, model.batch_size),
    ):
        responses.extend(
            model.predict_batch(
                q_batch,
                m_batch,
                system_prompt=system_prompt,
                media_type="image",
            )
        )

    m_labels = []
    s_labels = []
    for response in responses:
        m_val, s_val = _parse_unified_response(response)
        m_labels.append(m_val)
        s_labels.append(s_val)

    return {
        "queries": queries,
        "responses": responses,
        "metaphor_labels": m_labels,
        "sentiment_labels": s_labels,
    }


def _label_to_text(task: str, label: int) -> str:
    if task == "metaphor":
        return {0: "literal", 1: "metaphor"}.get(label, str(label))
    return {-1: "negative", 0: "neutral", 1: "positive"}.get(label, str(label))


def _build_combined_records(lang: str, test_df, unified_out: Dict) -> List[Dict]:
    records = []
    for idx, row in test_df.reset_index(drop=True).iterrows():
        base_record = {
            "language": lang,
            "sample_index": int(idx),
            "image": str(row["image"]),
            "text": str(row["text"]),
            "query": unified_out["queries"][idx],
            "model_response": unified_out["responses"][idx],
        }

        m_record = base_record.copy()
        m_record.update(
            {
                "task": "metaphor",
                "gt_label": int(row["metaphor"]),
                "pred_label": int(unified_out["metaphor_labels"][idx]),
                "gt_text": _label_to_text("metaphor", row["metaphor"]),
                "pred_text": _label_to_text("metaphor", unified_out["metaphor_labels"][idx]),
                "is_correct": int(row["metaphor"]) == int(unified_out["metaphor_labels"][idx]),
            }
        )
        records.append(m_record)

        s_record = base_record.copy()
        s_record.update(
            {
                "task": "sentiment",
                "gt_label": int(row["sentiment"]),
                "pred_label": int(unified_out["sentiment_labels"][idx]),
                "gt_text": _label_to_text("sentiment", row["sentiment"]),
                "pred_text": _label_to_text("sentiment", unified_out["sentiment_labels"][idx]),
                "is_correct": int(row["sentiment"]) == int(unified_out["sentiment_labels"][idx]),
            }
        )
        records.append(s_record)

    return records


def evaluate_unified(
    data_root: str,
    model_path: str,
    batch_size: int = 1,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    language: str = "both",
    fixed_test_dir_name: str = "fixed_test",
):
    model = Qwen3VL(
        model_path=model_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        temperature=0.0,
    )

    languages = ["EN", "CN"] if language.lower() == "both" else [language.upper()]
    results = {}
    all_records = []

    for lang in languages:
        print(f"\n[INFO] Starting Unified Evaluation for {lang}...")
        test_df = _load_fixed_test_df(data_root, lang, fixed_test_dir_name)
        unified_out = _predict_unified(model, test_df)
        all_records.extend(_build_combined_records(lang, test_df, unified_out))
        results[lang] = {
            "metaphor": _compute_metrics(
                test_df["metaphor"].tolist(),
                unified_out["metaphor_labels"],
                labels=_METAPHOR_LABELS,
            ),
            "sentiment": _compute_metrics(
                test_df["sentiment"].tolist(),
                unified_out["sentiment_labels"],
                labels=_SENTIMENT_LABELS,
            ),
        }

    return results, all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("EVAL_DATA_ROOT", "/root/autodl-tmp/MultiMM-master/data/fixed_test"),
        help="Path to the fixed test dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("EVAL_OUTPUT_DIR", str(Path(__file__).resolve().parent / "results")),
        help="Directory for evaluation csv/json outputs",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--language", type=str, default="both")
    args = parser.parse_args()

    ckpt_name = Path(args.model_path).name
    results, per_output_records = evaluate_unified(
        data_root=args.data_root,
        model_path=args.model_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        language=args.language,
    )

    print(f"\n========== Results for {ckpt_name} ==========")
    print(json.dumps(results, indent=4))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(per_output_records).to_csv(
        output_dir / f"{ckpt_name}_eval_{timestamp}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(output_dir / f"{ckpt_name}_metrics_{timestamp}.json", "w") as file:
        json.dump(results, file, indent=4)

    print(f"\n[SUCCESS] {ckpt_name} saved to {output_dir}")
