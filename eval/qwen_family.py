import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def _sanitize_runtime_env() -> None:
    def _fix_threads_var(name: str) -> None:
        val = os.environ.get(name, "")
        try:
            if int(val) < 1:
                raise ValueError
        except Exception:
            os.environ[name] = "1"

    _fix_threads_var("OMP_NUM_THREADS")
    _fix_threads_var("MKL_NUM_THREADS")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")


_sanitize_runtime_env()

_MULTIMM_SPLITS = {
    "EN": {"train": 3251, "val": 406, "test": 407},
    "CN": {"train": 3517, "val": 440, "test": 440},
}

_METAPHOR_LABELS = [0, 1]
_SENTIMENT_LABELS = [-1, 0, 1]
_SENTIMENT_QUESTIONS = (
    "What emotion is expressed in this image?",
    "Which emotion is most strongly conveyed by this image?",
    "What feeling does this image primarily communicate?",
    "What is the dominant emotion shown in this image?",
    "What emotion does this image evoke most clearly?",
    "Which emotional tone best matches this image?",
    "What core emotion is being conveyed in this image?",
    "What emotion is the image mainly expressing?",
)


class QwenVLBase:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        nframes: int = 100,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.nframes = nframes
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.001,
            max_tokens=self.max_tokens,
            stop_token_ids=[],
        )
        self.llm = None
        self.processor = None
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def prepare_messages(
        self,
        query: str,
        media_path: str,
        system_prompt: str,
        media_type: str = "video",
    ) -> List[Dict[str, Any]]:
        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported media_type: {media_type}")

        media_content = {
            "type": media_type,
            media_type: media_path,
            "max_pixels": 1280 * 28 * 28,
        }
        if media_type == "video":
            media_content["nframes"] = self.nframes

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    media_content,
                    {"type": "text", "text": query},
                ],
            },
        ]

    def predict_batch(
        self,
        queries: List[str],
        media_paths: List[str],
        system_prompt: str,
        media_type: str = "video",
    ) -> List[str]:
        raise NotImplementedError


class Qwen2_5VL(QwenVLBase):
    def load_model(self):
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def predict_batch(
        self,
        queries: List[str],
        media_paths: List[str],
        system_prompt: str,
        media_type: str = "video",
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info

        batch_messages = []
        for query, media_path in zip(queries, media_paths):
            batch_messages.append(
                self.prepare_messages(query, media_path, system_prompt, media_type=media_type)
            )

        prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batch_messages,
            return_video_kwargs=True,
        )

        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            sample_mm_data = {}
            if image_inputs is not None:
                sample_mm_data["image"] = image_inputs[idx]
            if video_inputs is not None:
                sample_mm_data["video"] = video_inputs[idx]

            sample_video_kw = {}
            for key, value in video_kwargs.items():
                if isinstance(value, (list, tuple)):
                    sample_video_kw[key] = value[idx]
                else:
                    sample_video_kw[key] = value

            llm_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                }
            )

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]


class Qwen3VL(QwenVLBase):
    def load_model(self):
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def predict_batch(
        self,
        queries: List[str],
        media_paths: List[str],
        system_prompt: str,
        media_type: str = "video",
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info

        llm_inputs = []
        for query, media_path in zip(queries, media_paths):
            messages = self.prepare_messages(query, media_path, system_prompt, media_type=media_type)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs.append(
                {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs,
                }
            )

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]


def _resolve_multimm_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    col_lower = {c.lower(): c for c in df.columns}

    def pick(candidates: Iterable[str]) -> Optional[str]:
        for name in candidates:
            if name in df.columns:
                return name
            lower = name.lower()
            if lower in col_lower:
                return col_lower[lower]
        return None

    img_col = pick(["Pic_id", "pic_id", "image", "img"])
    text_col = pick(["Text", "text"])
    metaphor_col = pick(["MetaphorOccurrence", "metaphor", "Unnamed: 2"])
    sentiment_col = pick(["SentimentCategory", "sentiment", "senti"])

    if not all([img_col, text_col, metaphor_col, sentiment_col]):
        missing = [
            name
            for name, column in [
                ("image", img_col),
                ("text", text_col),
                ("metaphor", metaphor_col),
                ("sentiment", sentiment_col),
            ]
            if column is None
        ]
        raise KeyError(f"Missing required MultiMM columns: {missing}")

    return img_col, text_col, metaphor_col, sentiment_col


def _normalize_image_path(path_str: str, image_root: Optional[Path] = None) -> str:
    path = Path(str(path_str))
    if path.exists():
        return str(path.resolve())

    if image_root is not None:
        candidate = image_root / path.name
        if candidate.exists():
            return str(candidate.resolve())

    return str(path)


def _load_multimm_split(
    data_root: str,
    language: str,
    split: str,
    seed: int = 42,
    image_subdir: Optional[str] = None,
) -> pd.DataFrame:
    lang = language.upper()
    if lang not in _MULTIMM_SPLITS:
        raise ValueError(f"Unsupported language: {language}")

    root = Path(data_root)
    csv_path = root / lang / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {csv_path}")

    image_root = root / image_subdir / lang if image_subdir else root / lang
    df = pd.read_csv(csv_path)
    img_col, text_col, metaphor_col, sentiment_col = _resolve_multimm_columns(df)

    out = pd.DataFrame(
        {
            "image": df[img_col].astype(str).map(lambda x: _normalize_image_path(x, image_root)),
            "text": df[text_col].fillna("").astype(str),
            "metaphor": df[metaphor_col].astype(int),
            "sentiment": df[sentiment_col].astype(int),
        }
    )

    rng = random.Random(seed)
    indices = list(range(len(out)))
    rng.shuffle(indices)

    counts = _MULTIMM_SPLITS[lang]
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split_map = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end : val_end + counts["test"]],
    }

    if split not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    return out.iloc[split_map[split]].reset_index(drop=True)


def _parse_label(text: str, task: str, default_label: int = 0) -> int:
    normalized = re.sub(r"[^a-z]+", " ", str(text).lower()).strip()

    if task == "metaphor":
        if any(token in normalized for token in ["literal", "non metaphor", "not metaphor"]):
            return 0
        if "metaphor" in normalized:
            return 1
        return default_label

    if any(token in normalized for token in ["negative", "sad", "anger", "angry", "fear", "disgust"]):
        return -1
    if any(token in normalized for token in ["positive", "happy", "joy", "surprise"]):
        return 1
    if "neutral" in normalized:
        return 0
    return default_label


def _compute_metrics(y_true: List[int], y_pred: List[int], labels: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
