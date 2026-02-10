from __future__ import annotations

import argparse
import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from api_sync import StreamGenerator
from api_sync.utils.parser import JSONParser

FINAL_JSONL_FIELDS = [
    "record_id",
    "image_path",
    "text",
    "is_metaphor",
    "metaphor_path",
    "emotion_type",
    "image_description",
    "problem",
    "options",
    "answer",
    "think",
]

EXAMPLE_RECORD = {
    "record_id": "multimeme_EN_000001",
    "image_path": "/abs/path/to/image.jpg",
    "text": "Original text from dataset",
    "is_metaphor": True,
    "metaphor_path": "sequential",
    "emotion_type": "positive",
    "image_description": "A smiling person holding a trophy.",
    "problem": "这张图片表达什么情感？",
    "options": ["negative", "neutral", "positive"],
    "answer": "positive",
    "think": "图片中的成功场景与文本一起暗示积极情感。",
}


@dataclass
class AnnotatorConfig:
    input_path: Path
    output_path: Path
    model_name: str
    api_keys: List[str]
    max_concurrent_per_key: int = 50
    max_retries: int = 5
    rational: bool = False


class BaseAnnotator(ABC):
    def __init__(self, config: AnnotatorConfig) -> None:
        self.config = config
        self.stream = StreamGenerator(
            model_name=config.model_name,
            api_keys=config.api_keys,
            max_concurrent_per_key=config.max_concurrent_per_key,
            max_retries=config.max_retries,
            rational=config.rational,
            with_unique_id=True,
        )

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, record: Dict[str, Any]) -> Union[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def build_output_record(self, record: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def handle_error_record(self, record: Dict[str, Any], response: str) -> Dict[str, Any]:
        output = self.build_output_record(record, {})
        output["annotation_error"] = "parse_failed"
        output["raw_response"] = response
        return output

    def load_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.config.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _build_prompts(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompts: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            prompts.append({
                "id": str(idx),
                "prompt": self.build_prompt(record),
            })
        return prompts

    def _validate_response(self, response: str) -> Optional[Dict[str, Any]]:
        if isinstance(response, dict):
            return response
        return JSONParser.parse(response)

    async def _run_async(self, records: Sequence[Dict[str, Any]]) -> None:
        prompts = self._build_prompts(records)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.config.output_path.open("w", encoding="utf-8") as handle:
            async for result in self.stream.generate_stream(
                prompts=prompts,
                system_prompt=self.system_prompt,
                validate_func=self._validate_response,
            ):
                record_index = int(result["id"])
                record = records[record_index]
                response = result["result"]

                if isinstance(response, dict):
                    annotation = response
                else:
                    annotation = self._validate_response(response)

                if annotation is None:
                    output_record = self.handle_error_record(record, str(response))
                else:
                    output_record = self.build_output_record(record, annotation)

                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    def run(self) -> None:
        records = self.load_records()
        asyncio.run(self._run_async(records))


def parse_api_keys(value: Optional[str]) -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    env_value = (
        os.getenv("MM_API_KEYS")
        or os.getenv("API_KEYS")
        or os.getenv("OPENAI_API_KEY")
    )
    if not env_value:
        raise ValueError("API keys not provided. Use --api-keys or set MM_API_KEYS/API_KEYS.")
    return [item.strip() for item in env_value.split(",") if item.strip()]


def _first_value(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


class MultiMemeAnnotator(BaseAnnotator):
    DATASET_NAME = "multimeme"
    PROBLEM = "这张图片表达什么情感？"
    EMOTION_OPTIONS = ("negative", "neutral", "positive")
    SYSTEM_PROMPT = "你是一个严谨的多模态标注助手，输出必须是JSON。"
    PROMPT_TEMPLATE = (
        "你将看到一张图片及其文本信息。\n"
        "文本: {text}\n"
        "情感类别: {sentiment_category}\n"
        "{offensiveness_block}"
        "任务: \n"
        "1. 用一句话客观描述图片内容。\n"
        "2. 判断隐喻理解路径，选择 direct / sequential / parallel 之一。\n"
        "   - direct: 直接从图像或文本得到隐喻含义。\n"
        "   - sequential: 先理解字面内容，再推导隐喻含义。\n"
        "   - parallel: 图文线索并行互相提示隐喻含义。\n"
        "如果不是隐喻，请输出 direct。\n"
        "仅输出JSON，格式如下: {\"image_description\": \"...\", \"metaphor_path\": \"direct\"}"
    )

    def __init__(self, *args, image_root: Optional[Path] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_root = Path(image_root) if image_root else None

    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def resolve_image_path(self, record: Dict[str, Any]) -> Optional[Path]:
        image_value = _first_value(
            record,
            ["image_path", "images_name", "Pic_id", "image", "path"],
        )
        if image_value is None:
            return None
        image_path = Path(str(image_value))
        if not image_path.is_absolute() and self.image_root:
            image_path = self.image_root / image_path
        return image_path

    def extract_text(self, record: Dict[str, Any]) -> str:
        value = _first_value(record, ["text", "Text"])
        return str(value) if value is not None else ""

    def extract_sentiment_category(self, record: Dict[str, Any]) -> Optional[str]:
        value = _first_value(record, ["sentiment category", "sentiment_category", "SentimentCategory"])
        if value is None:
            return None
        return str(value)

    def extract_intention_detection(self, record: Dict[str, Any]) -> Optional[str]:
        value = _first_value(record, ["intention detection", "intention_detection", "IntentionDetection"])
        if value is None:
            return None
        return str(value)

    def extract_offensiveness_degree(self, record: Dict[str, Any]) -> Optional[str]:
        value = _first_value(
            record,
            [
                "offensiveness degree",
                "offensiveness_degree",
                "offensiveness detection",
                "offensiveness_detection",
            ],
        )
        if value is None:
            return None
        return str(value)

    def extract_metaphor_occurrence(self, record: Dict[str, Any]) -> Optional[str]:
        value = _first_value(record, ["metaphor occurrence", "metaphor_occurrence", "MetaphorOccurrence"])
        if value is None:
            return None
        return str(value)

    def is_offensive(self, intention_detection: Optional[str]) -> bool:
        if not intention_detection:
            return False
        return "offensive" in intention_detection.lower()

    def normalize_emotion(self, sentiment_category: Optional[str]) -> Optional[str]:
        if sentiment_category is None:
            return None
        normalized = sentiment_category.strip().lower()
        if normalized in {"1", "1.0", "positive", "pos"}:
            return "positive"
        if normalized in {"0", "0.0", "neutral", "neu"}:
            return "neutral"
        if normalized in {"-1", "-1.0", "negative", "neg"}:
            return "negative"
        if "(" in normalized and ")" in normalized:
            start = normalized.find("(")
            end = normalized.find(")", start + 1)
            if start != -1 and end != -1:
                return normalized[start + 1:end]
        return sentiment_category

    def normalize_metaphor_occurrence(self, metaphor_occurrence: Optional[str]) -> Optional[bool]:
        if metaphor_occurrence is None:
            return None
        value = metaphor_occurrence.strip().lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        return None

    def build_prompt_text(self, record: Dict[str, Any]) -> str:
        text = self.extract_text(record)
        sentiment_category = self.extract_sentiment_category(record) or ""
        intention_detection = self.extract_intention_detection(record)
        offensiveness_degree = self.extract_offensiveness_degree(record)

        offensiveness_block = ""
        if self.is_offensive(intention_detection) and offensiveness_degree:
            offensiveness_block = f"Offensiveness degree: {offensiveness_degree}\n"

        return self.PROMPT_TEMPLATE.format(
            text=text,
            sentiment_category=sentiment_category,
            offensiveness_block=offensiveness_block,
        ).strip()

    def build_prompt(self, record: Dict[str, Any]) -> Union[str, List[Dict[str, Any]]]:
        prompt_text = self.build_prompt_text(record)
        image_path = self.resolve_image_path(record)
        if image_path:
            return [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ]
        return prompt_text

    def build_base_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.resolve_image_path(record)
        sentiment_category = self.extract_sentiment_category(record)
        emotion_type = self.normalize_emotion(sentiment_category)
        metaphor_occurrence = self.extract_metaphor_occurrence(record)

        base: Dict[str, Any] = {
            "record_id": str(
                _first_value(record, ["id", "image_id", "images_name", "Pic_id", "image_path"]) or ""
            ),
            "image_path": str(image_path) if image_path else "",
            "text": self.extract_text(record),
            "is_metaphor": self.normalize_metaphor_occurrence(metaphor_occurrence),
            "emotion_type": emotion_type,
            "problem": self.PROBLEM,
            "options": list(self.EMOTION_OPTIONS),
            "answer": emotion_type,
        }
        return base

    def build_output_record(self, record: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        output = self.build_base_record(record)
        output["image_description"] = annotation.get("image_description", "")
        output["metaphor_path"] = annotation.get("metaphor_path", "")
        output["think"] = ""
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate MultiMeme (stage 1: description + metaphor path).")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--image-root", default=None, help="Optional image root to resolve relative paths")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-keys", default=None, help="Comma-separated API keys")
    parser.add_argument("--max-concurrent", type=int, default=50, help="Max concurrent requests per key")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")

    args = parser.parse_args()

    config = AnnotatorConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model,
        api_keys=parse_api_keys(args.api_keys),
        max_concurrent_per_key=args.max_concurrent,
        max_retries=args.max_retries,
    )

    annotator = MultiMemeAnnotator(config, image_root=args.image_root)
    annotator.run()


if __name__ == "__main__":
    main()
