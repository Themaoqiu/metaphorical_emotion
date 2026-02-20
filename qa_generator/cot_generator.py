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


class MetMemeAnnotator(BaseAnnotator):
    EMOTION_OPTIONS = [
        "happiness",
        "love",
        "anger",
        "sorrow",
        "fear",
        "hate",
        "surprise",
        "neutral",
    ]
    DEFAULT_PROBLEM = "What metaphor is shown in this image, and which emotion category best matches its implied meaning?"

    SYSTEM_PROMPT = (
        "You are an expert metaphorical image emotion analysis assistant specialized in creating natural, "
        "flowing Chain of Thought reasoning.\n\n"
        "You will be given:\n"
        "- A question about the metaphor and emotion in one image\n"
        "- Answer options with emotion categories\n"
        "- The correct answer\n"
        "- Key fields: is_metaphor, text, emotion_type, caption\n\n"
        "Your task:\n"
        "- Analyze the image using provided key fields.\n"
        "- Use this exact tag order in your reasoning output:\n"
        "  1) <caption></caption>: initial visual perception only, no metaphor or emotion analysis.\n"
        "  2) <metaphor></metaphor>: check whether metaphor exists based on the provided information; if yes, explain your understanding path "
        "(direct, sequential, or parallel).\n"
        "  3) <think></think>: deeper reasoning that combines visual clues and metaphor judgment.\n"
        "  4) <answer></answer>: final answer as one emotion label from options.\n"
        "- Keep the reasoning natural and conversational, but concise and evidence-based.\n"
        "- Use English only.\n\n"
        "Example Output Style:\n"
        "<caption>A cartoon character stands under dark clouds while holding a tiny umbrella. The text says, "
        "\"My week in one picture.\"</caption>\n"
        "<metaphor>There is a metaphor. The dark cloud is not literal weather only; it maps to ongoing pressure "
        "and emotional burden. This fits a direct path because the visual convention is immediately recognizable."
        "</metaphor>\n"
        "<think>Let me connect the clues. The facial expression looks tired, the cloud follows the character, "
        "and the text generalizes the state to a whole week. Together they imply persistent stress rather than "
        "a single bad moment, so the dominant emotion is closer to sorrow than surprise or anger.</think>\n"
        "<answer>sorrow</answer>\n\n"
        "Output JSON only in this format:\n"
        "{\"analysis\": \"<caption>...</caption>\\n<metaphor>...</metaphor>\\n<think>...</think>\\n<answer>...</answer>\"}"
    )

    USER_PROMPT_TEMPLATE = (
        "Question: {problem}\n\n"
        "Answer Options:\n"
        "{options_text}\n\n"
        "Correct Answer: {correct_answer}\n\n"
        "Known Fields:\n"
        "- is_metaphor: {is_metaphor}\n"
        "- text: {text}\n"
        "- emotion_type: {emotion_type}\n"
        "- caption: {caption}\n\n"
        "Task: Generate a natural, conversational reasoning process for metaphorical emotion analysis. "
        "Use exactly these tags in order: <caption>, <metaphor>, <think>, <answer>."
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

    def _format_options_text(self, options: Any) -> str:
        if isinstance(options, dict):
            items = [(str(k), str(v)) for k, v in options.items()]
        elif isinstance(options, list):
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            items = [(labels[idx], str(v)) for idx, v in enumerate(options) if idx < len(labels)]
        else:
            items = [(label, emotion) for label, emotion in zip("ABCDEFGH", self.EMOTION_OPTIONS)]
        if not items:
            items = [(label, emotion) for label, emotion in zip("ABCDEFGH", self.EMOTION_OPTIONS)]
        return "\n".join(f"{label}. {value}" for label, value in items)

    def build_prompt_text(self, record: Dict[str, Any]) -> str:
        text = self.extract_text(record)
        emotion_type = record.get("emotion_type") or ""
        is_metaphor = record.get("is_metaphor", "")
        caption = record.get("caption", "")
        problem = str(record.get("problem", "")).strip() or self.DEFAULT_PROBLEM
        options_text = self._format_options_text(record.get("options"))

        return self.USER_PROMPT_TEMPLATE.format(
            problem=problem,
            options_text=options_text,
            correct_answer=emotion_type,
            is_metaphor=is_metaphor,
            text=text,
            emotion_type=emotion_type,
            caption=caption,
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

    def build_output_record(self, record: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        output = dict(record)
        output["think"] = annotation.get("analysis", "")
        return output

    def handle_error_record(self, record: Dict[str, Any], response: str) -> Dict[str, Any]:
        output = dict(record)
        output["annotation_error"] = "parse_failed"
        output["raw_response"] = response
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate MultiMeme (stage 2: chain-of-thought).")
    parser.add_argument("--input", required=True, help="Input JSONL path (stage1 output)")
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

    annotator = MetMemeAnnotator(config, image_root=args.image_root)
    annotator.run()


if __name__ == "__main__":
    main()
