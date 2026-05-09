from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import fire

from api_sync.api import StreamGenerator
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
    start_index: int = 1
    end_index: Optional[int] = None
    limit: int = 0


def parse_api_keys(value: Optional[str]) -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    env_value = (
        os.getenv("MM_API_KEYS")
        or os.getenv("API_KEYS")
        or os.getenv("OPENAI_API_KEY")
    )
    if not env_value:
        raise ValueError("API keys not provided. Use --api_keys or set MM_API_KEYS/API_KEYS.")
    return [item.strip() for item in env_value.split(",") if item.strip()]


def _first_value(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _format_field(label: str, value: Any) -> str:
    if value is None:
        value = ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return f"- {label}: {value}"


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _common_metaphor_fields(record: Dict[str, Any]) -> List[str]:
    return [
        _format_field("is_metaphor", record.get("is_metaphor", "")),
        _format_field("metaphor_path", record.get("metaphor_path", "")),
        _format_field("emotion_type", record.get("emotion_type", "")),
        _format_field("caption", record.get("caption", "")),
    ]



class BaseAnnotator(ABC):
    ANALYSIS_TAG_PATTERN = re.compile(
        r"^\s*<caption>.*?</caption>\s*<metaphor>.*?</metaphor>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
        re.DOTALL,
    )

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
    DEFAULT_PROBLEM = (
        "What metaphor is shown in this image, and which emotion category best matches its implied meaning?"
    )

    SYSTEM_PROMPT = (
        "You are an expert metaphorical image emotion analysis assistant specialized in creating natural, flowing Chain of Thought reasoning process.\n\n"
        "You will seen:\n"
        "- A question about the metaphor and emotion in one image\n"
        "- Answer options with emotion categories\n"
        "- The correct answer\n"
        "- Key information describing the image and its metaphorical content\n\n"
        "Your task:\n"
        "- Analyze the image using provided key fields.\n"
        "- Use this exact tag order in your reasoning output:\n"
        "  1) <caption></caption>: initial visual perception only, no metaphor or emotion analysis. Consistent basically with the visual content description of the image provided to you.\n"
        "  2) <metaphor></metaphor>: check whether metaphor exists based on the provided information; if yes, explain the understanding path using the pathway definitions above and keep it consistent with the given metaphor_path when provided; if not, output 'There is no metaphor', then also give a simple explanation in 1 sentence.\n"
        "Metaphor comprehension pathways:\n"
        "- direct: The image employs common idioms or fixed expressions, or its metaphor can be recognized at a glance without additional interpretation.\n"
        "- sequential: When reading the text sequentially and viewing the image, one first perceives the literal meaning of the picture. However, upon integrating the context and the content of the image, this literal meaning is revealed to be incorrect, and a cognitive shift is required to truly grasp the metaphorical meaning it conveys.\n"
        "- parallel: After examining the entire image, both its metaphorical and literal meanings are quite common, with roughly equal weight in comprehension. Unlike direct expressions, which only evoke one meaning (for instance, one does not think of the literal meaning when using an idiom).\n\n"
        "  3) <think></think>: deeper reasoning that combines visual clues and metaphor judgment for image's emotion analyse.\n"
        "  4) <answer></answer>: final answer as one emotion label from options.\n"
        "- Keep the reasoning natural, but concise and evidence-based.\n"
        "- Use English only.\n\n"
        "Example Output Style:\n"
        "<caption>In the top panel, a monkey is thoughtfully selecting one of several ropes to climb on in a forest. In the bottom panel, a human is in a subway station, pointing at a subway map as if choosing a route or getting information.</caption>\n"
        "<metaphor>There is a metaphor, and the path is parallel. Both scenes show route selection in different "
        "worlds, linking animal movement and urban navigation. Literal and metaphorical readings are both common "
        "and equally salient.</metaphor>\n"
        "<think>The focus is comparison, not emotional drama. Both figures look focused and calm, with no strong "
        "signals of joy, fear, anger, or sadness. So neutral fits best.</think>\n"
        "<answer>neutral</answer>\n\n"
        "Output JSON only in this format, and donot output any other tags:\n"
        "{\"analysis\": \"<caption>...</caption>\\n<metaphor>...</metaphor>\\n<think>...</think>\\n<answer>...</answer>\"}"
    )

    USER_PROMPT_TEMPLATE = (
        "A user may ask the following question about this image: {problem}\n\n"
        "Answer Options:\n"
        "{options_text}\n\n"
        "Correct emotion category: {correct_answer}\n\n"
        "Use the following information to understand the image and its metaphorical meaning:\n"
        "{known_fields}\n\n"
        "Task: Generate a natural, conversational chain of thought for metaphorical image's emotion analysis. "
        "Use exactly these tags in order: <caption>, <metaphor>, <think>, <answer>."
    )

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
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    @abstractmethod
    def build_known_fields(self, record: Dict[str, Any]) -> str:
        raise NotImplementedError

    def emotion_options(self) -> Sequence[str]:
        return self.EMOTION_OPTIONS

    def _format_options_text(self, options: Any) -> str:
        if isinstance(options, dict):
            items = [(str(k), str(v)) for k, v in options.items()]
        elif isinstance(options, list):
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            items = [(labels[idx], str(v)) for idx, v in enumerate(options) if idx < len(labels)]
        else:
            items = [(label, emotion) for label, emotion in zip("ABCDEFGH", self.emotion_options())]
        if not items:
            items = [(label, emotion) for label, emotion in zip("ABCDEFGH", self.emotion_options())]
        return "\n".join(f"{label}. {value}" for label, value in items)

    def build_prompt_text(self, record: Dict[str, Any]) -> str:
        problem = str(record.get("problem", "")).strip() or self.DEFAULT_PROBLEM
        options_text = self._format_options_text(record.get("options"))
        correct_answer = record.get("emotion_type") or ""
        known_fields = self.build_known_fields(record)
        return self.USER_PROMPT_TEMPLATE.format(
            problem=problem,
            options_text=options_text,
            correct_answer=correct_answer,
            known_fields=known_fields,
        ).strip()

    def build_prompt(self, record: Dict[str, Any]) -> Union[str, List[Dict[str, Any]]]:
        return self.build_prompt_text(record)

    def build_output_record(self, record: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        output = dict(record)
        output["think"] = annotation.get("analysis", "")
        return output

    def handle_error_record(self, record: Dict[str, Any], response: str) -> Dict[str, Any]:
        output = dict(record)
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
                if self.config.limit > 0 and len(records) >= self.config.limit:
                    break
        return records

    def get_resume_key(self, record: Dict[str, Any]) -> str:
        return str(record.get("image_path", "")).strip()

    def record_has_required_output(self, record: Dict[str, Any]) -> bool:
        think = record.get("think")
        if not isinstance(think, str) or not think.strip():
            return False
        return self.ANALYSIS_TAG_PATTERN.match(think) is not None

    def _load_existing_output(self) -> Dict[str, Dict[str, Any]]:
        if not self.config.output_path.exists() or self.config.output_path.stat().st_size == 0:
            return {}

        existing_by_key: Dict[str, Dict[str, Any]] = {}
        with self.config.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = self.get_resume_key(record)
                if key:
                    existing_by_key[key] = record
        return existing_by_key

    def _validate_response(self, response: str) -> Optional[Dict[str, Any]]:
        if isinstance(response, dict):
            return response
        return JSONParser.parse(response)

    def _validate_annotation_format(self, annotation: Dict[str, Any]) -> bool:
        analysis = annotation.get("analysis")
        if not isinstance(analysis, str):
            return False
        return self.ANALYSIS_TAG_PATTERN.match(analysis) is not None

    async def _run_async(self, records: Sequence[Dict[str, Any]]) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        annotations_by_idx: Dict[int, Dict[str, Any]] = {}
        existing_output_by_key = self._load_existing_output()
        resumed_output_by_idx: Dict[int, Dict[str, Any]] = {}
        pending_indices: List[int] = []
        for idx, record in enumerate(records):
            existing_output = existing_output_by_key.get(self.get_resume_key(record))
            if existing_output and self.record_has_required_output(existing_output):
                resumed_output_by_idx[idx] = existing_output
                continue
            pending_indices.append(idx)

        print(
            f"[cot_generator] total={len(records)}, to_annotate={len(pending_indices)}, "
            f"skipped={len(resumed_output_by_idx)}"
        )
        round_id = 0

        while pending_indices:
            round_id += 1
            prompts = [
                {"id": str(idx), "prompt": self.build_prompt(records[idx])}
                for idx in pending_indices
            ]
            next_pending: List[int] = []
            seen_indices = set()

            async for result in self.stream.generate_stream(
                prompts=prompts,
                system_prompt=self.system_prompt,
                validate_func=self._validate_response,
            ):
                record_index = int(result["id"])
                seen_indices.add(record_index)
                response = result["result"]
                annotation = response if isinstance(response, dict) else self._validate_response(response)

                if annotation is None or not self._validate_annotation_format(annotation):
                    next_pending.append(record_index)
                    continue
                annotations_by_idx[record_index] = annotation

            for idx in pending_indices:
                if idx not in seen_indices and idx not in next_pending:
                    next_pending.append(idx)

            pending_indices = next_pending
            print(f"[cot_generator] format-check round={round_id}, remaining={len(pending_indices)}")

        final_outputs: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            if idx in resumed_output_by_idx:
                final_outputs.append(resumed_output_by_idx[idx])
                continue
            output_record = self.build_output_record(record, annotations_by_idx[idx])
            final_outputs.append(output_record)

        with self.config.output_path.open("w", encoding="utf-8") as handle:
            for output_record in final_outputs:
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    def run(self) -> None:
        records = self.load_records()
        start = self.config.start_index
        end = self.config.end_index if self.config.end_index is not None else len(records)

        if start < 1:
            raise ValueError(f"--start must be >= 1, got {start}")
        if end < start:
            raise ValueError(f"--end must be >= --start, got start={start}, end={end}")

        selected_records = records[start - 1:end]
        asyncio.run(self._run_async(selected_records))



class MetMemeAnnotator(BaseAnnotator):
    def __init__(self, *args, image_root: Optional[Path] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_root = Path(image_root) if image_root else None

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

    def build_known_fields(self, record: Dict[str, Any]) -> str:
        text = _first_value(record, ["text", "Text"]) or ""
        is_metaphor_value = bool(record.get("is_metaphor", False))
        metaphor_path = _stringify_value(record.get("metaphor_path", ""))
        emotion_type = _stringify_value(record.get("emotion_type", ""))
        caption = _stringify_value(record.get("caption", ""))
        text_value = _stringify_value(text)
        metaphor_sentence = (
            "This image is a metaphorical image.\n"
            if is_metaphor_value
            else "This image does not contain metaphorical content.\n"
        )
        metaphor_path_sentence = (
            f"The metaphor understand path is {metaphor_path}.\n"
            if is_metaphor_value
            else ""
        )
        return (
            f"{metaphor_sentence}\n"
            f"The image's visual content is: {caption}\n"
            f"The text appearing in the image is: {text_value}.\n"
            f"{metaphor_path_sentence}"
        )


class CIIBenchAnnotator(BaseAnnotator):
    EMOTION_OPTIONS = ["positive", "negative", "neutral"]

    def emotion_options(self) -> Sequence[str]:
        return self.EMOTION_OPTIONS

    def build_known_fields(self, record: Dict[str, Any]) -> str:
        extra = record.get("extra_info") or {}
        is_metaphor = _stringify_value(record.get("is_metaphor", ""))
        metaphor_path = _stringify_value(record.get("metaphor_path", ""))
        emotion_type = _stringify_value(record.get("emotion_type", ""))
        caption = _stringify_value(record.get("caption", ""))
        explanation = _stringify_value(extra.get("explanation", ""))
        metaphorical_meaning = _stringify_value(extra.get("metaphorical_meaning", ""))
        return (
            f"This image is a metaphorical image.\n"
            f"The image's visual content is: {caption}\n"
            f"Its metaphor understanding path is {metaphor_path}.\n"
            f"Here is an explanation of the metaphorical content in the image: {explanation}\n"
            f"The intended metaphorical meaning is: {metaphorical_meaning}.\n"
        )


class ImageMetAnnotator(BaseAnnotator):
    def build_known_fields(self, record: Dict[str, Any]) -> str:
        extra = record.get("extra_info") or {}
        source = _stringify_value(extra.get("source", ""))
        target = _stringify_value(extra.get("target", ""))
        linguistic_metaphor = _stringify_value(extra.get("generated_linguistic_metaphor", ""))
        entailing_literal = _stringify_value(extra.get("entailing_literal", ""))
        literal_description = _stringify_value(extra.get("literal_description", ""))
        objects = _stringify_value(extra.get("objects", ""))
        properties = _stringify_value(extra.get("properties", ""))
        relations = _stringify_value(extra.get("relations", ""))
        return (
            f"This image is a metaphorical image.\n"
            f"The image's visual content is: {_stringify_value(record.get('caption', ''))}\n"
            f"Its metaphor understanding path is {_stringify_value(record.get('metaphor_path', ''))}.\n"
            f"In this metaphor, the source domain is: {source}.\n"
            f"The target domain is: {target}.\n"
            f"The linguistic metaphor meaning is: {linguistic_metaphor}. {entailing_literal}\n"
            f"The description of the situation is: {literal_description}.\n"
            f"The key objects involved are: {objects}.\n"
            f"The important properties highlighted by the metaphor are: {properties}.\n"
            f"The important relations between the objects are: {relations}.\n"
        )


class MemeCapAnnotator(BaseAnnotator):
    EXCLUDED_EXTRA_KEYS = {"img_captions"}

    def build_known_fields(self, record: Dict[str, Any]) -> str:
        extra = record.get("extra_info") or {}
        meme_captions_value = extra.get("meme_captions", "")
        if isinstance(meme_captions_value, list):
            meme_captions = " ".join(
                str(item).strip() for item in meme_captions_value if str(item).strip()
            )
        else:
            meme_captions = _stringify_value(meme_captions_value)
        return (
            f"This image is a metaphorical image.\n"
            f"The image's visual content is: {_stringify_value(record.get('caption', ''))}\n"
            f"Its metaphor understanding path is {_stringify_value(record.get('metaphor_path', ''))}.\n"
            f"Explanation of the implied meaning of the meme: {meme_captions}.\n"
        )


class VFluteAnnotator(BaseAnnotator):
    def build_known_fields(self, record: Dict[str, Any]) -> str:
        extra = record.get("extra_info") or {}
        return (
            f"This image is a metaphorical image.\n"
            f"The image's visual content is: {_stringify_value(record.get('caption', ''))}\n"
            f"Its metaphor understanding path is {_stringify_value(record.get('metaphor_path', ''))}.\n"
            f"Here is an explanation that may help you understand the metaphorical content in the image: "
            f"{_stringify_value(extra.get('explanation', ''))}.\n"
        )


ANNOTATOR_REGISTRY = {
    "metmeme": MetMemeAnnotator,
    "ciibench": CIIBenchAnnotator,
    "imagemet": ImageMetAnnotator,
    "memecap": MemeCapAnnotator,
    "vflute": VFluteAnnotator,
}


def run(
    dataset: str,
    input: str,
    output: str,
    model: str,
    api_keys: Optional[str] = None,
    max_concurrent: int = 50,
    max_retries: int = 5,
    start: int = 1,
    end: Optional[int] = None,
    limit: int = 0,
    image_root: Optional[str] = None,
) -> None:
    """Generate chain-of-thought annotations for a supported dataset.

    Args:
        dataset: One of metmeme, ciibench, imagemet, memecap, vflute.
        input: Input JSONL path (stage1 output).
        output: Output JSONL path.
        model: Model name.
        api_keys: Comma-separated API keys. Falls back to MM_API_KEYS/API_KEYS/OPENAI_API_KEY.
        max_concurrent: Max concurrent requests per key.
        max_retries: Max retries per request.
        start: Start record index (1-based, inclusive).
        end: End record index (1-based, inclusive). None means end of file.
        limit: Max number of input records to load before range selection.
        image_root: Only used by metmeme to resolve relative image paths.
    """
    dataset_key = dataset.lower()
    if dataset_key not in ANNOTATOR_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {sorted(ANNOTATOR_REGISTRY)}"
        )

    config = AnnotatorConfig(
        input_path=Path(input),
        output_path=Path(output),
        model_name=model,
        api_keys=parse_api_keys(api_keys),
        max_concurrent_per_key=max_concurrent,
        max_retries=max_retries,
        start_index=start,
        end_index=end,
        limit=limit,
    )

    annotator_cls = ANNOTATOR_REGISTRY[dataset_key]
    if annotator_cls is MetMemeAnnotator:
        annotator = annotator_cls(config, image_root=image_root)
    else:
        annotator = annotator_cls(config)
    annotator.run()


if __name__ == "__main__":
    fire.Fire(run)
