from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
from api_sync.api import StreamGenerator


SYSTEM_PROMPT = (
    "You are a professional image description assistant. Carefully inspect the image and produce a faithful, "
    "detailed visual description without adding unsupported interpretation."
)

OUTPUT_FIELDS = (
    "image_path",
    "text",
    "is_metaphor",
    "metaphor_path",
    "emotion_type",
    "caption",
    "extra_info",
    "think",
    "prompt_with_image",
)


def parse_api_keys(value: Optional[str], model_name: str = "") -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]

    preferred = (
        ["GEMINI_API_KEYS", "MM_API_KEYS", "API_KEYS", "OPENAI_API_KEY"]
        if model_name.lower().startswith("gemini")
        else ["MM_API_KEYS", "API_KEYS", "OPENAI_API_KEY", "GEMINI_API_KEYS"]
    )
    env_value = next((os.getenv(name) for name in preferred if os.getenv(name)), None)
    if not env_value:
        raise ValueError(
            "API keys not provided. Use --api_keys or set "
            "MM_API_KEYS / GEMINI_API_KEYS / API_KEYS."
        )
    return [item.strip() for item in env_value.split(",") if item.strip()]


def clean_caption_text(raw: str) -> str:
    cleaned = " ".join(str(raw).strip().split())
    if cleaned and "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[0].strip()
    return cleaned


class BaseCaptionDataset:
    USER_PROMPT = ""

    def __init__(self, input_path: Path, image_root: Optional[Path] = None) -> None:
        self.input_path = input_path
        self.image_root = image_root

    def load_records(self, limit: int = 0) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_image_filename(self, record: Dict[str, Any]) -> str:
        raise NotImplementedError

    def build_user_prompt(self, record: Dict[str, Any]) -> str:
        return self.USER_PROMPT

    def build_output_record(self, record: Dict[str, Any], caption: str) -> Dict[str, Any]:
        raise NotImplementedError

    def resolve_image_path(self, record: Dict[str, Any]) -> Path:
        filename = self.get_image_filename(record)
        image_path = Path(filename)
        if image_path.is_absolute():
            return image_path
        if self.image_root is None:
            raise ValueError("image_root is required when image_path is not absolute.")
        return self.image_root / image_path

    def get_resume_key(self, record: Dict[str, Any]) -> str:
        return self.get_image_filename(record)

    def get_output_image_path(self, record: Dict[str, Any]) -> str:
        return self.get_image_filename(record)


class MemeCapCaptionDataset(BaseCaptionDataset):
    USER_PROMPT = (
        "Describe the image in one or two sentences, including the elements necessary to understand the image meaning, without adding any interpretation of the image.\n\n"
        "Here is a rough caption for the image: {reference_captions}\n\n"
        "This caption is incomplete. Please observe the image carefully, refine and supplement the caption based on the original one, adding crucial details not mentioned but essential for understanding the image. Output only the final revised caption in 1-2 sentences."
    )

    def load_records(self, limit: int = 0) -> List[Dict[str, Any]]:
        with self.input_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {self.input_path}")
        if limit > 0:
            return records[:limit]
        return records

    def get_image_filename(self, record: Dict[str, Any]) -> str:
        image_name = str(record.get("img_fname") or "").strip()
        if not image_name:
            raise ValueError("memecap record is missing img_fname")
        return image_name

    def build_user_prompt(self, record: Dict[str, Any]) -> str:
        image_reference_captions = record.get("img_captions") or []
        meme_reference_captions = record.get("meme_captions") or []
        prompt_parts: List[str] = []
        if image_reference_captions:
            image_references = "\n".join(f"- {caption}" for caption in image_reference_captions)
            prompt_parts.append(f"Reference image captions:\n{image_references}")
        if meme_reference_captions:
            meme_references = "\n".join(f"- {caption}" for caption in meme_reference_captions)
            prompt_parts.append(f"Reference meme captions:\n{meme_references}")
        reference_captions = "\n\n".join(prompt_parts) if prompt_parts else ""
        return self.USER_PROMPT.format(reference_captions=reference_captions)

    def build_output_record(self, record: Dict[str, Any], caption: str) -> Dict[str, Any]:
        output_record = {
            "image_path": self.get_output_image_path(record),
            "text": "",
            "is_metaphor": True,
            "metaphor_path": "",
            "emotion_type": "",
            "caption": caption,
            "extra_info": {
                "category": record.get("category", ""),
                "title": record.get("title", ""),
                "img_captions": record.get("img_captions", []),
                "meme_captions": record.get("meme_captions", []),
            },
            "think": "",
            "prompt_with_image": "",
        }
        return {field: output_record[field] for field in OUTPUT_FIELDS}


class ImageMetCaptionDataset(BaseCaptionDataset):
    USER_PROMPT = (
        "Describe the image in one or two sentences, including the elements necessary to understand the image meaning, "
        "without adding any interpretation of the image.\n\n"
        "Here is the most important clue for understanding the image meaning: {reference_captions}\n\n"
        "This clue should be treated as the key starting point. Please observe the image carefully, make only simple "
        "modifications to turn it into a natural visual content description, and add crucial visible details needed "
        "for understanding the image. Output only the final revised caption in 1-2 sentences."
    )

    def load_records(self, limit: int = 0) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit > 0 and len(records) >= limit:
                    break
        return records

    def get_image_filename(self, record: Dict[str, Any]) -> str:
        image_name = str(record.get("image_path") or "").strip()
        if not image_name:
            raise ValueError("imagemet record is missing image_path")
        return image_name

    def build_user_prompt(self, record: Dict[str, Any]) -> str:
        visual_elaboration = str(record.get("visual_elaboration") or "").strip()
        reference_captions = visual_elaboration if visual_elaboration else "No clue provided."
        return self.USER_PROMPT.format(reference_captions=reference_captions)

    def build_output_record(self, record: Dict[str, Any], caption: str) -> Dict[str, Any]:
        output_record = {
            "image_path": self.get_output_image_path(record),
            "text": "",
            "is_metaphor": True,
            "metaphor_path": "",
            "emotion_type": "",
            "caption": caption,
            "extra_info": {
                "source": record.get("source", ""),
                "target": record.get("target", ""),
                "generated_linguistic_metaphor": record.get("generated_linguistic_metaphor", ""),
                "entailing_literal": record.get("entailing_literal", ""),
                "literal_description": record.get("literal_description", ""),
                "objects": record.get("objects", ""),
                "properties": record.get("properties", ""),
                "relations": record.get("relations", ""),
            },
            "think": "",
            "prompt_with_image": "",
        }
        return {field: output_record[field] for field in OUTPUT_FIELDS}


class MetMemeCaptionDataset(BaseCaptionDataset):
    USER_PROMPT = (
        "Please observe this meme image carefully and describe the image in one or two sentences, including all the elements which are necessary to understand the image meaning, "
        "without adding any interpretation of the image.\n\n"
    )

    def load_records(self, limit: int = 0) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit > 0 and len(records) >= limit:
                    break
        return records

    def get_image_filename(self, record: Dict[str, Any]) -> str:
        image_path = str(record.get("image_path") or record.get("images_name") or "").strip()
        if not image_path:
            raise ValueError("metmeme record is missing image_path/images_name")
        return image_path

    def build_output_record(self, record: Dict[str, Any], caption: str) -> Dict[str, Any]:
        output_record = {
            "image_path": self.get_output_image_path(record),
            "text": str(record.get("text") or ""),
            "is_metaphor": bool(record.get("metaphor occurrence") == "1"),
            "metaphor_path": "",
            "emotion_type": _normalize_emotion_type(record.get("sentiment category")),
            "caption": caption,
            "extra_info": {},
            "think": "",
            "prompt_with_image": "",
        }
        return {field: output_record[field] for field in OUTPUT_FIELDS}


class CiiBenchCaptionDataset(BaseCaptionDataset):
    USER_PROMPT = (
        "Describe the image in one or two sentences, including all the elements which are necessary to understand the image meaning\n\n"
        "This explanation reflects an understanding of the image meaning: {reference_explanation}\n\n"
        "Use it only as a clue to identify which visual evidence matters. Reference what can actually be seen in the "
        "image and emphasize the specific image elements that correspond to that clue. Do not mention any meaning or "
        "explain what the image suggests. Output only the final key elements visual description in 1-2 sentences."
    )

    def load_records(self, limit: int = 0) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit > 0 and len(records) >= limit:
                    break
        return records

    def get_image_filename(self, record: Dict[str, Any]) -> str:
        image_path = str(record.get("image_path") or "").strip()
        if not image_path:
            image_info = record.get("image")
            if isinstance(image_info, dict):
                nested_path = str(image_info.get("path") or "").strip()
                if nested_path:
                    image_path = nested_path.removeprefix("images/")
        if not image_path:
            raise ValueError("ciibench record is missing image_path")
        return image_path

    def build_user_prompt(self, record: Dict[str, Any]) -> str:
        extra_info = record.get("extra_info") or {}
        explanation = str(record.get("explanation") or extra_info.get("explanation") or "").strip()
        reference_explanation = explanation if explanation else "No clue provided."
        return self.USER_PROMPT.format(reference_explanation=reference_explanation)

    def build_output_record(self, record: Dict[str, Any], caption: str) -> Dict[str, Any]:
        extra_info = record.get("extra_info") or {}
        output_record = {
            "image_path": self.get_output_image_path(record),
            "text": str(record.get("text") or ""),
            "is_metaphor": bool(record.get("is_metaphor", True)),
            "metaphor_path": str(record.get("metaphor_path") or ""),
            "emotion_type": str(record.get("emotion_type") or ""),
            "caption": caption,
            "extra_info": {
                "emotion": record.get("emotion", extra_info.get("emotion", "")),
                "explanation": record.get("explanation", extra_info.get("explanation", "")),
                "metaphorical_meaning": record.get(
                    "metaphorical_meaning",
                    extra_info.get("metaphorical_meaning", ""),
                ),
            },
            "think": "",
            "prompt_with_image": "",
        }
        return {field: output_record[field] for field in OUTPUT_FIELDS}


class VFluteCaptionDataset(BaseCaptionDataset):
    USER_PROMPT = (
        "Describe the image in one or two sentences, including all the elements which are necessary to understand the image meaning\n\n"
        "This explanation reflects an understanding of the image meaning: {reference_explanation}\n\n"
        "Use it only as a clue to identify which visual evidence matters. Reference what can actually be seen in the "
        "image and emphasize the specific image elements that correspond to that clue. Do not mention any meaning or "
        "claim behind the image. Output only the final key elements visual description in 1-2 sentences."
    )

    def load_records(self, limit: int = 0) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit > 0 and len(records) >= limit:
                    break
        return records

    def get_image_filename(self, record: Dict[str, Any]) -> str:
        image_path = str(record.get("image_path") or "").strip()
        if not image_path:
            raise ValueError("vflute record is missing image_path")
        return image_path

    def build_user_prompt(self, record: Dict[str, Any]) -> str:
        extra_info = record.get("extra_info") or {}
        explanation = str(extra_info.get("explanation") or "").strip()
        reference_explanation = explanation if explanation else "No clue provided."
        return self.USER_PROMPT.format(reference_explanation=reference_explanation)

    def build_output_record(self, record: Dict[str, Any], caption: str) -> Dict[str, Any]:
        output_record = {
            "image_path": self.get_output_image_path(record),
            "text": str(record.get("text") or ""),
            "is_metaphor": bool(record.get("is_metaphor", True)),
            "metaphor_path": str(record.get("metaphor_path") or ""),
            "emotion_type": str(record.get("emotion_type") or ""),
            "caption": caption,
            "extra_info": dict(record.get("extra_info") or {}),
            "think": str(record.get("think") or ""),
            "prompt_with_image": str(record.get("prompt_with_image") or ""),
        }
        return {field: output_record[field] for field in OUTPUT_FIELDS}


def _normalize_emotion_type(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "(" in raw and raw.endswith(")"):
        return raw.split("(", 1)[-1].rstrip(")")
    return raw


DATASET_REGISTRY = {
    "ciibench": CiiBenchCaptionDataset,
    "memecap": MemeCapCaptionDataset,
    "imagemet": ImageMetCaptionDataset,
    "metmeme": MetMemeCaptionDataset,
    "vflute": VFluteCaptionDataset,
}


async def annotate_records_with_api(
    records: List[Dict[str, Any]],
    dataset: BaseCaptionDataset,
    output_path: Path,
    model_name: str,
    api_keys: List[str],
    max_concurrent: int,
    max_retries: int,
) -> None:
    stream = StreamGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent,
        max_retries=max_retries,
        rational=False,
        with_unique_id=True,
    )
    prompts: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompts.append(
            {
                "id": str(idx),
                "prompt": [
                    {"type": "image", "image": str(dataset.resolve_image_path(record))},
                    {"type": "text", "text": dataset.build_user_prompt(record)},
                ],
            }
        )

    existing_records, completed_keys = _load_existing_output(output_path)
    if existing_records:
        print(f"[captioner] resuming from existing output={output_path}, completed={len(completed_keys)}")

    pending_items: List[tuple[int, Dict[str, Any]]] = []
    for idx, record in enumerate(records):
        if dataset.get_output_image_path(record) in completed_keys:
            continue
        pending_items.append((idx, record))

    completed_ids: set[int] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() and output_path.stat().st_size > 0 else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        async for result in stream.generate_stream(
            prompts=[
                {
                    "id": str(idx),
                    "prompt": [
                        {"type": "image", "image": str(dataset.resolve_image_path(record))},
                        {"type": "text", "text": dataset.build_user_prompt(record)},
                    ],
                }
                for idx, record in pending_items
            ],
            system_prompt=SYSTEM_PROMPT,
        ):
            rec_idx = int(result["id"])
            raw = str(result.get("result") or "")
            if raw.startswith("__ERROR__:data_inspection_failed"):
                caption = ""
                annotation_error = "data_inspection_failed"
            elif raw.startswith("__ERROR__:request_failed"):
                caption = ""
                annotation_error = "request_failed"
            else:
                caption = clean_caption_text(raw)
                annotation_error = ""

            record = records[rec_idx]
            output_record = dataset.build_output_record(record, caption=caption)
            if annotation_error:
                output_record["annotation_error"] = annotation_error
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            completed_ids.add(rec_idx)
            completed = len(completed_ids)
            if completed % 50 == 0 or completed == len(pending_items):
                print(f"[captioner][api] progress {completed}/{len(records)}")

        for idx, record in pending_items:
            if idx in completed_ids:
                continue
            output_record = dataset.build_output_record(record, caption="")
            output_record["annotation_error"] = "request_failed"
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()


def _load_existing_output(
    output_path: Path,
) -> tuple[List[Dict[str, Any]], set[str]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return [], set()
    existing_records: List[Dict[str, Any]] = []
    completed_keys: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            existing_records.append(record)
            key = str(record.get("image_path") or "").strip()
            if key:
                completed_keys.add(key)
    return existing_records, completed_keys


class CLI:
    def run(
        self,
        input: str,
        output: str,
        dataset: str,
        image_root: Optional[str] = None,
        model_path: str = "",
        api_model_name: str = "",
        api_keys: Optional[str] = None,
        max_concurrent: int = 50,
        max_retries: int = 5,
        generate_limit: int = 0,
        limit: int = 0,
    ) -> None:
        dataset_cls = DATASET_REGISTRY.get(dataset.strip().lower())
        if dataset_cls is None:
            raise ValueError(f"Unsupported dataset: {dataset}. Available datasets: {sorted(DATASET_REGISTRY)}")

        input_path = Path(input)
        output_path = Path(output)
        image_root_path = Path(image_root) if image_root else None
        dataset_adapter = dataset_cls(input_path=input_path, image_root=image_root_path)

        records = dataset_adapter.load_records(limit=limit)
        if generate_limit > 0:
            records = records[:generate_limit]
        print(f"[captioner] loaded records={len(records)} from {input_path}")

        model_name = (api_model_name or model_path).strip()
        if not model_name:
            raise ValueError("Set --api_model_name or --model_path for API inference.")
        key_list = parse_api_keys(api_keys, model_name=model_name)

        asyncio.run(
            annotate_records_with_api(
                records=records,
                dataset=dataset_adapter,
                output_path=output_path,
                model_name=model_name,
                api_keys=key_list,
                max_concurrent=max_concurrent,
                max_retries=max_retries,
            )
        )
        print(f"[captioner] done output={output_path}")


if __name__ == "__main__":
    fire.Fire(CLI)
