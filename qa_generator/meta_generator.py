from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser

EXAMPLE_RECORD = {
    "image_path": "/abs/path/to/image.jpg",
    "text": "Original text from dataset",
    "is_metaphor": True,
    "metaphor_path": "sequential",
    "emotion_type": "love",
    "caption": "",
    "extra_info": {},
    "think": "",
}

SYSTEM_PROMPT = "You are a careful multimodal annotator. Return valid JSON only."


@dataclass
class AnnotatorConfig:
    input_path: Path
    output_path: Path
    model_name: str
    api_keys: List[str]
    max_concurrent_per_key: int = 50
    max_retries: int = 5
    limit: int = 0


def parse_api_keys(value: Optional[str]) -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    env_value = os.getenv("MM_API_KEYS") or os.getenv("API_KEYS") or os.getenv("OPENAI_API_KEY")
    if not env_value:
        raise ValueError("API keys not provided. Use --api_keys or set MM_API_KEYS/API_KEYS.")
    return [item.strip() for item in env_value.split(",") if item.strip()]


class BaseAnnotator(ABC):
    def __init__(self, config: AnnotatorConfig, image_root: Optional[Path] = None) -> None:
        self.config = config
        self.image_root = image_root
        self.stream = StreamGenerator(
            model_name=config.model_name,
            api_keys=config.api_keys,
            max_concurrent_per_key=config.max_concurrent_per_key,
            max_retries=config.max_retries,
            rational=False,
            with_unique_id=True,
        )


    @abstractmethod
    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, fields: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def load_records(self) -> List[Dict[str, Any]]:
        if self.config.input_path.suffix.lower() == ".json":
            with self.config.input_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
            else:
                raise ValueError(f"Unsupported JSON format in {self.config.input_path}")
            if self.config.limit > 0:
                records = records[: self.config.limit]
            return records

        records: List[Dict[str, Any]] = []
        with self.config.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                if self.config.limit > 0 and len(records) >= self.config.limit:
                    break
        return records

    def _parse_annotation(self, response: Any) -> Optional[Dict[str, Any]]:
        if isinstance(response, dict):
            return response
        return JSONParser.parse(str(response))

    def _build_prompt_payload(self, fields: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        prompt = self.build_prompt(fields)
        image_path = fields.get("image_path", "")
        if image_path:
            return [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ]
        return prompt

    async def annotate(self) -> None:
        records = self.load_records()
        fields_by_idx: Dict[int, Dict[str, Any]] = {}
        prompts: List[Dict[str, Any]] = []

        for idx, record in enumerate(records):
            fields = self.extract_fields(record)
            fields_by_idx[idx] = fields
            # For datasets with explicit metaphor labels, only annotate metaphor samples.
            if "is_metaphor" in fields and fields["is_metaphor"] is not True:
                continue
            prompts.append({"id": str(idx), "prompt": self._build_prompt_payload(fields)})

        total_records = len(records)
        total_to_annotate = len(prompts)
        total_skipped = total_records - total_to_annotate
        print(
            f"[meta_generator] total={total_records}, "
            f"to_annotate={total_to_annotate}, skipped={total_skipped}"
        )

        annotations_by_idx: Dict[int, Dict[str, Any]] = {}
        parse_failed_by_idx: Dict[int, str] = {}
        completed = 0

        if prompts:
            async for result in self.stream.generate_stream(
                prompts=prompts,
                system_prompt=SYSTEM_PROMPT,
                validate_func=self._parse_annotation,
            ):
                idx = int(result["id"])
                raw = result["result"]
                parsed = self._parse_annotation(raw)
                completed += 1
                if parsed is None:
                    annotations_by_idx[idx] = {}
                    parse_failed_by_idx[idx] = str(raw)
                else:
                    annotations_by_idx[idx] = parsed

                if completed % 50 == 0 or completed == total_to_annotate:
                    print(f"[meta_generator] progress: {completed}/{total_to_annotate}")
        else:
            print("[meta_generator] no records require model annotation")

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_path.open("w", encoding="utf-8") as handle:
            for idx in range(len(records)):
                fields = fields_by_idx[idx]
                annotation = annotations_by_idx.get(idx, {})
                output = self.build_output_record(fields, annotation)
                if idx in parse_failed_by_idx:
                    output["annotation_error"] = "parse_failed"
                    output["raw_response"] = parse_failed_by_idx[idx]
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

        parse_failed_count = len(parse_failed_by_idx)
        success_count = total_to_annotate - parse_failed_count
        print(
            f"[meta_generator] done: success={success_count}, parse_failed={parse_failed_count}, "
            f"skipped={total_skipped}, output={self.config.output_path}"
        )

    def run(self) -> None:
        asyncio.run(self.annotate())


class MetMemeAnnotator(BaseAnnotator):
    SENTIMENT_ID_TO_LABEL = {
        "1": "happiness",
        "2": "love",
        "3": "anger",
        "4": "sorrow",
        "5": "fear",
        "6": "hate",
        "7": "surprise",
    }

    PROMPT_TEMPLATE = (
        "You are annotating a social meme sample. "
        "The text on the picture says: {text}. "
        "It expresses {sentiment_category} emotion with a {sentiment_degree} degree, "
        "based on existing dataset annotations.\n\n"
        "Task: decide the metaphor understanding path as one of: direct, sequential, parallel.\n"
        "Use these practical criteria:\n"
        "- direct: the metaphor is recognized immediately at first glance from the image-text pair.\n"
        "- sequential: an obvious high-salience meaning appears first, but deeper thinking shows that first meaning is wrong and then the correct metaphorical meaning is reached.\n"
        "- parallel: at first glance there are multiple plausible interpretation directions, and it is not immediately clear which meaning is correct.\n\n"
        "Output JSON only in this format: {{\"metaphor_path\": \"direct\"}}"
    )

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = str(record.get("text", "")).strip()
        sentiment_raw = str(record.get("sentiment category", "")).strip()
        sentiment_degree = str(record.get("sentiment degree", "")).strip()
        metaphor_occurrence = str(record.get("metaphor occurrence", "")).strip().lower()
        image_path = str(record.get("image_path", "")).strip()

        if image_path and self.image_root and not Path(image_path).is_absolute():
            image_path = str(self.image_root / image_path)

        if "(" in sentiment_raw and ")" in sentiment_raw:
            emotion_type = sentiment_raw.split("(", 1)[1].split(")", 1)[0].strip().lower()
        elif sentiment_raw in self.SENTIMENT_ID_TO_LABEL:
            emotion_type = self.SENTIMENT_ID_TO_LABEL[sentiment_raw]
        else:
            emotion_type = sentiment_raw.lower()

        is_metaphor: Optional[bool] = None
        if metaphor_occurrence in {"1", "true", "yes"}:
            is_metaphor = True
        elif metaphor_occurrence in {"0", "false", "no"}:
            is_metaphor = False

        return {
            "image_path": image_path,
            "text": text,
            "emotion_type": emotion_type,
            "sentiment_degree": sentiment_degree,
            "is_metaphor": is_metaphor,
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        return self.PROMPT_TEMPLATE.format(
            text=fields["text"],
            sentiment_category=fields["emotion_type"],
            sentiment_degree=fields["sentiment_degree"],
        ).strip()

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": str(annotation.get("metaphor_path", "")),
            "emotion_type": fields["emotion_type"],
            "caption": "",
            "extra_info": {},
            "think": "",
        }


class YesButAnnotator(BaseAnnotator):
    EMOTION_PROMPT_TEMPLATE = (
        "You are annotating a YesBut sample from a JSON record with fields like "
        "description, caption, contradiction, and moral. "
        "All samples are metaphorical comics and you only need to classify the primary emotion category.\n\n"
        "The description of the image content is {caption}, and the implied metaphorical meaning is {moral}. "
        "Choose one emotion from: happiness, love, anger, sorrow, fear, hate, surprise, neutral. "
        "happiness means a sense of happiness, optimism, and relaxation, embracing feelings of tranquility and ecstasy. "
        "love means a profound and positive emotional and psychological state, signifying deep and sincere affection towards individuals or entities. "
        "This sentiment has the power to evoke warm attraction, intense passion, and selfless dedication. "
        "Typically, love manifests in interpersonal relationships, such as those between family members, friends, or romantic partners. "
        "anger means a potent emotion that surfaces when confronted with something bad or unjust. "
        "It encompasses feelings of trouble and rage, including annoyance and intense displeasure. "
        "sorrow is commonly employed to characterize the psychological state experienced when confronting negative emotions like loss and pain. "
        "This emotional state typically manifests as a psychological condition marked by feelings of frustration, pensiveness, or grief. "
        "fear conveys a negative sensation that arises in the face of danger or when confronted with something frightening. "
        "It encompasses emotions such as worry, anxiety, and panic, encapsulating a range of feelings including apprehension, anxiety, and terror. "
        "hate means a profound aversion towards someone or something deemed unacceptable, distasteful, or possessing unpleasant visual or olfactory qualities. "
        "This emotional response can encompass disinterest, dislike, or even a sense of loathing. "
        "surprise is the emotion elicited by unforeseen or sudden events, manifesting in a state of distraction and amazement. "
        "neutral indicates that the picture evokes no specific emotional response.\n"
        "Only output a JSON which only in this format: {{\"emotion_type\": \"happiness\"}}"
    )

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        caption = str(record.get("caption", "")).strip()
        moral = str(record.get("moral", "")).strip()
        image_path = str(record.get("image_file", "")).strip()

        return {
            "image_path": image_path,
            "caption": caption,
            "moral": moral,
            "is_metaphor": True,
            "metaphor_path": "sequential",
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        return self.EMOTION_PROMPT_TEMPLATE.format(
            caption=fields["caption"],
            moral=fields["moral"],
        ).strip()

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_path": fields["image_path"],
            "text": "",
            "is_metaphor": True,
            "metaphor_path": "sequential",
            "emotion_type": str(annotation.get("emotion_type", "")).strip().lower(),
            "caption": fields["caption"],
            "extra_info": {"moral": fields["moral"]} if fields["moral"] else {},
            "think": "",
        }


class CLI:
    def run(
        self,
        input: str,
        output: str,
        model: str,
        dataset: str,
        image_root: Optional[str] = None,
        api_keys: Optional[str] = None,
        max_concurrent: int = 50,
        max_retries: int = 5,
        limit: int = 0,
    ) -> None:
        config = AnnotatorConfig(
            input_path=Path(input),
            output_path=Path(output),
            model_name=model,
            api_keys=parse_api_keys(api_keys),
            max_concurrent_per_key=max_concurrent,
            max_retries=max_retries,
            limit=limit,
        )
        dataset_name = dataset.strip().lower()
        if dataset_name == "metmeme":
            annotator = MetMemeAnnotator(config, image_root=Path(image_root) if image_root else None)
        elif dataset_name == "yesbut":
            annotator = YesButAnnotator(config, image_root=Path(image_root) if image_root else None)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Use metmeme or yesbut.")
        annotator.run()


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
