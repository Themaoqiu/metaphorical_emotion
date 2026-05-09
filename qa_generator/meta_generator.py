from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

SYSTEM_PROMPT = (
    "You are a meticulous multimodal annotator specializing in visual metaphor and "
    "affective computing. Carefully ground every decision in the provided image and "
    "auxiliary context, choose labels strictly from the allowed options, and respond "
    "with a single valid JSON object that matches the requested schema exactly. "
    "Do not include explanations, markdown, code fences, or any text outside the JSON."
)


@dataclass
class AnnotatorConfig:
    input_path: Path
    output_path: Path
    model_name: str
    api_keys: List[str]
    max_concurrent_per_key: int = 50
    max_retries: int = 5
    start: int = 1
    end: int = 0
    limit: int = 0


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

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        raise NotImplementedError

    def _validate_range(self) -> None:
        if self.config.start < 1:
            raise ValueError("--start must be >= 1")
        if self.config.end != 0 and self.config.end < self.config.start:
            raise ValueError("--end must be 0 or >= --start")

    def _in_selected_range(self, one_based_idx: int) -> bool:
        if one_based_idx < self.config.start:
            return False
        if self.config.end != 0 and one_based_idx > self.config.end:
            return False
        return True

    def load_records(self) -> List[Dict[str, Any]]:
        self._validate_range()
        if self.config.input_path.suffix.lower() == ".json":
            with self.config.input_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                all_records = data
            elif isinstance(data, dict):
                all_records = [data]
            else:
                raise ValueError(f"Unsupported JSON format in {self.config.input_path}")
            records = [
                item
                for idx, item in enumerate(all_records, start=1)
                if self._in_selected_range(idx)
            ]
            if self.config.limit > 0:
                records = records[: self.config.limit]
            return records

        records: List[Dict[str, Any]] = []
        one_based_idx = 0
        with self.config.input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                one_based_idx += 1
                if not self._in_selected_range(one_based_idx):
                    if self.config.end != 0 and one_based_idx > self.config.end:
                        break
                    continue
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
        image_path = self.resolve_input_image_path(str(fields.get("image_path", "")))
        if image_path:
            return [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ]
        return prompt

    def get_resume_key(self, fields: Dict[str, Any]) -> str:
        return str(fields.get("image_path", "")).strip()

    def resolve_input_image_path(self, image_path: str) -> str:
        raw_path = str(image_path or "").strip()
        if not raw_path or self.image_root is None:
            return raw_path
        parsed = Path(raw_path)
        if parsed.is_absolute():
            if parsed.exists():
                return str(parsed)
            fallback = self.image_root / parsed.name
            if fallback.exists():
                return str(fallback)
            return raw_path
        return str(self.image_root / parsed)

    def _record_has_required_annotations(self, record: Dict[str, Any], fields: Dict[str, Any]) -> bool:
        required_fields = self.get_required_annotation_fields(fields)
        if not required_fields:
            return True
        for field_name in required_fields:
            value = record.get(field_name)
            if isinstance(value, str):
                if not value.strip():
                    return False
            elif value is None:
                return False
        return True

    def _load_existing_output(self) -> Dict[str, Dict[str, Any]]:
        output_path = self.config.output_path
        if not output_path.exists() or output_path.stat().st_size == 0:
            return {}

        existing_by_key: Dict[str, Dict[str, Any]] = {}
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = str(record.get("image_path", "")).strip()
                if key:
                    existing_by_key[key] = record
        return existing_by_key

    async def _retry_parse_failed_prompts(
        self,
        parse_failed_by_idx: Dict[int, str],
        prompts_by_idx: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
        recovered_annotations: Dict[int, Dict[str, Any]] = {}
        retry_failures = dict(parse_failed_by_idx)

        for attempt in range(1, self.config.max_retries + 1):
            if not retry_failures:
                break

            retry_prompts = [
                prompts_by_idx[idx]
                for idx in sorted(retry_failures)
            ]
            retry_results = await self.stream.generate(
                prompts=retry_prompts,
                system_prompt=SYSTEM_PROMPT,
                validate_func=self._parse_annotation,
            )

            next_failures: Dict[int, str] = {}
            for result in retry_results:
                idx = int(result["id"])
                raw = result["result"]
                parsed = self._parse_annotation(raw)
                if parsed is None:
                    next_failures[idx] = str(raw)
                    continue
                recovered_annotations[idx] = parsed

            retry_failures = next_failures
            if retry_failures:
                print(
                    f"[meta_generator] parse retry {attempt}/{self.config.max_retries} "
                    f"still_failed={len(retry_failures)}"
                )

        return recovered_annotations, retry_failures

    async def annotate(self) -> None:
        records = self.load_records()
        fields_by_idx: Dict[int, Dict[str, Any]] = {}
        prompts: List[Dict[str, Any]] = []
        prompts_by_idx: Dict[int, Dict[str, Any]] = {}
        existing_output_by_key = self._load_existing_output()
        resumed_output_by_idx: Dict[int, Dict[str, Any]] = {}

        for idx, record in enumerate(records):
            fields = self.extract_fields(record)
            fields_by_idx[idx] = fields
            # For datasets with explicit metaphor labels, only annotate metaphor samples.
            if "is_metaphor" in fields and fields["is_metaphor"] is not True:
                continue
            resume_key = self.get_resume_key(fields)
            existing_output = existing_output_by_key.get(resume_key)
            if existing_output and self._record_has_required_annotations(existing_output, fields):
                resumed_output_by_idx[idx] = existing_output
                continue
            prompt_item = {"id": str(idx), "prompt": self._build_prompt_payload(fields)}
            prompts.append(prompt_item)
            prompts_by_idx[idx] = prompt_item

        total_records = len(records)
        total_to_annotate = len(prompts)
        total_skipped = total_records - total_to_annotate
        print(
            f"[meta_generator] total={total_records}, "
            f"to_annotate={total_to_annotate}, skipped={total_skipped}, "
            f"start={self.config.start}, end={self.config.end}"
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

        if parse_failed_by_idx:
            recovered_annotations, parse_failed_by_idx = await self._retry_parse_failed_prompts(
                parse_failed_by_idx=parse_failed_by_idx,
                prompts_by_idx=prompts_by_idx,
            )
            annotations_by_idx.update(recovered_annotations)

        final_outputs: List[Dict[str, Any]] = []
        for idx in range(len(records)):
            if idx in resumed_output_by_idx:
                final_outputs.append(resumed_output_by_idx[idx])
                continue
            fields = fields_by_idx[idx]
            annotation = annotations_by_idx.get(idx, {})
            output = self.build_output_record(fields, annotation)
            if idx in parse_failed_by_idx:
                output["annotation_error"] = "parse_failed"
                output["raw_response"] = parse_failed_by_idx[idx]
            final_outputs.append(output)

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_path.open("w", encoding="utf-8") as handle:
            for output in final_outputs:
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
        "The source domain is {source_domain}, and the target domain is {target_domain}. "
        "based on existing dataset annotations.\n\n"
        "Task: Based on the image text provided, as well as the source domain and target domain of the metaphor, reflect on and present the metaphor comprehension pathway required to understand the image's meaning. The metaphor comprehension pathway is defined as follows:\n"
        "- direct: The image employs common idioms or fixed expressions, or its metaphor can be recognized at a glance without additional interpretation.\n"
        "- sequential: When reading the text sequentially and viewing the image, one first perceives the literal meaning of the picture. However, upon integrating the context and the content of the image, this literal meaning is revealed to be incorrect, and a cognitive shift is required to truly grasp the metaphorical meaning it conveys.\n"
        "- parallel: After examining the entire image, found that both its metaphorical and literal meanings are quite common, with roughly equal weight in comprehension. Unlike direct expressions, which only evoke one meaning (for instance, one does not think of the literal meaning when using an idiom).\n\n"
        "Output JSON only in this format: {{\"metaphor_path\": \"direct\"}}"
    )

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(record.get("image_path", "")).strip()
        text = str(record.get("text", "")).strip()
        caption = str(record.get("caption", "")).strip()

        extra_info = record.get("extra_info")
        extra = extra_info if isinstance(extra_info, dict) else {}

        emotion_type = str(record.get("emotion_type", "")).strip().lower()
        if not emotion_type:
            sentiment_raw = str(record.get("sentiment category", "")).strip()
            if "(" in sentiment_raw and ")" in sentiment_raw:
                emotion_type = sentiment_raw.split("(", 1)[1].split(")", 1)[0].strip().lower()
            elif sentiment_raw in self.SENTIMENT_ID_TO_LABEL:
                emotion_type = self.SENTIMENT_ID_TO_LABEL[sentiment_raw]
            else:
                emotion_type = sentiment_raw.lower()

        raw_is_metaphor = record.get("is_metaphor")
        if isinstance(raw_is_metaphor, bool):
            is_metaphor: Optional[bool] = raw_is_metaphor
        else:
            metaphor_occurrence = str(record.get("metaphor occurrence", "")).strip().lower()
            if metaphor_occurrence in {"1", "true", "yes"}:
                is_metaphor = True
            elif metaphor_occurrence in {"0", "false", "no"}:
                is_metaphor = False
            else:
                is_metaphor = None

        sentiment_degree = str(
            extra.get("sentiment_degree")
            or record.get("sentiment_degree")
            or record.get("sentiment degree")
            or ""
        ).strip()
        source_domain = str(
            extra.get("source_domain")
            or record.get("source_domain")
            or record.get("source domain")
            or ""
        ).strip()
        target_domain = str(
            extra.get("target_domain")
            or record.get("target_domain")
            or record.get("target domain")
            or ""
        ).strip()

        return {
            "image_path": image_path,
            "text": text,
            "emotion_type": emotion_type,
            "sentiment_degree": sentiment_degree,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "is_metaphor": is_metaphor,
            "caption": caption,
            "extra_info": extra,
            "think": str(record.get("think", "")).strip(),
            "prompt_with_image": str(record.get("prompt_with_image", "") or ""),
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        return self.PROMPT_TEMPLATE.format(
            text=fields["text"],
            source_domain=fields["source_domain"],
            target_domain=fields["target_domain"],
            sentiment_category=fields["emotion_type"],
            sentiment_degree=fields["sentiment_degree"],
        ).strip()

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        return ("metaphor_path",)

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": str(annotation.get("metaphor_path", "")).strip().lower(),
            "emotion_type": fields["emotion_type"],
            "caption": fields["caption"],
            "think": fields["think"],
            "prompt_with_image": fields["prompt_with_image"],
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
        "if you cannot see the image, output error in this format: {{\"emotion_type\": \"error\", \"error_message\": \"cannot see image\"}}"
    )

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        caption = str(record.get("caption", "")).strip()
        extra_info = record.get("extra_info")
        moral = ""
        if isinstance(extra_info, dict):
            moral = str(extra_info.get("moral", "")).strip()
        if not moral:
            moral = str(record.get("moral", "")).strip()

        image_path = str(record.get("image_path", "")).strip()

        return {
            "image_path": image_path,
            "text": str(record.get("text", "")).strip(),
            "is_metaphor": record.get("is_metaphor", True),
            "metaphor_path": str(record.get("metaphor_path", "")).strip(),
            "caption": caption,
            "moral": moral,
            "extra_info": extra_info if isinstance(extra_info, dict) else {},
            "think": str(record.get("think", "")).strip(),
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        return self.EMOTION_PROMPT_TEMPLATE.format(
            caption=fields["caption"],
            moral=fields["moral"],
        ).strip()

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        return ("emotion_type",)

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": fields["metaphor_path"],
            "emotion_type": str(annotation.get("emotion_type", "")).strip().lower(),
            "caption": fields["caption"],
            "extra_info": fields["extra_info"] if fields["extra_info"] else ({"moral": fields["moral"]} if fields["moral"] else {}),
            "think": fields["think"],
        }


class HummusAnnotator(BaseAnnotator):
    EMOTION_PROMPT_TEMPLATE = (
        "You are annotating a metaphor sample and you only need to classify the primary emotion category.\n\n"
        "The description of the image content is {caption}, and the humor effect is explained as {explanation}. "
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
        image_path = str(record.get("image_path", "")).strip()

        extra_info = record.get("extra_info")
        explanation = ""
        if isinstance(extra_info, dict):
            explanation = str(extra_info.get("explanation", "")).strip()

        return {
            "image_path": image_path,
            "text": str(record.get("text", "")).strip(),
            "is_metaphor": record.get("is_metaphor", True),
            "metaphor_path": str(record.get("metaphor_path", "")).strip(),
            "caption": str(record.get("caption", "")).strip(),
            "explanation": explanation,
            "extra_info": extra_info if isinstance(extra_info, dict) else {},
            "think": str(record.get("think", "")).strip(),
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        return self.EMOTION_PROMPT_TEMPLATE.format(
            caption=fields["caption"],
            explanation=fields["explanation"],
        ).strip()

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        return ("emotion_type",)

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": fields["metaphor_path"],
            "emotion_type": str(annotation.get("emotion_type", "")).strip().lower(),
            "caption": fields["caption"],
            "extra_info": fields["extra_info"],
            "think": fields["think"],
        }


class ImageMetAnnotator(BaseAnnotator):
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

    PROMPT_TEMPLATE = (
        "You are annotating a visual metaphor sample. All samples are metaphorical images. "
        "Use the image together with the auxiliary information below to jointly decide "
        "the metaphor comprehension pathway and the primary emotion category.\n\n"
        "Auxiliary information:\n"
        "- Image caption: {caption}\n"
        "- Metaphor source domain (vehicle): {source}\n"
        "- Metaphor target domain (tenor): {target}\n"
        "- Linguistic form of the metaphor: {linguistic_metaphor}\n"
        "- Entailing literal meaning: {entailing_literal}\n"
        "- Literal description of the implied meaning: {literal_description}\n"
        "- Relations between elements: {relations}\n\n"
        "Task 1 (metaphor_path): Reflect on the metaphor comprehension pathway required "
        "to understand this image. Pick exactly one of:\n"
        "- direct: The image employs common idioms or fixed expressions, or its metaphor can be recognized at a glance without additional interpretation.\n"
        "- sequential: When reading the text sequentially and viewing the image, one first perceives the literal meaning of the picture. However, upon integrating the context and the content of the image, this literal meaning is revealed to be incorrect, and a cognitive shift is required to truly grasp the metaphorical meaning it conveys.\n"
        "- parallel: After examining the entire image, found that both its metaphorical and literal meanings are quite common, with roughly equal weight in comprehension. Unlike direct expressions, which only evoke one meaning (for instance, one does not think of the literal meaning when using an idiom).\n\n"
        "Task 2 (emotion_type): Choose one emotion from: happiness, love, anger, sorrow, fear, hate, surprise, neutral. "
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
        "Output JSON only in this exact format: "
        "{{\"metaphor_path\": \"direct\", \"emotion_type\": \"happiness\"}}"
    )

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(record.get("image_path", "")).strip()

        extra_info = record.get("extra_info")
        extra = extra_info if isinstance(extra_info, dict) else {}

        return {
            "image_path": image_path,
            "text": str(record.get("text", "")).strip(),
            "is_metaphor": record.get("is_metaphor", True),
            "caption": str(record.get("caption", "")).strip(),
            "source": str(extra.get("source", "")).strip(),
            "target": str(extra.get("target", "")).strip(),
            "linguistic_metaphor": str(extra.get("generated_linguistic_metaphor", "")).strip(),
            "entailing_literal": str(extra.get("entailing_literal", "")).strip(),
            "literal_description": str(extra.get("literal_description", "")).strip(),
            "relations": str(extra.get("relations", "")).strip(),
            "extra_info": extra,
            "think": str(record.get("think", "")).strip(),
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        def _fallback(value: str) -> str:
            return value if value else "(not provided)"

        return self.PROMPT_TEMPLATE.format(
            caption=_fallback(fields["caption"]),
            source=_fallback(fields["source"]),
            target=_fallback(fields["target"]),
            linguistic_metaphor=_fallback(fields["linguistic_metaphor"]),
            entailing_literal=_fallback(fields["entailing_literal"]),
            literal_description=_fallback(fields["literal_description"]),
            relations=_fallback(fields["relations"]),
        ).strip()

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        return ("metaphor_path", "emotion_type")

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        emotion = str(annotation.get("emotion_type", "")).strip().lower()
        metaphor_path = str(annotation.get("metaphor_path", "")).strip().lower()
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": metaphor_path,
            "emotion_type": emotion,
            "caption": fields["caption"],
            "extra_info": fields["extra_info"],
            "think": fields["think"],
        }


class MemeCapAnnotator(BaseAnnotator):
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

    PROMPT_TEMPLATE = (
        "You are annotating a meme sample. All samples are metaphorical images. "
        "Use the image together with the auxiliary information below to jointly decide "
        "the metaphor comprehension pathway and the primary emotion category.\n\n"
        "Auxiliary information:\n"
        "- Image caption: {caption}\n"
        "- Meme title: {title}\n"
        "- Metaphorical meaning of the meme: {meme_captions}\n\n"
        "Task 1 (metaphor_path): Reflect on the metaphor comprehension pathway required "
        "to understand this image. Pick exactly one of:\n"
        "- direct: The image employs common idioms or fixed expressions, or its metaphor can be recognized at a glance without additional interpretation.\n"
        "- sequential: When reading the text sequentially and viewing the image, one first perceives the literal meaning of the picture. However, upon integrating the context and the content of the image, this literal meaning is revealed to be incorrect, and a cognitive shift is required to truly grasp the metaphorical meaning it conveys.\n"
        "- parallel: After examining the entire image, found that both its metaphorical and literal meanings are quite common, with roughly equal weight in comprehension. Unlike direct expressions, which only evoke one meaning (for instance, one does not think of the literal meaning when using an idiom).\n\n"
        "Task 2 (emotion_type): Choose one emotion from: happiness, love, anger, sorrow, fear, hate, surprise, neutral. "
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
        "Output JSON only in this exact format: "
        "{{\"metaphor_path\": \"direct\", \"emotion_type\": \"happiness\"}}"
    )

    @staticmethod
    def _join_list(value: Any) -> str:
        if isinstance(value, list):
            return " | ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(record.get("image_path", "")).strip()

        extra_info = record.get("extra_info")
        extra = extra_info if isinstance(extra_info, dict) else {}

        return {
            "image_path": image_path,
            "text": str(record.get("text", "")).strip(),
            "is_metaphor": record.get("is_metaphor", True),
            "existing_metaphor_path": str(record.get("metaphor_path", "")).strip().lower(),
            "caption": str(record.get("caption", "")).strip(),
            "title": str(extra.get("title", "")).strip(),
            "meme_captions": self._join_list(extra.get("meme_captions", "")),
            "extra_info": extra,
            "think": str(record.get("think", "")).strip(),
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        def _fallback(value: str) -> str:
            return value if value else "(not provided)"

        return self.PROMPT_TEMPLATE.format(
            caption=_fallback(fields["caption"]),
            title=_fallback(fields["title"]),
            meme_captions=_fallback(fields["meme_captions"]),
        ).strip()

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        if fields["existing_metaphor_path"]:
            return ("emotion_type",)
        return ("metaphor_path", "emotion_type")

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        emotion = str(annotation.get("emotion_type", "")).strip().lower()
        predicted_path = str(annotation.get("metaphor_path", "")).strip().lower()
        metaphor_path = fields["existing_metaphor_path"] or predicted_path
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": metaphor_path,
            "emotion_type": emotion,
            "caption": fields["caption"],
            "extra_info": fields["extra_info"],
            "think": fields["think"],
        }


class VFluteAnnotator(BaseAnnotator):
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

    METAPHOR_PATH_BLOCK = (
        "- direct: The image employs common idioms or fixed expressions, or its metaphor can be recognized at a glance without additional interpretation.\n"
        "- sequential: When reading the text sequentially and viewing the image, one first perceives the literal meaning of the picture. However, upon integrating the context and the content of the image, this literal meaning is revealed to be incorrect, and a cognitive shift is required to truly grasp the metaphorical meaning it conveys.\n"
        "- parallel: After examining the entire image, found that both its metaphorical and literal meanings are quite common, with roughly equal weight in comprehension. Unlike direct expressions, which only evoke one meaning (for instance, one does not think of the literal meaning when using an idiom).\n"
    )

    EMOTION_BLOCK = (
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
        "neutral indicates that the picture evokes no specific emotional response."
    )

    BOTH_PROMPT_TEMPLATE = (
        "You are annotating a visual metaphor sample. All samples are metaphorical images. "
        "Use the image together with the auxiliary information below to jointly decide "
        "the metaphor comprehension pathway and the primary emotion category.\n\n"
        "Auxiliary information:\n"
        "- Image caption: {caption}\n"
        "- Explanation of the figurative meaning: {explanation}\n\n"
        "Task 1 (metaphor_path): Reflect on the metaphor comprehension pathway required "
        "to understand this image. Pick exactly one of:\n"
        + METAPHOR_PATH_BLOCK
        + "\nTask 2 (emotion_type): "
        + EMOTION_BLOCK
        + "\nOutput JSON only in this exact format: "
        "{{\"metaphor_path\": \"direct\", \"emotion_type\": \"happiness\"}}"
    )

    EMOTION_ONLY_PROMPT_TEMPLATE = (
        "You are annotating a visual metaphor sample. All samples are metaphorical images. "
        "Use the image together with the auxiliary information below to decide the primary "
        "emotion category conveyed by the sample.\n\n"
        "Auxiliary information:\n"
        "- Image caption: {caption}\n"
        "- Explanation of the figurative meaning: {explanation}\n\n"
        "Task (emotion_type): "
        + EMOTION_BLOCK
        + "\nOutput JSON only in this exact format: "
        "{{\"emotion_type\": \"happiness\"}}"
    )

    def extract_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(record.get("image_path", "")).strip()

        extra_info = record.get("extra_info")
        extra = extra_info if isinstance(extra_info, dict) else {}

        return {
            "image_path": image_path,
            "text": str(record.get("text", "")).strip(),
            "is_metaphor": record.get("is_metaphor", True),
            "existing_metaphor_path": str(record.get("metaphor_path", "")).strip().lower(),
            "caption": str(record.get("caption", "")).strip(),
            "explanation": str(extra.get("explanation", "")).strip(),
            "extra_info": extra,
            "think": str(record.get("think", "")).strip(),
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        def _fallback(value: str) -> str:
            return value if value else "(not provided)"

        template = (
            self.EMOTION_ONLY_PROMPT_TEMPLATE
            if fields["existing_metaphor_path"]
            else self.BOTH_PROMPT_TEMPLATE
        )
        return template.format(
            caption=_fallback(fields["caption"]),
            explanation=_fallback(fields["explanation"]),
        ).strip()

    def get_required_annotation_fields(self, fields: Dict[str, Any]) -> Tuple[str, ...]:
        if fields["existing_metaphor_path"]:
            return ("emotion_type",)
        return ("metaphor_path", "emotion_type")

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        emotion = str(annotation.get("emotion_type", "")).strip().lower()
        predicted_path = str(annotation.get("metaphor_path", "")).strip().lower()
        metaphor_path = fields["existing_metaphor_path"] or predicted_path
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": metaphor_path,
            "emotion_type": emotion,
            "caption": fields["caption"],
            "extra_info": fields["extra_info"],
            "think": fields["think"],
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
        start: int = 1,
        end: int = 0,
        limit: int = 0,
    ) -> None:
        config = AnnotatorConfig(
            input_path=Path(input),
            output_path=Path(output),
            model_name=model,
            api_keys=parse_api_keys(api_keys, model_name=model),
            max_concurrent_per_key=max_concurrent,
            max_retries=max_retries,
            start=start,
            end=end,
            limit=limit,
        )
        dataset_name = dataset.strip().lower()
        root = Path(image_root) if image_root else None
        if dataset_name == "metmeme":
            annotator = MetMemeAnnotator(config, image_root=root)
        elif dataset_name == "yesbut":
            annotator = YesButAnnotator(config, image_root=root)
        elif dataset_name == "hummus":
            annotator = HummusAnnotator(config, image_root=root)
        elif dataset_name == "imagemet":
            annotator = ImageMetAnnotator(config, image_root=root)
        elif dataset_name == "memecap":
            annotator = MemeCapAnnotator(config, image_root=root)
        elif dataset_name == "vflute":
            annotator = VFluteAnnotator(config, image_root=root)
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset}. Use metmeme, yesbut, hummus, imagemet, memecap, or vflute."
            )
        annotator.run()


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
