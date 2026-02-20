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
    start: int = 1
    end: int = 0
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
        text = str(record.get("text", "")).strip()
        sentiment_raw = str(record.get("sentiment category", "")).strip()
        sentiment_degree = str(record.get("sentiment degree", "")).strip()
        source_domain = str(record.get("source domain", "")).strip()
        target_domain = str(record.get("target domain", "")).strip()
        metaphor_occurrence = str(record.get("metaphor occurrence", "")).strip().lower()
        image_path = str(record.get("image_path", "")).strip()
        if image_path and self.image_root:
            parsed = Path(image_path)
            if parsed.is_absolute():
                if not parsed.exists():
                    fallback = self.image_root / parsed.name
                    if fallback.exists():
                        image_path = str(fallback)
            else:
                image_path = str(self.image_root / parsed)

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
            "source_domain": source_domain,
            "target_domain": target_domain,
            "is_metaphor": is_metaphor,
        }

    def build_prompt(self, fields: Dict[str, Any]) -> str:
        return self.PROMPT_TEMPLATE.format(
            text=fields["text"],
            source_domain=fields["source_domain"],
            target_domain=fields["target_domain"],
            sentiment_category=fields["emotion_type"],
            sentiment_degree=fields["sentiment_degree"],
        ).strip()

    def build_output_record(self, fields: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        option_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        options = {
            label: emotion
            for label, emotion in zip(option_labels, self.EMOTION_OPTIONS)
        }
        return {
            "image_path": fields["image_path"],
            "text": fields["text"],
            "is_metaphor": fields["is_metaphor"],
            "metaphor_path": str(annotation.get("metaphor_path", "")),
            "emotion_type": fields["emotion_type"],
            "problem": "What metaphor is shown in this image, and which emotion category best matches its implied meaning?",
            "options": options,
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
        if image_path and self.image_root and not Path(image_path).is_absolute():
            image_path = str(self.image_root / image_path)

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
        if image_path and self.image_root and not Path(image_path).is_absolute():
            image_path = str(self.image_root / image_path)

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
            api_keys=parse_api_keys(api_keys),
            max_concurrent_per_key=max_concurrent,
            max_retries=max_retries,
            start=start,
            end=end,
            limit=limit,
        )
        dataset_name = dataset.strip().lower()
        if dataset_name == "metmeme":
            annotator = MetMemeAnnotator(config, image_root=Path(image_root) if image_root else None)
        elif dataset_name == "yesbut":
            annotator = YesButAnnotator(config, image_root=Path(image_root) if image_root else None)
        elif dataset_name == "hummus":
            annotator = HummusAnnotator(config, image_root=Path(image_root) if image_root else None)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Use metmeme, yesbut, or hummus.")
        annotator.run()


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
