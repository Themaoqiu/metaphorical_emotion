import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from pipelines.base_pipeline import BasePipeline
from prompts import (
    EMOTION_QUESTION,
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_emotion_to_sentiment_label,
    parse_metaphor_label,
)
from utils.metrics import compute_macro_f1
from utils.multimm import ensure_multimm_csvs, get_image_dir, read_multimm_csv, resolve_multimm_columns


logger = logging.getLogger(__name__)


def compute_four_f1_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    metric_definitions = {
        "metaphor_zh_f1": ("CN", "metaphor", [0, 1]),
        "metaphor_en_f1": ("EN", "metaphor", [0, 1]),
        "emotion_zh_f1": ("CN", "emotion", [-1, 0, 1]),
        "emotion_en_f1": ("EN", "emotion", [-1, 0, 1]),
    }

    output: Dict[str, float] = {}
    for metric_name, (language, task, labels) in metric_definitions.items():
        subset = [item for item in results if item["language"] == language]
        y_true = [item["labels"][task] for item in subset]
        y_pred = [item["prediction"][task] for item in subset]
        output[metric_name] = compute_macro_f1(y_true=y_true, y_pred=y_pred, labels=labels)

    return output


class MultiMMPipeline(BasePipeline):
    def __init__(
        self,
        model,
        model_name: str,
        data_name: str,
        annotation_path: str,
        image_dir: str,
        output_dir: str,
        batch_size: int = 1,
        num_rounds: int = 10,
        cn_sample_size: int = 440,
        en_sample_size: int = 407,
        random_seed: int = 42,
    ):
        super().__init__(model, model_name, data_name, annotation_path, image_dir, output_dir, batch_size=batch_size)
        self.num_rounds = num_rounds
        self.cn_sample_size = cn_sample_size
        self.en_sample_size = en_sample_size
        self.random_seed = random_seed
        self.system_prompt = SYSTEM_PROMPT

    def get_dataset_name(self) -> str:
        return "MultiMM"

    def load_data(self) -> List[Dict[str, Any]]:
        data_root = Path(self.annotation_path)
        csv_paths = ensure_multimm_csvs(data_root)

        datasets: List[Dict[str, Any]] = []
        missing_images = 0
        for language in ("CN", "EN"):
            rows = read_multimm_csv(csv_paths[language])
            columns = resolve_multimm_columns(rows)
            image_dir = get_image_dir(data_root, language, self.media_dir)
            for sample_index, row in enumerate(rows):
                image_name = str(row[columns["image"]])
                image_path = image_dir / Path(image_name).name
                if not image_path.exists():
                    missing_images += 1
                    continue

                datasets.append(
                    {
                        "language": language,
                        "sample_index": int(sample_index),
                        "image_name": image_name,
                        "image_path": str(Path(image_dir.name) / Path(image_name).name),
                        "media_path": str(image_path.resolve()),
                        "labels": {
                            "metaphor": int(row[columns["metaphor"]]),
                            "emotion": int(row[columns["emotion"]]),
                        },
                    }
                )

        logger.info("Loaded %s MultiMM samples from %s", len(datasets), data_root)
        if missing_images:
            logger.warning("Skipped %s MultiMM rows because images were missing", missing_images)
        return datasets

    def run_evaluation(self):
        logger.info("Starting %s evaluation", self.get_dataset_name())
        samples = self.load_data()
        by_language = {
            "CN": [item for item in samples if item["language"] == "CN"],
            "EN": [item for item in samples if item["language"] == "EN"],
        }
        if len(by_language["CN"]) < self.cn_sample_size:
            raise ValueError(
                f"CN sample size {self.cn_sample_size} exceeds available samples {len(by_language['CN'])}"
            )
        if len(by_language["EN"]) < self.en_sample_size:
            raise ValueError(
                f"EN sample size {self.en_sample_size} exceeds available samples {len(by_language['EN'])}"
            )

        all_results: List[Dict[str, Any]] = []
        round_summaries: List[Dict[str, Any]] = []

        for round_index in range(self.num_rounds):
            logger.info("Processing round %s/%s", round_index + 1, self.num_rounds)
            round_rng = random.Random(self.random_seed + round_index)
            round_samples = []
            round_samples += round_rng.sample(by_language["CN"], self.cn_sample_size)
            round_samples += round_rng.sample(by_language["EN"], self.en_sample_size)

            round_results = self._run_one_round(round_samples, round_index)
            all_results += round_results
            round_summary = {"round": round_index + 1, **compute_four_f1_metrics(round_results)}
            round_summaries.append(round_summary)

        avg_metrics = self._compute_average_metrics(round_summaries)
        avg_metrics["per_round"] = round_summaries
        self._save_results(all_results, avg_metrics)
        logger.info("Evaluation completed")
        return all_results, avg_metrics

    def _run_one_round(self, samples: List[Dict[str, Any]], round_index: int) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size

        for batch_start in range(0, len(samples), self.batch_size):
            batch = samples[batch_start : batch_start + self.batch_size]
            logger.info(
                "Round %s batch %s/%s",
                round_index + 1,
                batch_start // self.batch_size + 1,
                total_batches,
            )
            all_results.extend(self._process_batch(batch, round_index))

        return all_results

    def _process_batch(self, batch: List[Dict[str, Any]], round_index: int) -> List[Dict[str, Any]]:
        queries = []
        media_paths = []
        for item in batch:
            queries.append(build_user_prompt(EMOTION_QUESTION))
            media_paths.append(item["media_path"])

        responses = self.model.predict_batch(
            queries=queries,
            media_paths=media_paths,
            system_prompt=self.system_prompt,
            media_type="image",
        )
        raw_responses = getattr(self.model, "last_raw_responses", responses)

        results: List[Dict[str, Any]] = []
        for index, sample in enumerate(batch):
            result = {
                "round": round_index + 1,
                "language": sample["language"],
                "sample_index": sample["sample_index"],
                "image_name": sample["image_name"],
                "image_path": sample["image_path"],
                "raw_response": raw_responses[index],
                "prediction": {
                    "metaphor": parse_metaphor_label(raw_responses[index]),
                    "emotion": parse_emotion_to_sentiment_label(raw_responses[index]),
                },
                "labels": sample["labels"],
            }
            results.append(result)

        return results

    def _compute_average_metrics(self, round_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_names = [
            "metaphor_zh_f1",
            "metaphor_en_f1",
            "emotion_zh_f1",
            "emotion_en_f1",
        ]
        return {
            metric_name: float(sum(item[metric_name] for item in round_summaries) / len(round_summaries))
            for metric_name in metric_names
        }
