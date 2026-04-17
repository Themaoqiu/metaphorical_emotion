import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    def __init__(
        self,
        model,
        model_name: str,
        data_name: str,
        annotation_path: str,
        media_dir: str,
        output_dir: str,
        batch_size: int = 1,
    ):
        self.model = model
        self.model_name = model_name
        self.data_name = data_name
        self.annotation_path = Path(annotation_path)
        self.media_dir = Path(media_dir) if media_dir else None
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def _compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def run_evaluation(self):
        logger.info("Starting %s evaluation", self.get_dataset_name())
        samples = self.load_data()
        all_results: List[Dict[str, Any]] = []

        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        for index in range(0, len(samples), self.batch_size):
            batch = samples[index : index + self.batch_size]
            logger.info("Processing batch %s/%s", index // self.batch_size + 1, total_batches)
            all_results.extend(self._process_batch(batch))

        avg_metrics = self._compute_average_metrics(all_results)
        self._save_results(all_results, avg_metrics)
        logger.info("Evaluation completed")
        return all_results, avg_metrics

    def _save_results(self, results: List[Dict[str, Any]], avg_metrics: Dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = self.get_dataset_name().lower().replace("-", "").replace(" ", "")
        eval_folder_name = f"{dataset_name}_{self.model_name}_{timestamp}"
        eval_folder = self.output_dir / eval_folder_name
        eval_folder.mkdir(parents=True, exist_ok=True)

        results_file = eval_folder / "results.jsonl"
        with results_file.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "dataset": self.get_dataset_name(),
            "model": self.model_name,
            "num_samples": len(results),
            "timestamp": timestamp,
            "average_metrics": avg_metrics,
        }
        summary_file = eval_folder / "status.json"
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        logger.info("Detailed results saved to %s", results_file)
        logger.info("Summary saved to %s", summary_file)
