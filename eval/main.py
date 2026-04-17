import logging
import sys

import fire


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MetaThinkerEvaluator:
    @staticmethod
    def _log_metric_summary(metrics: dict) -> None:
        logger.info("Final Metrics:")
        for metric_name in ["metaphor_zh_f1", "metaphor_en_f1", "emotion_zh_f1", "emotion_en_f1"]:
            if metric_name in metrics:
                logger.info("  %s: %.6f", metric_name, metrics[metric_name])

    def run(
        self,
        model_name: str,
        model_path: str,
        data_name: str,
        annotation_path: str,
        image_dir: str = "",
        output_dir: str = "./results",
        batch_size: int = 1,
        max_tokens: int = 1024,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        num_rounds: int = 10,
        cn_sample_size: int = 440,
        en_sample_size: int = 407,
        random_seed: int = 42,
    ):
        logger.info("Model: %s", model_name)
        logger.info("Model Path: %s", model_path)
        logger.info("Data Name: %s", data_name)
        logger.info("Annotation Path: %s", annotation_path)
        logger.info("Image Dir: %s", image_dir or "<auto>")
        logger.info("Output Dir: %s", output_dir)
        logger.info("Batch Size: %s", batch_size)
        logger.info("Num Rounds: %s", num_rounds)
        logger.info("CN Sample Size: %s", cn_sample_size)
        logger.info("EN Sample Size: %s", en_sample_size)
        logger.info("Random Seed: %s", random_seed)

        if data_name.lower() in ["multimm", "multi-mm"]:
            from utils.multimm import ensure_multimm_csvs

            csv_paths = ensure_multimm_csvs(annotation_path)
            logger.info("Prepared MultiMM csv files: %s", csv_paths)

        model_key = model_name.lower()
        if model_key in ["qwen2.5vl", "qwen2.5-vl"]:
            from models.qwen_family import Qwen2_5VL

            model = Qwen2_5VL(
                model_path=model_path,
                batch_size=batch_size,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        elif model_key in ["qwen3vl", "qwen3-vl", "qwen3.5", "qwen3.5vl", "qwen3.5-vl"]:
            from models.qwen_family import Qwen3VL

            model = Qwen3VL(
                model_path=model_path,
                batch_size=batch_size,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if data_name.lower() in ["multimm", "multi-mm"]:
            from pipelines.multimm import MultiMMPipeline

            pipeline = MultiMMPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                image_dir=image_dir,
                output_dir=output_dir,
                batch_size=batch_size,
                num_rounds=num_rounds,
                cn_sample_size=cn_sample_size,
                en_sample_size=en_sample_size,
                random_seed=random_seed,
            )
            _, metrics = pipeline.run_evaluation()
            self._log_metric_summary(metrics)
            return None

        raise ValueError(f"Unknown dataset: {data_name}")


def main() -> None:
    fire.Fire(MetaThinkerEvaluator)


if __name__ == "__main__":
    main()
