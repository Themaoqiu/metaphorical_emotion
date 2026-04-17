import os
from typing import Any, Dict, List

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def _get_batched_value(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if index < len(value):
            return value[index]
        return value[-1]
    return value


class QwenVLBase:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.001,
            max_tokens=self.max_tokens,
            stop_token_ids=[],
        )
        self.llm = None
        self.processor = None
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def prepare_messages(
        self,
        query: str,
        media_path: str,
        system_prompt: str,
        media_type: str = "image",
    ) -> List[Dict[str, Any]]:
        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported media_type: {media_type}")

        media_content = {
            "type": media_type,
            media_type: media_path,
            "max_pixels": 1280 * 28 * 28,
        }

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    media_content,
                    {"type": "text", "text": query},
                ],
            },
        ]

    def predict_batch(
        self,
        queries: List[str],
        media_paths: List[str],
        system_prompt: str,
        media_type: str = "image",
    ) -> List[str]:
        raise NotImplementedError


class Qwen2_5VL(QwenVLBase):
    def load_model(self):
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 0},
            trust_remote_code=True,
            dtype="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def predict_batch(
        self,
        queries: List[str],
        media_paths: List[str],
        system_prompt: str,
        media_type: str = "image",
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info

        batch_messages = []
        for query, media_path in zip(queries, media_paths):
            batch_messages.append(self.prepare_messages(query, media_path, system_prompt, media_type=media_type))

        self.last_user_prompts = list(queries)
        prompts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in batch_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batch_messages,
            return_video_kwargs=True,
        )

        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            sample_mm_data = {}
            if image_inputs is not None:
                sample_image = _get_batched_value(image_inputs, idx)
                if sample_image is not None:
                    sample_mm_data["image"] = sample_image
            if video_inputs is not None:
                sample_video = _get_batched_value(video_inputs, idx)
                if sample_video is not None:
                    sample_mm_data["video"] = sample_video

            sample_video_kw = {}
            for key, value in video_kwargs.items():
                sample_value = _get_batched_value(value, idx)
                if sample_value is not None:
                    sample_video_kw[key] = sample_value

            llm_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                }
            )

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        self.last_raw_responses = [output.outputs[0].text for output in outputs]
        return self.last_raw_responses


class Qwen3VL(QwenVLBase):
    def load_model(self):
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def predict_batch(
        self,
        queries: List[str],
        media_paths: List[str],
        system_prompt: str,
        media_type: str = "image",
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info

        llm_inputs = []
        self.last_user_prompts = list(queries)
        for query, media_path in zip(queries, media_paths):
            messages = self.prepare_messages(query, media_path, system_prompt, media_type=media_type)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs.append(
                {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs,
                }
            )

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        self.last_raw_responses = [output.outputs[0].text for output in outputs]
        return self.last_raw_responses
