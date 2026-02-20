from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire
from api_sync.api import StreamGenerator


SYSTEM_PROMPT = (
    "You are a professional image description assistant, and you are required to describe the visual content in the image."
)

USER_PROMPT = (
    "Describe the image in one or two sentences, including all elements necessary to understand the image content, without adding any interpretation of the image."
)


def parse_api_keys(value: Optional[str]) -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    env_value = os.getenv("MM_API_KEYS") or os.getenv("API_KEYS") or os.getenv("OPENAI_API_KEY")
    if not env_value:
        raise ValueError("API keys not provided. Use --api_keys or set MM_API_KEYS/API_KEYS.")
    return [item.strip() for item in env_value.split(",") if item.strip()]


class CLI:
    def run(
        self,
        input: str,
        output: str,
        model_path: str = "",
        image_root: Optional[str] = None,
        provider: str = "vllm",
        batch_size: int = 8,
        max_tokens: int = 64,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        api_model_name: str = "",
        api_keys: Optional[str] = None,
        max_concurrent: int = 50,
        max_retries: int = 5,
        generate_limit: int = 0,
        limit: int = 0,
    ) -> None:
        input_path = Path(input)
        output_path = Path(output)
        image_root_path = Path(image_root) if image_root else None

        records: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit > 0 and len(records) >= limit:
                    break
        print(f"[captioner] loaded records={len(records)} from {input_path}")

        pending: List[Tuple[int, str]] = []
        for idx, record in enumerate(records):
            if "image_path" in record and record.get("image_path"):
                raw = str(record["image_path"]).strip()
            elif isinstance(record.get("image"), dict) and record["image"].get("path"):
                raw = str(record["image"]["path"]).strip()
            else:
                continue

            image_path = Path(raw)
            if not image_path.is_absolute() and image_root_path is not None:
                image_path = image_root_path / image_path
            pending.append((idx, str(image_path)))

        if generate_limit > 0:
            pending = pending[:generate_limit]

        print(f"[captioner] captionable records={len(pending)}")

        provider_name = provider.strip().lower()
        captions: Dict[int, str] = {}

        if provider_name == "api":
            model_name = (api_model_name or model_path).strip()
            if not model_name:
                raise ValueError("For provider=api, set --api_model_name or --model_path.")
            key_list = parse_api_keys(api_keys)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pending_map: Dict[int, str] = {idx: image_path for idx, image_path in pending}
            completed_ids: set[int] = set()

            async def _run_api_stream(handle) -> None:
                if not pending:
                    return

                stream = StreamGenerator(
                    model_name=model_name,
                    api_keys=key_list,
                    max_concurrent_per_key=max_concurrent,
                    max_retries=max_retries,
                    rational=False,
                    with_unique_id=True,
                )
                prompts: List[Dict[str, Any]] = []
                for idx, image_path in pending:
                    prompts.append(
                        {
                            "id": str(idx),
                            "prompt": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": USER_PROMPT},
                            ],
                        }
                    )

                completed = 0
                async for result in stream.generate_stream(prompts=prompts, system_prompt=SYSTEM_PROMPT):
                    rec_idx = int(result["id"])
                    raw = str(result.get("result") or "")
                    out = dict(records[rec_idx])

                    if raw.startswith("__ERROR__:data_inspection_failed"):
                        out["caption"] = ""
                        out["annotation_error"] = "data_inspection_failed"
                    elif raw.startswith("__ERROR__:request_failed"):
                        out["caption"] = ""
                        out["annotation_error"] = "request_failed"
                    else:
                        cleaned = " ".join(raw.strip().split())
                        if cleaned and "\n" in cleaned:
                            cleaned = cleaned.split("\n", 1)[0].strip()
                        out["caption"] = cleaned

                    handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed_ids.add(rec_idx)
                    completed += 1
                    if completed % 50 == 0 or completed == len(prompts):
                        print(f"[captioner][api] progress {completed}/{len(prompts)}")

            with output_path.open("w", encoding="utf-8") as handle:
                # Save non-image records first so they are never lost.
                for idx, record in enumerate(records):
                    if idx in pending_map:
                        continue
                    out = dict(record)
                    if "caption" not in out:
                        out["caption"] = ""
                    handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                    handle.flush()

                asyncio.run(_run_api_stream(handle))

                # Any prompt still unresolved is persisted as failed to avoid silent loss.
                for idx in pending_map:
                    if idx in completed_ids:
                        continue
                    out = dict(records[idx])
                    out["caption"] = ""
                    out["annotation_error"] = "request_failed"
                    handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                    handle.flush()

            print(f"[captioner] done output={output_path}")
            return
        elif provider_name == "vllm":
            if not model_path.strip():
                raise ValueError("For provider=vllm, --model_path is required.")
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
            from qwen_vl_utils import process_vision_info

            llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                mm_processor_kwargs={
                    "min_pixels": 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                limit_mm_per_prompt={"image": 1, "video": 1},
                trust_remote_code=True,
                dtype="auto",
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.001,
                max_tokens=max_tokens,
                stop_token_ids=[],
            )

            chunks: List[List[Tuple[int, str]]] = [
                pending[i : i + batch_size] for i in range(0, len(pending), batch_size)
            ]
            for chunk_id, chunk in enumerate(chunks, start=1):
                llm_inputs: List[Dict[str, Any]] = []
                order: List[int] = []

                for idx, image_path in chunk:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": USER_PROMPT},
                            ],
                        },
                    ]
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        image_patch_size=processor.image_processor.patch_size,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )

                    mm_data: Dict[str, Any] = {}
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
                    order.append(idx)

                outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
                for rec_idx, output in zip(order, outputs):
                    raw = output.outputs[0].text if output.outputs else ""
                    cleaned = " ".join(raw.strip().split())
                    if cleaned and "\n" in cleaned:
                        cleaned = cleaned.split("\n", 1)[0].strip()
                    captions[rec_idx] = cleaned

                print(f"[captioner][vllm] progress {chunk_id}/{len(chunks)}")
        else:
            raise ValueError("provider must be one of: vllm, api")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for idx, record in enumerate(records):
                out = dict(record)
                if idx in captions:
                    out["caption"] = captions[idx]
                elif "caption" not in out:
                    out["caption"] = ""
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"[captioner] done output={output_path}")


if __name__ == "__main__":
    fire.Fire(CLI)
