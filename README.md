# Metaphorical Emotion

## Annotator

1. Download original image
    ```bash
    hf download Sakigo/MetaThinker-dataset --repo-type dataset --local-dir /path/to/data/MetaThinker-dataset
    ```

2. Configure API
    ```bash
    git clone https://github.com/Themaoqiu/metaphorical_emotion.git
    cd metaphorical_emotion
    touch .env
    ```

    Internal dedicated API .env content:
    ```bash
    # DeepSeek via MAAS
    MM_API_KEYS=""
    MM_API_BASE_URL=""
    MM_USER_EMAIL=""
    MM_APP_ID=""

    # Gemini via Runway
    GEMINI_API_KEYS="<runway api-key>"
    GEMINI_API_BASE_URL=""
    ```

    Common Openai API .env content:
    ```bash
    MM_API_KEYS=""
    VISION_API_BASE_URL=""
    ```

3. Install dependencies
    ```bash
    cd metaphorical_emotion
    uv sync
    ```

4. Run annotation, Modify the path to the locally downloaded [Sakigo/MetaThinker-data](https://huggingface.co/datasets/Sakigo/MetaThinker-data) dataset on your machine before running
    ```bash
    bash qa_generator/run_full_pipeline.sh
    ```

## Evaluation

Support datasets:
- MultiMM
- ...

Support models:
- Qwen2.5VL series
- Qwen3VL series
- Qwen3.5 series
- ...

### Environment Setup

1. Install dependencies:
    ```bash
    cd MetaThinker
    uv sync
    ```
2. Prepare MultiMM dataset:
    ```bash
    cd /path/to/data
    git clone https://github.com/DUTIR-YSQ/MultiMM.git
    ```
    It is recommended to keep the original path of the dataset repository.
3. Modify the parameters in the [run_eval.sh](MetaThinker/eval/run_eval.sh):

    ```bash
    python main.py run \
    --model_name qwen2.5VL \
    --model_path /models/Qwen/Qwen2.5-VL-7B-Instruct \
    --data_name multimm \
    --annotation_path /MultiMM/data \ # your path to the annotation files
    --image_dir /MultiMM/data \ # your path to the image folders imags_CN and images_EN
    --output_dir /eval/results \
    --batch_size 64 \
    --max_tokens 2048 \
    --max_model_len 8192 \
    --temperature 0.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --num_rounds 10 \ # It is recommended to use 10-fold cross-validation for MultiMM
    --cn_sample_size 440 \
    --en_sample_size 407 \
    --random_seed 42
    ```