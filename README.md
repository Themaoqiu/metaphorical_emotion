# Metaphorical Emotion
## Environment
```bash
cd metaphorical_emotion
uv sync
```

This repository keeps the project-specific training and evaluation entrypoints, but it does not vendor the full upstream runtimes they depend on.

Environment requirements:
- `trainer/run.sh` must be executed inside a local [LLaMA-Factory](/root/metaphorical_emotion/trainer/run.sh) checkout. It expects a working LLaMA-Factory virtualenv and launches `llamafactory-cli train`.
- `eval/eval.sh` must be executed with the DORO-STVG runtime available locally. It activates the DORO-STVG environment first, then runs this repository's `eval/eval.py`.
- The corresponding environment snapshots are stored in [trainer/ENVIRONMENT.md](/root/metaphorical_emotion/trainer/ENVIRONMENT.md) and [eval/ENVIRONMENT.md](/root/metaphorical_emotion/eval/ENVIRONMENT.md), together with upstream and locked dependency lists.
- A source-only copy of LLaMA-Factory is vendored under [trainer/LLaMA-Factory](/root/metaphorical_emotion/trainer/LLaMA-Factory). Large local artifacts such as `.venv/`, `saves/`, and `.git/` are intentionally excluded.

## Training
Training code is organized under `trainer/`.

```bash
cd trainer
bash run.sh
```

What `trainer/run.sh` depends on:
- A local LLaMA-Factory checkout. Default path: `/root/autodl-tmp/OneMLLM/LLaMA-Factory`
- A LLaMA-Factory virtualenv under that checkout. Default path: `/root/autodl-tmp/OneMLLM/LLaMA-Factory/.venv`
- `llamafactory-cli` installed in that virtualenv
- `deepspeed` available in that same virtualenv

How to configure the training environment:
```bash
cd /root/autodl-tmp/OneMLLM/LLaMA-Factory
uv venv .venv --python 3.11
source .venv/bin/activate
pip install -e ".[metrics,deepspeed]" --no-build-isolation
```

If you want to use the vendored source copy in this repository instead, the equivalent setup is:
```bash
cd /root/metaphorical_emotion/trainer/LLaMA-Factory
uv venv .venv --python 3.11
source .venv/bin/activate
pip install -e ".[metrics,deepspeed]" --no-build-isolation
```

Then point the wrapper script to that checkout:
```bash
export LLAMA_FACTORY_DIR=/root/metaphorical_emotion/trainer/LLaMA-Factory
cd /root/metaphorical_emotion/trainer
bash run.sh
```

Useful environment variables:
- `LLAMA_FACTORY_DIR`: local LLaMA-Factory path
- `TRAIN_MODEL_PATH`: base model path
- `TRAIN_DATASET_DIR`: dataset directory
- `TRAIN_OUTPUT_DIR`: checkpoint output directory
- `AUTO_SHUTDOWN=1`: shut down after training succeeds

## Evaluation
Evaluation code is organized under `eval/`.

```bash
cd eval
bash eval.sh /path/to/checkpoint
```

What `eval/eval.sh` depends on:
- A local DORO-STVG checkout. Default path: `/root/autodl-tmp/DORO-STVG`
- A DORO-STVG Python environment. Default path: `/root/.virtualenvs/stvg/dev`
- DORO-STVG's activation helper at `scripts/activate_env.sh`
- The DORO-STVG eval dependencies installed into that environment

How to configure the evaluation environment:
```bash
cd /root/autodl-tmp/DORO-STVG
bash ./scripts/setup_env.sh /root/.virtualenvs/stvg/dev
source ./scripts/activate_env.sh /root/.virtualenvs/stvg/dev
uv sync --active
```

Then run evaluation from this repository:
```bash
cd /root/metaphorical_emotion/eval
bash eval.sh /path/to/checkpoint
```

Useful environment variables:
- `STVG_PROJECT_DIR`: local DORO-STVG path
- `STVG_ENV_PATH`: evaluation virtualenv path
- `EVAL_DATA_ROOT`: fixed test dataset root
- `EVAL_OUTPUT_DIR`: directory for metrics and csv outputs
