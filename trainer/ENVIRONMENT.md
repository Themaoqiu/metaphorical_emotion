# Trainer Environment

`trainer/run.sh` does not create a standalone training runtime. It expects a local LLaMA-Factory checkout and uses that project's virtualenv plus `llamafactory-cli`.

Expected upstream runtime:
- Project checkout: `/root/autodl-tmp/OneMLLM/LLaMA-Factory`
- Python env: `/root/autodl-tmp/OneMLLM/LLaMA-Factory/.venv`
- Upstream dependency file: `LLaMA-Factory/pyproject.toml`

Vendored source tree:
- A source-only copy of LLaMA-Factory is included in `trainer/LLaMA-Factory/`.
- Excluded on purpose: `.git/`, `.venv/`, `saves/`, `__pycache__/`, and other local runtime artifacts.

Setup steps for the upstream checkout:
1. Create the virtual environment:
```bash
cd /root/autodl-tmp/OneMLLM/LLaMA-Factory
uv venv .venv --python 3.11
```
2. Activate it:
```bash
cd /root/autodl-tmp/OneMLLM/LLaMA-Factory
source .venv/bin/activate
```
3. Install training dependencies, including DeepSpeed:
```bash
cd /root/autodl-tmp/OneMLLM/LLaMA-Factory
pip install -e ".[metrics,deepspeed]" --no-build-isolation
```
4. Run the wrapper in this repository:
```bash
cd /root/metaphorical_emotion/trainer
bash run.sh
```

Setup steps for the vendored source copy:
```bash
cd /root/metaphorical_emotion/trainer/LLaMA-Factory
uv venv .venv --python 3.11
source .venv/bin/activate
pip install -e ".[metrics,deepspeed]" --no-build-isolation
export LLAMA_FACTORY_DIR=/root/metaphorical_emotion/trainer/LLaMA-Factory
cd /root/metaphorical_emotion/trainer
bash run.sh
```

Files in this folder:
- `requirements.upstream.txt`: core and optional dependencies extracted from `LLaMA-Factory/pyproject.toml`
- `requirements.lock.txt`: package snapshot exported from the current `LLaMA-Factory/.venv` environment on this machine

Reproduction note:
- If you want to reproduce the exact training environment used here, start from the LLaMA-Factory checkout, create its Python 3.11 virtualenv, then install packages to match `requirements.lock.txt`.
