# Trainer Environment

`trainer/run.sh` does not create a standalone training runtime. It expects a local LLaMA-Factory checkout and uses that project's virtualenv plus `llamafactory-cli`.

Expected upstream runtime:
- Project checkout: `/root/autodl-tmp/OneMLLM/LLaMA-Factory`
- Python env: `/root/autodl-tmp/OneMLLM/LLaMA-Factory/.venv`
- Upstream dependency file: `LLaMA-Factory/pyproject.toml`

Vendored source tree:
- A source-only copy of LLaMA-Factory is included in `trainer/LLaMA-Factory/`.
- Excluded on purpose: `.git/`, `.venv/`, `saves/`, `__pycache__/`, and other local runtime artifacts.

Files in this folder:
- `requirements.upstream.txt`: core and optional dependencies extracted from `LLaMA-Factory/pyproject.toml`
- `requirements.lock.txt`: package snapshot exported from the current `LLaMA-Factory/.venv` environment on this machine

Reproduction note:
- If you want to reproduce the exact training environment used here, start from the LLaMA-Factory checkout, create its Python 3.11 virtualenv, then install packages to match `requirements.lock.txt`.
