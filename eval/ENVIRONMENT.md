# Eval Environment

`eval/eval.sh` does not create its own runtime from scratch. It depends on the local DORO-STVG project and activates the DORO-STVG environment before running this repository's evaluation code.

Expected upstream runtime:
- Project checkout: `/root/autodl-tmp/DORO-STVG`
- Python env: `/root/.virtualenvs/stvg/dev`
- Upstream dependency file: `DORO-STVG/eval/pyproject.toml`

Setup steps:
1. Create the virtual environment from the DORO-STVG checkout:
```bash
cd /root/autodl-tmp/DORO-STVG
bash ./scripts/setup_env.sh /root/.virtualenvs/stvg/dev
```
2. Activate the environment:
```bash
cd /root/autodl-tmp/DORO-STVG
source ./scripts/activate_env.sh /root/.virtualenvs/stvg/dev
```
3. Install the DORO-STVG eval dependencies into the active environment:
```bash
cd /root/autodl-tmp/DORO-STVG
uv sync --active
```
4. Run this repository's wrapper:
```bash
cd /root/metaphorical_emotion/eval
bash eval.sh /path/to/checkpoint
```

Files in this folder:
- `requirements.upstream.txt`: dependencies declared by `DORO-STVG/eval/pyproject.toml`
- `requirements.lock.txt`: package snapshot exported from the current `/root/.virtualenvs/stvg/dev` environment on this machine

Reproduction note:
- If you want to reproduce the exact environment used here, start from the DORO-STVG project, create its Python 3.11 environment, then install packages to match `requirements.lock.txt`.
