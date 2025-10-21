This project uses `uv` for Python package management with `pyproject.toml`.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS
brew install uv 

# Or via pip
pip install uv
```

### Setup Development Environment

```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## Usage

Run examples to export HLO for different layers:

```bash
# supported layers: rmsnorm, dense_general, mlp_block
python entrypoints.py \
  --capture layer \
  --target rmsnorm \
  --output_dir ./hlo_outputs

# change model config
python entrypoints.py \
  --capture layer \
  --target dense_general \
  --base_emb_dim 256 \
  --base_mlp_dim 256

# run forward+backward and compute loss/gradients
python entrypoints.py \
  --capture layer \
  --target mlp_block \
  --with_grads
```

## Testing

```bash
# Run pytest with coverage
python -m pytest --cov --cov-report=term-missing --tb=short

# Or
pytest ./
```

## Git Commit Message Format

This project enforces a structured commit message format to ensure clear and traceable project history. Commit messages are validated by CI using `check_pr_logs.py`.

### Required Format

```
<type>[<SCOPE>]: <short-summary>

Problem:
<description of the problem being solved>

Solution:
<description of the solution implemented>

Test:
<description of how the change was tested>

JIRA: <ISSUE-123>
```
