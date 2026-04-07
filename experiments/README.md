# Experiments

Each experiment lives in a numbered folder (`NN_description/`) and is self-contained.

## Structure

```
experiments/
├── _template/          # Copy to start a new experiment
├── 01_gemma_sparsity_sweep/
├── 02_activation_sparsity_sweep/
├── ...
```

Each experiment folder contains:

| File | Purpose |
|------|---------|
| `run.py` | Self-contained entry point with hardcoded config |
| `README.md` | Goal, setup, results, and analysis |
| `results/` | JSON outputs (gitignored, regenerable) |
| `figures/` | Plots (gitignored, regenerable) |
| `logs/` | Training logs (gitignored) |

## Workflow

### Creating a new experiment

```bash
# 1. Copy the template
cp -r experiments/_template experiments/07_my_experiment

# 2. Edit run.py with your config and logic
# 3. Edit README.md with your hypothesis and setup

# 4. Run
uv run python experiments/07_my_experiment/run.py
```

### Rules

1. **Immutability**: Never modify a completed experiment's `run.py`. Create a new numbered folder instead.
2. **Self-contained configs**: All hyperparameters are hardcoded in `run.py` (no external YAML).
3. **Outputs are gitignored**: `results/`, `figures/`, `logs/` can be regenerated from `run.py`.
4. **Document findings**: Update `README.md` with results and analysis after running.
5. **Use shared utils**: Import from `molt.utils` instead of duplicating boilerplate.

### Running with wandb

Set `wandb_enabled=True` in your config and ensure `WANDB_API_KEY` is set:

```python
config = MOLTConfig(
    wandb_enabled=True,
    wandb_project="repro-molts",
    ...
)
```
