# AP-BMM Repository Structure

This document summarizes the retained code structure for the AP-BMM paper implementation.

## Core modules kept in the repository

- `src/evoMI/mi_opt_unified.py`: main optimization entry for AP-BMM and optimization baselines.
- `src/evoMI/mi_opt_optimizer.py`: prior construction and block partition helpers.
- `src/evoMI/async_merge_evaluator.py`: asynchronous evaluation orchestration for the AP-BMM pipeline.
- `src/evoMI/evaluation_utils.py`: evaluation configuration, caching, and task execution helpers.
- `src/evoMI/runtime_artifacts.py`: checkpoint/runtime artifact generation used by the core pipeline.
- `src/evoMI/task_diff_analyzer.py`: discrepancy analysis used to form AP-BMM priors.
- `src/evoMI/optimization_reporting.py`: core optimization result visualization.
- `src/ta_methos/model_level_fusion_test.py`: model-level baselines in the paper.

## Runtime directories

- `models/`: user-provided base/task models.
- `checkpoints/`: optimization checkpoints and runtime artifacts.
- `output/`: evaluation cache and temporary runtime files.
- `.mplconfig/`: local matplotlib cache.

These directories are intentionally lightweight in Git and preserved with `.gitkeep` files.

## Removed non-core components

The repository intentionally excludes:

- checkpoint replay / checkpoint evaluation utilities
- checkpoint analysis scripts
- manuscript LaTeX sources
- paper-only figure regeneration scripts
- exploratory debugging utilities outside the main paper workflow

## Quick checklist

1. Create a Python 3.10+ environment.
2. Install dependencies with `pip install -e .`.
3. Set `PYTHONPATH=.` and `MPLCONFIGDIR=$PWD/.mplconfig`.
4. Put the required models under `models/`.
5. Run `python scripts/apbmm_quickstart.py`.
6. Launch AP-BMM from `src/evoMI/mi_opt_unified.py` or run `src/ta_methos/model_level_fusion_test.py` for model-level baselines.
