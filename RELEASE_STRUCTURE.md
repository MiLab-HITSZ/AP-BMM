# APBMM Release Structure

This file describes the release-oriented layout of the APBMM repository.

## Core code kept in the release

- `src/evoMI/`: AP-BMM main method, optimization baselines, checkpoint replay, and evaluation helpers.
- `src/ta_methos/`: model-level fusion baselines used for comparison.
- `evalscope/`: local evaluation toolkit required by the runtime.
- `mergekit/`: local model-merging backend required by the runtime.
- `scripts/apbmm_quickstart.py`: lightweight environment and CLI sanity checker.

## Runtime directories kept as placeholders

- `models/`: user-provided base / task models.
- `checkpoints/`: optimization checkpoints and generated merged models.
- `output/`: cache files and other temporary runtime artifacts.
- `.mplconfig/`: optional local matplotlib cache directory.

These directories are intentionally almost empty in Git and are preserved with `.gitkeep` files only.

## Excluded from this release

- manuscript and supplementary LaTeX sources
- figure-generation and redraw scripts used only for the paper
- exploratory debugging scripts
- generated logs and experiment outputs

## Minimal release checklist

1. Create a Python 3.10+ environment.
2. Install dependencies with `pip install -e .`.
3. Set `PYTHONPATH=.` and `MPLCONFIGDIR=$PWD/.mplconfig`.
4. Put the required models under `models/`.
5. Run `python scripts/apbmm_quickstart.py`.
6. Launch AP-BMM or a baseline from `src/evoMI/mi_opt_unified.py`.
