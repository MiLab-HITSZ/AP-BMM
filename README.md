# APBMM

APBMM is the minimal runnable code release for the AP-BMM paper. This repository keeps the optimization code, fusion baselines, evaluation pipeline, and local dependencies needed for reproduction, while excluding paper writing assets, plotting scripts, and other non-core utilities.

## Included in this release

- `src/evoMI/mi_opt_unified.py`: unified entry for AP-BMM and the paper's optimization baselines.
- `src/evoMI/mi_opt_optimizer.py`: prior construction and block-partition helpers used by the main pipeline.
- `src/evoMI/model_reproduction.py`: checkpoint replay, merged-model generation, and cached evaluation.
- `src/evoMI/evaluation_script.py`: sequential evaluation helper for multiple algorithms.
- `src/evoMI/task_diff_analyzer.py`: layer/block discrepancy analysis used to build priors and partitions.
- `src/evoMI/optimization_reporting.py`: runtime reporting helper used by the optimization flow.
- `src/ta_methos/model_level_fusion_test.py`: model-level comparison baselines.
- `evalscope/`: vendored local evaluation toolkit.
- `mergekit/`: vendored local model-merging backend.
- `scripts/apbmm_quickstart.py`: quickstart sanity checker.

## Intentionally excluded

This release does **not** include:

- paper manuscript sources (`APBMM_paper/`)
- plotting / redraw / figure-generation scripts used only for the paper
- ad-hoc debugging scripts and one-off experiments
- generated logs, checkpoints, cached results, and model weights

## Repository layout

```text
APBMM/
├── checkpoints/
├── evalscope/
├── mergekit/
├── models/
├── output/
├── scripts/
│   └── apbmm_quickstart.py
├── src/
│   ├── config_manager.py
│   ├── evoMI/
│   └── ta_methos/
├── .mplconfig/
├── pyproject.toml
├── README.md
└── RELEASE_STRUCTURE.md
```

`checkpoints/`, `models/`, `output/`, and `.mplconfig/` are kept as empty placeholder directories for GitHub.

## Environment setup

```bash
cd /path/to/APBMM
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
export PYTHONPATH=.
export MPLCONFIGDIR=$PWD/.mplconfig
```

If you use `uv`, the equivalent install step is also fine.

## Prepare models

Put the required base and task models under `models/`. The default code path assumes:

```text
models/
├── Qwen3-4B-Instruct-2507/
└── Qwen3-4B-thinking-2507/
```

You can also override model paths from the command line, or edit `src/config_manager.py`.

## Quick sanity check

Before a full run:

```bash
python scripts/apbmm_quickstart.py
```

This checks the release layout, imports the core modules, and runs `mi_opt_unified.py --help`.

## Verified smoke tests

The following lightweight commands were verified in a local environment. They validate the repository layout and the paper-related CLI entry points:

```bash
python scripts/apbmm_quickstart.py
python src/evoMI/mi_opt_unified.py --help
python src/ta_methos/model_level_fusion_test.py --help
python src/evoMI/evaluation_script.py \
  --plan-only \
  --algorithms apbmm qnehvi momm moead_cmaes \
  --eval-profile gsm8k_gpqa \
  --gsm8k-limit 2 \
  --gpqa-limit 2 \
  --initial-samples 4 \
  --batch-size 4 \
  --max-evaluations 8
```

The full experiment commands below additionally require local model weights, available GPUs, and benchmark data caches.

## Run AP-BMM

The AP-BMM setting used in the paper corresponds to:

- checkpoint/run alias: `sass_prior_bo_wo_update_gap_async`
- normalized optimizer name: `prior_saas_bo`

This preset enables:

- asynchronous dispatch
- SAAS prior
- blueprint / importance prior
- no importance update
- no importance-guided acquisition
- no importance-prior cutoff
- gap-aware postprocessing

Main paper-style AP-BMM command:

```bash
python src/evoMI/mi_opt_unified.py \
  --algorithm sass_prior_bo_wo_update_gap_async \
  --base-model-path models/Qwen3-4B-Instruct-2507 \
  --task-model-paths models/Qwen3-4B-thinking-2507 models/Qwen3-4B-Instruct-2507 \
  --num-blocks 36 \
  --initial-samples 8 \
  --batch-size 4 \
  --max-evaluations 88 \
  --num-restarts 10 \
  --raw-samples 512 \
  --mc-samples 128 \
  --eval-profile aime_gpqa \
  --eval-aime-limit 5 \
  --eval-gpqa-limit 60 \
  --available-gpus 0 1 2 3
```

For paper ablations, `priorbo` is also retained as the non-SAAS prior-guided BO entry so that sync / async variants can still be reproduced from flags.

## Optimization baselines

`src/evoMI/mi_opt_unified.py` also supports:

- `priorbo` (for paper ablations)
- `prior_saas_bo`
- `qnehvi`
- `momm`
- `moead_cmaes`

Example:

```bash
python src/evoMI/mi_opt_unified.py \
  --algorithm qnehvi \
  --base-model-path models/Qwen3-4B-Instruct-2507 \
  --task-model-paths models/Qwen3-4B-thinking-2507 models/Qwen3-4B-Instruct-2507 \
  --num-blocks 36 \
  --initial-samples 8 \
  --batch-size 4 \
  --max-evaluations 88 \
  --eval-profile aime_gpqa \
  --available-gpus 0 1 2 3
```

## Model-level comparison baselines

Run the paper's model-level comparison methods with:

```bash
python src/ta_methos/model_level_fusion_test.py \
  --fusion_method ties \
  --num_blocks 8 \
  --budget 88 \
  --batch_size 4
```

Supported `--fusion_method` values:

- `task_arithmetic`
- `ties`
- `dare_ties`
- `dare_linear`
- `breadcrumbs`
- `breadcrumbs_ties`
- `della`
- `della_linear`

`model_level_fusion_test.py` keeps its historical filename, but it is the runnable baseline entry rather than a unit test.

## Sequential evaluation after checkpoints

After generating checkpoints, you can run:

```bash
python src/evoMI/evaluation_script.py \
  --algorithms apbmm qnehvi momm moead_cmaes \
  --checkpoint-root ./checkpoints/statistical_eval \
  --cache-root ./output/statistical_eval_cache \
  --eval-profile gsm8k_gpqa \
  --gsm8k-limit 100 \
  --gpqa-limit 100 \
  --available-gpus 0 1 2 3
```

Supported `evaluation_script.py --algorithms` values are:

- `apbmm`
- `qnehvi`
- `momm`
- `moead_cmaes`

## Outputs

- checkpoints: `checkpoints/`
- evaluation cache and temporary files: `output/`
- user-provided model weights: `models/`

These directories are ignored by Git except for the placeholder `.gitkeep` files.

## Notes

- The code is designed for local GPU evaluation with vLLM.
- `evalscope` and `mergekit` are vendored here to keep the release self-contained.
- The default benchmark profile in the paper code is `aime_gpqa`.
- If you migrate to another machine, the main things to update are model paths, available GPUs, and dataset cache locations.
