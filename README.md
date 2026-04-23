# AP-BMM

This repository contains the core implementation of **AP-BMM (Asynchronous Prior-guided Bayesian Model Merging)**.

AP-BMM targets the capability-efficiency trade-off in LLM merging. The method follows the paper design: it performs **layer-wise model merging**, builds a **discrepancy-derived prior** to guide high-dimensional Bayesian optimization, and uses an **asynchronous pending-aware dispatch loop** to better utilize GPUs under highly heterogeneous evaluation latency.

The code in this repository is organized as the paper's open-source implementation: only the main optimization pipeline, the necessary baselines, the model-level comparison methods, and the supporting runtime/evaluation utilities are retained.

## What is included

- `src/evoMI/mi_opt_unified.py`: unified entry for AP-BMM and the optimization baselines used in the paper.
- `src/evoMI/mi_opt_optimizer.py`: prior construction and block partition logic.
- `src/evoMI/mi_opt_saasbo2.py`: asynchronous evaluation and SAAS-prior-related orchestration.
- `src/evoMI/evaluation_utils.py`: evaluation config, caching, and task-launch utilities.
- `src/evoMI/runtime_artifacts.py`: checkpoint/runtime artifact helpers used by the core optimization flow.
- `src/evoMI/task_diff_analyzer.py`: layer discrepancy analysis for building the AP-BMM prior.
- `src/evoMI/optimization_reporting.py`: optimization result visualization utilities.
- `src/ta_methos/model_level_fusion_test.py`: model-level baselines used in the paper.
- `evalscope/`: local evaluation toolkit.
- `mergekit/`: local model merging backend.
- `scripts/apbmm_quickstart.py`: quick environment and CLI sanity checker.

## What is intentionally removed

The following non-core utilities are not kept in this release:

- checkpoint post-analysis / replay scripts
- checkpoint re-evaluation scripts
- paper-only plotting / redraw scripts
- manuscript LaTeX sources
- debugging / exploratory scripts unrelated to the main paper pipeline

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

`checkpoints/`, `models/`, `output/`, and `.mplconfig/` are kept as placeholder directories.

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

## Prepare models

Put the required models under `models/`. The default layout is:

```text
models/
├── Qwen3-4B-Instruct-2507/
└── Qwen3-4B-thinking-2507/
```

You can also override model paths from the command line.

## Quick sanity check

```bash
python scripts/apbmm_quickstart.py
```

This verifies the repository layout, imports the core modules, and checks `mi_opt_unified.py --help`.

## Verified basic checks

```bash
python scripts/apbmm_quickstart.py
python src/evoMI/mi_opt_unified.py --help
python src/ta_methos/model_level_fusion_test.py --help
```

## AP-BMM paper setting

The AP-BMM configuration used in the paper corresponds to:

- alias: `sass_prior_bo_wo_update_gap_async`
- normalized optimizer: `prior_saas_bo`

This preset enables:

- asynchronous dispatch
- SAAS prior
- blueprint / importance prior
- no importance update
- no importance-guided acquisition
- no importance-prior cutoff
- gap-aware postprocessing

Main command:

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

## Optimization baselines

`src/evoMI/mi_opt_unified.py` also supports the main baselines used in the paper:

- `priorbo`
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

## Model-level baselines

Run the model-level comparison methods with:

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

## Outputs

- optimization checkpoints and runtime artifacts: `checkpoints/`
- evaluation cache and temporary files: `output/`
- user-provided model weights: `models/`

## Notes

- The code assumes local GPU evaluation with vLLM.
- `evalscope` and `mergekit` are vendored to keep the project self-contained.
- The default benchmark profile used by the main paper experiments is `aime_gpqa`.
- If you move the project to another machine, usually only model paths, GPU ids, and dataset caches need to be updated.
