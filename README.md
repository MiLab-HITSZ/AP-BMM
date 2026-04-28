# AP-BMM

Official implementation of **AP-BMM: Asynchronous Prior-guided Bayesian Model Merging**.

> AP-BMM approximates capability-efficiency Pareto sets of LLMs through layer-wise model merging, discrepancy-derived surrogate priors, and pending-aware asynchronous Bayesian optimization.

## Overview

Large language model deployment often requires balancing **capability** and **efficiency** rather than optimizing only one of them. Existing model merging methods are dominated by coarse **model-level operators**, which are simple to apply but provide limited control over the shape of the capability-efficiency trade-off frontier. Layer-wise merging is more expressive, yet prior methods still suffer from two major limitations:

1. **Modeling inefficiency**: the high-dimensional layer-wise merge space is treated as an unstructured black box.
2. **System inefficiency**: synchronous optimization wastes wall-clock time under heterogeneous LLM evaluation latency.

AP-BMM addresses these two bottlenecks jointly. It constructs a **discrepancy-derived importance prior** from layer-wise parameter and activation discrepancies, uses this prior to initialize the geometry of the GP surrogate, and couples the surrogate with a **pending-aware asynchronous multi-objective Bayesian optimization loop**. A lightweight **frontier-coverage reranking** stage further improves dispatch diversity and reduces redundant proposals.

Under the common evaluation protocol described in the paper, AP-BMM is designed to produce stronger Pareto-set approximations than synchronous layer-wise baselines while also reducing runtime overhead caused by stragglers.

## Main components

The retained codebase corresponds to the core design described in the main paper.

### 1. Prior-guided layer-wise search

AP-BMM builds an importance prior from:

- **parameter discrepancy** between source models
- **reasoning-set activation discrepancy** across layers

These signals are fused into a layer-wise prior that initializes surrogate sensitivity instead of directly constraining the feasible decision space.

### 2. Pending-aware asynchronous optimization

The optimizer maintains separate **completed** and **pending** evaluation sets. Whenever a worker becomes available, AP-BMM updates the surrogate on completed observations and dispatches the next candidate without waiting for the slowest model in a batch.

### 3. Frontier-coverage reranking

Before dispatch, AP-BMM reranks candidate pools with three signals:

- acquisition value
- frontier-gap reward
- decision-space proximity penalty

This encourages broader frontier coverage and reduces proposal collisions.

## Repository scope

This repository intentionally keeps only the paper's runnable core implementation.

### Core AP-BMM pipeline

- `src/evoMI/mi_opt_unified.py` — unified CLI entry for AP-BMM and the retained layer-wise baselines
- `src/evoMI/async_merge_evaluator.py` — asynchronous merge construction, evaluation scheduling, and objective extraction
- `src/evoMI/mi_opt_optimizer.py` — prior construction and prior-guided optimization helpers
- `src/evoMI/task_diff_analyzer.py` — discrepancy analysis used to build the AP-BMM prior
- `src/evoMI/evaluation_utils.py` — benchmark configuration, caching, and task execution helpers
- `src/evoMI/runtime_artifacts.py` — runtime traces, hypervolume curves, and checkpoint-side runtime summaries

### Layer-wise baselines retained for Experiment 1

- `src/evoMI/qnehvi_optimizer.py` — synchronous qNEHVI baseline
- `src/evoMI/momm_optimizer.py` — synchronous MOMM-style baseline
- `src/evoMI/moead_cmaes_prior_optimizer.py` — MOEA-D+CMA/ES baseline
- `src/evoMI/optimizer.py` — prior BO baseline support
- `src/evoMI/saasbo_qnehvi_optimizer.py` — SAAS-prior optimization backend used by AP-BMM

Experiment 1 in `APBMM.tex` compares AP-BMM against:

- MOMM
- MOEA-D+CMA/ES
- qNEHVI
- TA as a model-level reference

### Model-level baselines retained for Experiment 2

- `src/ta_methos/model_level_fusion_test.py`

Experiment 2 compares AP-BMM against the model-level methods listed in the main paper:

- Task Arithmetic (TA)
- TIES
- DARE
- Breadcrumbs
- DELLA

## Intentionally excluded

This release does **not** include:

- checkpoint replay or checkpoint re-evaluation utilities
- checkpoint-only post-analysis scripts
- paper-only figure regeneration scripts
- manuscript sources and unrelated exploratory tooling

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

`models/`, `checkpoints/`, `output/`, and `.mplconfig/` are intentionally lightweight placeholders in Git.

## Setup

```bash
cd /path/to/APBMM
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
export PYTHONPATH=.
export MPLCONFIGDIR=$PWD/.mplconfig
```

## Required local models

The default local layout is:

```text
models/
├── Qwen3-4B-Instruct-2507/
└── Qwen3-4B-thinking-2507/
```

You may override model paths from the command line.

## Sanity checks

```bash
python scripts/apbmm_quickstart.py
python src/evoMI/mi_opt_unified.py --help
python src/ta_methos/model_level_fusion_test.py --help
```

## Paper-aligned AP-BMM setting

In the retained implementation, the paper's AP-BMM setting corresponds to:

- algorithm alias: `sass_prior_bo_wo_update_gap_async`
- normalized optimizer family: `prior_saas_bo`

The main-paper configuration in `APBMM.tex` uses:

- shared initial design: `8` points
- reporting budget: `40` completed evaluations after initialization
- dispatch width: `q = 4`
- worker pool: `4` GPUs
- acquisition restarts: `10`
- raw samples: `512`
- Sobol-QMC samples: `128`
- reranking weights: `lambda_gap = 0.25`, `lambda_prox = 0.15`
- candidate-pool multiplier: `3`
- async completion threshold: `0.15`
- prior fusion weights: parameter / activation = `0.5 / 0.5`

Example AP-BMM command:

```bash
python src/evoMI/mi_opt_unified.py \
  --algorithm sass_prior_bo_wo_update_gap_async \
  --base-model-path models/Qwen3-4B-Instruct-2507 \
  --task-model-paths models/Qwen3-4B-thinking-2507 models/Qwen3-4B-Instruct-2507 \
  --num-blocks 36 \
  --initial-samples 8 \
  --batch-size 4 \
  --max-evaluations 48 \
  --num-restarts 10 \
  --raw-samples 512 \
  --mc-samples 128 \
  --eval-profile aime_gpqa \
  --eval-aime-limit 5 \
  --eval-gpqa-limit 60 \
  --available-gpus 0 1 2 3
```

`48 = 8` shared initial-design evaluations `+ 40` completed evaluations after initialization, matching the Experiment 1 budget in the main paper.

## Baseline commands

### Layer-wise baselines

`src/evoMI/mi_opt_unified.py` also exposes the retained Experiment 1 baselines:

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
  --max-evaluations 48 \
  --eval-profile aime_gpqa \
  --available-gpus 0 1 2 3
```

### Model-level baselines

```bash
python src/ta_methos/model_level_fusion_test.py \
  --fusion_method ties \
  --num_blocks 8 \
  --budget 80 \
  --batch_size 4
```

`80` matches the model-level search budget used in Experiment 2 of `APBMM.tex`.

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

- `checkpoints/` — optimization checkpoints and runtime artifacts
- `output/` — temporary merged models, evaluation cache, and intermediate runtime files
- `models/` — local model weights

## Notes

- The runtime assumes local GPU evaluation with persistent vLLM workers.
- `evalscope/` and `mergekit/` are vendored to keep the repository self-contained.
- The default benchmark profile used in the main paper is `aime_gpqa`.
- On a new machine, the usual adjustments are model paths, GPU ids, and local dataset cache locations.
