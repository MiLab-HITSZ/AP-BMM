# AP-BMM

Official code release for **AP-BMM (Asynchronous Prior-guided Bayesian Model Merging)**.

AP-BMM targets the capability-efficiency trade-off in LLM merging. Compared with coarse model-level merging, it performs **layer-wise optimization** and addresses two bottlenecks emphasized in the paper:

1. **Modeling inefficiency** in the high-dimensional layer-wise merge space.
2. **System inefficiency** caused by synchronous straggler effects under heterogeneous LLM evaluation latency.

The released code keeps only the paper's core pipeline: AP-BMM itself, the optimization baselines used in the layer-wise comparison, the model-level baselines used in the granularity comparison, and the necessary evaluation/runtime utilities.

## Method overview

AP-BMM combines three coupled components from the paper:

- **Discrepancy-derived importance prior**  
  Layer-wise parameter discrepancy and reasoning-set activation discrepancy are fused into a prior that initializes surrogate sensitivity.
- **Pending-aware asynchronous Bayesian optimization**  
  The optimizer refits on completed observations and dispatches new candidates without waiting for the slowest batch element.
- **Frontier-coverage reranking**  
  Candidate pools are reranked with acquisition, frontier-gap reward, and decision-space proximity penalty to improve coverage and reduce redundant dispatches.

## Paper-aligned implementation scope

### Core AP-BMM pipeline

- `src/evoMI/mi_opt_unified.py` — unified CLI entry for AP-BMM and layer-wise baselines.
- `src/evoMI/async_merge_evaluator.py` — asynchronous merge construction, evaluation scheduling, and objective extraction.
- `src/evoMI/mi_opt_optimizer.py` — prior construction and prior-guided optimization helpers.
- `src/evoMI/task_diff_analyzer.py` — parameter / activation discrepancy analysis for prior construction.
- `src/evoMI/runtime_artifacts.py` — runtime traces, hypervolume curves, and checkpoint-side runtime summaries.
- `src/evoMI/evaluation_utils.py` — benchmark configuration, caching, and task execution helpers.

### Layer-wise baselines in Experiment 1

- `src/evoMI/qnehvi_optimizer.py` — synchronous qNEHVI baseline.
- `src/evoMI/momm_optimizer.py` — synchronous MOMM-style baseline.
- `src/evoMI/moead_cmaes_prior_optimizer.py` — MOEA-D+CMA/ES baseline.
- `src/evoMI/optimizer.py` — prior BO baseline support.
- `src/evoMI/saasbo_qnehvi_optimizer.py` — SAAS-prior optimization backend used by AP-BMM.

### Model-level baselines in Experiment 2

- `src/ta_methos/model_level_fusion_test.py`

Supported model-level methods:

- Task Arithmetic (TA)
- TIES
- DARE
- Breadcrumbs
- DELLA

## What is intentionally excluded

This release does **not** keep:

- checkpoint replay / re-evaluation utilities
- checkpoint-only analysis scripts
- paper figure regeneration scripts
- manuscript LaTeX sources
- exploratory or debugging-only scripts outside the paper workflow

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

## Required models

Place the required models under `models/`. The default paper-style local layout is:

```text
models/
├── Qwen3-4B-Instruct-2507/
└── Qwen3-4B-thinking-2507/
```

You may also override model paths from the CLI.

## Sanity checks

```bash
python scripts/apbmm_quickstart.py
python src/evoMI/mi_opt_unified.py --help
python src/ta_methos/model_level_fusion_test.py --help
```

## AP-BMM paper configuration

The paper's AP-BMM setting corresponds to:

- algorithm alias: `sass_prior_bo_wo_update_gap_async`
- normalized optimizer family: `prior_saas_bo`

This setting matches the paper's core design:

- shared initial design: `8` points
- reporting budget: `40` completed evaluations after initialization
- dispatch width: `q = 4`
- worker pool: `4` GPUs
- acquisition restarts: `10`
- raw samples: `512`
- Sobol-QMC MC samples: `128`
- reranking weights: `lambda_gap = 0.25`, `lambda_prox = 0.15`
- candidate-pool multiplier: `3`
- async completion threshold: `0.15`
- prior fusion weights: parameter / activation = `0.5 / 0.5`

Example command:

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

`88 = 8` initial evaluations `+ 80` total evaluations, which matches the retained implementation's paper-style run setup.

## Layer-wise baseline runs

`src/evoMI/mi_opt_unified.py` also exposes the main layer-wise baselines used in the paper:

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

## Model-level baseline runs

```bash
python src/ta_methos/model_level_fusion_test.py \
  --fusion_method ties \
  --num_blocks 8 \
  --budget 88 \
  --batch_size 4
```

Supported `--fusion_method` values include:

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

- The runtime is designed for local GPU evaluation with persistent vLLM workers.
- `evalscope/` and `mergekit/` are vendored so the repository remains self-contained.
- The default benchmark profile used by the main paper experiments is `aime_gpqa`.
- For a new machine, the usual changes are model paths, GPU ids, and local dataset cache locations.
