#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evoMI.mi_opt_unified import main_optimization
from src.evoMI.model_reproduction import build_eval_cache_config, build_eval_setting_id


DEFAULT_ALGORITHMS = [
    {"name": "apbmm", "algorithm": "sass_prior_bo_wo_update_gap_async", "async_mode": True},
    {"name": "qnehvi", "algorithm": "qnehvi", "async_mode": False},
    {"name": "momm", "algorithm": "momm", "async_mode": False},
    {"name": "moead_cmaes", "algorithm": "moead_cmaes", "async_mode": False},
]


def _sanitize_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def _resolve_limit(raw_value, use_full=False):
    if use_full:
        return None
    if raw_value is None:
        return None
    return int(raw_value)


def _build_run_cache_dir(cache_root, eval_setting_id, run_id):
    short_eval_dir = hashlib.sha1(str(eval_setting_id).encode("utf-8")).hexdigest()[:12]
    short_run_dir = hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_root, short_eval_dir, short_run_dir)


def _build_run_plan(args):
    eval_profile = str(args.eval_profile).strip().lower()
    if eval_profile == "math500_level5_gpqa":
        eval_limits = {
            "math_500": _resolve_limit(args.math_limit, args.math_full),
            "gpqa_diamond": _resolve_limit(args.gpqa_limit, args.gpqa_full),
        }
        eval_repeats = {
            "math_500": int(args.math_repeats),
            "gpqa_diamond": int(args.gpqa_repeats),
        }
    else:
        eval_limits = {
            "gsm8k": _resolve_limit(args.gsm8k_limit, args.gsm8k_full),
            "gpqa_diamond": _resolve_limit(args.gpqa_limit, args.gpqa_full),
        }
        eval_repeats = {
            "gsm8k": int(args.gsm8k_repeats),
            "gpqa_diamond": int(args.gpqa_repeats),
        }
    eval_config = build_eval_cache_config(
        eval_profile=eval_profile,
        eval_limit=eval_limits,
        repeats=eval_repeats,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    eval_setting_id = build_eval_setting_id(eval_config)
    n_batch = max(0, int(math.ceil(max(args.max_evaluations - args.initial_samples, 0) / max(args.batch_size, 1))))
    algorithm_names = set(args.algorithms) if args.algorithms else None
    selected_algorithms = []
    for item in DEFAULT_ALGORITHMS:
        if algorithm_names is None or item["name"] in algorithm_names:
            selected_algorithms.append(item)
    plan = []
    for order, item in enumerate(selected_algorithms, start=1):
        run_name = f"{order:02d}_{item['name']}_{eval_setting_id}_seed{int(args.seed)}"
        run_id = _sanitize_name(run_name)
        plan.append(
            {
                "order": order,
                "name": item["name"],
                "algorithm": item["algorithm"],
                "async_mode": bool(item["async_mode"]),
                "run_id": run_id,
                "eval_limits": eval_limits,
                "eval_repeats": eval_repeats,
                "eval_setting_id": eval_setting_id,
                "n_batch": n_batch,
            }
        )
    return {
        "eval_profile": eval_profile,
        "eval_limits": eval_limits,
        "eval_repeats": eval_repeats,
        "eval_setting_id": eval_setting_id,
        "initial_samples": int(args.initial_samples),
        "batch_size": int(args.batch_size),
        "max_evaluations": int(args.max_evaluations),
        "n_batch": n_batch,
        "algorithms": plan,
    }


def _collect_result_summary(result):
    pareto_x = result.get("pareto_x")
    hypervolume_history = result.get("hypervolume_history", []) or []
    all_metrics = result.get("all_metrics", []) or []
    return {
        "pareto_count": int(len(pareto_x)) if pareto_x is not None else 0,
        "evaluated_count": int(len(all_metrics)),
        "best_hypervolume": float(max(hypervolume_history)) if len(hypervolume_history) > 0 else None,
        "last_hypervolume": float(hypervolume_history[-1]) if len(hypervolume_history) > 0 else None,
    }


def _write_json(file_path, payload):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_evaluations(args):
    plan = _build_run_plan(args)
    if len(plan["algorithms"]) == 0:
        raise ValueError("没有可执行的算法，请检查 --algorithms 参数")
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    checkpoint_root = os.path.abspath(args.checkpoint_root)
    cache_root = os.path.abspath(args.cache_root)
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan": plan,
        "runs": [],
    }
    summary_path = os.path.join(
        checkpoint_root,
        f"evaluation_summary_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )

    for run_spec in plan["algorithms"]:
        run_cache_dir = _build_run_cache_dir(
            cache_root,
            run_spec["eval_setting_id"],
            run_spec["run_id"],
        )
        started_at = time.time()
        run_record = {
            "name": run_spec["name"],
            "algorithm": run_spec["algorithm"],
            "async_mode": run_spec["async_mode"],
            "run_id": run_spec["run_id"],
            "status": "running",
            "checkpoint_dir": os.path.join(checkpoint_root, run_spec["run_id"]),
            "cache_dir": run_cache_dir,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary["runs"].append(run_record)
        _write_json(summary_path, summary)
        try:
            result = main_optimization(
                algorithm=run_spec["algorithm"],
                run_id=run_spec["run_id"],
                checkpoint_root=checkpoint_root,
                cache_dir=run_cache_dir,
                fusion_method="task_arithmetic",
                initial_samples=args.initial_samples,
                BATCH_SIZE=args.batch_size,
                N_BATCH=run_spec["n_batch"],
                max_evaluations=args.max_evaluations,
                eval_profile=plan["eval_profile"],
                full_eval_limits=run_spec["eval_limits"],
                eval_repeats=run_spec["eval_repeats"],
                async_mode=run_spec["async_mode"],
                wait_for_completion_threshold=args.wait_for_completion_threshold,
                seed=args.seed,
                num_blocks=args.num_blocks,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
                available_gpus=args.available_gpus,
                base_model_path=args.base_model_path,
                task_model_paths=args.task_model_paths,
                base_model=args.base_model,
                expert_model=args.expert_model,
                verbose=not args.quiet,
            )
            run_record["status"] = "completed"
            run_record["duration_sec"] = float(time.time() - started_at)
            run_record["result"] = _collect_result_summary(result)
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["duration_sec"] = float(time.time() - started_at)
            run_record["error"] = str(exc)
            run_record["traceback"] = traceback.format_exc()
            _write_json(summary_path, summary)
            if not args.continue_on_error:
                raise
        _write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="顺序统计评测脚本")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=[item["name"] for item in DEFAULT_ALGORITHMS],
    )
    parser.add_argument("--checkpoint-root", type=str, default="./checkpoints/statistical_eval")
    parser.add_argument("--cache-root", type=str, default="./output/statistical_eval_cache")
    parser.add_argument("--initial-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-evaluations", type=int, default=40)
    parser.add_argument("--eval-profile", type=str, default="gsm8k_gpqa", choices=["gsm8k_gpqa", "math500_level5_gpqa"])
    parser.add_argument("--gsm8k-limit", type=int, default=100)
    parser.add_argument("--gsm8k-full", action="store_true", default=False)
    parser.add_argument("--math-limit", type=int, default=100)
    parser.add_argument("--math-full", action="store_true", default=False)
    parser.add_argument("--gpqa-limit", type=int, default=100)
    parser.add_argument("--gpqa-full", action="store_true", default=False)
    parser.add_argument("--gsm8k-repeats", type=int, default=1)
    parser.add_argument("--math-repeats", type=int, default=1)
    parser.add_argument("--gpqa-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-blocks", type=int, default=36)
    parser.add_argument("--max-tokens", type=int, default=35000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--available-gpus", nargs="+", type=int, default=None)
    parser.add_argument("--base-model-path", type=str, default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--task-model-paths", nargs="+", default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--base-model", nargs="+", default=["models/Qwen3-4B-Instruct-2507", "models/Qwen3-4B-thinking-2507"])
    parser.add_argument("--expert-model", nargs="+", default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--wait-for-completion-threshold", type=float, default=0.15)
    parser.add_argument("--plan-only", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_evaluations(args)


if __name__ == "__main__":
    main()
