#!/usr/bin/env python3

import os
import sys
import time
import json
import gc
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evoMI.optimizer import prior_bo_optimizer
from src.evoMI.mi_block_fusion import calculate_merged_blocks
from src.evoMI.task_diff_analyzer import TaskDiffAnalyzer
from src.evoMI.mi_opt_saasbo2 import (
    initialize_model_evaluations,
    create_optimizer_config,
    model_merge_optimization_function,
    create_iteration_callback,
    save_optimization_results,
    save_settings,
    visualize_optimization_results,
    set_available_gpus,
    get_shared_vllm_manager,
    get_idle_gpu_count,
    shutdown_shared_vllm_manager,
)


def _default_reasoning_specs(limit=None):
    aime_limit = 5 if limit is None else limit
    gpqa_limit = 60 if limit is None else limit
    return [
        {
            "dataset_id": "opencompass/AIME2025",
            "split": "test",
            "text_field": "question",
            "subset_field": "subset",
            "subset_values": ["AIME2025-I"],
            "limit": aime_limit,
        },
        {
            "dataset_id": "opencompass/AIME2025",
            "split": "test",
            "text_field": "question",
            "subset_field": "subset",
            "subset_values": ["AIME2025-II"],
            "limit": aime_limit,
        },
        {
            "dataset_id": "AI-ModelScope/gpqa_diamond",
            "split": "train",
            "text_field": "Question",
            "limit": gpqa_limit,
        },
    ]


def _default_general_specs(limit=None):
    gpqa_limit = 60 if limit is None else limit
    return [
        {
            "dataset_id": "AI-ModelScope/gpqa_diamond",
            "split": "train",
            "text_field": "Question",
            "limit": gpqa_limit,
        }
    ]


def _align_prior_length(values, dim):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] < dim:
        repeat_n = int(np.ceil(dim / arr.shape[0]))
        return np.tile(arr, repeat_n)[:dim]
    return arr[:dim]


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _release_gpu_memory(tag):
    if not torch.cuda.is_available():
        return
    gc.collect()
    device_count = torch.cuda.device_count()
    for device_idx in range(device_count):
        torch.cuda.set_device(device_idx)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    free_gb, total_gb = torch.cuda.mem_get_info()
    print(f"[GPU Memory] {tag} free={free_gb/1024**3:.2f}GiB total={total_gb/1024**3:.2f}GiB")


def _build_blueprint_priors(
    base_model_path,
    reason_model_path,
    device,
    output_dir,
    reasoning_specs,
    general_specs,
    proxy_batch_size=2,
    proxy_max_length=256,
    proxy_pooling="last_token",
    proxy_dtype="float16",
    patch_topk=8,
    beta_shared=1.0,
    lambda_spec=1.0,
    w_param=0.30,
    w_shared=0.40,
    w_patch_abs=0.30,
    v_spec=0.40,
    v_ratio=0.20,
    v_patch=0.40,
):
    visualizer = TaskDiffAnalyzer(device=device)
    reasoning_texts = visualizer.build_proxy_prompt_sets(reasoning_specs)
    general_texts = visualizer.build_proxy_prompt_sets(general_specs)
    if len(general_texts) == 0:
        general_texts = reasoning_texts

    tokenizer, reason_model = visualizer.load_causal_model_and_tokenizer(reason_model_path, dtype_str=proxy_dtype)
    _, base_model = visualizer.load_causal_model_and_tokenizer(base_model_path, dtype_str=proxy_dtype)
    num_layers = reason_model.config.num_hidden_layers

    reason_tensors = visualizer.load_model_tensors(reason_model_path)
    base_tensors = visualizer.load_model_tensors(base_model_path)
    param_score = visualizer.compute_param_delta_scores(reason_tensors, base_tensors, num_layers)
    act_reason = visualizer.compute_activation_distance_scores(
        reason_model=reason_model,
        base_model=base_model,
        tokenizer=tokenizer,
        texts=reasoning_texts,
        num_layers=num_layers,
        batch_size=proxy_batch_size,
        max_length=proxy_max_length,
        pooling=proxy_pooling,
    )
    act_general = visualizer.compute_activation_distance_scores(
        reason_model=reason_model,
        base_model=base_model,
        tokenizer=tokenizer,
        texts=general_texts,
        num_layers=num_layers,
        batch_size=proxy_batch_size,
        max_length=proxy_max_length,
        pooling=proxy_pooling,
    )

    act_shared = (act_reason + beta_shared * act_general) / (1.0 + beta_shared)
    act_spec = act_reason - lambda_spec * act_general
    act_ratio_log = np.log(act_reason + 1e-8) - np.log(act_general + 1e-8)
    candidate_layers = np.argsort(-act_shared)[: min(patch_topk, num_layers)].tolist()
    patch_score = visualizer.compute_patch_scores(
        reason_model=reason_model,
        base_model=base_model,
        tokenizer=tokenizer,
        texts=reasoning_texts,
        candidate_layers=candidate_layers,
        num_layers=num_layers,
        batch_size=proxy_batch_size,
        max_length=proxy_max_length,
    )

    p_norm = visualizer.robust_normalize(param_score)
    a_shared_norm = visualizer.robust_normalize(act_shared)
    a_spec_norm = visualizer.robust_normalize(act_spec)
    a_ratio_norm = visualizer.robust_normalize(act_ratio_log)
    t_signed_norm = visualizer.robust_normalize(patch_score)
    t_abs_norm = visualizer.robust_normalize(np.abs(patch_score))

    u_prior = w_param * p_norm + w_shared * a_shared_norm + w_patch_abs * t_abs_norm
    u_prior = visualizer.smooth_scores(u_prior)
    u_prior = np.clip(u_prior, 0.05, 1.0)

    m_prior = v_spec * a_spec_norm + v_ratio * a_ratio_norm + v_patch * t_signed_norm
    m_prior = visualizer.smooth_scores(m_prior)
    m_prior = np.clip(m_prior, 0.05, 0.95)

    proxy_metrics = {
        "layer_numbers": list(range(num_layers)),
        "param_score": param_score.tolist(),
        "act_reason": act_reason.tolist(),
        "act_general": act_general.tolist(),
        "act_shared": act_shared.tolist(),
        "act_spec": act_spec.tolist(),
        "act_ratio_log": act_ratio_log.tolist(),
        "patch_score": patch_score.tolist(),
        "importance_score": u_prior.tolist(),
        "u_prior": u_prior.tolist(),
        "m_prior": m_prior.tolist(),
        "candidate_layers": candidate_layers,
        "reasoning_text_count": len(reasoning_texts),
        "general_text_count": len(general_texts),
    }
    visualizer.visualize_proxy_metrics(proxy_metrics, output_dir=output_dir)
    with open(os.path.join(output_dir, "model_proxy_metrics.json"), "w") as fp:
        json.dump(proxy_metrics, fp, indent=2, ensure_ascii=False)

    del reason_model, base_model
    del tokenizer, reason_tensors, base_tensors
    del visualizer, reasoning_texts, general_texts
    _release_gpu_memory("after_proxy_metrics")
    return m_prior, u_prior, proxy_metrics


def main_optimization(
    custom_initial_solutions=None,
    num_blocks=36,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=20,
    verbose=True,
    device="cuda",
    dtype=torch.double,
    initial_samples=4,
    noise_level=0.0001,
    run_id="priorbo_real_36_4_20_aime10_gpqa100",
    cache_dir="output/mi_optimization_temp",
    alpha=1.0,
    beta=0.0,
    base_model=["models/Qwen3-4B-Instruct-2507", "models/Qwen3-4B-thinking-2507"],
    expert_model=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"],
    base_model_path="models/Qwen3-4B-Instruct-2507",
    task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"],
    fusion_method="task_arithmetic",
    optimize_density=1,
    seed=42,
    rho=0.5,
    topk=6,
    n_groups=4,
    enable_grouping=False,
    max_tokens=35000,
    max_model_len=48000,
    available_gpus=None,
    reasoning_specs=None,
    general_specs=None,
    reasoning_limit=None,
    general_limit=None,
    proxy_batch_size=2,
    proxy_max_length=256,
    proxy_pooling="last_token",
    proxy_dtype="float16",
    patch_topk=8,
    beta_shared=1.0,
    lambda_spec=1.0,
    w_param=0.30,
    w_shared=0.40,
    w_patch_abs=0.30,
    v_spec=0.40,
    v_ratio=0.20,
    v_patch=0.40,
    max_evaluations=88,
    eval_aime_limit=4,
    eval_gpqa_limit=20,
    async_mode=False,
    wait_for_completion_threshold=0.15,
):
    _set_global_seed(seed)
    start_time = time.time()
    checkpoint_root = "./checkpoints"
    run_dir = os.path.join(checkpoint_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, "output", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    params = {
        "custom_initial_solutions": custom_initial_solutions,
        "num_blocks": num_blocks,
        "num_objectives": num_objectives,
        "batch_size": BATCH_SIZE,
        "num_restarts": NUM_RESTARTS,
        "raw_samples": RAW_SAMPLES,
        "mc_samples": MC_SAMPLES,
        "n_batch": N_BATCH,
        "verbose": verbose,
        "device": device,
        "dtype": str(dtype),
        "initial_samples": initial_samples,
        "noise_level": noise_level,
        "run_id": run_id,
        "cache_dir": cache_dir,
        "alpha": alpha,
        "beta": beta,
        "base_model": base_model,
        "expert_model": expert_model,
        "base_model_path": base_model_path,
        "task_model_paths": task_model_paths,
        "fusion_method": fusion_method,
        "optimize_density": optimize_density,
        "seed": seed,
        "algorithm": "prior_bo_qnehvi",
        "rho": rho,
        "topk": topk,
        "n_groups": n_groups,
        "enable_grouping": enable_grouping,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "available_gpus": available_gpus,
        "reasoning_specs": reasoning_specs,
        "general_specs": general_specs,
        "reasoning_limit": reasoning_limit,
        "general_limit": general_limit,
        "proxy_batch_size": proxy_batch_size,
        "proxy_max_length": proxy_max_length,
        "proxy_pooling": proxy_pooling,
        "proxy_dtype": proxy_dtype,
        "patch_topk": patch_topk,
        "beta_shared": beta_shared,
        "lambda_spec": lambda_spec,
        "w_param": w_param,
        "w_shared": w_shared,
        "w_patch_abs": w_patch_abs,
        "v_spec": v_spec,
        "v_ratio": v_ratio,
        "v_patch": v_patch,
        "max_evaluations": max_evaluations,
        "eval_aime_limit": eval_aime_limit,
        "eval_gpqa_limit": eval_gpqa_limit,
        "async_mode": async_mode,
        "wait_for_completion_threshold": wait_for_completion_threshold,
    }
    save_settings(params, output_dir)

    if reasoning_specs is None:
        reasoning_specs = _default_reasoning_specs(reasoning_limit)
    if general_specs is None:
        general_specs = _default_general_specs(general_limit)

    if available_gpus is None:
        available_gpus = list(range(torch.cuda.device_count()))
    set_available_gpus(available_gpus)
    get_shared_vllm_manager(max_model_len=max_model_len if max_model_len is not None else (max_tokens + 3000))

    merged_blocks = calculate_merged_blocks(
        task_model_paths=task_model_paths,
        num_blocks=num_blocks,
        alpha=alpha,
        beta=beta,
        checkpoint_dir=cache_dir,
        metric="L2-norm",
        partition_method="hybrid",
    )

    config = create_optimizer_config(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=num_blocks,
        num_objectives=num_objectives,
        BATCH_SIZE=BATCH_SIZE,
        NUM_RESTARTS=NUM_RESTARTS,
        RAW_SAMPLES=RAW_SAMPLES,
        MC_SAMPLES=MC_SAMPLES,
        N_BATCH=N_BATCH,
        verbose=verbose,
        device=device,
        dtype=dtype,
        initial_samples=initial_samples,
        noise_level=noise_level,
        run_id=run_id,
        checkpoint_dir=run_dir,
        optimize_density=optimize_density,
    )
    config["seed"] = seed
    config["scheduler_gpu_count"] = len(available_gpus)
    config["max_evaluations"] = max_evaluations
    config["async_mode"] = async_mode
    config["wait_for_completion_threshold"] = wait_for_completion_threshold
    config["full_eval_limits"] = {"aime25": eval_aime_limit, "gpqa_diamond": eval_gpqa_limit}

    reason_model_path = task_model_paths[0] if len(task_model_paths) > 0 else base_model_path
    m_prior, u_prior, _ = _build_blueprint_priors(
        base_model_path=base_model_path,
        reason_model_path=reason_model_path,
        device=device,
        output_dir=output_dir,
        reasoning_specs=reasoning_specs,
        general_specs=general_specs,
        proxy_batch_size=proxy_batch_size,
        proxy_max_length=proxy_max_length,
        proxy_pooling=proxy_pooling,
        proxy_dtype=proxy_dtype,
        patch_topk=patch_topk,
        beta_shared=beta_shared,
        lambda_spec=lambda_spec,
        w_param=w_param,
        w_shared=w_shared,
        w_patch_abs=w_patch_abs,
        v_spec=v_spec,
        v_ratio=v_ratio,
        v_patch=v_patch,
    )
    m_prior = _align_prior_length(m_prior, config["dim"])
    u_prior = _align_prior_length(u_prior, config["dim"])
    _release_gpu_memory("before_formal_evaluation")

    global base_model_results, expert_model_results
    base_model_results, expert_model_results = initialize_model_evaluations(
        base_model,
        expert_model,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        eval_limit={"aime25": eval_aime_limit, "gpqa_diamond": eval_gpqa_limit},
    )

    iteration_callback = create_iteration_callback(cache_dir=cache_dir, cleanup_interval=1, keep_dirs=["important", "checkpoint"], exclude_patterns=["pareto", "best", "important"])

    def wrapped_optimization_function(x, eval_limit=None, eval_mode="full", estimated_tokens=None):
        return model_merge_optimization_function(
            x,
            merged_blocks=merged_blocks,
            num_blocks=num_blocks,
            cache_dir=cache_dir,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            fusion_method=fusion_method,
            base_model_results=base_model_results,
            expert_model_results=expert_model_results,
            optimize_density=optimize_density,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            eval_limit=eval_limit,
            eval_mode=eval_mode,
            estimated_tokens=estimated_tokens,
        )
    wrapped_optimization_function.get_idle_gpu_count = get_idle_gpu_count

    result = prior_bo_optimizer(
        wrapped_optimization_function,
        iteration_callback=iteration_callback,
        m_prior=m_prior,
        u_prior=u_prior,
        rho=rho,
        topk=topk,
        n_groups=n_groups,
        enable_grouping=enable_grouping,
        **config,
    )

    elapsed_time = time.time() - start_time
    print(f"\n优化完成！总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/3600:.2f} 小时)")

    result_dict = {
        "pareto_x": result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        "pareto_y": result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        "all_x": result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        "all_y": result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        "all_metrics": result[2],
        "hypervolume_history": result[3] if len(result) > 3 else [],
        "problem_ref_point": result[4].tolist() if isinstance(result[4], torch.Tensor) else result[4],
        "run_id": result[5] if len(result) > 5 else None,
    }

    save_optimization_results(result_dict, output_dir)
    visualize_optimization_results(result_dict, output_dir)

    pareto_x = result_dict.get("pareto_x", np.array([]))
    print(f"\n找到 {len(pareto_x)} 个帕累托最优解")
    print("\n=== 优化统计信息 ===")
    print(f"总评估次数: {len(result_dict.get('all_x', []))}")
    print(f"初始样本数: {config['initial_samples']}")
    print(f"最大评估次数: {config['max_evaluations']}")
    print(f"每次候选数: {config['BATCH_SIZE']}")
    try:
        hypervolume_history = result_dict.get("hypervolume_history", [0])
        best_hypervolume = max(hypervolume_history) if hypervolume_history else 0
        print(f"最佳超体积: {best_hypervolume}")
    except Exception:
        print("最佳超体积: 计算失败")
    print(f"\n所有结果已保存到: {output_dir}")
    shutdown_shared_vllm_manager()

    return result_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型合并优化工具 - 使用Prior-BO+qNEHVI算法")
    parser.add_argument("--custom-initial-solutions", type=str, default=None)
    parser.add_argument("--num-blocks", type=int, default=36)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--num-objectives", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--initial-samples", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.0001)
    parser.add_argument("--run-id", type=str, default="priorbo_real_36_4_20_aime10_gpqa100")
    parser.add_argument("--cache-dir", type=str, default="output/mi_optimization_temp")
    parser.add_argument("--base-model-path", type=str, default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--task-model-paths", nargs="+", default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--base-model", nargs="+", default=["models/Qwen3-4B-Instruct-2507", "models/Qwen3-4B-thinking-2507"])
    parser.add_argument("--expert-model", nargs="+", default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--fusion-method", type=str, default="task_arithmetic")
    parser.add_argument("--optimize-density", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--n-groups", type=int, default=4)
    parser.add_argument("--enable-grouping", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=35000)
    parser.add_argument("--max-model-len", type=int, default=48000)
    parser.add_argument("--available-gpus", type=str, default="auto")
    parser.add_argument("--reasoning-limit", type=int, default=None)
    parser.add_argument("--general-limit", type=int, default=None)
    parser.add_argument("--proxy-batch-size", type=int, default=2)
    parser.add_argument("--proxy-max-length", type=int, default=256)
    parser.add_argument("--proxy-pooling", choices=["last_token", "mean"], default="last_token")
    parser.add_argument("--proxy-dtype", type=str, default="float16")
    parser.add_argument("--patch-topk", type=int, default=8)
    parser.add_argument("--beta-shared", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=1.0)
    parser.add_argument("--w-param", type=float, default=0.30)
    parser.add_argument("--w-shared", type=float, default=0.40)
    parser.add_argument("--w-patch-abs", type=float, default=0.30)
    parser.add_argument("--v-spec", type=float, default=0.40)
    parser.add_argument("--v-ratio", type=float, default=0.20)
    parser.add_argument("--v-patch", type=float, default=0.40)
    parser.add_argument("--max-evaluations", type=int, default=88)
    parser.add_argument("--eval-aime-limit", type=int, default=4)
    parser.add_argument("--eval-gpqa-limit", type=int, default=20)
    parser.add_argument("--async-opt", dest="async_opt", action="store_true")
    parser.add_argument("--sync-opt", dest="async_opt", action="store_false")
    parser.set_defaults(async_opt=False)
    parser.add_argument("--wait-for-completion-threshold", type=float, default=0.15)
    parser.add_argument("--reasoning-specs-json", type=str, default=None)
    parser.add_argument("--general-specs-json", type=str, default=None)

    args = parser.parse_args()

    custom_initial_solutions = None
    if args.custom_initial_solutions:
        custom_initial_solutions = [float(x) for x in args.custom_initial_solutions.split(",")]
    if args.available_gpus == "auto":
        available_gpus = list(range(torch.cuda.device_count()))
    else:
        available_gpus = [int(x.strip()) for x in args.available_gpus.split(",") if x.strip()]
    reasoning_specs = json.loads(args.reasoning_specs_json) if args.reasoning_specs_json else None
    general_specs = json.loads(args.general_specs_json) if args.general_specs_json else None

    main_optimization(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=args.num_blocks,
        num_objectives=args.num_objectives,
        BATCH_SIZE=args.batch_size,
        NUM_RESTARTS=args.num_restarts,
        RAW_SAMPLES=args.raw_samples,
        MC_SAMPLES=args.mc_samples,
        N_BATCH=1,
        verbose=args.verbose,
        device=args.device,
        dtype=torch.double,
        initial_samples=args.initial_samples,
        noise_level=args.noise_level,
        run_id=args.run_id,
        cache_dir=args.cache_dir,
        alpha=args.alpha,
        beta=args.beta,
        base_model=args.base_model,
        expert_model=args.expert_model,
        base_model_path=args.base_model_path,
        task_model_paths=args.task_model_paths,
        fusion_method=args.fusion_method,
        optimize_density=args.optimize_density,
        seed=args.seed,
        rho=args.rho,
        topk=args.topk,
        n_groups=args.n_groups,
        enable_grouping=args.enable_grouping,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        available_gpus=available_gpus,
        reasoning_specs=reasoning_specs,
        general_specs=general_specs,
        reasoning_limit=args.reasoning_limit,
        general_limit=args.general_limit,
        proxy_batch_size=args.proxy_batch_size,
        proxy_max_length=args.proxy_max_length,
        proxy_pooling=args.proxy_pooling,
        proxy_dtype=args.proxy_dtype,
        patch_topk=args.patch_topk,
        beta_shared=args.beta_shared,
        lambda_spec=args.lambda_spec,
        w_param=args.w_param,
        w_shared=args.w_shared,
        w_patch_abs=args.w_patch_abs,
        v_spec=args.v_spec,
        v_ratio=args.v_ratio,
        v_patch=args.v_patch,
        max_evaluations=args.max_evaluations,
        eval_aime_limit=args.eval_aime_limit,
        eval_gpqa_limit=args.eval_gpqa_limit,
        async_mode=args.async_opt,
        wait_for_completion_threshold=args.wait_for_completion_threshold,
    )
