#!/usr/bin/env python3
"""Utilities for evaluation configuration, caching, and vLLM task execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from evalscope import run_task


def list_eval_profile_datasets(eval_profile: str = "aime_gpqa") -> List[str]:
    profile = str(eval_profile or "aime_gpqa").strip().lower()
    if profile == "gsm8k_gpqa":
        return ["gsm8k", "gpqa_diamond"]
    if profile == "math500_level5_gpqa":
        return ["math_500", "gpqa_diamond"]
    if profile == "aime_gpqa":
        return ["aime25", "gpqa_diamond"]
    raise ValueError(f"不支持的评测配置: {eval_profile}")


def normalize_eval_limit(
    eval_profile: str = "aime_gpqa",
    eval_limit: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Optional[int]]:
    datasets = list_eval_profile_datasets(eval_profile)
    if eval_limit is None:
        if eval_profile == "gsm8k_gpqa":
            eval_limit = {"gsm8k": 50, "gpqa_diamond": 50}
        elif eval_profile == "math500_level5_gpqa":
            eval_limit = {"math_500": None, "gpqa_diamond": None}
        else:
            eval_limit = {"aime25": 5, "gpqa_diamond": 60}
    normalized = {}
    for dataset in datasets:
        value = eval_limit.get(dataset, None) if isinstance(eval_limit, dict) else None
        normalized[dataset] = None if value is None else int(value)
    return normalized


def normalize_eval_repeats(
    eval_profile: str = "aime_gpqa",
    repeats: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    datasets = list_eval_profile_datasets(eval_profile)
    if repeats is None:
        repeats = {dataset: 1 for dataset in datasets}
    normalized = {}
    for dataset in datasets:
        normalized[dataset] = int(repeats.get(dataset, 1))
    return normalized


def build_eval_config(
    eval_profile: str = "aime_gpqa",
    eval_limit: Optional[Dict[str, Optional[int]]] = None,
    repeats: Optional[Dict[str, int]] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    profile = str(eval_profile or "aime_gpqa").strip().lower()
    return {
        "eval_profile": profile,
        "datasets": list_eval_profile_datasets(profile),
        "limit": normalize_eval_limit(profile, eval_limit),
        "repeats": normalize_eval_repeats(profile, repeats),
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "seed": None if seed is None else int(seed),
    }


def build_eval_namespace(eval_config: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(eval_config, dict) or not eval_config:
        return "legacy"
    normalized_payload = {
        "eval_profile": str(eval_config.get("eval_profile", "unknown")).strip().lower(),
        "datasets": [str(item) for item in eval_config.get("datasets", [])],
        "limit": eval_config.get("limit", {}),
        "repeats": eval_config.get("repeats", {}),
        "max_tokens": eval_config.get("max_tokens"),
        "seed": eval_config.get("seed"),
    }
    digest = hashlib.md5(
        json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    dataset_part = "-".join(normalized_payload["datasets"]) if normalized_payload["datasets"] else "datasets"
    limit_part = "_".join(
        f"{key}{'all' if value is None else int(value)}"
        for key, value in sorted((normalized_payload["limit"] or {}).items())
    ) or "limit"
    repeat_part = "_".join(
        f"{key}x{int(value)}"
        for key, value in sorted((normalized_payload["repeats"] or {}).items())
    ) or "repeat"
    token_part = "tokall" if normalized_payload["max_tokens"] is None else f"tok{int(normalized_payload['max_tokens'])}"
    seed_part = "seednone" if normalized_payload["seed"] is None else f"seed{int(normalized_payload['seed'])}"
    raw_namespace = (
        f"{normalized_payload['eval_profile']}__{dataset_part}__{limit_part}__"
        f"{repeat_part}__{token_part}__{seed_part}__{digest}"
    )
    return re.sub(r"[^0-9A-Za-z._-]+", "_", raw_namespace)


def build_eval_setting_id(eval_config: Optional[Dict[str, Any]] = None) -> str:
    return build_eval_namespace(eval_config)


def create_eval_task_config(
    model_path: str,
    max_tokens: Optional[int] = None,
    eval_profile: str = "aime_gpqa",
    eval_limit: Optional[Dict[str, Optional[int]]] = None,
    repeats: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
):
    from src.config_manager import config_manager

    profile = str(eval_profile or "aime_gpqa").strip().lower()
    normalized_limit = normalize_eval_limit(profile, eval_limit)
    normalized_repeats = normalize_eval_repeats(profile, repeats)
    if profile == "gsm8k_gpqa":
        return config_manager.create_gsm8k_gpqa_task_config(
            model_path=model_path,
            max_tokens=max_tokens,
            limit=normalized_limit,
            repeats=normalized_repeats,
            seed=42 if seed is None else int(seed),
        )
    if profile == "math500_level5_gpqa":
        return config_manager.create_math500_level5_gpqa_task_config(
            model_path=model_path,
            max_tokens=max_tokens,
            limit=normalized_limit,
            repeats=normalized_repeats,
            seed=42 if seed is None else int(seed),
        )
    if profile == "aime_gpqa":
        return config_manager.create_aime_gpqa_task_config(
            model_path=model_path,
            max_tokens=max_tokens,
            limit=normalized_limit,
            repeats=normalized_repeats,
            seed=seed,
        )
    raise ValueError(f"不支持的评测配置: {eval_profile}")


def collect_dataset_metrics(result_obj: Any, dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    if isinstance(result_obj, dict) and "processed_results" in result_obj:
        results_list = result_obj["processed_results"]
    elif isinstance(result_obj, list):
        results_list = result_obj
    else:
        results_list = [result_obj]

    for result in results_list:
        if not isinstance(result, dict):
            continue
        for dataset_name in dataset_names:
            dataset_metrics = result.get(dataset_name)
            if not isinstance(dataset_metrics, dict):
                continue
            metrics.setdefault(dataset_name, {}).update(dataset_metrics)
    return metrics


def generate_model_cache_key(model_path: str, eval_config: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {"model_path": model_path}
    if eval_config is not None:
        payload["eval_config"] = eval_config
    return hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def get_model_cache_path(checkpoint_dir: str, model_key: str, cache_type: str = "original") -> str:
    cache_dir = os.path.join(checkpoint_dir, "evaluation_cache", cache_type)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model_key}.json")


def get_model_cache_candidate_paths(
    checkpoint_dir: str,
    model_key: str,
    cache_type: str = "original",
    eval_config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    candidates: List[str] = []
    if eval_config is not None:
        namespaced_dir = os.path.join(
            checkpoint_dir,
            "evaluation_cache",
            cache_type,
            build_eval_namespace(eval_config),
        )
        os.makedirs(namespaced_dir, exist_ok=True)
        candidates.append(os.path.join(namespaced_dir, f"{model_key}.json"))
    legacy_path = get_model_cache_path(checkpoint_dir, model_key, cache_type)
    if legacy_path not in candidates:
        candidates.append(legacy_path)
    return candidates


def get_compatible_model_cache_candidate_paths(
    checkpoint_dir: str,
    model_path: str,
    cache_type: str = "original",
    eval_config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    candidates: List[str] = []
    primary_key = generate_model_cache_key(model_path, eval_config=eval_config)
    for candidate in get_model_cache_candidate_paths(
        checkpoint_dir,
        primary_key,
        cache_type,
        eval_config=eval_config,
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    legacy_key = generate_model_cache_key(model_path)
    legacy_path = get_model_cache_path(checkpoint_dir, legacy_key, cache_type)
    if legacy_path not in candidates:
        candidates.append(legacy_path)
    return candidates


def load_cached_results(
    cache_path: str,
    expected_eval_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            print(f"Loading cached results from: {cache_path}")
            cached = json.load(handle)
        if expected_eval_config is None:
            return cached
        cached_eval_config = cached.get("eval_config")
        if cached_eval_config is None:
            return None
        if build_eval_namespace(cached_eval_config) != build_eval_namespace(expected_eval_config):
            return None
        return cached
    except Exception as exc:
        print(f"Error loading cache from {cache_path}: {exc}")
        return None


def load_compatible_cached_results(
    cache_path: str,
    expected_eval_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    cached = load_cached_results(cache_path, expected_eval_config=expected_eval_config)
    if cached is not None:
        return cached
    if expected_eval_config is None or not os.path.exists(cache_path):
        return None
    legacy_cached = load_cached_results(cache_path, expected_eval_config=None)
    if not isinstance(legacy_cached, dict) or "metrics" not in legacy_cached:
        return None
    return legacy_cached


def save_results_to_cache(
    cache_path: str,
    results: Dict[str, Any],
    eval_config: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cached_data = dict(results)
        cached_data["cached_at"] = datetime.now().isoformat()
        if eval_config is not None:
            cached_data["eval_config"] = eval_config
            cached_data["cache_namespace"] = build_eval_namespace(eval_config)
            cached_data["cache_schema_version"] = 2
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cached_data, handle, indent=2, ensure_ascii=False)
        print(f"Results cached to: {cache_path}")
    except Exception as exc:
        print(f"Error saving cache to {cache_path}: {exc}")


def run_task_with_server(port: int, task_cfg: Any, served_model_name: Optional[str] = None) -> Dict[str, Any]:
    task_cfg.api_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    if served_model_name:
        task_cfg.model = served_model_name
    print(
        f"在端口 {port} 上执行任务: 模型={task_cfg.model}, "
        f"数据集={task_cfg.datasets}, repeats={task_cfg.repeats}"
    )
    try:
        result = run_task(task_cfg=task_cfg)
        print(f"端口 {port} 上的任务执行完成")
        return result
    except Exception as exc:
        print(f"端口 {port} 上的任务执行出错: {exc}")
        return {"error": str(exc)}


# Backward-compatible aliases kept for internal callers during refactor.
build_eval_cache_config = build_eval_config
build_eval_cache_namespace = build_eval_namespace
get_eval_profile_datasets = list_eval_profile_datasets
