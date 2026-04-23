#!/usr/bin/env python3
"""Minimal quickstart helper for the APBMM release."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = REPO_ROOT / '.mplconfig'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault('PYTHONPATH', str(REPO_ROOT))
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIGDIR))

CORE_MODULES = [
    'src.config_manager',
    'src.evoMI.mi_opt_unified',
    'src.evoMI.optimization_reporting',
    'src.evoMI.task_diff_analyzer',
    'src.ta_methos.model_level_fusion_test',
]


def check_layout() -> None:
    required_paths = [
        REPO_ROOT / 'src' / 'evoMI' / 'mi_opt_unified.py',
        REPO_ROOT / 'src' / 'ta_methos' / 'model_level_fusion_test.py',
        REPO_ROOT / 'evalscope',
        REPO_ROOT / 'mergekit',
        REPO_ROOT / 'models',
        REPO_ROOT / 'checkpoints',
        REPO_ROOT / 'output',
        REPO_ROOT / '.mplconfig',
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required release paths: {', '.join(missing)}")


def check_imports() -> None:
    for module_name in CORE_MODULES:
        importlib.import_module(module_name)
        print(f'[ok] import {module_name}')


def print_demo_commands() -> None:
    main_cmd = [
        'python',
        'src/evoMI/mi_opt_unified.py',
        '--algorithm',
        'sass_prior_bo_wo_update_gap_async',
        '--base-model-path',
        'models/Qwen3-4B-Instruct-2507',
        '--task-model-paths',
        'models/Qwen3-4B-thinking-2507',
        'models/Qwen3-4B-Instruct-2507',
        '--num-blocks',
        '36',
        '--initial-samples',
        '8',
        '--batch-size',
        '4',
        '--max-evaluations',
        '16',
        '--eval-profile',
        'aime_gpqa',
        '--eval-aime-limit',
        '2',
        '--eval-gpqa-limit',
        '8',
        '--available-gpus',
        '0',
    ]
    baseline_cmd = [
        'python',
        'src/ta_methos/model_level_fusion_test.py',
        '--fusion_method',
        'ties',
        '--num_blocks',
        '8',
        '--budget',
        '16',
        '--batch_size',
        '4',
    ]
    print('Recommended AP-BMM smoke-test command:')
    print('PYTHONPATH=. ' + ' '.join(main_cmd))
    print('Recommended model-level baseline smoke-test command:')
    print('PYTHONPATH=. ' + ' '.join(baseline_cmd))


def run_help() -> None:
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault('MPLCONFIGDIR', str(MPLCONFIGDIR))
    command = [sys.executable, str(REPO_ROOT / 'src' / 'evoMI' / 'mi_opt_unified.py'), '--help']
    print('[run] ' + ' '.join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Quickstart checks for the APBMM release.')
    parser.add_argument('--skip-help', action='store_true', help="Skip the final '--help' subprocess check.")
    parser.add_argument('--skip-imports', action='store_true', help='Skip module import checks.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_layout()
    print(f'[ok] repository root: {REPO_ROOT}')
    print(f'[ok] MPLCONFIGDIR: {os.environ.get("MPLCONFIGDIR")}')
    if not any((REPO_ROOT / 'models').iterdir()):
        print('[warn] models/ is empty; add base/task models before a real run')
    if not args.skip_imports:
        check_imports()
    print_demo_commands()
    if not args.skip_help:
        run_help()
    print('[done] quickstart checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
