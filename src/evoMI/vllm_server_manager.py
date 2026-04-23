#!/usr/bin/env python3
"""
VLLM服务器管理器模块

该模块提供了一个管理器类，用于集中管理多个vLLM服务器进程，
支持自动回收资源，并提供添加、查询等功能。
"""

import os
import signal
import socket
import subprocess
import shutil
import sys
import time
import threading
import queue
import uuid
import asyncio
import inspect
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import requests
import json
import numpy as np

    # 内置简单实现，确保即使无法导入start_vllm_server也能工作
def set_gpu_exclusive_mode(gpus: List[int]) -> bool:
    """
    设置GPU为独占进程模式
    
    参数:
        gpus: 需要设置为独占模式的GPU列表
        
    返回:
        bool: 是否成功设置所有GPU为独占模式
    """
    success = True
    for gpu_id in gpus:
        try:
            # 使用nvidia-smi命令设置GPU为独占进程模式（值3）
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "-c", "3"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"成功将GPU {gpu_id} 设置为独占进程模式")
        except subprocess.CalledProcessError as e:
            print(f"设置GPU {gpu_id} 为独占进程模式失败: {e.stderr}")
            success = False
        except Exception as e:
            print(f"设置GPU {gpu_id} 为独占进程模式时发生错误: {str(e)}")
            success = False
    return success

def set_gpu_default_mode(gpus: List[int]) -> bool:
    """
    设置GPU为默认模式
    
    参数:
        gpus: 需要设置为默认模式的GPU列表
        
    返回:
        bool: 是否成功设置所有GPU为默认模式
    """
    success = True
    for gpu_id in gpus:
        try:
            # 使用nvidia-smi命令设置GPU为默认模式
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "-c", "default"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"成功将GPU {gpu_id} 恢复为默认模式")
        except subprocess.CalledProcessError as e:
            print(f"恢复GPU {gpu_id} 为默认模式失败: {e.stderr}")
            success = False
        except Exception as e:
            print(f"恢复GPU {gpu_id} 为默认模式时发生错误: {str(e)}")
            success = False
    return success


class InProcessVllmRuntime:
    _startup_lock = threading.RLock()

    def __init__(
        self,
        args_list: List[str],
        env_overrides: Dict[str, Optional[str]],
        log_file: Optional[str] = None,
    ):
        self.args_list = list(args_list)
        self.env_overrides = dict(env_overrides)
        self.log_file = log_file
        self._pid = os.getpid()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Any = None
        self._engine_client: Any = None
        self._startup_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._exit_code: Optional[int] = None
        self._startup_error: Optional[BaseException] = None

    @property
    def pid(self) -> int:
        return self._pid

    def start(self) -> "InProcessVllmRuntime":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._thread_main, name=f"inproc-vllm-{uuid.uuid4().hex[:8]}")
        self._thread.daemon = True
        self._thread.start()
        return self

    def poll(self) -> Optional[int]:
        if self._thread is not None and self._thread.is_alive() and self._exit_code is None:
            return None
        return 0 if self._exit_code is None else self._exit_code

    def communicate(self, timeout: Optional[float] = None) -> Tuple[str, str]:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        stderr = ""
        if self._startup_error is not None:
            stderr = str(self._startup_error)
        return "", stderr

    def is_ready(self) -> bool:
        return (
            self._startup_event.is_set()
            and self.poll() is None
            and self._engine_client is not None
            and bool(getattr(self._server, "started", False))
        )

    def stop(self, timeout: float = 30) -> bool:
        if self._thread is None:
            self._exit_code = 0
            self._shutdown_event.set()
            return True
        if not self._thread.is_alive():
            self._shutdown_event.set()
            if self._exit_code is None:
                self._exit_code = 0
            return True
        if self._loop is None:
            self._thread.join(timeout=timeout)
            alive = self._thread.is_alive()
            if not alive and self._exit_code is None:
                self._exit_code = 0
            self._shutdown_event.set()
            return not alive
        try:
            future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
            future.result(timeout=timeout)
        except Exception:
            pass
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        if not alive and self._exit_code is None:
            self._exit_code = 0
        self._shutdown_event.set()
        return not alive

    def sleep(self, level: int = 1, timeout: int = 120) -> bool:
        self._run_async(self._sleep_async(level=level), timeout=timeout)
        return True

    def wake_up(self, tags: Optional[List[str]] = None, timeout: int = 120) -> bool:
        self._run_async(self._wake_up_async(tags=tags), timeout=timeout)
        return True

    def reload_weights(self, timeout: int = 300) -> bool:
        self._run_async(self._reload_weights_async(), timeout=timeout)
        return True

    def reset_prefix_cache(self, timeout: int = 120) -> bool:
        self._run_async(self._reset_prefix_cache_async(), timeout=timeout)
        return True

    def is_sleeping(self, timeout: int = 5) -> Optional[bool]:
        return bool(self._run_async(self._is_sleeping_async(), timeout=timeout))

    def supports_direct_collective_rpc(self) -> bool:
        return self._engine_client is not None and hasattr(self._engine_client, "collective_rpc")

    def _run_async(self, coro: Any, timeout: Optional[float] = None) -> Any:
        if self._loop is None:
            raise RuntimeError("vLLM 进程内 runtime 尚未初始化完成")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _sleep_async(self, level: int) -> None:
        engine_client = self._engine_client
        if engine_client is None:
            raise RuntimeError("engine_client 尚未就绪")
        await engine_client.sleep(level=level)

    async def _wake_up_async(self, tags: Optional[List[str]]) -> None:
        engine_client = self._engine_client
        if engine_client is None:
            raise RuntimeError("engine_client 尚未就绪")
        await engine_client.wake_up(tags=tags)

    async def _reload_weights_async(self) -> Any:
        engine_client = self._engine_client
        if engine_client is None:
            raise RuntimeError("engine_client 尚未就绪")
        collective_rpc = getattr(engine_client, "collective_rpc", None)
        if collective_rpc is not None:
            try:
                return await collective_rpc("reload_weights")
            except NotImplementedError:
                pass
        wake_up = getattr(engine_client, "wake_up", None)
        if wake_up is None:
            raise RuntimeError("当前 engine_client 不支持权重热加载")
        await wake_up(tags=["weights"])
        return None

    async def _reset_prefix_cache_async(self) -> None:
        engine_client = self._engine_client
        if engine_client is None:
            raise RuntimeError("engine_client 尚未就绪")
        await engine_client.reset_prefix_cache()

    async def _is_sleeping_async(self) -> bool:
        engine_client = self._engine_client
        if engine_client is None:
            raise RuntimeError("engine_client 尚未就绪")
        return await engine_client.is_sleeping()

    async def _shutdown_async(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        await asyncio.sleep(0)

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_async())
            if self._exit_code is None:
                self._exit_code = 0
        except BaseException as exc:
            self._startup_error = exc
            if self._exit_code is None:
                self._exit_code = 1
            self._startup_event.set()
        finally:
            pending = []
            try:
                pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            except Exception:
                pending = []
            for task in pending:
                task.cancel()
            if pending:
                try:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            self._loop.close()
            self._shutdown_event.set()

    def _apply_env_overrides(self) -> Dict[str, Optional[str]]:
        previous: Dict[str, Optional[str]] = {}
        for key, value in self.env_overrides.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return previous

    def _apply_selected_env_overrides(self, keys: List[str]) -> Dict[str, Optional[str]]:
        previous: Dict[str, Optional[str]] = {}
        for key in keys:
            if key not in self.env_overrides:
                continue
            value = self.env_overrides[key]
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return previous

    def _restore_env_overrides(self, previous: Dict[str, Optional[str]]):
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    async def _main_async(self):
        persistent_keys = ["VLLM_SERVER_DEV_MODE", "VLLM_USE_V1"]
        transient_keys = [
            key for key in self.env_overrides.keys()
            if key not in persistent_keys
        ]
        persistent_previous: Dict[str, Optional[str]] = {}
        engine_client_cm = None
        engine_client = None
        try:
            with self.__class__._startup_lock:
                persistent_previous = self._apply_selected_env_overrides(persistent_keys)
                import uvicorn
                from vllm.config import VllmConfig
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.entrypoints.openai.api_server import (
                    build_app,
                    build_async_engine_client_from_engine_args,
                    init_app_state,
                )
                from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
                from vllm.usage.usage_lib import UsageContext
                from vllm.utils import FlexibleArgumentParser

                parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
                parser = make_arg_parser(parser)
                args = parser.parse_args(self.args_list)
                validate_parsed_serve_args(args)
                engine_args = AsyncEngineArgs.from_cli_args(args)
                vllm_config: Optional[VllmConfig] = None
                engine_client_cm = build_async_engine_client_from_engine_args(
                    engine_args,
                    usage_context=UsageContext.OPENAI_API_SERVER,
                    disable_frontend_multiprocessing=True,
                    client_config=None,
                )
                transient_previous = self._apply_selected_env_overrides(transient_keys)
                try:
                    vllm_config = engine_args.create_engine_config(
                        usage_context=UsageContext.OPENAI_API_SERVER
                    )
                    engine_client = await engine_client_cm.__aenter__()
                finally:
                    self._restore_env_overrides(transient_previous)
            self._engine_client = engine_client
            app = build_app(args)
            await init_app_state(engine_client, vllm_config, app.state, args)
            config = uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                access_log=not args.disable_uvicorn_access_log,
                timeout_keep_alive=getattr(args, "timeout_keep_alive", 5),
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs,
            )
            server = uvicorn.Server(config)
            server.install_signal_handlers = lambda: None
            self._server = server
            self._startup_event.set()
            await server.serve()
        finally:
            self._engine_client = None
            self._server = None
            if engine_client is not None and engine_client_cm is not None:
                await engine_client_cm.__aexit__(None, None, None)
            self._restore_env_overrides(persistent_previous)

def start_vllm_server(
    model_path: str = "models/Qwen3-4B-Instruct-2507",
    port: int = 8000,
    served_model_name: str = "Qwen3-4B-Instruct",
    tensor_parallel_size: int = 8,
    max_model_len: int = 48000,
    worker_multiproc_method: str = "spawn",
    use_modelscope: bool = True,
    gpus: Optional[List[int]] = None,
    additional_args: Optional[List[str]] = None,
    blocking: bool = False,
    log_file: Optional[str] = None,
    exclusive_gpu_mode: bool = False
) -> Any:
    """start_vllm_server函数的实现，包含CUDA环境变量配置和编译器标志设置"""
    requested_args = list(additional_args) if additional_args else []
    model_path_abs = os.path.abspath(model_path)
    if os.path.lexists(model_path_abs) and not os.path.exists(model_path_abs):
        raise FileNotFoundError(f"vLLM 启动前检测到模型别名存在但目标缺失: {model_path_abs}")
    use_modelscope_effective = bool(use_modelscope)
    # Local model directories should be loaded directly instead of going through ModelScope repo resolution.
    if os.path.exists(model_path_abs):
        use_modelscope_effective = False
    env_overrides: Dict[str, Optional[str]] = {
        "VLLM_WORKER_MULTIPROC_METHOD": worker_multiproc_method,
        "VLLM_USE_MODELSCOPE": "True" if use_modelscope_effective else "False",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_USE_V1": "0",
    }
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "--enable-sleep-mode" in requested_args:
        if "expandable_segments:True" in alloc_conf:
            filtered_parts = [
                part.strip()
                for part in alloc_conf.split(",")
                if part.strip() and part.strip() != "expandable_segments:True"
            ]
            if filtered_parts:
                env_overrides["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(filtered_parts)
            else:
                env_overrides["PYTORCH_CUDA_ALLOC_CONF"] = None
    elif alloc_conf:
        env_overrides["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

    if gpus is not None:
        gpu_list = gpus
        env_overrides["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    else:
        gpu_list = []

    if exclusive_gpu_mode:
        set_gpu_exclusive_mode(gpu_list)

    args_list = [
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(0.85),
        "--enable-prefix-caching",
    ]

    if "--disable-frontend-multiprocessing" not in requested_args:
        args_list.append("--disable-frontend-multiprocessing")

    if requested_args:
        args_list.extend(requested_args)

    stdout_handle = None
    stderr_handle = None
    try:
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            stdout_handle = open(log_file, "a", encoding="utf-8")
            stderr_handle = subprocess.STDOUT

        process_env = os.environ.copy()
        for key, value in env_overrides.items():
            if value is None:
                process_env.pop(key, None)
            else:
                process_env[key] = value
        process_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            *args_list,
        ]

        process = subprocess.Popen(
            command,
            env=process_env,
            stdout=stdout_handle if stdout_handle is not None else subprocess.DEVNULL,
            stderr=stderr_handle if stdout_handle is not None else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        if stdout_handle is not None:
            stdout_handle.close()
        return process
    except Exception:
        if stdout_handle is not None:
            stdout_handle.close()
        raise


@dataclass
class Task:
    """任务数据类"""
    task_id: str
    model_path: str
    params_dict: Dict[str, Any]
    func_handle: Callable
    gpu_count: int
    estimated_tokens: float = -1.0
    result: Any = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: float = 0.0  # 任务开始时间
    end_time: float = 0.0
    assigned_gpus: List[int] = field(default_factory=list)
    server_port: Optional[int] = None
    cleanup_model_dirs: List[str] = field(default_factory=list)
    cleanup_server_after_completion: bool = False


@dataclass
class VllmServerInfo:
    """存储vLLM服务器信息的数据类"""
    process: Any
    model_path: str
    launch_model_path: Optional[str]
    port: int
    served_model_name: str
    tensor_parallel_size: int
    max_model_len: int
    worker_multiproc_method: str
    use_modelscope: bool
    gpus: Optional[List[int]]
    additional_args: Optional[List[str]]
    log_file: Optional[str] = None
    current_task: Optional[Task] = None
    start_time: float = field(default_factory=time.time)
    exclusive_gpu_mode: bool = False
    pool_slot_key: Tuple[int, ...] = field(default_factory=tuple)
    available_routes: Optional[List[str]] = None
    
    def is_running(self) -> bool:
        """检查进程是否仍在运行"""
        return self.process.poll() is None
    
    def is_ready(self, timeout: int = 2) -> bool:
        """
        检查vLLM服务器是否已准备好接受请求
        
        参数:
            timeout: API调用超时时间（秒）
            
        返回:
            bool: 如果服务器已准备好则返回True，否则返回False
        """
        try:
            if not self.is_running():
                return False
            response = requests.get(
                f"http://localhost:{self.port}/v1/models",
                timeout=timeout
            )
            if response.status_code != 200:
                return False
            payload = response.json()
            data = payload.get("data", [])
            if not isinstance(data, list) or len(data) == 0:
                return False
            model_ids = {
                str(item.get("id"))
                for item in data
                if isinstance(item, dict) and item.get("id") is not None
            }
            if not model_ids:
                return False
            if self.served_model_name:
                return str(self.served_model_name) in model_ids
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False
        except Exception:
            return False
            
    def is_idle(self, timeout: int = 5) -> Optional[bool]:
        """
        检查vLLM服务器是否空闲（没有正在处理的请求）
        
        参数:
            timeout: HTTP请求超时时间（秒）
            
        返回:
            Optional[bool]: 如果服务器空闲返回True，如果有活跃任务返回False，
                          如果无法连接到服务器返回None
        """
        if not self.is_running():
            return None
        
        try:
            # vLLM的API接口通常提供了stats端点来查看服务器状态
            url = f"http://localhost:{self.port}/v1/internal/stats"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                stats = response.json()
                # 检查是否有活跃的生成任务
                # 根据vLLM的实现，活跃任务数可能在不同字段中
                active_requests = stats.get('active_requests', 0)
                return active_requests == 0
            else:
                # 如果没有stats端点，尝试使用健康检查端点
                health_url = f"http://localhost:{self.port}/health"
                health_response = requests.get(health_url, timeout=timeout)
                # 注意：健康检查只能确认服务器运行，但不能直接判断是否空闲
                # 这里作为后备方案，返回True表示假设服务器空闲
                # 实际使用中，最好使用stats端点
                return health_response.status_code == 200
        except Exception as e:
            print(f"检查服务器 {self.port} 空闲状态时出错: {e}")
            return None
    
    def get_pid(self) -> int:
        """获取进程ID"""
        return int(getattr(self.process, "pid", -1))

    def supports_direct_collective_rpc(self) -> bool:
        return bool(getattr(self.process, "supports_direct_collective_rpc", lambda: False)())

    def refresh_available_routes(self, timeout: int = 5) -> List[str]:
        if not self.is_running():
            self.available_routes = []
            return []
        response = requests.get(
            f"http://localhost:{self.port}/openapi.json",
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        routes = sorted(str(path) for path in payload.get("paths", {}).keys())
        self.available_routes = routes
        return routes

    def supports_endpoint(self, path: str, timeout: int = 5) -> bool:
        routes = self.available_routes
        if routes is None:
            try:
                routes = self.refresh_available_routes(timeout=timeout)
            except Exception:
                return False
        return str(path) in set(routes)

    def _post_dev_endpoint(
        self,
        path: str,
        timeout: int = 30,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        response = requests.post(
            f"http://localhost:{self.port}{path}",
            params=params,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response

    def sleep(self, level: int = 2, timeout: int = 120) -> bool:
        if not self.is_running():
            return False
        if hasattr(self.process, "sleep"):
            return bool(self.process.sleep(level=level, timeout=timeout))
        self._post_dev_endpoint("/sleep", timeout=timeout, params={"level": int(level)})
        return True

    def wake_up(self, tags: Optional[List[str]] = None, timeout: int = 120) -> bool:
        if not self.is_running():
            return False
        if hasattr(self.process, "wake_up"):
            return bool(self.process.wake_up(tags=tags, timeout=timeout))
        params = None
        if tags:
            params = [("tags", str(tag)) for tag in tags]
        self._post_dev_endpoint("/wake_up", timeout=timeout, params=params)
        return True

    def reload_weights(self, timeout: int = 300) -> bool:
        if not self.is_running():
            return False
        if hasattr(self.process, "reload_weights"):
            return bool(self.process.reload_weights(timeout=timeout))
        try:
            self._post_dev_endpoint(
                "/collective_rpc",
                timeout=timeout,
                payload={"method": "reload_weights"},
            )
        except Exception:
            self.wake_up(tags=["weights"], timeout=timeout)
        return True

    def reset_prefix_cache(self, timeout: int = 120) -> bool:
        if not self.is_running():
            return False
        if hasattr(self.process, "reset_prefix_cache"):
            return bool(self.process.reset_prefix_cache(timeout=timeout))
        self._post_dev_endpoint("/reset_prefix_cache", timeout=timeout)
        return True

    def is_sleeping(self, timeout: int = 5) -> Optional[bool]:
        if not self.is_running():
            return None
        if hasattr(self.process, "is_sleeping"):
            return self.process.is_sleeping(timeout=timeout)
        try:
            response = requests.get(
                f"http://localhost:{self.port}/is_sleeping",
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:
                text = response.text.strip().lower()
                if text in ("true", "1"):
                    return True
                if text in ("false", "0"):
                    return False
                return None
            if isinstance(data, bool):
                return data
            if isinstance(data, dict):
                for key in ("is_sleeping", "sleeping", "value"):
                    if key in data:
                        return bool(data[key])
            return None
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """将服务器信息转换为字典格式"""
        return {
            "pid": self.get_pid(),
            "model_path": self.model_path,
            "launch_model_path": self.launch_model_path,
            "port": self.port,
            "served_model_name": self.served_model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "worker_multiproc_method": self.worker_multiproc_method,
            "use_modelscope": self.use_modelscope,
            "gpus": self.gpus,
            "additional_args": self.additional_args,
            "log_file": self.log_file,
            "start_time": self.start_time,
            "is_running": self.is_running(),
            "exclusive_gpu_mode": self.exclusive_gpu_mode,
            "pool_slot_key": list(self.pool_slot_key),
        }


class VllmServerManager:
    """
    vLLM服务器管理器
    
    功能：
    1. 管理多个vLLM服务器进程
    2. 自动清理资源（进程）
    3. 提供添加、查询服务器的接口
    4. 支持上下文管理器模式
    5. 支持任务队列和GPU资源管理
    """
    _global_resource_lock = threading.RLock()
    _global_used_gpus: Dict[str, List[int]] = {}
    _global_used_ports: set = set()
    _global_progress_log_lock = threading.RLock()
    _singleton_lock = threading.RLock()
    _singleton_instance: Optional["VllmServerManager"] = None
    
    @classmethod
    def get_shared_manager(cls, available_gpus: Optional[List[int]] = None, max_model_len: int = 48000) -> "VllmServerManager":
        with cls._singleton_lock:
            if cls._singleton_instance is not None and (not cls._singleton_instance._destroyed):
                return cls._singleton_instance
            return cls(available_gpus=available_gpus, max_model_len=max_model_len)

    def __init__(self, available_gpus: Optional[List[int]] = None, max_model_len: int = 48000, allow_multiple_instances: bool = False):
        """初始化管理器"""
        with self.__class__._singleton_lock:
            active_instance = self.__class__._singleton_instance
            if (not allow_multiple_instances) and active_instance is not None and (not active_instance._destroyed):
                raise RuntimeError("当前进程只允许一个VllmServerManager实例，请复用共享管理器")
            self.__class__._singleton_instance = self
        self._instance_id = uuid.uuid4().hex
        # 存储服务器信息的字典，使用端口作为键
        self._servers: Dict[int, VllmServerInfo] = {}
        # 线程锁，确保线程安全
        self._lock = threading.RLock()
        # 标记管理器是否已被销毁
        self._destroyed = False
        # 可用GPU列表
        self._available_gpus = available_gpus if available_gpus is not None else list(range(8))  # 默认8个GPU
        # 正在使用的GPU映射
        self._used_gpus: Dict[int, List[int]] = {}
        # 任务队列
        self._task_queue = queue.Queue()
        # 任务结果字典
        self._task_results: Dict[str, Any] = {}
        # 任务线程
        self._task_thread: Optional[threading.Thread] = None
        # 任务处理是否运行中
        self._tasks_running = False
        # 结果事件，用于通知任务完成
        self._result_events: Dict[str, threading.Event] = {}
        # 所有任务列表
        self._all_tasks: Dict[str, Task] = {}
        # 进度日志文件路径
        self._progress_log_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output'), 'vllm_servers_logs.log')
        self._gpu_stats_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output'), 'vllm_gpu_usage_stats.json')
        # 进度更新线程
        self._progress_thread: Optional[threading.Thread] = None
        # 进度更新是否运行中
        self._progress_running = False
        # 最大模型长度
        self._max_model_len = max_model_len
        self._series_start_time = 0.0
        self._series_end_time = 0.0
        self._gpu_busy_time_sec: Dict[int, float] = {gpu: 0.0 for gpu in self._available_gpus}
        self._gpu_task_count: Dict[int, int] = {gpu: 0 for gpu in self._available_gpus}
        self._gpu_task_records: List[Dict[str, Any]] = []
        self._task_runtime_records: List[Dict[str, Any]] = []
        self._task_completion_cursor: int = 0
        self._gpu_wait_log_interval_sec = 30.0
        self._task_wait_retry_count: Dict[str, int] = {}
        self._task_wait_first_ts: Dict[str, float] = {}
        self._task_wait_last_log_ts: Dict[str, float] = {}
        self._gpu_release_ts: Dict[int, float] = {gpu: 0.0 for gpu in self._available_gpus}
        self._gpu_reuse_cooldown_sec = 20.0
        self._gpu_round_robin_cursor = 0
        self._gpu_wait_retry_sleep_sec = 0.2
        self._dispatch_wakeup_event = threading.Event()
        self._reserved_ports: set = set()
        self._port_search_start = 8000 + (min(self._available_gpus) * 100 if self._available_gpus else 0)
        self._last_progress_snapshot = ""
        self._server_pool_ports: Dict[Tuple[int, ...], int] = {}
        self._persistent_pool_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'output',
            'vllm_server_pool',
        )
        self._sleep_idle_level = 1
        
        self._clear_progress_log()
        
        print(f"vLLM服务器管理器已初始化，可用GPU: {self._available_gpus}, 最大模型长度: {self._max_model_len}")
    
    def __del__(self):
        """析构函数，确保所有进程被清理"""
        if hasattr(self, "_lock"):
            self.destroy()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保清理资源"""
        self.destroy()
        return False  # 不抑制异常

    def _global_gpu_key(self, port: int) -> str:
        return f"{self._instance_id}:{int(port)}"

    def _is_port_available(self, port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, int(port)))
            except OSError:
                return False
        return True

    def _reserve_port(self, start_port: int = 8000) -> int:
        with self.__class__._global_resource_lock:
            port = int(start_port)
            while True:
                port_reserved = port in self.__class__._global_used_ports
                port_available = self._is_port_available(port)
                if (not port_reserved) and port_available:
                    break
                if (not port_reserved) and (not port_available):
                    print(f"端口 {port} 当前已被系统占用，自动尝试下一个端口")
                port += 1
            self.__class__._global_used_ports.add(port)
            self._reserved_ports.add(port)
            return port

    def _release_port(self, port: int):
        port = int(port)
        if port not in self._reserved_ports:
            return
        with self.__class__._global_resource_lock:
            self._reserved_ports.discard(port)
            self.__class__._global_used_ports.discard(port)

    def _rebind_allocated_gpus(self, old_port: int, new_port: int):
        with self.__class__._global_resource_lock:
            if old_port not in self._used_gpus:
                return
            gpus = self._used_gpus.pop(old_port)
            self._used_gpus[new_port] = gpus
            old_key = self._global_gpu_key(old_port)
            new_key = self._global_gpu_key(new_port)
            self.__class__._global_used_gpus.pop(old_key, None)
            self.__class__._global_used_gpus[new_key] = list(gpus)

    def _bind_allocated_gpus(self, port: int, gpus: List[int]):
        normalized_gpus = [int(gpu) for gpu in gpus]
        with self.__class__._global_resource_lock:
            self._used_gpus[int(port)] = normalized_gpus
            global_key = self._global_gpu_key(port)
            self.__class__._global_used_gpus[global_key] = list(normalized_gpus)

    def _normalize_gpu_key(self, gpus: Optional[List[int]]) -> Tuple[int, ...]:
        if not gpus:
            return tuple()
        return tuple(sorted(int(gpu) for gpu in gpus))

    def _ensure_sleep_mode_args(self, additional_args: Optional[List[str]]) -> List[str]:
        args = list(additional_args) if additional_args else []
        if "--enable-sleep-mode" not in args and "--no-enable-sleep-mode" not in args:
            args.append("--enable-sleep-mode")
        return args

    def _build_pool_model_alias_path(self, slot_key: Tuple[int, ...]) -> str:
        slot_name = "slot_" + "_".join(str(gpu) for gpu in slot_key) if slot_key else "slot_cpu"
        return os.path.join(self._persistent_pool_root, slot_name, "current_model")

    def _set_model_alias(self, alias_path: str, model_path: str) -> str:
        target_path = os.path.abspath(model_path)
        os.makedirs(os.path.dirname(alias_path), exist_ok=True)
        temp_alias = f"{alias_path}.tmp.{uuid.uuid4().hex}"
        if os.path.lexists(temp_alias):
            os.unlink(temp_alias)
        os.symlink(target_path, temp_alias)
        os.replace(temp_alias, alias_path)
        return alias_path

    def _detach_server_from_pool(self, port: int):
        stale_keys = [
            key for key, value in self._server_pool_ports.items()
            if int(value) == int(port)
        ]
        for key in stale_keys:
            self._server_pool_ports.pop(key, None)

    def _wait_server_ready(self, server_info: VllmServerInfo, wait_seconds: int) -> bool:
        print(f"等待服务器启动并准备就绪，端口: {server_info.port}...")
        start_time = time.time()
        ready = False
        while time.time() - start_time < wait_seconds:
            if not server_info.is_running():
                print(f"警告: 服务器进程已退出，端口: {server_info.port}")
                stdout, stderr = server_info.process.communicate(timeout=1)
                if stderr:
                    print(f"错误输出:\n{stderr[:500]}..." if len(stderr) > 500 else f"错误输出:\n{stderr}")
                break
            if server_info.is_ready():
                ready = True
                print(f"服务器已准备就绪，端口: {server_info.port}")
                break
            time.sleep(0.5)
        if not ready:
            if server_info.is_running():
                print(f"警告: 在{wait_seconds}秒内服务器未完全就绪，但进程仍在运行，端口: {server_info.port}")
            else:
                print(f"警告: 服务器启动失败，端口: {server_info.port}")
        return ready

    def _sleep_server(self, server_info: VllmServerInfo, level: Optional[int] = None) -> bool:
        if not server_info.is_running():
            return False
        sleep_level = int(self._sleep_idle_level if level is None else level)
        server_info.sleep(level=sleep_level)
        return True

    def _restart_pooled_server(
        self,
        server_info: VllmServerInfo,
        model_path: str,
        served_model_name: Optional[str] = None,
        wait_seconds: int = 60000,
    ) -> VllmServerInfo:
        alias_path = server_info.launch_model_path or self._build_pool_model_alias_path(server_info.pool_slot_key)
        self._set_model_alias(alias_path, model_path)
        port = int(server_info.port)
        slot_key = tuple(server_info.pool_slot_key)
        current_task = server_info.current_task
        additional_args = self._ensure_sleep_mode_args(server_info.additional_args)
        gpus = list(server_info.gpus) if server_info.gpus else None
        restarted_served_model_name = served_model_name if served_model_name is not None else model_path
        if not self.stop_server(port, wait_for_idle=False, preserve_pool_state=True):
            raise RuntimeError(f"停止端口 {port} 的常驻服务器失败")
        try:
            restarted_server = self.add_server(
                model_path=model_path,
                launch_model_path=alias_path,
                port=port,
                served_model_name=restarted_served_model_name,
                tensor_parallel_size=server_info.tensor_parallel_size,
                max_model_len=server_info.max_model_len,
                worker_multiproc_method=server_info.worker_multiproc_method,
                use_modelscope=server_info.use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                wait_for_start=True,
                wait_seconds=wait_seconds,
                log_file=server_info.log_file,
                exclusive_gpu_mode=server_info.exclusive_gpu_mode,
            )
        except Exception:
            self._detach_server_from_pool(port)
            self._release_gpus(port)
            self._release_port(port)
            if server_info.exclusive_gpu_mode and gpus:
                set_gpu_default_mode(gpus)
            raise
        restarted_server.pool_slot_key = slot_key
        restarted_server.current_task = current_task
        self._server_pool_ports[slot_key] = port
        return restarted_server

    def _prepare_server_for_model(
        self,
        server_info: VllmServerInfo,
        model_path: str,
        served_model_name: Optional[str] = None,
        wait_seconds: int = 60000,
    ) -> VllmServerInfo:
        alias_path = server_info.launch_model_path
        if not alias_path:
            raise RuntimeError(f"端口 {server_info.port} 的服务器未配置可重载模型路径")
        if not server_info.is_running():
            raise RuntimeError(f"端口 {server_info.port} 的服务器已停止，无法热加载")
        next_served_model_name = (
            served_model_name if served_model_name is not None else server_info.served_model_name
        )
        if os.path.abspath(server_info.model_path) == os.path.abspath(model_path):
            print(f"复用端口 {server_info.port} 的常驻vLLM服务器，直接唤醒同模型: {model_path}")
            if server_info.is_sleeping() is True:
                server_info.wake_up()
            server_info.model_path = model_path
            server_info.served_model_name = next_served_model_name
            self._wait_server_ready(server_info, wait_seconds=wait_seconds)
            return server_info
        print(f"复用端口 {server_info.port} 的常驻vLLM服务器，热加载模型: {model_path}")
        try:
            self._sleep_server(server_info, level=2)
            self._set_model_alias(alias_path, model_path)
            server_info.reload_weights()
        except Exception as exc:
            print(f"端口 {server_info.port} 的热加载失败，改为同端口重启: {exc}")
            return self._restart_pooled_server(
                server_info,
                model_path=model_path,
                served_model_name=next_served_model_name,
                wait_seconds=wait_seconds,
            )
        try:
            server_info.reset_prefix_cache()
        except Exception as exc:
            print(f"重置端口 {server_info.port} 的 prefix cache 失败，继续执行: {exc}")
        server_info.wake_up(tags=["kv_cache"])
        server_info.model_path = model_path
        server_info.served_model_name = next_served_model_name
        self._wait_server_ready(server_info, wait_seconds=wait_seconds)
        return server_info

    def _acquire_or_create_pooled_server(
        self,
        task: Task,
        gpus: List[int],
        temp_port: int,
    ) -> VllmServerInfo:
        slot_key = self._normalize_gpu_key(gpus)
        pooled_port = self._server_pool_ports.get(slot_key)
        if pooled_port is not None:
            pooled_server = self._servers.get(pooled_port)
            if pooled_server is not None and pooled_server.is_running():
                self._rebind_allocated_gpus(temp_port, pooled_server.port)
                pooled_server.current_task = task
                try:
                    return self._prepare_server_for_model(
                        pooled_server,
                        model_path=task.model_path,
                        served_model_name=pooled_server.served_model_name,
                        wait_seconds=60000,
                    )
                except Exception as exc:
                    print(f"复用端口 {pooled_server.port} 的常驻服务器失败，将回退为冷启动: {exc}")
                    try:
                        stopped = self.stop_server(pooled_server.port, wait_for_idle=False)
                        if not stopped:
                            raise RuntimeError(f"停止端口 {pooled_server.port} 的失效常驻服务器失败")
                        self._bind_allocated_gpus(temp_port, gpus)
                    except Exception as stop_exc:
                        print(f"停止端口 {pooled_server.port} 的失效常驻服务器失败: {stop_exc}")
                        raise
            self._detach_server_from_pool(pooled_port)

        port = self._reserve_port(self._port_search_start)
        self._rebind_allocated_gpus(temp_port, port)
        alias_path = self._set_model_alias(
            self._build_pool_model_alias_path(slot_key),
            task.model_path,
        )
        print(f"为任务 {task.task_id} 冷启动常驻vLLM服务器，端口: {port}，模型: {task.model_path}，GPU: {gpus}")
        server_info = self.add_server(
            model_path=task.model_path,
            launch_model_path=alias_path,
            port=port,
            served_model_name=alias_path,
            tensor_parallel_size=task.gpu_count,
            max_model_len=self._max_model_len,
            gpus=gpus,
            additional_args=self._ensure_sleep_mode_args(None),
            wait_for_start=True,
            wait_seconds=60000,
        )
        server_info.pool_slot_key = slot_key
        server_info.current_task = task
        self._server_pool_ports[slot_key] = port
        return server_info
    
    def add_server(
        self,
        model_path: str = "models/Qwen3-4B-Instruct-2507",
        launch_model_path: Optional[str] = None,
        port: int = 8000,
        served_model_name: str = "Qwen3-4B-Instruct",
        tensor_parallel_size: int = 8,
        max_model_len: Optional[int] = None,
        worker_multiproc_method: str = "spawn",
        use_modelscope: bool = True,
        gpus: Optional[List[int]] = None,
        additional_args: Optional[List[str]] = None,
        wait_for_start: bool = False,
        wait_seconds: int = 5,
        log_file: Optional[str] = None,
        exclusive_gpu_mode: bool = False
    ) -> VllmServerInfo:
        """
        添加并启动一个新的vLLM服务器
        
        参数:
            model_path: 模型路径
            port: 服务器端口（必须唯一）
            served_model_name: 服务模型名称
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大模型长度
            worker_multiproc_method: 工作进程多处理方法
            use_modelscope: 是否使用ModelScope
            gpus: 可见的GPU列表
            additional_args: 额外的命令行参数列表
            wait_for_start: 是否等待服务器启动（检查进程是否仍在运行）
            wait_seconds: 等待服务器启动的秒数
            
        返回:
            VllmServerInfo: 服务器信息对象
            
        异常:
            ValueError: 如果端口已被使用
        """
        with self._lock:
            # 检查管理器是否已被销毁
            if self._destroyed:
                raise RuntimeError("管理器已被销毁，无法添加新服务器")
            
            # 检查端口是否已被使用
            if port in self._servers:
                existing_server = self._servers[port]
                if existing_server.is_running():
                    raise ValueError(f"端口 {port} 已被使用")
                else:
                    print(f"注意: 端口 {port} 上的服务器已停止，将替换为新服务器")
            
            # 启动服务器
            print(f"正在启动vLLM服务器，端口: {port}，模型: {model_path}")
            # 如果没有指定日志文件，生成默认日志文件名
            if log_file is None:
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output/vllm_manager_logs')
                model_name = os.path.basename(model_path)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"vllm_server_{model_name}_{port}_{timestamp}.log")
                print(f"未指定日志文件，将使用默认日志文件: {log_file}")
            
            # 如果没有指定max_model_len，使用实例变量中的值
            if max_model_len is None:
                max_model_len = self._max_model_len
                print(f"未指定max_model_len，将使用管理器默认值: {max_model_len}")
            effective_model_path = launch_model_path or model_path
            additional_args = self._ensure_sleep_mode_args(additional_args)
            
            process = start_vllm_server(
                model_path=effective_model_path,
                port=port,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                worker_multiproc_method=worker_multiproc_method,
                use_modelscope=use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                blocking=False,
                log_file=log_file,
                exclusive_gpu_mode=exclusive_gpu_mode,
            )
            
            # 创建服务器信息对象
            server_info = VllmServerInfo(
                process=process,
                model_path=model_path,
                launch_model_path=launch_model_path,
                port=port,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                worker_multiproc_method=worker_multiproc_method,
                use_modelscope=use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                log_file=log_file,
                exclusive_gpu_mode=exclusive_gpu_mode
            )
            
            # 存储服务器信息
            self._servers[port] = server_info
            
            # 等待服务器启动
            if wait_for_start:
                self._wait_server_ready(server_info, wait_seconds=wait_seconds)
            
            print(f"vLLM服务器已添加，端口: {port}，进程ID: {process.pid}")
            return server_info
    
    def get_server_info(self, port: int) -> Optional[VllmServerInfo]:
        """
        获取指定端口的服务器信息
        
        参数:
            port: 服务器端口
            
        返回:
            Optional[VllmServerInfo]: 服务器信息对象，如果不存在则返回None
        """
        with self._lock:
            return self._servers.get(port)
    
    def list_servers(self) -> List[VllmServerInfo]:
        """
        列出所有服务器信息
        
        返回:
            List[VllmServerInfo]: 服务器信息列表
        """
        with self._lock:
            return list(self._servers.values())
    
    def list_active_servers(self) -> List[VllmServerInfo]:
        """
        列出所有活跃（正在运行）的服务器
        
        返回:
            List[VllmServerInfo]: 活跃服务器信息列表
        """
        with self._lock:
            # 先清理已停止的服务器
            self._cleanup_stopped_servers()
            # 返回仍在运行的服务器
            return [server for server in self._servers.values() if server.is_running()]
    
    def get_available_processes(self) -> Dict[int, Dict[str, Any]]:
        """
        获取当前可用的vLLM进程信息
        
        返回:
            Dict[int, Dict[str, Any]]: 以端口为键，服务器信息字典为值的映射
        """
        with self._lock:
            # 清理已停止的服务器
            self._cleanup_stopped_servers()
            # 返回所有活跃服务器的信息字典
            return {
                port: server.to_dict() 
                for port, server in self._servers.items() 
                if server.is_running()
            }
    
    def stop_server(
        self,
        port: int,
        wait_for_idle: bool = True,
        idle_timeout: int = 300,
        preserve_pool_state: bool = False,
    ) -> bool:
        """
        停止指定端口的服务器
        
        参数:
            port: 服务器端口
            wait_for_idle: 是否等待服务器空闲后再停止
            idle_timeout: 等待服务器空闲的最大时间（秒）
            
        返回:
            bool: 如果服务器被成功停止则返回True，否则返回False
        """
        with self._lock:
            if port not in self._servers:
                print(f"警告: 端口 {port} 上没有找到服务器")
                return False
            
            server_info = self._servers[port]
            # 保存GPU信息和独占模式设置
            gpus_to_release = server_info.gpus
            is_exclusive_mode = server_info.exclusive_gpu_mode
            
            if not server_info.is_running():
                print(f"注意: 端口 {port} 上的服务器已经停止")
                del self._servers[port]
                if not preserve_pool_state:
                    self._detach_server_from_pool(port)
                    self._release_gpus(port)
                    self._release_port(port)
                if is_exclusive_mode and gpus_to_release and not preserve_pool_state:
                    set_gpu_default_mode(gpus_to_release)
                return True
            
            try:
                print(f"正在停止服务器，端口: {port}，进程ID: {server_info.get_pid()}")
                # 如果需要等待服务器空闲
                if wait_for_idle:
                    print(f"等待服务器 {port} 空闲中...")
                    start_wait_time = time.time()
                    while time.time() - start_wait_time < idle_timeout:
                        # 检查服务器是否空闲
                        is_idle_result = server_info.is_idle()
                        if is_idle_result is True:
                            print(f"服务器 {port} 已空闲，准备停止")
                            break
                        elif is_idle_result is False:
                            # 服务器正在处理任务，继续等待
                            print(f"服务器 {port} 正在处理任务，等待中...")
                            time.sleep(5)  # 每5秒检查一次
                        else:
                            # 无法确定服务器状态，可能已经停止或有其他问题
                            print(f"无法确定服务器 {port} 状态，继续停止流程")
                            break
                    
                    # 检查是否超时
                    if time.time() - start_wait_time >= idle_timeout:
                        print(f"等待服务器 {port} 空闲超时，强制停止")
                
                if hasattr(server_info.process, "stop"):
                    if not server_info.process.stop(timeout=30):
                        raise RuntimeError(f"进程内 vLLM runtime 停止超时，端口: {port}")
                else:
                    os.kill(server_info.get_pid(), signal.SIGTERM)
                    wait_timeout = 5
                    start_time = time.time()
                    while time.time() - start_time < wait_timeout:
                        if not server_info.is_running():
                            break
                        time.sleep(0.1)
                    if server_info.is_running():
                        print(f"强制终止服务器，端口: {port}")
                        os.kill(server_info.get_pid(), signal.SIGKILL)
                
                # 从记录中移除
                del self._servers[port]
                if not preserve_pool_state:
                    self._detach_server_from_pool(port)
                    self._release_gpus(port)
                    self._release_port(port)
                if is_exclusive_mode and gpus_to_release and not preserve_pool_state:
                    set_gpu_default_mode(gpus_to_release)
                    
                print(f"服务器已停止，端口: {port}")
                return True
            except Exception as e:
                print(f"停止服务器失败，端口: {port}，错误: {e}")
                # 即使出错，也尝试恢复GPU模式
                if is_exclusive_mode and gpus_to_release:
                    set_gpu_default_mode(gpus_to_release)
                return False
    
    def stop_all_servers(self) -> int:
        """
        停止所有服务器
        
        返回:
            int: 成功停止的服务器数量
        """
        with self._lock:
            ports = list(self._servers.keys())
            success_count = 0
            
            for port in ports:
                if self.stop_server(port):
                    success_count += 1
            
            print(f"已停止 {success_count}/{len(ports)} 个服务器")
            return success_count
    
    def _cleanup_stopped_servers(self):
        """
        清理已停止的服务器记录
        """
        stopped_ports = [
            port for port, server in self._servers.items()
            if not server.is_running()
        ]

        for port in stopped_ports:
            server_info = self._servers[port]
            print(f"清理已停止的服务器记录，端口: {port}")
            del self._servers[port]
            self._detach_server_from_pool(port)
            self._release_port(port)
            self._release_gpus(port)
            if server_info.exclusive_gpu_mode and server_info.gpus:
                set_gpu_default_mode(server_info.gpus)
    
    def update_server(
        self,
        port: int,
        model_path: str,
        served_model_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        worker_multiproc_method: Optional[str] = None,
        use_modelscope: Optional[bool] = None,
        gpus: Optional[List[int]] = None,
        additional_args: Optional[List[str]] = None,
        wait_for_start: bool = True,
        wait_seconds: int = 10,
        wait_for_idle: bool = True,
        idle_timeout: int = 300,
        log_file: Optional[str] = None,
        exclusive_gpu_mode: bool = False
    ) -> VllmServerInfo:
        """
        更新指定端口的vLLM服务器模型
        
        参数:
            port: 服务器端口
            model_path: 新的模型路径
            served_model_name: 新的服务模型名称（如果为None则使用model_path）
            tensor_parallel_size: 新的张量并行大小（如果为None则使用原有值）
            max_model_len: 新的最大模型长度（如果为None则使用原有值）
            worker_multiproc_method: 新的工作进程多处理方法（如果为None则使用原有值）
            use_modelscope: 是否使用ModelScope（如果为None则使用原有值）
            gpus: 新的可见GPU列表（如果为None则使用原有值）
            additional_args: 额外的命令行参数列表（如果为None则使用原有值）
            wait_for_start: 是否等待服务器启动
            wait_seconds: 等待服务器启动的秒数
            
        返回:
            VllmServerInfo: 更新后的服务器信息对象
            
        异常:
            ValueError: 如果指定端口不存在服务器
            RuntimeError: 如果管理器已被销毁
        """
        with self._lock:
            # 检查管理器是否已被销毁
            if self._destroyed:
                raise RuntimeError("管理器已被销毁，无法更新服务器")
            
            # 获取现有服务器信息
            existing_server = self.get_server_info(port)
            if not existing_server:
                raise ValueError(f"端口 {port} 上没有找到服务器")
            
            # 使用新参数或保留原有参数
            if served_model_name is None:
                served_model_name = model_path
            if tensor_parallel_size is None:
                tensor_parallel_size = existing_server.tensor_parallel_size
            if max_model_len is None:
                max_model_len = existing_server.max_model_len
            if worker_multiproc_method is None:
                worker_multiproc_method = existing_server.worker_multiproc_method
            if use_modelscope is None:
                use_modelscope = existing_server.use_modelscope
            if gpus is None:
                gpus = existing_server.gpus
            if additional_args is None:
                additional_args = existing_server.additional_args
            
            print(f"准备更新端口 {port} 的服务器，从模型 {existing_server.model_path} 更新到 {model_path}")
            additional_args = self._ensure_sleep_mode_args(additional_args)
            launch_model_path = existing_server.launch_model_path
            can_hot_reload = (
                existing_server.is_running()
                and launch_model_path is not None
                and existing_server.tensor_parallel_size == tensor_parallel_size
                and existing_server.max_model_len == max_model_len
                and existing_server.worker_multiproc_method == worker_multiproc_method
                and existing_server.use_modelscope == use_modelscope
                and self._normalize_gpu_key(existing_server.gpus) == self._normalize_gpu_key(gpus)
            )
            if can_hot_reload:
                existing_server.additional_args = additional_args
                return self._prepare_server_for_model(
                    existing_server,
                    model_path=model_path,
                    served_model_name=served_model_name,
                    wait_seconds=wait_seconds,
                )
            
            # 停止现有服务器
            if not self.stop_server(port, wait_for_idle=wait_for_idle, idle_timeout=idle_timeout):
                raise RuntimeError(f"停止端口 {port} 的服务器失败")
            
            # 确保端口已被释放（短暂等待）
            print(f"等待端口 {port} 释放...")
            # 检查端口是否可用
            start_time = time.time()
            max_wait = 5  # 最多等待5秒
            while time.time() - start_time < max_wait:
                # 检查是否有进程在使用该端口
                try:
                    import socket
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        # 尝试绑定端口，如果能成功绑定，说明端口可用
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind(("localhost", port))
                        print(f"端口 {port} 已释放并可用")
                        break
                except socket.error:
                    # 端口被占用，继续等待
                    time.sleep(0.5)
            else:
                print(f"警告: 在{max_wait}秒内端口 {port} 可能尚未完全释放，但继续启动")
            
            # 在相同端口上启动新服务器
            print(f"在端口 {port} 上启动新服务器，模型: {model_path}")
            # 如果没有指定新的日志文件，生成一个新的默认日志文件名
            if log_file is None:
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'vllm_logs')
                model_name = os.path.basename(model_path)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"vllm_server_{model_name}_{port}_{timestamp}_updated.log")
                print(f"更新服务器时未指定日志文件，将使用新的默认日志文件: {log_file}")
            
            new_server_info = self.add_server(
                model_path=model_path,
                launch_model_path=launch_model_path,
                port=port,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                worker_multiproc_method=worker_multiproc_method,
                use_modelscope=use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                wait_for_start=False,  # 先不等待，我们会自己检查
                log_file=log_file,
                exclusive_gpu_mode=exclusive_gpu_mode
            )
            
            # 等待服务器启动并准备就绪
            if wait_for_start:
                self._wait_server_ready(new_server_info, wait_seconds=wait_seconds)
            
            return new_server_info
    
    def _find_available_gpus(self, required_count: int) -> Optional[List[int]]:
        """
        查找可用的GPU
        
        参数:
            required_count: 需要的GPU数量
            
        返回:
            Optional[List[int]]: 可用的GPU列表，如果没有足够的GPU返回None
        """
        used_gpus_set = set()
        with self.__class__._global_resource_lock:
            for gpus in self.__class__._global_used_gpus.values():
                used_gpus_set.update(gpus)
        available_gpus = [gpu for gpu in self._available_gpus if gpu not in used_gpus_set]
        if len(available_gpus) < required_count:
            return None
        if len(self._available_gpus) > 0:
            rr_start = self._gpu_round_robin_cursor % len(self._available_gpus)
            rr_order = self._available_gpus[rr_start:] + self._available_gpus[:rr_start]
        else:
            rr_order = []
        now_ts = time.time()
        cold_candidates = [
            gpu for gpu in rr_order
            if gpu in available_gpus and (now_ts - self._gpu_release_ts.get(gpu, 0.0)) >= self._gpu_reuse_cooldown_sec
        ]
        warm_candidates = [gpu for gpu in rr_order if gpu in available_gpus and gpu not in cold_candidates]
        free_mem_map = self._query_gpu_free_memory_mib()
        order_map = {gpu: idx for idx, gpu in enumerate(rr_order)}
        cold_candidates.sort(key=lambda gpu: (order_map.get(gpu, len(order_map)), -free_mem_map.get(gpu, -1.0)))
        warm_candidates.sort(key=lambda gpu: (order_map.get(gpu, len(order_map)), -free_mem_map.get(gpu, -1.0)))
        ordered_candidates = cold_candidates + warm_candidates
        if len(ordered_candidates) < required_count:
            return None
        return ordered_candidates[:required_count]

    def _query_gpu_free_memory_mib(self) -> Dict[int, float]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return {}
        free_mem_map: Dict[int, float] = {}
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            gpu_idx = int(parts[0])
            free_mib = float(parts[1])
            free_mem_map[gpu_idx] = free_mib
        return free_mem_map
    
    def _allocate_gpus(self, port: int, gpu_count: int) -> Optional[List[int]]:
        """
        为指定端口的服务器分配GPU
        
        参数:
            port: 服务器端口
            gpu_count: 需要的GPU数量
            
        返回:
            Optional[List[int]]: 分配的GPU列表，如果无法分配返回None
        """
        with self.__class__._global_resource_lock:
            gpus = self._find_available_gpus(gpu_count)
            if gpus:
                self._used_gpus[port] = gpus
                global_key = self._global_gpu_key(port)
                self.__class__._global_used_gpus[global_key] = list(gpus)
                if len(self._available_gpus) > 0:
                    last_gpu = gpus[-1]
                    if last_gpu in self._available_gpus:
                        last_idx = self._available_gpus.index(last_gpu)
                        self._gpu_round_robin_cursor = (last_idx + 1) % len(self._available_gpus)
                print(f"为端口 {port} 分配GPU: {gpus}")
            return gpus
    
    def _release_gpus(self, port: int):
        """
        释放为指定端口分配的GPU
        
        参数:
            port: 服务器端口
        """
        if port in self._used_gpus:
            released_gpus = self._used_gpus.pop(port)
            with self.__class__._global_resource_lock:
                global_key = self._global_gpu_key(port)
                self.__class__._global_used_gpus.pop(global_key, None)
            release_ts = time.time()
            for gpu in released_gpus:
                self._gpu_release_ts[int(gpu)] = release_ts
            self._dispatch_wakeup_event.set()
            print(f"释放端口 {port} 的GPU: {released_gpus}")

    def _record_waiting_for_gpu(self, task: Task) -> None:
        now_ts = time.time()
        task_id = task.task_id
        self._task_wait_retry_count[task_id] = self._task_wait_retry_count.get(task_id, 0) + 1
        if task_id not in self._task_wait_first_ts:
            self._task_wait_first_ts[task_id] = now_ts
        last_log_ts = self._task_wait_last_log_ts.get(task_id, 0.0)
        if (now_ts - last_log_ts) >= self._gpu_wait_log_interval_sec:
            waited = now_ts - self._task_wait_first_ts[task_id]
            retry = self._task_wait_retry_count[task_id]
            print(
                f"任务 {task_id} 等待GPU中（已等待 {waited:.1f}s，重试 {retry} 次，队列 {self._task_queue.qsize()}）"
            )
            self._task_wait_last_log_ts[task_id] = now_ts

    def _record_waiting_finished(self, task: Task) -> None:
        task_id = task.task_id
        if task_id not in self._task_wait_first_ts:
            return
        now_ts = time.time()
        waited = max(now_ts - self._task_wait_first_ts[task_id], 0.0)
        retry = self._task_wait_retry_count.get(task_id, 0)
        print(f"任务 {task_id} 获取到GPU（累计等待 {waited:.1f}s，重试 {retry} 次）")
        self._task_wait_retry_count.pop(task_id, None)
        self._task_wait_first_ts.pop(task_id, None)
        self._task_wait_last_log_ts.pop(task_id, None)
    
    def _process_tasks(self):
        """处理任务队列"""
        while self._tasks_running:
            try:
                # 尝试获取任务，超时1秒
                task = self._task_queue.get(timeout=1)
                
                # 尝试分配GPU
                gpus = self._allocate_gpus(-1, task.gpu_count)  # 使用临时端口-1进行分配
                
                if gpus:
                    # 分配成功，开始执行任务
                    self._record_waiting_finished(task)
                    task.status = "running"
                    task.start_time = time.time()  # 记录任务开始时间
                    task.assigned_gpus = list(gpus)
                    temp_port = -int(uuid.uuid4().int % 1_000_000_000)
                    self._rebind_allocated_gpus(-1, temp_port)

                    def execute_task(task=task, gpus=list(gpus), temp_port=temp_port):
                        server_info = None
                        port = None
                        server_stopped = False
                        try:
                            server_info = self._acquire_or_create_pooled_server(task, gpus, temp_port=temp_port)
                            port = int(server_info.port)
                            task.params_dict['port'] = port
                            task.server_port = port
                            server_info.current_task = task
                            call_params = dict(task.params_dict)
                            if "served_model_name" in inspect.signature(task.func_handle).parameters:
                                call_params["served_model_name"] = server_info.served_model_name
                            result = task.func_handle(**call_params)
                            task.result = result
                            task.status = "completed"
                            task.end_time = time.time()
                            print(f"任务 {task.task_id} 执行完成")
                        except Exception as e:
                            task.result = str(e)
                            task.status = "failed"
                            task.end_time = time.time()
                            if server_info is None:
                                print(f"启动任务 {task.task_id} 的服务器失败: {e}")
                            else:
                                print(f"任务 {task.task_id} 执行失败: {e}")
                        finally:
                            if not task.end_time:
                                task.end_time = time.time()
                            runtime_sec = max(task.end_time - task.start_time, 0.0)
                            server_port = int(task.server_port) if task.server_port is not None else int(port if port is not None else -1)
                            server_port_record = server_port if server_port >= 0 else None
                            if server_info is not None:
                                try:
                                    server_info.current_task = None
                                    if task.cleanup_server_after_completion and server_port >= 0:
                                        server_stopped = self.stop_server(server_port, wait_for_idle=False)
                                    else:
                                        self._sleep_server(server_info, level=self._sleep_idle_level)
                                except Exception as sleep_err:
                                    print(f"休眠任务 {task.task_id} 的常驻服务器失败，将尝试停止服务器: {sleep_err}")
                                    try:
                                        if server_port >= 0:
                                            server_stopped = self.stop_server(server_port, wait_for_idle=False)
                                    except Exception as stop_err:
                                        print(f"停止任务 {task.task_id} 服务器时出错: {stop_err}")
                            else:
                                if server_port >= 0:
                                    self._release_port(server_port)
                            try:
                                if server_port >= 0 and not server_stopped:
                                    self._release_gpus(server_port)
                                elif not server_stopped:
                                    self._release_gpus(temp_port)
                            except Exception as release_err:
                                print(f"释放任务 {task.task_id} GPU时出错: {release_err}")
                            if task.cleanup_model_dirs:
                                self._cleanup_task_model_dirs(task.cleanup_model_dirs)
                            with self._lock:
                                for gpu_id in task.assigned_gpus:
                                    if gpu_id not in self._gpu_busy_time_sec:
                                        self._gpu_busy_time_sec[gpu_id] = 0.0
                                    if gpu_id not in self._gpu_task_count:
                                        self._gpu_task_count[gpu_id] = 0
                                    self._gpu_busy_time_sec[gpu_id] += runtime_sec
                                    self._gpu_task_count[gpu_id] += 1
                                    self._gpu_task_records.append({
                                        "task_id": task.task_id,
                                        "gpu_id": int(gpu_id),
                                        "status": task.status,
                                        "start_time": float(task.start_time),
                                        "end_time": float(task.end_time),
                                        "runtime_sec": float(runtime_sec),
                                        "server_port": server_port_record,
                                    })
                                self._task_runtime_records.append({
                                    "task_id": task.task_id,
                                    "status": task.status,
                                    "gpu_ids": [int(g) for g in task.assigned_gpus],
                                    "start_time": float(task.start_time),
                                    "end_time": float(task.end_time),
                                    "runtime_sec": float(runtime_sec),
                                    "server_port": server_port_record,
                                    "estimated_tokens": float(task.estimated_tokens),
                                })
                                self._task_results[task.task_id] = task.result
                                if task.task_id in self._result_events:
                                    self._result_events[task.task_id].set()
                                self._task_queue.task_done()

                    task_thread = threading.Thread(target=execute_task)
                    task_thread.daemon = True
                    task_thread.start()
                else:
                    # 没有可用GPU，将任务重新放回队列
                    self._record_waiting_for_gpu(task)
                    self._task_queue.put(task)
                    self._dispatch_wakeup_event.wait(timeout=self._gpu_wait_retry_sleep_sec)
                    self._dispatch_wakeup_event.clear()
                    
            except queue.Empty:
                # 队列为空，继续循环
                self._dispatch_wakeup_event.wait(timeout=self._gpu_wait_retry_sleep_sec)
                self._dispatch_wakeup_event.clear()
            except Exception as e:
                print(f"处理任务时出错: {e}")
    
    def _clear_progress_log(self):
        """清空并初始化进度日志文件"""
        try:
            log_dir = os.path.dirname(self._progress_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with self.__class__._global_progress_log_lock:
                with open(self._progress_log_file, 'w', encoding='utf-8') as f:
                    f.write("========== vLLM服务器进度日志 ==========\n")
                    f.write(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"实例ID: {self._instance_id}\n")
                    f.write(f"可用GPU: {self._available_gpus}\n\n")
        except Exception as e:
            print(f"初始化进度日志文件时出错: {e}")

    def _cleanup_task_model_dirs(self, model_dirs: List[str]) -> None:
        cleaned_dirs = set()
        for model_dir in model_dirs:
            if not isinstance(model_dir, str) or len(model_dir) == 0:
                continue
            model_dir_abs = os.path.abspath(model_dir)
            if model_dir_abs in cleaned_dirs or not os.path.exists(model_dir_abs):
                continue
            try:
                if os.path.isdir(model_dir_abs):
                    shutil.rmtree(model_dir_abs)
                else:
                    os.remove(model_dir_abs)
                cleaned_dirs.add(model_dir_abs)
                print(f"已清理融合模型缓存: {model_dir_abs}")
                parent_dir = os.path.dirname(model_dir_abs)
                if os.path.isdir(parent_dir) and len(os.listdir(parent_dir)) == 0:
                    os.rmdir(parent_dir)
                    print(f"已清理空缓存目录: {parent_dir}")
            except Exception as cleanup_err:
                print(f"清理融合模型缓存失败 {model_dir_abs}: {cleanup_err}")

    def _estimate_runtime_by_tokens(self, task: Task) -> float:
        estimated_tokens = task.estimated_tokens
        if estimated_tokens <= 0:
            return -1.0
        completed_with_tokens = [
            rec for rec in self._task_runtime_records
            if rec.get("estimated_tokens", None) not in (None, 0) and rec.get("runtime_sec", 0) > 0
        ]
        token_rate_samples = [
            float(rec["runtime_sec"]) / max(float(rec["estimated_tokens"]), 1.0)
            for rec in completed_with_tokens
        ]
        running_with_tokens = [
            t for t in self._all_tasks.values()
            if t.status == "running" and t.estimated_tokens > 0 and t.start_time > 0
        ]
        now_ts = time.time()
        for running_task in running_with_tokens:
            runtime_sec = max(now_ts - running_task.start_time, 0.0)
            if runtime_sec <= 0:
                continue
            token_rate_samples.append(runtime_sec / max(float(running_task.estimated_tokens), 1.0))
        if len(token_rate_samples) == 0:
            return -1.0
        sec_per_token = np.mean(token_rate_samples)
        return float(estimated_tokens) * float(sec_per_token)

    def get_realtime_task_stats(self) -> Dict[str, Any]:
        with self._lock:
            now_ts = time.time()
            pending_tasks = [task for task in self._all_tasks.values() if task.status == "pending"]
            running_tasks = [task for task in self._all_tasks.values() if task.status == "running"]
            completed_tasks = [task for task in self._all_tasks.values() if task.status == "completed"]
            failed_tasks = [task for task in self._all_tasks.values() if task.status == "failed"]
            total_tasks = len(self._all_tasks)
            done_tasks = len(completed_tasks) + len(failed_tasks)
            elapsed = max(now_ts - self._series_start_time, 0.0) if self._series_start_time > 0 else 0.0
            tqdm_eta = -1.0
            if done_tasks > 0 and total_tasks > done_tasks:
                tqdm_eta = (elapsed / done_tasks) * (total_tasks - done_tasks)
            running_details = []
            for task in running_tasks:
                runtime = max(now_ts - task.start_time, 0.0)
                est_total = self._estimate_runtime_by_tokens(task)
                eta_sec = -1.0
                finish_ts = -1.0
                if est_total > 0:
                    eta_sec = max(est_total - runtime, 0.0)
                    finish_ts = now_ts + eta_sec
                running_details.append({
                    "task_id": task.task_id,
                    "runtime_sec": float(runtime),
                    "estimated_total_sec": float(est_total),
                    "remaining_sec": float(eta_sec),
                    "estimated_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_ts)) if finish_ts > 0 else "",
                    "gpu_ids": [int(g) for g in task.assigned_gpus],
                    "estimated_tokens": float(task.estimated_tokens),
                })
            series_start = float(self._series_start_time) if self._series_start_time > 0 else float(now_ts)
            gpu_segments_map: Dict[int, List[Dict[str, Any]]] = {int(gpu_id): [] for gpu_id in self._available_gpus}
            for rec in self._gpu_task_records:
                gpu_id = int(rec.get("gpu_id", -1))
                if gpu_id < 0:
                    continue
                if gpu_id not in gpu_segments_map:
                    gpu_segments_map[gpu_id] = []
                start_ts = max(float(rec.get("start_time", now_ts)), series_start)
                end_ts = max(float(rec.get("end_time", start_ts)), start_ts)
                gpu_segments_map[gpu_id].append({
                    "task_id": str(rec.get("task_id", "")),
                    "status": str(rec.get("status", "")),
                    "start_time": float(start_ts),
                    "end_time": float(end_ts),
                    "start_offset_sec": float(start_ts - series_start),
                    "end_offset_sec": float(end_ts - series_start),
                    "duration_sec": float(end_ts - start_ts),
                    "running": False,
                })
            for task in running_tasks:
                start_ts = max(float(task.start_time), series_start)
                end_ts = max(float(now_ts), start_ts)
                for gpu_id in task.assigned_gpus:
                    gpu_id = int(gpu_id)
                    if gpu_id not in gpu_segments_map:
                        gpu_segments_map[gpu_id] = []
                    gpu_segments_map[gpu_id].append({
                        "task_id": str(task.task_id),
                        "status": "running",
                        "start_time": float(start_ts),
                        "end_time": float(end_ts),
                        "start_offset_sec": float(start_ts - series_start),
                        "end_offset_sec": float(end_ts - series_start),
                        "duration_sec": float(end_ts - start_ts),
                        "running": True,
                    })
            gpu_timeline = []
            for gpu_id in sorted(gpu_segments_map.keys()):
                segments = sorted(gpu_segments_map[gpu_id], key=lambda item: (item["start_time"], item["task_id"]))
                busy_sec = float(sum(float(seg["duration_sec"]) for seg in segments))
                gpu_timeline.append({
                    "gpu_id": int(gpu_id),
                    "busy_sec": busy_sec,
                    "segments": segments,
                })
            return {
                "timestamp": float(now_ts),
                "total_tasks": int(total_tasks),
                "pending_tasks": int(len(pending_tasks)),
                "running_tasks": int(len(running_tasks)),
                "completed_tasks": int(len(completed_tasks)),
                "failed_tasks": int(len(failed_tasks)),
                "queue_size": int(self._task_queue.qsize()),
                "used_gpus": int(sum(len(gpus) for gpus in self._used_gpus.values() if gpus)),
                "available_gpus": int(len(self._available_gpus) - sum(len(gpus) for gpus in self._used_gpus.values() if gpus)),
                "tqdm_eta_sec": float(tqdm_eta),
                "running_details": running_details,
                "gpu_timeline": gpu_timeline,
            }

    def is_queue_empty(self) -> bool:
        with self._lock:
            return self._task_queue.qsize() == 0

    def has_idle_gpu(self) -> bool:
        with self._lock:
            return self._find_available_gpus(1) is not None

    def get_idle_gpu_count(self) -> int:
        with self._lock:
            used_count = int(sum(len(gpus) for gpus in self._used_gpus.values() if gpus))
            return max(0, int(len(self._available_gpus) - used_count))

    def collect_newly_completed_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._task_completion_cursor >= len(self._task_runtime_records):
                return []
            new_records = self._task_runtime_records[self._task_completion_cursor :]
            self._task_completion_cursor = len(self._task_runtime_records)
            return [dict(record) for record in new_records]
    
    def _get_progress_info(self) -> str:
        """获取当前进度信息"""
        with self._lock:
            realtime_stats = self.get_realtime_task_stats()
            # 统计GPU使用情况
            total_gpus = len(self._available_gpus)
            used_gpus_count = sum(len(gpus) for gpus in self._used_gpus.values() if gpus)
            available_gpus_count = total_gpus - used_gpus_count
            
            # 统计任务状态
            pending_tasks = []
            running_tasks = []
            completed_tasks = []
            failed_tasks = []
            
            # 从所有任务列表中获取状态
            for task_id, task in self._all_tasks.items():
                if task.status == "pending":
                    pending_tasks.append(task)
                elif task.status == "running":
                    running_tasks.append(task)
                elif task.status == "completed":
                    completed_tasks.append(task)
                elif task.status == "failed":
                    failed_tasks.append(task)
            
            # 还需要考虑队列中的任务
            queue_size = self._task_queue.qsize()
            
            # 构建进度信息
            progress_info = []
            progress_info.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 进度更新")
            progress_info.append("-" * 50)
            
            # GPU使用情况
            progress_info.append(f"GPU使用情况:")
            progress_info.append(f"  总共: {total_gpus}")
            progress_info.append(f"  已用: {used_gpus_count}")
            progress_info.append(f"  可用: {available_gpus_count}")
            gpu_timeline = realtime_stats.get("gpu_timeline", [])
            if gpu_timeline:
                progress_info.append("  实时GPU时间片:")
                for gpu_item in gpu_timeline:
                    running_count = sum(1 for seg in gpu_item["segments"] if seg.get("running", False))
                    progress_info.append(
                        f"    GPU{int(gpu_item['gpu_id'])}: busy={float(gpu_item['busy_sec']):.1f}s, "
                        f"segments={len(gpu_item['segments'])}, running={running_count}"
                    )
                    latest_segments = gpu_item["segments"][-3:]
                    if latest_segments:
                        progress_info.append("      最近时间段:")
                        for seg in latest_segments:
                            start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(seg["start_time"])))
                            end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(seg["end_time"])))
                            progress_info.append(
                                f"        - {str(seg['task_id'])} [{str(seg['status'])}] "
                                f"{start_str} ~ {end_str} ({float(seg['duration_sec']):.1f}s)"
                            )
            
            # 任务统计
            total_tasks = len(self._all_tasks)
            progress_info.append(f"\n任务统计:")
            progress_info.append(f"  总共: {total_tasks}")
            progress_info.append(f"  待处理: {len(pending_tasks)}")
            progress_info.append(f"  运行中: {len(running_tasks)}")
            progress_info.append(f"  已完成: {len(completed_tasks)}")
            progress_info.append(f"  失败: {len(failed_tasks)}")
            
            # 剩余任务数
            remaining_tasks = len(pending_tasks) + len(running_tasks)
            progress_info.append(f"  剩余: {remaining_tasks}")
            
            # 完成百分比
            if total_tasks > 0:
                completion_percentage = (len(completed_tasks) / total_tasks) * 100
                progress_info.append(f"  完成率: {completion_percentage:.1f}%")
                if realtime_stats["tqdm_eta_sec"] > 0:
                    eta_finish = time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(time.time() + realtime_stats["tqdm_eta_sec"])
                    )
                    progress_info.append(f"  tqdm剩余时间估计: {realtime_stats['tqdm_eta_sec']:.1f}s")
                    progress_info.append(f"  tqdm预计完成时间: {eta_finish}")
            
            # 正在运行的任务详情
            if running_tasks:
                progress_info.append("\n正在运行的任务:")
                for task in running_tasks:
                    runtime = time.time() - task.start_time
                    hours, remainder = divmod(runtime, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    progress_info.append(f"  - {task.task_id}")
                    progress_info.append(f"    模型: {task.model_path}")
                    progress_info.append(f"    GPU数量: {task.gpu_count}")
                    progress_info.append(f"    运行时间: {runtime_str}")
                    est_total = self._estimate_runtime_by_tokens(task)
                    if est_total > 0:
                        remaining = max(est_total - runtime, 0.0)
                        finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining))
                        progress_info.append(f"    预计总时长(基于tokens): {est_total:.1f}s")
                        progress_info.append(f"    预计剩余: {remaining:.1f}s")
                        progress_info.append(f"    预计完成时间: {finish_time}")
            
            # 已完成的任务
            if completed_tasks:
                progress_info.append("\n最近完成的任务:")
                # 只显示最近5个完成的任务
                for task in completed_tasks[-5:]:
                    progress_info.append(f"  - {task.task_id} (成功)")
            
            # 失败的任务
            if failed_tasks:
                progress_info.append("\n失败的任务:")
                for task in failed_tasks:
                    progress_info.append(f"  - {task.task_id} (失败)")
                    # 只显示部分错误信息
                    error_info = str(task.result)[:100] + "..." if len(str(task.result)) > 100 else str(task.result)
                    progress_info.append(f"    错误: {error_info}")
            
            progress_info.append("=" * 50 + "\n")
            
            return "\n".join(progress_info)
    
    def _update_progress_log(self, force: bool = False):
        """更新进度日志"""
        try:
            progress_info = self._get_progress_info()
            if (not force) and progress_info == self._last_progress_snapshot:
                return
            self._last_progress_snapshot = progress_info
            with self.__class__._global_progress_log_lock:
                with open(self._progress_log_file, 'w', encoding='utf-8') as f:
                    f.write("========== vLLM服务器进度日志 ==========\n")
                    f.write(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"实例ID: {self._instance_id}\n\n")
                    f.write(progress_info)
        except Exception as e:
            print(f"更新进度日志时出错: {e}")
    
    def _progress_monitor(self):
        """进度监控线程函数"""
        while self._progress_running:
            try:
                self._update_progress_log()
                time.sleep(5)  # 每5秒更新一次进度
            except Exception as e:
                print(f"进度监控线程出错: {e}")
                time.sleep(1)

    def _reset_series_state(self):
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except queue.Empty:
                break
        self._all_tasks.clear()
        self._task_results.clear()
        self._result_events.clear()
        self._series_start_time = time.time()
        self._series_end_time = 0.0
        self._gpu_busy_time_sec = {gpu: 0.0 for gpu in self._available_gpus}
        self._gpu_task_count = {gpu: 0 for gpu in self._available_gpus}
        self._gpu_task_records = []
        self._task_runtime_records = []
        self._task_completion_cursor = 0
        self._last_progress_snapshot = ""
        self._clear_progress_log()

    def _has_active_work(self) -> bool:
        if not self._task_queue.empty():
            return True
        for task in self._all_tasks.values():
            if task.status in ("pending", "running"):
                return True
        for server in self._servers.values():
            if server.is_running() and server.current_task is not None:
                return True
        return False

    def _ensure_workers_started(self):
        if not self._progress_running:
            self._progress_running = True
            self._progress_thread = threading.Thread(target=self._progress_monitor)
            self._progress_thread.daemon = True
            self._progress_thread.start()
        if not self._tasks_running:
            self._tasks_running = True
            self._task_thread = threading.Thread(target=self._process_tasks)
            self._task_thread.daemon = True
            self._task_thread.start()

    def submit_tasks(self, tasks: List[Dict[str, Any]], reset_series: bool = False) -> List[str]:
        with self._lock:
            if self._destroyed:
                raise RuntimeError("管理器已被销毁，无法提交任务")
            if reset_series:
                if self._has_active_work():
                    print("检测到仍有任务或服务器运行，跳过reset_series并追加任务")
                    if self._series_start_time <= 0:
                        self._series_start_time = time.time()
                else:
                    self._reset_series_state()
            elif self._series_start_time <= 0:
                self._series_start_time = time.time()
            self._ensure_workers_started()
            total_gpus = len(self._available_gpus)
            task_count = len(tasks)
            gpu_allocation = [int(task_info.get("gpu_count", 1)) for task_info in tasks]
            print(f"GPU分配计划: 可用GPU总数={total_gpus}, 任务总数={task_count}, 各任务GPU分配={gpu_allocation}")
            task_ids: List[str] = []
            for task_info in tasks:
                required_fields = ['task_id', 'model_path', 'params_dict', 'func_handle']
                for field in required_fields:
                    if field not in task_info:
                        raise ValueError(f"任务缺少必要字段: {field}")
                task_id = str(task_info['task_id'])
                if task_id in self._all_tasks:
                    raise ValueError(f"任务ID重复: {task_id}")
                gpu_count = max(1, int(task_info.get("gpu_count", 1)))
                task = Task(
                    task_id=task_id,
                    model_path=task_info['model_path'],
                    params_dict=task_info['params_dict'],
                    func_handle=task_info['func_handle'],
                    gpu_count=gpu_count,
                    estimated_tokens=float(task_info.get('estimated_tokens', -1.0)),
                    cleanup_model_dirs=list(task_info.get('cleanup_model_dirs', [])),
                    cleanup_server_after_completion=bool(task_info.get('cleanup_server_after_completion', False)),
                )
                print(f"任务 {task.task_id} 分配了 {task.gpu_count} 个GPU")
                self._task_queue.put(task)
                self._all_tasks[task.task_id] = task
                self._result_events[task.task_id] = threading.Event()
                task_ids.append(task.task_id)
            print(f"已将 {len(tasks)} 个任务添加到队列")
            self._dispatch_wakeup_event.set()
            return task_ids

    def submit_task(self, task: Dict[str, Any]) -> str:
        task_ids = self.submit_tasks([task], reset_series=False)
        return task_ids[0]

    def wait_for_tasks(self, task_ids: List[str], clear_events: bool = True) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for task_id in task_ids:
            event = self._result_events.get(task_id)
            if event is None:
                raise KeyError(f"任务不存在或事件已清理: {task_id}")
            event.wait()
            results[task_id] = self._task_results.get(task_id)
            if clear_events:
                self._result_events.pop(task_id, None)
        return results
    
    def wait_for_any_task(
        self,
        task_ids: List[str],
        clear_event: bool = True,
        poll_interval_sec: float = 0.2,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], Any]:
        pending_task_ids = [task_id for task_id in task_ids if task_id]
        if not pending_task_ids:
            return None, None
        start_time = time.time()
        while True:
            for task_id in pending_task_ids:
                event = self._result_events.get(task_id)
                if event is None:
                    if task_id in self._task_results:
                        return task_id, self._task_results.get(task_id)
                    raise KeyError(f"任务不存在或事件已清理: {task_id}")
                if event.is_set():
                    result = self._task_results.get(task_id)
                    if clear_event:
                        self._result_events.pop(task_id, None)
                    return task_id, result
            if timeout is not None and (time.time() - start_time) >= timeout:
                return None, None
            time.sleep(max(0.01, float(poll_interval_sec)))
    
    def run_series_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行一系列任务，自动分配GPU资源
        
        参数:
            tasks: 任务列表，每个任务包含：
                - task_id: 任务ID
                - model_path: 模型路径
                - params_dict: 参数字典
                - func_handle: 函数句柄
                
        返回:
            Dict[str, Any]: 所有任务的执行结果
        """
        task_ids = self.submit_tasks(tasks, reset_series=True)
        results = self.wait_for_tasks(task_ids, clear_events=True)
        self._series_end_time = time.time()
        self._update_progress_log(force=True)
        self.export_gpu_usage_stats()
        
        return results

    def export_gpu_usage_stats(self) -> Dict[str, Any]:
        with self._lock:
            if self._series_start_time > 0 and self._series_end_time > 0:
                window_sec = max(self._series_end_time - self._series_start_time, 1e-9)
            else:
                window_sec = 1e-9
            per_gpu = []
            for gpu_id in sorted(self._gpu_busy_time_sec.keys()):
                busy = float(self._gpu_busy_time_sec.get(gpu_id, 0.0))
                utilization = max(0.0, min(1.0, busy / window_sec))
                per_gpu.append({
                    "gpu_id": int(gpu_id),
                    "busy_time_sec": busy,
                    "idle_time_sec": max(window_sec - busy, 0.0),
                    "utilization": utilization,
                    "task_count": int(self._gpu_task_count.get(gpu_id, 0)),
                })
            payload = {
                "series_start_time": float(self._series_start_time),
                "series_end_time": float(self._series_end_time),
                "series_window_sec": float(window_sec),
                "gpu_summary": per_gpu,
                "task_runtime_records": self._task_runtime_records,
                "gpu_task_records": self._gpu_task_records,
            }
            os.makedirs(os.path.dirname(self._gpu_stats_file), exist_ok=True)
            with open(self._gpu_stats_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return payload
    
    def destroy(self):
        """
        销毁管理器，停止所有服务器并清理资源
        """
        with self._lock:
            if not self._destroyed:
                print("销毁vLLM服务器管理器，停止所有服务器...")
                # 停止进度监控线程
                if self._progress_running:
                    self._progress_running = False
                    if self._progress_thread:
                        self._progress_thread.join(timeout=5)
                # 停止任务处理
                self._tasks_running = False
                if self._task_thread:
                    self._task_thread.join(timeout=5)
                # 停止所有服务器
                self.stop_all_servers()
                # 清理GPU使用记录
                self._used_gpus.clear()
                with self.__class__._global_resource_lock:
                    for global_key in list(self.__class__._global_used_gpus.keys()):
                        if global_key.startswith(f"{self._instance_id}:"):
                            self.__class__._global_used_gpus.pop(global_key, None)
                    for port in list(self._reserved_ports):
                        self.__class__._global_used_ports.discard(int(port))
                self._reserved_ports.clear()
                
                # 清理所有日志文件
                for server_info in list(self._servers.values()):
                    # 注意：这里不直接关闭日志文件，因为在subprocess.Popen中处理
                    # 当进程结束时，相关的文件句柄会被自动关闭
                    pass
                    
                self._destroyed = True
                print("vLLM服务器管理器已销毁")
                
                # 清空数据结构
                self._servers.clear()
                self._task_results.clear()
                self._result_events.clear()
                self._all_tasks.clear()
                with self.__class__._singleton_lock:
                    if self.__class__._singleton_instance is self:
                        self.__class__._singleton_instance = None


if __name__ == "__main__":
    # 运行示例
    example_usage()
