import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from safetensors.torch import load_file
import re
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

class LayerNormDifferenceAnalyzer:
    """
    LayerNorm参数差异分析器
    专门用于比较两个Transformer块中LayerNorm的差异
    """
    
    def __init__(self, device='cpu'):
        """
        初始化分析器
        
        参数: 
        device: 计算设备
        """
        self.device = device
        
    def extract_layernorm_parameters_from_tensors(self, tensors, param_type='both'):
        """
        从模型张量字典中提取所有LayerNorm的参数
        
        参数:
        tensors: 模型张量字典
        param_type: 'gamma'(权重), 'beta'(偏置), 'both'(两者)
        
        返回:
        参数字典: {layer_name: {'gamma': numpy数组, 'beta': numpy数组}}
        """
        params_dict = {}
        
        for name, tensor in tensors.items():
            if 'norm' in name.lower() and ('weight' in name or 'bias' in name):
                # 解析层名称和参数类型
                layer_name_parts = name.split('.')
                param_type_in_name = 'gamma' if 'weight' in name else 'beta'
                
                # 构造层名称（去掉最后一个参数类型部分）
                layer_name = '.'.join(layer_name_parts[:-1])
                
                if layer_name not in params_dict:
                    params_dict[layer_name] = {'gamma': None, 'beta': None}
                
                # 提取参数，确保转换为float32以避免BFloat16等不支持的类型问题
                if param_type_in_name == 'gamma' and param_type in ['gamma', 'both']:
                    params_dict[layer_name]['gamma'] = tensor.to(dtype=torch.float32).detach().cpu().numpy()
                
                if param_type_in_name == 'beta' and param_type in ['beta', 'both']:
                    params_dict[layer_name]['beta'] = tensor.to(dtype=torch.float32).detach().cpu().numpy()
        
        # 过滤掉没有参数的层
        params_dict = {k: v for k, v in params_dict.items() if v['gamma'] is not None or v['beta'] is not None}
        
        return params_dict
    
    def compute_wasserstein_distance(self, params1, params2):
        """
        计算两个参数向量的Wasserstein距离
        适合比较分布形状
        """
        # 展平
        p1_flat = params1.flatten()
        p2_flat = params2.flatten()
        
        # 计算1D Wasserstein距离
        w_dist = wasserstein_distance(p1_flat, p2_flat)
        
        return w_dist
    
    def compute_kl_divergence_estimate(self, params1, params2, bins=50, eps=1e-10):
        """
        估计KL散度（需要小心处理零概率）
        通过直方图估计概率分布
        """
        # 展平
        p1_flat = params1.flatten()
        p2_flat = params2.flatten()
        
        # 确定共同的取值范围
        min_val = min(p1_flat.min(), p2_flat.min())
        max_val = max(p1_flat.max(), p2_flat.max())
        
        # 计算直方图
        hist1, _ = np.histogram(p1_flat, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(p2_flat, bins=bins, range=(min_val, max_val), density=True)
        
        # 添加小常数避免零
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        
        # 计算KL散度
        kl_12 = np.sum(hist1 * np.log(hist1 / hist2))
        kl_21 = np.sum(hist2 * np.log(hist2 / hist1))
        
        # 返回对称KL散度
        return (kl_12 + kl_21) / 2
    
    def calculate_layernorm_diffs(self, model1_tensors, model2_tensors):
        """
        计算两个模型张量之间的LayerNorm差异
        
        参数:
        model1_tensors, model2_tensors: 两个模型的张量字典
        
        返回:
        差异字典: {layer_num: {'layernorm_kl': float, 'layernorm_wasserstein': float}}
        """
        # 提取LayerNorm参数
        ln_params1 = self.extract_layernorm_parameters_from_tensors(model1_tensors)
        ln_params2 = self.extract_layernorm_parameters_from_tensors(model2_tensors)
        
        # 获取共同的LayerNorm层
        common_ln_layers = set(ln_params1.keys()) & set(ln_params2.keys())
        print(f"找到 {len(common_ln_layers)} 个共同的LayerNorm层")
        
        # 计算差异
        layer_diffs = {}
        
        for layer_name in common_ln_layers:
            # 获取层号
            layer_match = re.search(r"layers\.([0-9]+)", layer_name)
            if layer_match:
                layer_num = int(layer_match.group(1))
            else:
                # 跳过非Transformer层的LayerNorm
                continue
            
            # 获取当前层的参数
            params1 = ln_params1[layer_name]
            params2 = ln_params2[layer_name]
            
            # 计算权重和偏置的差异
            gamma_kl = gamma_wasserstein = beta_kl = beta_wasserstein = 0.0
            
            if params1['gamma'] is not None and params2['gamma'] is not None:
                gamma_kl = self.compute_kl_divergence_estimate(params1['gamma'], params2['gamma'])
                gamma_wasserstein = self.compute_wasserstein_distance(params1['gamma'], params2['gamma'])
            
            if params1['beta'] is not None and params2['beta'] is not None:
                beta_kl = self.compute_kl_divergence_estimate(params1['beta'], params2['beta'])
                beta_wasserstein = self.compute_wasserstein_distance(params1['beta'], params2['beta'])
            
            # 综合权重和偏置的差异（取平均值）
            avg_kl = (gamma_kl + beta_kl) / 2
            avg_wasserstein = (gamma_wasserstein + beta_wasserstein) / 2
            
            # 存储差异
            if layer_num not in layer_diffs:
                layer_diffs[layer_num] = {
                    'layernorm_kl': avg_kl,
                    'layernorm_wasserstein': avg_wasserstein
                }
            else:
                # 如果同一层有多个LayerNorm，取平均值
                layer_diffs[layer_num]['layernorm_kl'] = (layer_diffs[layer_num]['layernorm_kl'] + avg_kl) / 2
                layer_diffs[layer_num]['layernorm_wasserstein'] = (layer_diffs[layer_num]['layernorm_wasserstein'] + avg_wasserstein) / 2
        
        return layer_diffs

class TaskDiffAnalyzer:
    def __init__(self, device=None, alpha=1.0, beta=0.005):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_functions = {
            "Fisher": self.fisher_information_distance,
            "Grassmann": self.geodesic_distance_on_grassmann,
            "L2-norm": torch.norm,
            "Block": torch.norm
        }
        self.norm_params = {
            "Fisher": {},
            "Grassmann": {},
            "L2-norm": {"p": 2},
            "Block": {"p": float("inf")}
        }
        self.alpha = alpha  # 方差权重，默认1.0
        self.beta = beta  # 均衡权重，默认1.0

    def robust_normalize(self, values):
        x = np.asarray(values, dtype=np.float32)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8
        z = np.clip((x - med) / mad, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def smooth_scores(self, values, kernel=(1.0, 2.0, 1.0)):
        x = np.asarray(values, dtype=np.float32)
        k = np.asarray(kernel, dtype=np.float32)
        k = k / k.sum()
        y = np.copy(x)
        for i in range(len(x)):
            weighted_vals = []
            weighted_coeff = []
            for j, w in zip([i - 1, i, i + 1], k):
                if 0 <= j < len(x):
                    weighted_vals.append(x[j] * w)
                    weighted_coeff.append(w)
            y[i] = float(np.sum(weighted_vals) / np.sum(weighted_coeff))
        return y

    def build_variability_prior(self, importance_score, floor=0.05, ceil=1.0):
        x = np.asarray(importance_score, dtype=np.float32)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return floor + (ceil - floor) * x

    def build_lengthscale_prior(self, importance_score):
        values = np.asarray(importance_score, dtype=np.float32).reshape(-1)
        lengthscales = np.empty(values.shape[0], dtype=np.float32)
        if values.size == 0:
            return lengthscales
        order = np.argsort(-values, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(values.shape[0], dtype=np.int32)
        rank_ratio = rank.astype(np.float32) / max(values.shape[0] - 1, 1)
        lengthscales[rank_ratio <= 0.15] = 0.5
        lengthscales[(rank_ratio > 0.15) & (rank_ratio <= 0.40)] = 1.0
        lengthscales[(rank_ratio > 0.40) & (rank_ratio <= 0.70)] = 2.0
        lengthscales[rank_ratio > 0.70] = 5.0
        return lengthscales

    def build_direction_prior(self, act_gap, patch_score):
        signed = 0.5 * self.robust_normalize(act_gap) + 0.5 * self.robust_normalize(patch_score)
        direction = 0.2 + 0.6 * signed
        return np.clip(direction, 0.05, 0.95)

    def load_benchmark_texts(self, dataset_id, split, text_field, limit, subset_field=None, subset_values=None):
        cache_root = os.path.expanduser("~/.cache/evalscope/datasets")
        cache_pattern = os.path.join(cache_root, f"{dataset_id.replace('/', '_')}-*")
        cache_dirs = sorted(glob(cache_pattern))
        if len(cache_dirs) > 0:
            records = []
            for cache_dir in cache_dirs:
                cached_dataset = load_from_disk(cache_dir)
                records.extend(cached_dataset.to_list())
        else:
            dataset = load_dataset(dataset_id, split=split, local_files_only=True)
            records = dataset
        if subset_field is not None and subset_values is not None:
            subset_values_set = set(subset_values)
            if len(records) > 0 and subset_field in records[0]:
                records = [row for row in records if row[subset_field] in subset_values_set]
        texts = []
        limit_value = limit if limit is not None and limit > 0 else None
        for row in records:
            text = row[text_field]
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            if limit_value is not None and len(texts) >= limit_value:
                break
        return texts

    def load_local_benchmark_texts(self, spec):
        local_type = spec["local_type"]
        limit = spec.get("limit")
        limit_value = limit if limit is not None and limit > 0 else None
        texts = []
        if local_type == "bbh_cot_prompts":
            prompt_files = sorted(glob(spec["glob_pattern"]))
            for prompt_file in prompt_files:
                with open(prompt_file, "r", encoding="utf-8") as fp:
                    for line in fp:
                        text = line.strip()
                        if text.startswith("Q: "):
                            texts.append(text[3:].strip())
                            if limit_value is not None and len(texts) >= limit_value:
                                return texts
            return texts
        if local_type == "trivia_qa_samples":
            with open(spec["file_path"], "r", encoding="utf-8") as fp:
                for line in fp:
                    row = json.loads(line)
                    input_messages = row["input"]
                    question = input_messages[-1]["content"].strip()
                    if question:
                        texts.append(question)
                    if limit_value is not None and len(texts) >= limit_value:
                        return texts
            return texts
        raise ValueError(f"Unsupported local benchmark type: {local_type}")

    def build_proxy_prompt_sets(self, reasoning_specs):
        reasoning_texts = []
        for spec in reasoning_specs:
            if spec.get("source") == "local":
                reasoning_texts.extend(
                    self.load_local_benchmark_texts(spec)
                )
            else:
                reasoning_texts.extend(
                    self.load_benchmark_texts(
                        dataset_id=spec["dataset_id"],
                        split=spec["split"],
                        text_field=spec["text_field"],
                        limit=spec["limit"],
                        subset_field=spec.get("subset_field"),
                        subset_values=spec.get("subset_values"),
                    )
                )
        return reasoning_texts

    def get_transformer_layers_from_model(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        raise ValueError("Unsupported transformer architecture: cannot locate decoder layers")

    def load_causal_model_and_tokenizer(self, model_path, dtype_str="float16"):
        torch_dtype = getattr(torch, dtype_str)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def compute_param_delta_scores(self, reason_tensors, base_tensors, num_layers):
        layer_scores = {layer_idx: [] for layer_idx in range(num_layers)}
        common_keys = set(reason_tensors.keys()) & set(base_tensors.keys())
        for key in tqdm(common_keys, desc="Computing parameter delta"):
            layer_type, layer_num = self.get_layer_key_info(key)
            if layer_type != "transformer":
                continue
            tensor_reason = reason_tensors[key].to(dtype=torch.float32)
            tensor_base = base_tensors[key].to(dtype=torch.float32)
            if tensor_reason.shape != tensor_base.shape:
                continue
            delta_norm = torch.norm(tensor_reason - tensor_base, p=2).item()
            base_norm = torch.norm(tensor_base, p=2).item()
            score = delta_norm / (base_norm + 1e-8)
            layer_scores[layer_num].append(score)

        result = np.zeros(num_layers, dtype=np.float32)
        for layer_idx in range(num_layers):
            if len(layer_scores[layer_idx]) > 0:
                result[layer_idx] = float(np.mean(layer_scores[layer_idx]))
        return result

    def pool_hidden_states(self, hidden, attention_mask, pooling):
        if pooling == "last_token":
            last_indices = attention_mask.sum(dim=1) - 1
            last_indices = last_indices.to(device=hidden.device, dtype=torch.long)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_indices, last_indices, :]
        return hidden.mean(dim=1)

    def compute_activation_distance_scores(
        self,
        reason_model,
        base_model,
        tokenizer,
        texts,
        num_layers,
        batch_size=2,
        max_length=256,
        pooling="last_token",
    ):
        scores = np.zeros(num_layers, dtype=np.float32)
        count = 0
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Computing activation distances"):
                batch_texts = texts[start:start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                reason_outputs = reason_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                base_outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                for layer_idx in range(num_layers):
                    reason_h = reason_outputs.hidden_states[layer_idx + 1]
                    base_h = base_outputs.hidden_states[layer_idx + 1]
                    pooled_reason = self.pool_hidden_states(reason_h, attention_mask, pooling)
                    pooled_base = self.pool_hidden_states(base_h, attention_mask, pooling)
                    cosine = torch.nn.functional.cosine_similarity(pooled_reason, pooled_base, dim=-1)
                    distance = 1.0 - cosine
                    scores[layer_idx] += distance.mean().item()
                count += 1
                del input_ids, attention_mask, reason_outputs, base_outputs
                torch.cuda.empty_cache()
        return scores / max(count, 1)

    def compute_patch_scores(
        self,
        reason_model,
        base_model,
        tokenizer,
        texts,
        candidate_layers,
        num_layers,
        batch_size=2,
        max_length=256,
    ):
        score_sum = np.zeros(num_layers, dtype=np.float32)
        score_count = np.zeros(num_layers, dtype=np.float32)
        transformer_layers = self.get_transformer_layers_from_model(base_model)

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Computing patch transport"):
                batch_texts = texts[start:start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                reason_outputs = reason_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                base_outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                reason_logits = reason_outputs.logits[:, -1, :]
                base_logits = base_outputs.logits[:, -1, :]
                baseline_similarity = torch.nn.functional.cosine_similarity(
                    base_logits, reason_logits, dim=-1
                ).mean().item()

                for layer_idx in candidate_layers:
                    patched_hidden = reason_outputs.hidden_states[layer_idx + 1]

                    def patch_hook(module, inputs, output):
                        if isinstance(output, tuple):
                            return (patched_hidden,) + output[1:]
                        return patched_hidden

                    handle = transformer_layers[layer_idx].register_forward_hook(patch_hook)
                    patched_outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    handle.remove()
                    patched_logits = patched_outputs.logits[:, -1, :]
                    patched_similarity = torch.nn.functional.cosine_similarity(
                        patched_logits, reason_logits, dim=-1
                    ).mean().item()
                    score_sum[layer_idx] += patched_similarity - baseline_similarity
                    score_count[layer_idx] += 1.0
                    del patched_outputs, patched_logits

                del input_ids, attention_mask, reason_outputs, base_outputs, reason_logits, base_logits
                torch.cuda.empty_cache()

        patch_scores = np.zeros(num_layers, dtype=np.float32)
        nonzero = score_count > 0
        patch_scores[nonzero] = score_sum[nonzero] / score_count[nonzero]
        return patch_scores

    def combine_proxy_scores(
        self,
        param_score,
        act_reason,
        patch_score,
        w_param=0.35,
        w_act_reason=0.35,
        w_patch_abs=0.30,
        v_act_reason=0.60,
        v_patch=0.40,
    ):
        p = self.robust_normalize(param_score)
        a_reason = self.robust_normalize(act_reason)
        t_signed = self.robust_normalize(patch_score)
        t_abs = self.robust_normalize(np.abs(patch_score))

        fused_score = w_param * p + w_act_reason * a_reason + w_patch_abs * t_abs
        smoothed_score = self.smooth_scores(fused_score)
        importance = np.clip(smoothed_score, 0.05, 1.0)

        m_prior = v_act_reason * a_reason + v_patch * t_signed
        m_prior = self.smooth_scores(m_prior)
        m_prior = np.clip(m_prior, 0.05, 0.95)
        return {
            "param_score_norm": p,
            "act_reason_norm": a_reason,
            "patch_score_norm": t_signed,
            "patch_score_abs_norm": t_abs,
            "fused_score": fused_score,
            "smoothed_score": smoothed_score,
            "importance_score": importance,
            "importance_prior": importance,
            "u_prior": importance,
            "m_prior": m_prior,
        }

    def build_model_proxy_metrics(
        self,
        reason_model_path,
        base_model_path,
        reasoning_specs,
        batch_size=2,
        max_length=256,
        patch_topk=8,
        pooling="last_token",
        dtype_str="float16",
        w_param=0.35,
        w_act_reason=0.35,
        w_patch_abs=0.30,
        v_act_reason=0.60,
        v_patch=0.40,
        output_dir=None,
    ):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        reasoning_texts = self.build_proxy_prompt_sets(reasoning_specs)
        tokenizer, reason_model = self.load_causal_model_and_tokenizer(reason_model_path, dtype_str=dtype_str)
        _, base_model = self.load_causal_model_and_tokenizer(base_model_path, dtype_str=dtype_str)
        num_layers = reason_model.config.num_hidden_layers
        layer_numbers = list(range(num_layers))

        reason_tensors = self.load_model_tensors(reason_model_path)
        base_tensors = self.load_model_tensors(base_model_path)
        param_score = self.compute_param_delta_scores(reason_tensors, base_tensors, num_layers)

        act_reason = self.compute_activation_distance_scores(
            reason_model,
            base_model,
            tokenizer,
            reasoning_texts,
            num_layers=num_layers,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
        )
        top_k = min(patch_topk, num_layers)
        candidate_layers = np.argsort(-act_reason)[:top_k].tolist()
        patch_score = self.compute_patch_scores(
            reason_model,
            base_model,
            tokenizer,
            reasoning_texts,
            candidate_layers=candidate_layers,
            num_layers=num_layers,
            batch_size=batch_size,
            max_length=max_length,
        )
        score_pack = self.combine_proxy_scores(
            param_score=param_score,
            act_reason=act_reason,
            patch_score=patch_score,
            w_param=w_param,
            w_act_reason=w_act_reason,
            w_patch_abs=w_patch_abs,
            v_act_reason=v_act_reason,
            v_patch=v_patch,
        )
        u_prior = score_pack["u_prior"]
        m_prior = score_pack["m_prior"]

        result = {
            "layer_numbers": layer_numbers,
            "reasoning_text_count": len(reasoning_texts),
            "patch_candidate_layers": candidate_layers,
            "param_score": param_score.tolist(),
            "act_score": act_reason.tolist(),
            "act_reason": act_reason.tolist(),
            "patch_score": patch_score.tolist(),
            "param_score_norm": score_pack["param_score_norm"].tolist(),
            "act_reason_norm": score_pack["act_reason_norm"].tolist(),
            "patch_score_norm": score_pack["patch_score_norm"].tolist(),
            "patch_score_abs_norm": score_pack["patch_score_abs_norm"].tolist(),
            "fused_score": score_pack["fused_score"].tolist(),
            "smoothed_score": score_pack["smoothed_score"].tolist(),
            "importance_score": score_pack["importance_score"].tolist(),
            "importance_prior": score_pack["importance_prior"].tolist(),
            "lengthscale": self.build_lengthscale_prior(score_pack["importance_prior"]).tolist(),
            "u_prior": u_prior.tolist(),
            "m_prior": m_prior.tolist(),
        }

        if output_dir:
            with open(os.path.join(output_dir, "model_proxy_metrics.json"), "w") as fp:
                json.dump(result, fp, indent=2, ensure_ascii=False)

        del reason_model, base_model
        torch.cuda.empty_cache()
        return result

    def visualize_proxy_metrics(self, proxy_result, output_dir=None):
        layer_numbers = proxy_result["layer_numbers"]
        if "act_score" not in proxy_result or "smoothed_score" not in proxy_result or "importance_prior" not in proxy_result:
            patch_score = proxy_result.get("patch_score")
            if patch_score is None:
                raise KeyError("proxy_result 缺少 patch_score，无法从旧版结果恢复 smoothed_score")
            score_pack = self.combine_proxy_scores(
                param_score=np.asarray(proxy_result["param_score"], dtype=np.float32),
                act_reason=np.asarray(proxy_result.get("act_score", proxy_result.get("act_reason")), dtype=np.float32),
                patch_score=np.asarray(patch_score, dtype=np.float32),
            )
            proxy_result = dict(proxy_result)
            proxy_result["act_score"] = proxy_result.get("act_reason")
            proxy_result["smoothed_score"] = score_pack["smoothed_score"].tolist()
            proxy_result["importance_prior"] = score_pack["importance_prior"].tolist()
        if "lengthscale" not in proxy_result:
            proxy_result = dict(proxy_result)
            proxy_result["lengthscale"] = self.build_lengthscale_prior(
                np.asarray(proxy_result["importance_prior"], dtype=np.float32)
            ).tolist()

        metric_specs = [
            ("param_score", "Parameter Discrepancy"),
            ("act_score", "Reasoning Activation Discrepancy"),
            ("smoothed_score", "Smoothed Fused Score"),
            ("lengthscale", "Lengthscale"),
        ]
        metric_matrix = np.array([proxy_result[name] for name, _ in metric_specs], dtype=np.float32)
        row_min = metric_matrix.min(axis=1, keepdims=True)
        row_max = metric_matrix.max(axis=1, keepdims=True)
        metric_matrix_norm = (metric_matrix - row_min) / (row_max - row_min + 1e-8)

        plt.figure(figsize=(max(12, len(layer_numbers) * 0.35), max(4.8, 0.7 * len(metric_specs))))
        sns.heatmap(
            metric_matrix_norm,
            cmap="mako",
            yticklabels=[label for _, label in metric_specs],
            xticklabels=layer_numbers,
            cbar=True,
            linewidths=0,
        )
        plt.xlabel("Layer")
        plt.ylabel("")
        plt.tight_layout()
        if output_dir:
            heatmap_path = os.path.join(output_dir, "model_proxy_metrics_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(max(12, len(layer_numbers) * 0.35), 4))
        plt.plot(layer_numbers, proxy_result["smoothed_score"], label="Smoothed Fused Score", linewidth=2.0)
        plt.plot(layer_numbers, proxy_result["importance_prior"], label="Importance Prior", linewidth=2.0)
        plt.xlabel("Layer")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.tight_layout()
        if output_dir:
            curve_path = os.path.join(output_dir, "model_proxy_priors_curve.png")
            plt.savefig(curve_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_optimizer_layer_metrics(
        self,
        saasbo_checkpoint_path,
        priorbo_proxy_path,
        output_path,
        prior_metric_names=None,
        figure_title=None,
    ):
        if prior_metric_names is None:
            prior_metric_names = ["param_score", "act_reason", "patch_score"]

        with open(saasbo_checkpoint_path, "r", encoding="utf-8") as fp:
            saasbo_checkpoint = json.load(fp)
        with open(priorbo_proxy_path, "r", encoding="utf-8") as fp:
            priorbo_proxy = json.load(fp)

        saasbo_importance = np.asarray(saasbo_checkpoint.get("importance", []), dtype=np.float32).reshape(-1)
        if saasbo_importance.size == 0:
            raise ValueError("SAASBO checkpoint 中未找到 importance 字段")

        layer_numbers = priorbo_proxy.get("layer_numbers")
        if layer_numbers is None:
            layer_numbers = list(range(min(len(saasbo_importance), len(priorbo_proxy[prior_metric_names[0]]))))

        aligned_lengths = [len(layer_numbers), len(saasbo_importance)]
        for metric_name in prior_metric_names:
            if metric_name not in priorbo_proxy:
                raise ValueError(f"PriorBO proxy 文件中缺少指标: {metric_name}")
            aligned_lengths.append(len(priorbo_proxy[metric_name]))
        compare_len = int(min(aligned_lengths))
        layer_numbers = layer_numbers[:compare_len]

        row_labels = ["SAASBO importance"] + [f"PriorBO {metric_name}" for metric_name in prior_metric_names]
        raw_rows = [saasbo_importance[:compare_len]]
        raw_rows.extend(np.asarray(priorbo_proxy[name], dtype=np.float32)[:compare_len] for name in prior_metric_names)
        norm_rows = np.stack([self.robust_normalize(row) for row in raw_rows], axis=0)

        plt.style.use("seaborn-v0_8-paper")
        fig = plt.figure(figsize=(max(14, compare_len * 0.32), max(3.8, 0.9 * len(row_labels))))
        gs = fig.add_gridspec(len(row_labels), 2, width_ratios=[0.97, 0.03], hspace=0.12, wspace=0.04)
        axes = []
        for row_idx in range(len(row_labels)):
            share_ax = axes[0] if axes else None
            axes.append(fig.add_subplot(gs[row_idx, 0], sharex=share_ax))
        cbar_ax = fig.add_subplot(gs[:, 1])

        for row_idx, (row_label, row_values) in enumerate(zip(row_labels, norm_rows)):
            sns.heatmap(
                row_values.reshape(1, -1),
                cmap="mako",
                yticklabels=[row_label],
                xticklabels=layer_numbers if row_idx == len(row_labels) - 1 else False,
                cbar=(row_idx == 0),
                cbar_ax=cbar_ax if row_idx == 0 else None,
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
                ax=axes[row_idx],
            )
            axes[row_idx].tick_params(axis="y", labelsize=8, rotation=0)
            axes[row_idx].set_aspect("auto")
            if row_idx != len(row_labels) - 1:
                axes[row_idx].tick_params(axis="x", bottom=False, labelbottom=False)

        cbar_ax.set_ylabel("Row-wise normalized intensity", fontsize=8, labelpad=6)
        cbar_ax.tick_params(labelsize=7)
        axes[-1].set_xlabel("Transformer Layer", fontsize=9)
        axes[-1].tick_params(axis="x", labelrotation=45, labelsize=7)

        if figure_title is None:
            figure_title = "SAASBO vs PriorBO Layer Metrics"
        fig.suptitle(figure_title, fontsize=11, y=0.995)
        fig.subplots_adjust(top=0.90, bottom=0.18, left=0.10, right=0.96)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return {
            "compare_len": compare_len,
            "layer_numbers": layer_numbers,
            "row_labels": row_labels,
            "output_path": output_path,
        }
    
    def fisher_information_distance(self, model1_layer, model2_layer, **kwargs):
        """
        基于Fisher信息的距离
        衡量两个概率分布之间的差异
        
        参数:
            model1_layer: 第一个模型的层张量
            model2_layer: 第二个模型的层张量
            **kwargs: 额外参数（用于兼容接口）
        
        返回:
            float: Fisher-Rao距离
        """
        # 计算均值和标准差
        mu1 = model1_layer.mean().item()
        sigma1 = model1_layer.std().item()
        
        mu2 = model2_layer.mean().item()
        sigma2 = model2_layer.std().item()
        
        # 计算Fisher-Rao距离
        if sigma1 > 0 and sigma2 > 0:
            distance = np.sqrt(2 * np.log((sigma1 + sigma2) / (2 * np.sqrt(sigma1 * sigma2))) +
                              (mu1 - mu2)**2 / (sigma1 + sigma2))
        else:
            # 如果标准差为0，使用绝对距离
            distance = np.abs(mu1 - mu2)
        
        return distance
    
    def geodesic_distance_on_grassmann(self, model1_layer, model2_layer, **kwargs):
        """
        在格拉斯曼流形上计算测地线距离
        适用于比较子空间结构
        
        参数:
            model1_layer: 第一个模型的层张量
            model2_layer: 第二个模型的层张量
            **kwargs: 额外参数（用于兼容接口）
        
        返回:
            float: 格拉斯曼流形上的测地线距离
        """
        import numpy as np
        from scipy.linalg import svd
        
        # 将PyTorch张量转换为NumPy数组
        W1 = model1_layer.cpu().numpy()
        W2 = model2_layer.cpu().numpy()
        
        # 检查是否为2D矩阵
        if W1.ndim != 2 or W2.ndim != 2:
            # 如果不是矩阵，返回0或跳过（这里返回0作为默认值）
            return 0.0
        
        try:
            # 奇异值分解
            U1, S1, V1 = svd(W1, full_matrices=False)
            U2, S2, V2 = svd(W2, full_matrices=False)
            
            # 计算主角度
            cos_theta = svd(U1.T @ U2, compute_uv=False)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            # 测地线距离
            geodesic_dist = np.sqrt(np.sum(theta**2))
            
            return geodesic_dist
        except Exception as e:
            # 处理SVD计算可能出现的异常
            print(f"Warning: Failed to compute Grassmann distance: {e}")
            return 0.0
    
    def load_model_tensors(self, model_path):
        """
        Load all tensors from a model directory
        """
        model_files = glob(os.path.join(model_path, "*.safetensors"))
        if not model_files:
            raise ValueError(f"No safetensors files found in {model_path}")
        
        model_tensors = {}
        for file_path in model_files:
            try:
                print(f"Loading {file_path}...")
                # Load to CPU first to save GPU memory
                tensors = load_file(file_path, device="cpu")
                model_tensors.update(tensors)
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(model_tensors)} tensors from {model_path}")
        return model_tensors
    
    def get_layer_key_info(self, key):
        """
        Extract layer information from tensor key
        """
        # Check for special layer types
        if "embed_tokens" in key or "rotary_emb" in key:
            return "embeddings", -1
        if "norm" in key and ".layers." not in key:
            return "norm", -1
        if "lm_head" in key:
            return "lm_head", -1
        
        # Extract layer number from transformer layers
        layer_match = re.search(r"layers\.([0-9]+)", key)
        if layer_match:
            layer_num = int(layer_match.group(1))
            return "transformer", layer_num
        
        return "other", -1
    
    def calculate_task_diffs(self, model1_tensors, model2_tensors, metric="L2-norm"):
        """
        Calculate differences between two task models
        """
        layer_diffs = {}
        
        with torch.no_grad():
            # Get common keys between both models
            common_keys = set(model1_tensors.keys()) & set(model2_tensors.keys())
            print(f"Found {len(common_keys)} common keys between models")
            
            # Process each common key
            for key in tqdm(common_keys, desc="Processing tensors"):
                # Get layer info
                layer_type, layer_num = self.get_layer_key_info(key)
                
                # Skip non-transformer layers
                if layer_type != "transformer":
                    continue
                
                try:
                    # Get tensors
                    tensor1 = model1_tensors[key].to(dtype=torch.float32, device=self.device)
                    tensor2 = model2_tensors[key].to(dtype=torch.float32, device=self.device)
                    
                    # Check shape match
                    if tensor1.shape != tensor2.shape:
                        if "embed_tokens" in key:
                            try:
                                tensor2 = tensor2[:tensor1.shape[0], :tensor1.shape[1]]
                            except:
                                torch.cuda.empty_cache()
                                continue
                        else:
                            torch.cuda.empty_cache()
                            continue
                    
                    # Calculate norms based on the specified metric
                    layer_diff = {}
                    
                    # 计算基础度量，用于支持Grassmann-Wasserstein混合指标
                    if metric in ["Fisher", "Grassmann-Wasserstein"]:
                        fisher_diff = self.fisher_information_distance(tensor1, tensor2)
                        layer_diff["Fisher"] = fisher_diff
                    
                    if metric in ["Grassmann", "Grassmann-Wasserstein"]:
                        grassmann_diff = self.geodesic_distance_on_grassmann(tensor1, tensor2)
                        layer_diff["Grassmann"] = grassmann_diff
                    
                    if metric in ["L2-norm"]:
                        l2_diff = torch.norm(tensor1 - tensor2, p=2).item()
                        layer_diff["L2-norm"] = l2_diff
                    
                    if metric in ["Block"]:
                        linf_diff = torch.norm(tensor1 - tensor2, p=float("inf")).item()
                        layer_diff["Block"] = linf_diff
                    
                    # LayerNorm-Wasserstein直接在后续处理中计算，这里不需要处理
                    # 对于其他未指定的度量，默认使用L2-norm
                    if not layer_diff:
                        l2_diff = torch.norm(tensor1 - tensor2, p=2).item()
                        layer_diff["L2-norm"] = l2_diff
                    
                    # Store in layer_diffs
                    if layer_num not in layer_diffs:
                        layer_diffs[layer_num] = {}
                    
                    if key not in layer_diffs[layer_num]:
                        layer_diffs[layer_num][key] = layer_diff
                    
                    # Free memory
                    del tensor1, tensor2
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: CUDA OOM for key {key}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        # Average over all tensors in each layer
        avg_layer_diffs = {}
        for layer_num, tensor_diffs in layer_diffs.items():
            # 计算每个指标的平均值
            avg_metrics = {}
            for metric_key in list(tensor_diffs.values())[0].keys():
                avg_metrics[metric_key] = np.mean([tensor_diff[metric_key] for tensor_diff in tensor_diffs.values()])
            
            avg_layer_diffs[layer_num] = avg_metrics
        
        # 添加LayerNorm差异计算
        print(f"\nCalculating LayerNorm differences...")
        ln_analyzer = LayerNormDifferenceAnalyzer(self.device)
        ln_diffs = ln_analyzer.calculate_layernorm_diffs(model1_tensors, model2_tensors)
        
        # 将LayerNorm差异合并到avg_layer_diffs中
        for layer_num, ln_diff in ln_diffs.items():
            if layer_num in avg_layer_diffs:
                avg_layer_diffs[layer_num].update(ln_diff)
            else:
                # 如果该层没有其他差异，只添加LayerNorm差异
                avg_layer_diffs[layer_num] = ln_diff
        
        # 计算Grassmann-Wasserstein混合指标
        if metric == "Grassmann-Wasserstein":
            print(f"\nCalculating Grassmann-Wasserstein hybrid metric...")
            # 收集Grassmann和Wasserstein值用于归一化
            grassmann_values = []
            wasserstein_values = []
            
            for layer_num, diffs in avg_layer_diffs.items():
                if "Grassmann" in diffs and "layernorm_wasserstein" in diffs:
                    grassmann_values.append(diffs["Grassmann"])
                    wasserstein_values.append(diffs["layernorm_wasserstein"])
            
            # 归一化并计算混合指标
            if grassmann_values and wasserstein_values:
                # 归一化Grassmann值
                grassmann_min, grassmann_max = min(grassmann_values), max(grassmann_values)
                # 归一化Wasserstein值
                wasserstein_min, wasserstein_max = min(wasserstein_values), max(wasserstein_values)
                
                for layer_num, diffs in avg_layer_diffs.items():
                    if "Grassmann" in diffs and "layernorm_wasserstein" in diffs:
                        # 归一化到[0, 1]
                        norm_grassmann = (diffs["Grassmann"] - grassmann_min) / (grassmann_max - grassmann_min) if grassmann_max > grassmann_min else 0.5
                        norm_wasserstein = (diffs["layernorm_wasserstein"] - wasserstein_min) / (wasserstein_max - wasserstein_min) if wasserstein_max > wasserstein_min else 0.5
                        
                        # 加权混合：0.9 * Grassmann + 0.1 * Wasserstein
                        grassmann_wasserstein = 0.5 * norm_grassmann + 0.5 * norm_wasserstein
                        
                        # 添加到差异字典中
                        diffs["Grassmann-Wasserstein"] = grassmann_wasserstein
        
        # 处理LayerNorm-Wasserstein作为单独的度量
        if metric == "LayerNorm-Wasserstein":
            print(f"\nUsing LayerNorm-Wasserstein as primary metric...")
            # 确保每个层都有layernorm_wasserstein值
            for layer_num, diffs in avg_layer_diffs.items():
                if "layernorm_wasserstein" in diffs:
                    # 将LayerNorm-Wasserstein值复制到主度量位置
                    diffs["LayerNorm-Wasserstein"] = diffs["layernorm_wasserstein"]
        
        return avg_layer_diffs
    
    def hybrid_optimization(self, values, k, alpha=1.0, beta=1.0):
        """
        General Optimization Strategy using DP.
        Cost = alpha * Variance_Cost + beta * Balance_Cost
        
        - alpha=0, beta=1: Pure Balance (Equal Info)
        - alpha=1, beta=0: Pure Variance (Fisher-Jenks / Homogeneity)
        - alpha=1, beta=2: Hybrid
        """
        n = len(values)
        if k >= n: return list(range(n + 1))
        
        prefix_sum = np.concatenate(([0], np.cumsum(values)))
        target_block_sum = prefix_sum[n] / k
        
        # Precompute variance cost (SSE)
        var_cost = np.zeros((n, n))
        for i in range(n):
            s1 = s2 = 0
            for j in range(i, n):
                val = values[j]
                s1 += val
                s2 += val * val
                var = s2 - (s1 * s1) / (j - i + 1)
                var_cost[i, j] = var

        dp = np.full((n + 1, k + 1), np.inf)
        path = np.zeros((n + 1, k + 1), dtype=int)
        dp[0, 0] = 0

        for j in range(1, k + 1):
            for i in range(1, n + 1):
                for m in range(i):
                    # Cost components
                    c_var = var_cost[m, i - 1]
                    block_sum = prefix_sum[i] - prefix_sum[m]
                    c_balance = (block_sum - target_block_sum) ** 2
                    
                    # Weighted total cost
                    total_cost = (alpha * c_var) + (beta * c_balance)
                    
                    if dp[m, j - 1] + total_cost < dp[i, j]:
                        dp[i, j] = dp[m, j - 1] + total_cost
                        path[i, j] = m

        cuts = [n]
        curr = n
        for j in range(k, 0, -1):
            curr = path[curr, j]
            cuts.append(curr)
        return sorted(cuts)
    
    def prepare_visualization_data(self, layer_diffs, num_blocks=8):
        """
        Prepare data for visualization with current metric
        """
        all_layers = sorted(layer_diffs.keys())
        
        # 获取当前使用的metric（假设layer_diffs中只有一个metric）
        current_metric = list(layer_diffs[all_layers[0]].keys())[0]
        
        # Define row labels
        norm_types = [current_metric, "Balance", "Variance", "Hybrid"]
        
        num_layers = len(all_layers)
        data_matrix = np.zeros((4, num_layers))
        
        # 1. Fill current metric (Row 0)
        for j, layer_num in enumerate(all_layers):
            data_matrix[0, j] = layer_diffs[layer_num][current_metric]
        
        # Normalize current metric
        normalized_matrix = np.zeros((1, num_layers))
        row = data_matrix[0, :]
        row_min, row_max = np.min(row), np.max(row)
        if row_max > row_min:
            normalized_matrix[0, :] = (row - row_min) / (row_max - row_min)
        else:
            normalized_matrix[0, :] = row
        data_matrix[0:1, :] = normalized_matrix
        
        # Input for optimization algorithms
        total_normalized_diffs = normalized_matrix.sum(axis=0)
        
        # 2. Run Strategies
        print(f"Running Optimization Strategies for {num_blocks} blocks...")
        
        # Strategy 1: Pure Balance (Row 1) - alpha=0, beta=1
        cuts_balance = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=0.0, beta=1.0)
        
        # Strategy 2: Pure Variance (Row 2) - alpha=1, beta=0
        cuts_variance = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=1.0, beta=0.0)
        
        # Strategy 3: Hybrid (Row 3) - 使用实例的 alpha 和 beta 值
        cuts_hybrid = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=self.alpha, beta=self.beta)
        
        # 3. Fill Matrix Blocks
        strategies = [
            (1, cuts_balance),
            (2, cuts_variance),
            (3, cuts_hybrid)
        ]
        
        for row_idx, cuts in strategies:
            for i in range(len(cuts) - 1):
                start, end = cuts[i], cuts[i+1]
                # Fill with block index (for coloring)
                for k in range(start, end):
                    data_matrix[row_idx, k] = i
        
        # 4. Prepare return value (Hybrid blocks for downstream usage)
        merged_blocks = []
        for i in range(len(cuts_hybrid) - 1):
            start_idx, end_idx = cuts_hybrid[i], cuts_hybrid[i+1]
            block_layers = [all_layers[k] for k in range(start_idx, end_idx)]
            block_diff = np.sum(total_normalized_diffs[start_idx:end_idx])
            merged_blocks.append((block_layers, block_diff))
            
        return data_matrix, norm_types, all_layers, merged_blocks
    
    def save_layer_sorted_result(self, layer_diffs, output_path, num_blocks=8):
        """
        Save layer sorted result to txt file, sorted by merged block differences
        """
        # Get all unique layer numbers (sorted by layer number)
        all_layers = sorted(layer_diffs.keys())
        
        # 获取当前使用的metric（假设layer_diffs中只有一个metric）
        current_metric = list(layer_diffs[all_layers[0]].keys())[0]
        
        # Only use current metric
        norm_types = [current_metric]  # 只使用当前metric
        
        # Calculate total difference for each layer (sum of current metric)
        layer_total_diffs = {}
        for layer_num, diffs in layer_diffs.items():
            total_diff = sum(diffs[norm] for norm in norm_types)
            layer_total_diffs[layer_num] = total_diff
        
        # Calculate total normalized difference for each layer
        # First calculate normalized values
        normalized_diffs = {}
        for norm in norm_types:
            norm_values = [layer_diffs[layer][norm] for layer in all_layers]
            norm_min = min(norm_values)
            norm_max = max(norm_values)
            if norm_max > norm_min:
                normalized_diffs[norm] = [(v - norm_min) / (norm_max - norm_min) for v in norm_values]
            else:
                normalized_diffs[norm] = [0.0 for _ in norm_values]
        
        # Calculate total normalized difference for each layer
        total_normalized_diffs = [sum(normalized_diffs[norm][i] for norm in norm_types) for i in range(len(all_layers))]
        
        # Run Hybrid Optimization for the report, using instance's alpha and beta values
        cuts = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=self.alpha, beta=self.beta)
        
        merged_blocks = []
        for i in range(len(cuts) - 1):
            start_idx, end_idx = cuts[i], cuts[i+1]
            block_layers = [all_layers[k] for k in range(start_idx, end_idx)]
            block_diff = sum(total_normalized_diffs[start_idx:end_idx])
            block_total_raw_diff = sum(sum(layer_diffs[l][norm] for norm in norm_types) for l in block_layers)
            merged_blocks.append((block_layers, block_diff, block_total_raw_diff))
        
        merged_blocks.sort(key=lambda x: x[2], reverse=True)
        
        # Write to txt file
        with open(output_path, 'w') as f:
            f.write("Layer Blocks (Hybrid Strategy) Sorted by Difference\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Block':<8} {'Layers':<20} {'Total Difference':<20} {'Block Size':<12}\n")
            f.write("-" * 60 + "\n")
            for block_idx, (block_layers, block_diff, block_total_diff) in enumerate(merged_blocks):
                layer_range = f"{block_layers[0]}-{block_layers[-1]}" if len(block_layers) > 1 else f"{block_layers[0]}"
                f.write(f"{block_idx+1:<8} {layer_range:<20} {block_total_diff:<20.6f} {len(block_layers):<12}\n")
        
        print(f"Layer sorted result saved to: {output_path}")
    
    def plot_horizontal_heatmap(self, data_matrix, norm_types, layer_numbers, output_path=None):
        """
        Plot horizontal heatmap with variable number of rows
        """
        num_rows = data_matrix.shape[0]
        
        # Adjust figure height based on number of rows (approx 0.7 inch per row + margins)
        plt.figure(figsize=(6, 0.7 * num_rows))
        
        # GridSpec with dynamic rows
        gs = plt.GridSpec(num_rows, 2, width_ratios=[0.95, 0.05], 
                         height_ratios=[1] * num_rows, wspace=0.03, hspace=0.1)
        
        axes = []
        # Create axes sharing x-axis
        ax0 = plt.subplot(gs[0, 0])
        axes.append(ax0)
        for i in range(1, num_rows):
            axes.append(plt.subplot(gs[i, 0], sharex=ax0))
            
        # Colorbar next to L2 row (index 1) if it exists
        cbar_ax = plt.subplot(gs[1, 1]) if num_rows > 1 else None
        
        plt.style.use('seaborn-v0_8-paper')
        
        for i, row_label in enumerate(norm_types):
            # Determine if this is a Block row (Indices 2+)
            is_block_row = i >= 2
            
            if is_block_row:
                # Discrete colormap for blocks
                unique_blocks = np.unique(data_matrix[i, :])
                # 使用高对比度的离散颜色映射
                color_palettes = ["tab20", "tab20b", "tab20c"]
                # 根据需要组合多个颜色映射以获得足够的颜色
                all_colors = []
                for palette in color_palettes:
                    all_colors.extend(sns.color_palette(palette))
                # 确保有足够的颜色
                while len(all_colors) < len(unique_blocks):
                    # 如果还不够，循环使用现有颜色
                    all_colors.extend(all_colors[:len(unique_blocks) - len(all_colors)])
                # 创建自定义颜色映射
                custom_cmap = sns.color_palette(all_colors[:len(unique_blocks)], as_cmap=True)
                
                sns.heatmap(
                    data_matrix[i:i+1, :],
                    annot=False,
                    cmap=custom_cmap,
                    xticklabels=layer_numbers,
                    yticklabels=[row_label],
                    cbar=False,
                    square=False,
                    linewidths=0,  # 移除块之间的分隔线
                    ax=axes[i],
                    vmin=0, vmax=max(unique_blocks) if len(unique_blocks) > 0 else 1
                )
            elif i == 0:  # L1 row (No colorbar)
                sns.heatmap(
                    data_matrix[i:i+1, :],
                    annot=False,
                    cmap="viridis",
                    xticklabels=False,
                    yticklabels=[row_label],
                    cbar=False,
                    square=False,
                    linewidths=0,
                    ax=axes[i],
                    vmin=0, vmax=1
                )
            elif i == 1:  # L2 row (With colorbar)
                sns.heatmap(
                    data_matrix[i:i+1, :],
                    annot=False,
                    cmap="viridis",
                    xticklabels=False,
                    yticklabels=[row_label],
                    cbar=True,
                    cbar_ax=cbar_ax,
                    cbar_kws={"label": "Norm Diff", "shrink": 1.0}, # Adjusted shrink
                    square=False,
                    linewidths=0,
                    ax=axes[i],
                    vmin=0, vmax=1
                )
        
        # Colorbar styling
        if cbar_ax is not None:
            cbar_ax.set_ylabel('Norm Diff', fontsize=7, labelpad=5)
            cbar_ax.tick_params(labelsize=6)
        
        # Axis styling
        for ax in axes:
            ax.tick_params(axis='y', labelsize=7, rotation=0) # Ensure y-labels are horizontal
            ax.set_aspect("auto")
            # Hide x-ticks for all but the last row
            if ax != axes[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
        
        # Bottom label
        axes[-1].set_xlabel("Transformer Layer", fontsize=9)
        plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
        
        plt.tight_layout()
        # Adjust margins
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.92)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def visualize_task_diffs(self, model1_path, model2_path, output_dir=None, num_blocks=8, metric="L2-norm"):
        """
        Visualize differences between two task models with automatic layer merging
        Returns: merged_blocks - list of tuples containing (layer_numbers, block_diff)
        """
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load models
        print(f"Loading model 1 from {model1_path}...")
        model1_tensors = self.load_model_tensors(model1_path)
        
        print(f"\nLoading model 2 from {model2_path}...")
        model2_tensors = self.load_model_tensors(model2_path)
        
        # Calculate task differences
        print(f"\nCalculating differences between models using {metric} metric...")
        layer_diffs = self.calculate_task_diffs(model1_tensors, model2_tensors, metric=metric)
        
        # Save layer sorted result to txt file, sorted by merged block differences
        print(f"\nSaving layer sorted result...")
        sorted_result_path = os.path.join(output_dir, "task_diff_layer_sorted.txt") if output_dir else "task_diff_layer_sorted.txt"
        self.save_layer_sorted_result(layer_diffs, sorted_result_path, num_blocks)
        
        # Prepare visualization data with automatic layer merging
        print(f"\nPreparing visualization data with {num_blocks} merged blocks...")
        data_matrix, norm_types, layer_numbers, merged_blocks = self.prepare_visualization_data(layer_diffs, num_blocks)
        
        # Plot heatmap
        print(f"\nPlotting heatmap...")
        output_path = os.path.join(output_dir, "task_diff_heatmap.png") if output_dir else None
        self.plot_horizontal_heatmap(data_matrix, norm_types, layer_numbers, output_path)
        
        print(f"\nAutomatic layer merging completed. {num_blocks} blocks created.")
        return merged_blocks
    
    def generate_multiple_block_configs(self, model1_path, model2_path, block_numbers, output_dir=None, metric="L2-norm", partition_method="hybrid"):
        """
        Generate multiple block configurations from fine to coarse, loading models only once.
        
        Args:
            model1_path: Path to the first model
            model2_path: Path to the second model
            block_numbers: List of block numbers in ascending order (e.g., [6, 12, 24, 36])
            output_dir: Directory to save visualization results
            metric: Distance metric to use for calculating layer differences
            partition_method: Partition method to use ("hybrid", "balance", "variance")
            
        Returns:
            Dict[int, List[Tuple[List[int], float]]]: Dictionary mapping block numbers to their merged blocks
        """
        # Validate inputs
        if not os.path.exists(model1_path):
            raise ValueError(f"Model 1 path does not exist: {model1_path}")
        
        if not os.path.exists(model2_path):
            raise ValueError(f"Model 2 path does not exist: {model2_path}")
        
        if not isinstance(block_numbers, list) or len(block_numbers) < 2:
            raise ValueError("block_numbers must be a list with at least 2 elements")
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load models once
        print(f"Loading model 1 from {model1_path}...")
        model1_tensors = self.load_model_tensors(model1_path)
        
        print(f"\nLoading model 2 from {model2_path}...")
        model2_tensors = self.load_model_tensors(model2_path)
        
        # Calculate task differences once with specified metric
        print(f"\nCalculating differences between models using {metric} metric...")
        layer_diffs = self.calculate_task_diffs(model1_tensors, model2_tensors, metric=metric)
        
        # Get all unique layer numbers (sorted by layer number)
        all_layers = sorted(layer_diffs.keys())
        num_layers = len(all_layers)
        
        # Calculate total normalized difference for each layer
        # 只使用指定的metric来计算差异
        norm_types = [metric]  # 只使用指定的度量
        normalized_diffs = {}
        for norm in norm_types:
            norm_values = [layer_diffs[layer].get(norm, 0.0) for layer in all_layers]
            norm_min = min(norm_values)
            norm_max = max(norm_values)
            if norm_max > norm_min:
                normalized_diffs[norm] = [(v - norm_min) / (norm_max - norm_min) for v in norm_values]
            else:
                normalized_diffs[norm] = [0.0 for _ in norm_values]
        total_normalized_diffs = [sum(normalized_diffs[norm][i] for norm in norm_types) for i in range(len(all_layers))]
        
        # Generate block configurations from fine to coarse
        block_configs = {}
        
        # Sort block numbers in ascending order (fine to coarse)
        sorted_block_numbers = sorted(block_numbers)
        
        # Generate the finest partition first
        finest_blocks = sorted_block_numbers[-1]
        print(f"\nGenerating finest partition: {finest_blocks} blocks...")
        
        # Get data for finest partition
        data_matrix, _, _, merged_blocks = self.prepare_visualization_data(layer_diffs, finest_blocks)
        block_configs[finest_blocks] = merged_blocks
        
        # Generate coarser partitions using the previous partition as input
        for i in range(len(sorted_block_numbers) - 2, -1, -1):
            current_blocks = sorted_block_numbers[i]
            previous_blocks = sorted_block_numbers[i + 1]
            
            print(f"\nGenerating partition: {current_blocks} blocks (from {previous_blocks} blocks)...")
            
            # Get the previous partition
            prev_merged_blocks = block_configs[previous_blocks]
            
            # Convert previous merged blocks to block boundaries
            prev_block_boundaries = []
            current_layer_idx = 0
            for block_layers, _ in prev_merged_blocks:
                prev_block_boundaries.append(current_layer_idx)
                current_layer_idx += len(block_layers)
            prev_block_boundaries.append(current_layer_idx)
            
            # Calculate block-level differences
            block_diff_values = []
            for block_idx in range(len(prev_merged_blocks)):
                start = prev_block_boundaries[block_idx]
                end = prev_block_boundaries[block_idx + 1]
                block_diff = sum(total_normalized_diffs[start:end])
                block_diff_values.append(block_diff)
            
            # Apply optimization based on partition method
            if partition_method == "balance":
                cuts = self.hybrid_optimization(block_diff_values, current_blocks, alpha=0.0, beta=1.0)
            elif partition_method == "variance":
                cuts = self.hybrid_optimization(block_diff_values, current_blocks, alpha=1.0, beta=0.0)
            else:  # hybrid
                cuts = self.hybrid_optimization(block_diff_values, current_blocks, alpha=self.alpha, beta=self.beta)
            
            # Convert cuts from block indices to layer indices
            new_block_boundaries = [prev_block_boundaries[cut] for cut in cuts]
            
            # Generate new merged blocks based on the new boundaries
            new_merged_blocks = []
            for j in range(len(new_block_boundaries) - 1):
                start_idx, end_idx = new_block_boundaries[j], new_block_boundaries[j + 1]
                block_layers = all_layers[start_idx:end_idx]
                block_diff = sum(total_normalized_diffs[start_idx:end_idx])
                new_merged_blocks.append((block_layers, block_diff))
            
            block_configs[current_blocks] = new_merged_blocks
        
        # Generate combined heatmap with L1, L2, and all block configurations
        self.generate_combined_block_heatmap(
            layer_diffs, 
            block_configs, 
            all_layers, 
            output_dir=output_dir
        )
        
        # Save individual visualizations for each block configuration
        for num_blocks in block_configs.keys():
            if output_dir:
                # Prepare data matrix for visualization
                data_matrix, norm_types, layer_numbers, _ = self.prepare_visualization_data(layer_diffs, num_blocks)
                output_path = os.path.join(output_dir, f"task_diff_heatmap_{num_blocks}.png")
                self.plot_horizontal_heatmap(data_matrix, norm_types, layer_numbers, output_path)
                
                sorted_result_path = os.path.join(output_dir, f"task_diff_layer_sorted_{num_blocks}.txt")
                self.save_layer_sorted_result(layer_diffs, sorted_result_path, num_blocks)
        
        return block_configs
    
    def generate_combined_block_heatmap(self, layer_diffs, block_configs, all_layers, output_dir=None):
        """
        Generate a combined heatmap with dynamic rows:
        Row 1: Current metric
        Row 2: Block configuration 1 (finest)
        Row 3: Block configuration 2
        Row 4: Block configuration 3
        Row 5: Block configuration 4 (coarsest)
        
        Args:
            layer_diffs: Layer differences dictionary
            block_configs: Dictionary mapping block numbers to merged blocks
            all_layers: List of all layer numbers
            output_dir: Directory to save the visualization
        """
        print("\nGenerating combined block heatmap...")
        
        # Get current metric from layer_diffs
        current_metric = list(layer_diffs[all_layers[0]].keys())[0]
        
        # Sort block numbers in descending order (finest to coarsest)
        sorted_block_nums = sorted(block_configs.keys(), reverse=True)
        
        # Calculate total normalized difference for each layer using current metric
        norm_types = [current_metric]
        normalized_diffs = {}
        for norm in norm_types:
            norm_values = [layer_diffs[layer][norm] for layer in all_layers]
            norm_min = min(norm_values)
            norm_max = max(norm_values)
            if norm_max > norm_min:
                normalized_diffs[norm] = [(v - norm_min) / (norm_max - norm_min) for v in norm_values]
            else:
                normalized_diffs[norm] = [0.0 for _ in norm_values]
        
        # Create combined data matrix with 5 rows (1 metric + 4 block configs)
        num_layers = len(all_layers)
        combined_data = np.zeros((5, num_layers))
        
        # Row 0: Current metric
        combined_data[0, :] = normalized_diffs[current_metric]
        
        # Rows 1-4: Block configurations (finest to coarsest)
        for i, num_blocks in enumerate(sorted_block_nums[:4]):
            merged_blocks = block_configs[num_blocks]
            
            # Create layer-to-block mapping
            layer_block_map = {}
            for block_idx, (block_layers, _) in enumerate(merged_blocks):
                for layer in block_layers:
                    layer_idx = all_layers.index(layer)
                    layer_block_map[layer_idx] = block_idx
            
            # Fill in the data for this block configuration
            for layer_idx in range(num_layers):
                combined_data[1 + i, layer_idx] = layer_block_map.get(layer_idx, 0)
        
        # Create row labels with dynamic block configurations
        row_labels = [
            current_metric
        ]
        
        # Add labels only for the actual number of block configurations
        for num_blocks in sorted_block_nums[:4]:
            row_labels.append(f"Blocks-{num_blocks}")
        
        # Fill remaining rows with empty labels if needed
        while len(row_labels) < 5:
            row_labels.append("")
        
        # Plot the combined heatmap
        if output_dir:
            output_path = os.path.join(output_dir, "task_diff_heatmap_combined.png")
            self.plot_horizontal_heatmap(combined_data, row_labels, all_layers, output_path)
        else:
            self.plot_horizontal_heatmap(combined_data, row_labels, all_layers)
    
    def run(self, model1_path, model2_path, output_dir=None, num_blocks=8, metric="L2-norm", partition_method="hybrid"):
        """
        Run the complete visualization pipeline with automatic layer merging
        Returns: merged_blocks - list of tuples containing (layer_numbers, block_diff)
        """
        # Validate inputs
        if not os.path.exists(model1_path):
            raise ValueError(f"Model 1 path does not exist: {model1_path}")
        
        if not os.path.exists(model2_path):
            raise ValueError(f"Model 2 path does not exist: {model2_path}")
        
        # Run visualization with automatic layer merging and return merged blocks
        merged_blocks = self.visualize_task_diffs(model1_path, model2_path, output_dir, num_blocks, metric=metric)
        return merged_blocks

    def run_model_proxy(
        self,
        reason_model_path,
        base_model_path,
        output_dir,
        reasoning_specs,
        batch_size=2,
        max_length=256,
        patch_topk=8,
        pooling="last_token",
        dtype_str="float16",
        w_param=0.35,
        w_act_reason=0.35,
        w_patch_abs=0.30,
        v_act_reason=0.60,
        v_patch=0.40,
    ):
        proxy_result = self.build_model_proxy_metrics(
            reason_model_path=reason_model_path,
            base_model_path=base_model_path,
            reasoning_specs=reasoning_specs,
            batch_size=batch_size,
            max_length=max_length,
            patch_topk=patch_topk,
            pooling=pooling,
            dtype_str=dtype_str,
            w_param=w_param,
            w_act_reason=w_act_reason,
            w_patch_abs=w_patch_abs,
            v_act_reason=v_act_reason,
            v_patch=v_patch,
            output_dir=output_dir,
        )
        self.visualize_proxy_metrics(proxy_result, output_dir=output_dir)
        return proxy_result


def main():
    """
    Main function to run the visualizer
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Model Difference Visualizer")
    parser.add_argument("--mode", choices=["diff", "proxy", "optimizer_compare"], default="diff")
    parser.add_argument("--model1", default=None, help="Path to first task model directory")
    parser.add_argument("--model2", default=None, help="Path to second task model directory")
    parser.add_argument("--output", default=None, help="Output directory for heatmap")
    parser.add_argument("--device", default=None, help="Device to use (cpu or cuda)")
    parser.add_argument("--metric", type=str, default="L2-norm", help="Distance metric to use: Fisher, Grassmann, L2-norm, Block")
    parser.add_argument("--partition-method", type=str, default="hybrid", help="Partition method to use: hybrid, balance, variance")
    parser.add_argument("--num-blocks", type=int, default=8, help="Number of blocks to merge into")
    parser.add_argument("--proxy-batch-size", type=int, default=2)
    parser.add_argument("--proxy-max-length", type=int, default=256)
    parser.add_argument("--proxy-patch-topk", type=int, default=8)
    parser.add_argument("--proxy-pooling", choices=["last_token", "mean"], default="last_token")
    parser.add_argument("--proxy-dtype", default="float16")
    parser.add_argument("--reasoning-limit", type=int, default=30)
    parser.add_argument("--saasbo-checkpoint", default=None, help="Path to SAASBO checkpoint JSON")
    parser.add_argument("--priorbo-proxy", default=None, help="Path to PriorBO model_proxy_metrics.json")
    parser.add_argument("--compare-metrics", nargs="+", default=["param_score", "act_reason", "patch_score"])
    
    args = parser.parse_args()
    
    # Create visualizer instance
    visualizer = TaskDiffAnalyzer(device=args.device)
    
    if args.mode == "diff":
        if args.model1 is None or args.model2 is None:
            raise ValueError("diff 模式需要提供 --model1 和 --model2")
        visualizer.run(args.model1, args.model2, args.output, args.num_blocks, args.metric, args.partition_method)
    elif args.mode == "proxy":
        if args.model1 is None or args.model2 is None:
            raise ValueError("proxy 模式需要提供 --model1 和 --model2")
        reasoning_specs = [
            {
                "dataset_id": "opencompass/AIME2025",
                "split": "test",
                "text_field": "question",
                "subset_field": "subset",
                "subset_values": ["AIME2025-I", "AIME2025-II"],
                "limit": args.reasoning_limit,
            },
            {
                "dataset_id": "AI-ModelScope/gpqa_diamond",
                "split": "train",
                "text_field": "Question",
                "limit": args.reasoning_limit,
            },
        ]
        visualizer.run_model_proxy(
            reason_model_path=args.model2,
            base_model_path=args.model1,
            output_dir=args.output,
            reasoning_specs=reasoning_specs,
            batch_size=args.proxy_batch_size,
            max_length=args.proxy_max_length,
            patch_topk=args.proxy_patch_topk,
            pooling=args.proxy_pooling,
            dtype_str=args.proxy_dtype,
        )
    else:
        if args.saasbo_checkpoint is None or args.priorbo_proxy is None or args.output is None:
            raise ValueError("optimizer_compare 模式需要提供 --saasbo-checkpoint、--priorbo-proxy 和 --output")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        visualizer.visualize_optimizer_layer_metrics(
            saasbo_checkpoint_path=args.saasbo_checkpoint,
            priorbo_proxy_path=args.priorbo_proxy,
            output_path=args.output,
            prior_metric_names=args.compare_metrics,
        )

if __name__ == "__main__":
    main()
