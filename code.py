from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import heapq
import random

GPU_MEM_SIZE = 80  # GB

@dataclass
class Model:
    model_name: str
    model_size: int   # GB
    req_rate: int     # requests per second
    slo: int          # service level objective (latency target)
    cur_gpu_id: int   # current GPU assignment (can be ignored)

def compute_model_placement(gpu_num: int, models: List[Model]) -> Dict[int, List[Model]]:
    if not models:
        return {g: [] for g in range(gpu_num)}
    
    n = len(models)
    # Precompute per-model weight (req_rate / slo)
    weights = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]
    
    # Initial assignment: probabilistic greedy by high pressure-to-space ratio
    placement: Dict[int, List[int]] = {g: [] for g in range(gpu_num)}
    used_memory = [0] * gpu_num
    used_weights = [0.0] * gpu_num
    
    # Sort models by (weight / (M - size), size, weight) descending
    order = sorted(range(n), key=lambda i: (weights[i] / max(GPU_MEM_SIZE - sizes[i], 1e-9), sizes[i], weights[i]), reverse=True)
    
    # Use decaying temperature for softmax selection
    temp = 1.0
    decay = 0.95
    for i in order:
        candidates = []
        for g in range(gpu_num):
            if sizes[i] + used_memory[g] <= GPU_MEM_SIZE:
                new_used_mem = used_memory[g] + sizes[i]
                new_used_w = used_weights[g] + weights[i]
                if new_used_mem == GPU_MEM_SIZE:
                    continue
                kvpr_g = new_used_w / (GPU_MEM_SIZE - new_used_mem)
                candidates.append((kvpr_g, g))
        
        if not candidates:
            # Fallback: best-fit by space
            best_gpu = max(range(gpu_num), key=lambda g: GPU_MEM_SIZE - used_memory[g] if sizes[i] <= GPU_MEM_SIZE - used_memory[g] else -1)
            if sizes[i] > GPU_MEM_SIZE - used_memory[best_gpu]:
                raise ValueError("No feasible placement found.")
            placement[best_gpu].append(i)
            used_memory[best_gpu] += sizes[i]
            used_weights[best_gpu] += weights[i]
            temp *= decay
            continue
            
        # Softmax over KVPR scores (lower is better)
        kvprs = [score for score, _ in candidates]
        min_kvpr = min(kvprs)
        # Convert to probabilities: lower KVPR → higher probability
        exp_scores = [max(0.0, min(1e5, 1.0 - (s - min_kvpr) / (1e-6 + max(1e-6, max(kvprs) - min_kvpr)))) for s in kvprs]
        total = sum(exp_scores)
        probs = [s / total for s in exp_scores]
        
        # Sample GPU using softmax probabilities
        selected_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        best_gpu = candidates[selected_idx][1]
        
        placement[best_gpu].append(i)
        used_memory[best_gpu] += sizes[i]
        used_weights[best_gpu] += weights[i]
        temp *= decay
    
    # Local search: try to improve by swapping models between GPUs
    # Use a heap to prioritize high-impact swaps
    improvements = []
    # Only consider swaps involving models from top-3 highest KVPR GPUs
    kvpr_scores = [
        used_weights[g] / (GPU_MEM_SIZE - used_memory[g]) if used_memory[g] < GPU_MEM_SIZE else 0.0
        for g in range(gpu_num)
    ]
    top_gpu_indices = sorted(range(gpu_num), key=lambda g: kvpr_scores[g], reverse=True)[:3]
    
    for g in top_gpu_indices:
        for i_idx in placement[g]:
            for g2 in range(gpu_num):
                if g == g2:
                    continue
                for j_idx in placement[g2]:
                    # Check if swapping improves max KVPR
                    if sizes[i_idx] + used_memory[g2] > GPU_MEM_SIZE or sizes[j_idx] + used_memory[g] > GPU_MEM_SIZE:
                        continue
                    # Calculate new KVPRs after swap
                    # GPU g before and after
                    old_kv_g = used_weights[g] / (GPU_MEM_SIZE - used_memory[g])
                    new_w_g = used_weights[g] - weights[i_idx] + weights[j_idx]
                    new_m_g = used_memory[g] - sizes[i_idx] + sizes[j_idx]
                    new_kv_g = new_w_g / (GPU_MEM_SIZE - new_m_g)
                    # GPU g2 before and after
                    old_kv_g2 = used_weights[g2] / (GPU_MEM_SIZE - used_memory[g2])
                    new_w_g2 = used_weights[g2] - weights[j_idx] + weights[i_idx]
                    new_m_g2 = used_memory[g2] - sizes[j_idx] + sizes[i_idx]
                    new_kv_g2 = new_w_g2 / (GPU_MEM_SIZE - new_m_g2)
                    # Check if max KVPR decreases
                    old_max = max(old_kv_g, old_kv_g2)
                    new_max = max(new_kv_g, new_kv_g2)
                    if new_max < old_max - 1e-9:
                        heapq.heappush(improvements, (new_max - old_max, g, i_idx, g2, j_idx))
    
    # Apply top improvements if they improve the solution
    seen = set()
    while improvements:
        diff, g1, i, g2, j = heapq.heappop(improvements)
        state_key = (g1, i, g2, j)
        if state_key in seen:
            continue
        seen.add(state_key)
        # Perform swap
        placement[g1].remove(i)
        placement[g2].remove(j)
        placement[g1].append(j)
        placement[g2].append(i)
        used_memory[g1] = sum(sizes[idx] for idx in placement[g1])
        used_memory[g2] = sum(sizes[idx] for idx in placement[g2])
        used_weights[g1] = sum(weights[idx] for idx in placement[g1])
        used_weights[g2] = sum(weights[idx] for idx in placement[g2])
        
        # Re-scan for new improvements, but only from top-3 GPUs
        kvpr_scores = [
            used_weights[g] / (GPU_MEM_SIZE - used_memory[g]) if used_memory[g] < GPU_MEM_SIZE else 0.0
            for g in range(gpu_num)
        ]
        top_gpu_indices = sorted(range(gpu_num), key=lambda g: kvpr_scores[g], reverse=True)[:3]
        for g in top_gpu_indices:
            for idx1 in placement[g]:
                for g2 in range(gpu_num):
                    if g == g2:
                        continue
                    for idx2 in placement[g2]:
                        if sizes[idx1] + used_memory[g2] > GPU_MEM_SIZE or sizes[idx2] + used_memory[g] > GPU_MEM_SIZE:
                            continue
                        new_w_g1 = used_weights[g] - weights[idx1] + weights[idx2]
                        new_m_g1 = used_memory[g] - sizes[idx1] + sizes[idx2]
                        new_kv_g1 = new_w_g1 / (GPU_MEM_SIZE - new_m_g1)
                        new_w_g2 = used_weights[g2] - weights[idx2] + weights[idx1]
                        new_m_g2 = used_memory[g2] - sizes[idx2] + sizes[idx1]
                        new_kv_g2 = new_w_g2 / (GPU_MEM_SIZE - new_m_g2)
                        old_max = max(used_weights[g] / (GPU_MEM_SIZE - used_memory[g]), used_weights[g2] / (GPU_MEM_SIZE - used_memory[g2]))
                        new_max = max(new_kv_g1, new_kv_g2)
                        if new_max < old_max - 1e-9:
                            heapq.heappush(improvements, (new_max - old_max, g, idx1, g2, idx2))
    
    # Map back to Model objects
    result: Dict[int, List[Model]] = {g: [] for g in range(gpu_num)}
    for g in range(gpu_num):
        result[g] = [models[i] for i in placement[g]]
    return result