#!/usr/bin/env python3
"""
Synapse Forge - Autonomous Cognitive Core (R2)
Implements:
  1. Oja's Normalized Hebbian Synaptic Plasticity across a 13x13 model topology
  2. Multi-Hop Lateral Spreading Associative Activation
  3. Continuous Modern Hopfield Free-Energy Objective with Log-Sum-Exp Stabilization
  4. Bayesian Surprise Gating via 1024-dim BGE-M3 Semantic Vector Embeddings
  5. Objective Composite AGI Quotient (AGIQ) Metric [0.0, 200.0]

Strict zero-NaN, zero-divergence, and bounded numerical guarantees using pure NumPy.
"""

import json
import math
import urllib.request
import urllib.error
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

# Canonical 13-node specialist model topology
FLEET_MODELS_13 = [
    "Qwen3.8-27B-UD-Q4_K_M",         # 0: Architect / General high-context
    "Qwen3.8-27B-Uncensored",        # 1: Uncensored security / exploit auditor
    "Qwen2.5-Coder-32B-Instruct",    # 2: Flagship code factory
    "DeepSeek-R1-Distill-Qwen-32B",  # 3: Deep reasoning / mathematical critic
    "Qwen2.5-Coder-14B-Instruct",    # 4: Mid-tier fast coder
    "DeepSeek-R1-Distill-Qwen-14B",  # 5: Mid-tier fast reasoning
    "Qwen2.5-VL-7B-Instruct",        # 6: Vision / visual UI inspection
    "sqlcoder-7b-2",                 # 7: Relational database / SQL expert
    "Qwen2.5-Coder-14B-abliterated", # 8: Security / unrestricted coder
    "Hermes-3-Llama-3.1-8B",         # 9: Agent tool orchestration / strict JSON
    "gemma-2-27b-it",                # 10: Creative / technical copywriting
    "bge-m3",                        # 11: 1024-dim semantic embedding / memory
    "Qwen3-30B-A3B"                  # 12: Fast MoE hybrid / dynamic assistant
]

# Model role / alias mapping to index
MODEL_ALIAS_MAP = {
    "architect": 0, "general": 0, "qwen3.8-27b": 0, "qwen3.8-27b-ud-q4_k_m": 0,
    "security": 1, "uncensored": 1, "unfiltered": 1, "qwen3.8-27b-uncensored": 1,
    "coder": 2, "coding": 2, "coder-max": 2, "coder32b": 2, "qwen2.5-coder-32b-instruct": 2,
    "critic": 3, "reasoning": 3, "r1": 3, "r1-32b": 3, "deepseek-r1-distill-qwen-32b": 3,
    "coder14": 4, "coder14b": 4, "qwen2.5-coder-14b-instruct": 4,
    "r1-14": 5, "reasoning14": 5, "deepseek-r1-distill-qwen-14b": 5,
    "vision": 6, "vl": 6, "qwen2.5-vl-7b-instruct": 6,
    "sql": 7, "database": 7, "sqlcoder": 7, "sqlcoder-7b-2": 7,
    "security-coder": 8, "coder-abliterated": 8, "qwen2.5-coder-14b-abliterated": 8,
    "agent": 9, "hermes": 9, "hermes-3-llama-3.1-8b": 9,
    "creative": 10, "marketer": 10, "copywriter": 10, "gemma": 10, "gemma-2-27b-it": 10,
    "embed": 11, "embedding": 11, "memory": 11, "bge-m3": 11,
    "moe": 12, "turbo": 12, "fast-coder": 12, "qwen3-30b-a3b": 12
}

DEFAULT_EMBEDDING_URL = "http://127.0.0.1:18434/v1/embeddings"
EPSILON = 1e-7


class CognitiveCore:
    """
    Bio-inspired Autonomous Cognitive Core.
    Maintains synaptic plasticity, associative memory retrieval,
    epistemic surprise evaluation, and objective AGIQ calculation.
    """

    def __init__(
        self,
        num_models: int = 13,
        learning_rate: float = 0.12,
        decay_rate: float = 0.005,
        alpha_diffusion: float = 0.45,
        beta_hopfield: float = 1.0,
        gamma_surprise: float = 0.65,
        surprise_threshold: float = 0.42,
        embedding_url: str = DEFAULT_EMBEDDING_URL
    ):
        self.num_models = num_models
        self.learning_rate = float(learning_rate)
        self.decay_rate = float(decay_rate)
        self.alpha_diffusion = float(alpha_diffusion)
        self.beta_hopfield = float(beta_hopfield)
        self.gamma_surprise = float(gamma_surprise)
        self.surprise_threshold = float(surprise_threshold)
        self.embedding_url = embedding_url

        # Canonical model list
        self.models = list(FLEET_MODELS_13[:num_models])

    @staticmethod
    def initial_weights(num_models: int = 13, initial_val: float = 0.10) -> np.ndarray:
        """
        Creates an initial NxN synaptic weight matrix with zero self-connections (diagonal=0.0).
        """
        w = np.full((num_models, num_models), float(initial_val), dtype=np.float64)
        np.fill_diagonal(w, 0.0)
        return w

    @staticmethod
    def resolve_model_index(model_identifier: Union[int, str]) -> int:
        """
        Resolves model alias, canonical name, or index to 0..12 index.
        """
        if isinstance(model_identifier, (int, np.integer)):
            return int(model_identifier) % len(FLEET_MODELS_13)
        normalized = str(model_identifier).strip().lower()
        if normalized in MODEL_ALIAS_MAP:
            return MODEL_ALIAS_MAP[normalized]
        for idx, m in enumerate(FLEET_MODELS_13):
            if m.lower() == normalized or normalized in m.lower():
                return idx
        return 0

    def compute_oja_update(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        lr: Optional[float] = None,
        decay: Optional[float] = None,
        w_min: float = 0.001,
        w_max: float = 1.000
    ) -> np.ndarray:
        """
        Oja's Normalized Hebbian Learning Update:
          Delta W_{ij} = eta * (a_i * a_j - W_{ij} * a_j^2)
          W^{(t+1)} = clip((W + Delta W) * (1 - lambda), w_min, w_max)
          diag(W) = 0.0

        Guaranteed: zero-NaN, bounded in [w_min, w_max], diagonal preserved at 0.0.
        """
        eta = float(lr if lr is not None else self.learning_rate)
        lam = float(decay if decay is not None else self.decay_rate)

        # Sanitize inputs
        w_curr = np.nan_to_num(np.array(weights, dtype=np.float64), nan=0.10, posinf=w_max, neginf=w_min)
        act = np.nan_to_num(np.array(activations, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        act = np.clip(act, 0.0, 1.0)

        # Ensure correct shapes
        n = self.num_models
        if w_curr.shape != (n, n):
            w_curr = np.resize(w_curr, (n, n))
            np.fill_diagonal(w_curr, 0.0)
        if act.shape[0] != n:
            act = np.resize(act, (n,))

        # Vectorized Oja calculation:
        # Hebbian term: P_{ij} = a_i * a_j
        hebbian = np.outer(act, act)

        # Homeostatic decay term: W_{ij} * a_j^2
        # act**2 broadcast across rows: shape (1, n)
        act_sq = (act ** 2).reshape(1, n)
        homeostatic = w_curr * act_sq

        # Delta W
        delta_w = eta * (hebbian - homeostatic)

        # Apply update with passive forgetting decay
        w_next = (w_curr + delta_w) * (1.0 - lam)

        # Enforce bounds and zero diagonal
        w_bounded = np.clip(w_next, w_min, w_max)
        np.fill_diagonal(w_bounded, 0.0)

        # Absolute zero-NaN check
        if np.isnan(w_bounded).any() or np.isneginf(w_bounded).any() or np.isposinf(w_bounded).any():
            w_bounded = np.nan_to_num(w_bounded, nan=0.10, posinf=w_max, neginf=w_min)
            np.fill_diagonal(w_bounded, 0.0)

        return w_bounded

    def spreading_activation(
        self,
        initial_activations: np.ndarray,
        weights: np.ndarray,
        alpha: Optional[float] = None,
        hops: int = 2
    ) -> np.ndarray:
        """
        Associative Lateral Spreading Activation:
          A^{(t+1)} = Normalize((1 - alpha) * A^{(0)} + alpha * W^T @ A^{(t)})
          Normalize(v) = v / (sum(|v|) + epsilon)
        """
        diff = float(alpha if alpha is not None else self.alpha_diffusion)
        diff = np.clip(diff, 0.0, 1.0)

        a0 = np.nan_to_num(np.array(initial_activations, dtype=np.float64), nan=0.0)
        a0 = np.clip(a0, 0.0, 1.0)
        norm_a0 = np.sum(np.abs(a0)) + EPSILON
        a0 = a0 / norm_a0

        w = np.nan_to_num(np.array(weights, dtype=np.float64), nan=0.0)
        np.fill_diagonal(w, 0.0)

        a_t = np.copy(a0)
        for _ in range(max(1, hops)):
            propagated = w.T @ a_t
            a_next = (1.0 - diff) * a0 + diff * propagated
            a_next = np.clip(a_next, 0.0, None)
            norm = np.sum(np.abs(a_next)) + EPSILON
            a_t = a_next / norm

        return np.nan_to_num(a_t, nan=1.0 / self.num_models)

    def continuous_hopfield_energy(
        self,
        query_vector: np.ndarray,
        memory_vectors: Union[np.ndarray, List[Any]],
        beta: Optional[float] = None
    ) -> float:
        """
        Modern Continuous Hopfield Free Energy (Ramsauer et al. 2020):
          E(xi) = -beta^-1 * log(sum_k exp(beta * xi^T v_k)) + 0.5 * ||xi||^2

        Stabilized using Log-Sum-Exp:
          log(sum exp(s_k)) = m + log(sum exp(s_k - m)) where m = max(s_k).
        """
        b = float(beta if beta is not None else self.beta_hopfield)
        q = np.nan_to_num(np.array(query_vector, dtype=np.float64).flatten(), nan=0.0)
        norm_q = np.linalg.norm(q) + EPSILON
        q = q / norm_q

        q_energy_term = 0.5 * float(np.dot(q, q))

        if memory_vectors is None or len(memory_vectors) == 0:
            return q_energy_term

        mems = np.nan_to_num(np.array(memory_vectors, dtype=np.float64), nan=0.0)
        if mems.ndim == 1:
            mems = mems.reshape(1, -1)

        # Normalize memory vectors
        m_norms = np.linalg.norm(mems, axis=1, keepdims=True) + EPSILON
        mems = mems / m_norms

        # Dot products s_k = beta * (q @ v_k)
        dot_prods = b * (mems @ q)

        # Log-Sum-Exp stabilization
        m = float(np.max(dot_prods))
        exp_terms = np.exp(dot_prods - m)
        sum_exp = float(np.sum(exp_terms)) + EPSILON
        lse = m + math.log(sum_exp)

        free_energy = -(1.0 / max(b, EPSILON)) * lse + q_energy_term
        return float(np.nan_to_num(free_energy, nan=q_energy_term))

    def get_embedding(self, text: str, timeout: float = 6.0) -> np.ndarray:
        """
        Fetches 1024-dim BGE-M3 embedding from local inference proxy (http://127.0.0.1:18434/v1/embeddings).
        Falls back gracefully to a deterministic normalized pseudo-embedding if unreachable.
        """
        clean_text = str(text).strip() if text else "synapse forge cognitive task"
        try:
            payload = json.dumps({"model": "bge-m3", "input": clean_text}).encode("utf-8")
            req = urllib.request.Request(
                self.embedding_url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = data["data"][0]["embedding"]
                vec = np.array(emb, dtype=np.float64)
                norm = np.linalg.norm(vec) + EPSILON
                return np.nan_to_num(vec / norm, nan=0.0)
        except Exception:
            # Deterministic, normalized fallback vector derived from SHA-256
            # Guarantees robust, zero-crash operation during unit tests or offline runs
            h = hashlib.sha256(clean_text.encode("utf-8")).digest()
            raw = np.frombuffer(h * 32, dtype=np.uint8)[:1024].astype(np.float64)
            raw = (raw / 255.0) - 0.5
            norm = np.linalg.norm(raw) + EPSILON
            return raw / norm

    def compute_surprise(
        self,
        task_text: str,
        prior_embeddings: Optional[List[Any]] = None,
        td_error: float = 0.0,
        gamma: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Bayesian Surprise & Free-Energy Gating:
          S_{semantic}(x) = clip(1.0 - max_k cos(e_x, v_k), 0.0, 1.0)
          E_{prediction} = clip(|delta_{TD}|, 0.0, 1.0)
          S_{comp}(x) = gamma * S_{semantic} + (1 - gamma) * E_{prediction}
          Gate triggered if S_{comp} >= threshold.
        """
        gam = float(gamma if gamma is not None else self.gamma_surprise)
        gam = np.clip(gam, 0.0, 1.0)

        # 1. Compute query vector
        query_vec = self.get_embedding(task_text)

        # 2. Semantic Novelty
        if prior_embeddings and len(prior_embeddings) > 0:
            mems = np.array(prior_embeddings, dtype=np.float64)
            if mems.ndim == 1:
                mems = mems.reshape(1, -1)
            # Cosine similarities
            m_norms = np.linalg.norm(mems, axis=1, keepdims=True) + EPSILON
            normalized_mems = mems / m_norms
            sims = normalized_mems @ query_vec
            max_sim = float(np.max(sims))
            novelty = float(np.clip(1.0 - max_sim, 0.0, 1.0))
        else:
            novelty = 1.0  # Greenfield task is completely novel

        # 3. Prediction Error
        clean_td = float(np.nan_to_num(td_error, nan=0.0))
        pred_error = float(np.clip(abs(clean_td), 0.0, 1.0))

        # 4. Composite Surprise
        composite = float(gam * novelty + (1.0 - gam) * pred_error)
        composite = float(np.clip(composite, 0.0, 1.0))

        # 5. Hopfield Free Energy
        hopfield_e = self.continuous_hopfield_energy(query_vec, prior_embeddings or [])

        # 6. Gating Decision & Effective Learning Rate
        gating_triggered = bool(composite >= self.surprise_threshold)
        effective_lr = float(self.learning_rate * (1.0 + 1.5 * composite)) if gating_triggered else float(self.learning_rate * 0.25)

        return {
            "surprise_score": composite,
            "novelty_score": novelty,
            "td_error": clean_td,
            "hopfield_energy": hopfield_e,
            "gating_triggered": gating_triggered,
            "effective_lr": effective_lr,
            "query_embedding": query_vec.tolist()
        }

    def calculate_agiq(self, components: Dict[str, Any]) -> float:
        """
        Objective Composite AGI Quotient (AGIQ) Metric [0.0, 200.0]:
          AGIQ = 200.0 * sum_{m=1}^6 (omega_m * phi_m)
          phi_1 (code_correctness): weight 0.25
          phi_2 (arch_coherence): weight 0.20
          phi_3 (plasticity_stability): weight 0.15
          phi_4 (surprise_adaptability): weight 0.15
          phi_5 (execution_speed): weight 0.10
          phi_6 (economic_viability): weight 0.15

        Guaranteed: bounded [0.0, 200.0], zero NaN/Inf.
        """
        phi_code = float(np.clip(components.get("code_score", 0.75), 0.0, 1.0))
        phi_arch = float(np.clip(components.get("arch_score", 0.70), 0.0, 1.0))
        phi_plast = float(np.clip(components.get("plasticity_score", 0.65), 0.0, 1.0))
        phi_surp = float(np.clip(components.get("surprise_score", 0.60), 0.0, 1.0))
        phi_speed = float(np.clip(components.get("speed_score", 0.80), 0.0, 1.0))
        phi_econ = float(np.clip(components.get("econ_score", 0.70), 0.0, 1.0))

        # Convex combination
        raw_composite = (
            0.25 * phi_code +
            0.20 * phi_arch +
            0.15 * phi_plast +
            0.15 * phi_surp +
            0.10 * phi_speed +
            0.15 * phi_econ
        )

        agiq = 200.0 * float(np.clip(raw_composite, 0.0, 1.0))
        return float(np.nan_to_num(agiq, nan=100.0, posinf=200.0, neginf=0.0))

    def evaluate_plasticity_score(self, weights: np.ndarray) -> float:
        """
        Calculates the synaptic network density and eigenvalue stability phi_3:
          phi_3 = clip((Mean(W) / 0.35) * (1.0 - Var(W)), 0.0, 1.0)
        """
        w = np.nan_to_num(np.array(weights, dtype=np.float64), nan=0.0)
        n = w.shape[0]
        # Ignore diagonal (which is 0)
        off_diag = w[~np.eye(n, dtype=bool)]
        if len(off_diag) == 0:
            return 0.50

        mean_w = float(np.mean(off_diag))
        var_w = float(np.var(off_diag))
        score = (mean_w / 0.35) * (1.0 - min(1.0, var_w))
        return float(np.clip(np.nan_to_num(score, nan=0.50), 0.0, 1.0))

    def evaluate_speed_score(self, latency_ms: float) -> float:
        """
        Calculates execution throughput score phi_5:
          phi_5 = clip(1.0 - (latency_ms - 200) / (2000 - 200), 0.0, 1.0)
        """
        lat = float(np.nan_to_num(latency_ms, nan=500.0))
        score = 1.0 - (lat - 200.0) / (2000.0 - 200.0)
        return float(np.clip(score, 0.0, 1.0))

    def decompose_agiq(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decomposes AGIQ into its weighted dimensional breakdown.
        """
        total = self.calculate_agiq(components)
        return {
            "composite_agiq": round(total, 2),
            "code_score": round(float(components.get("code_score", 0.75)), 4),
            "arch_score": round(float(components.get("arch_score", 0.70)), 4),
            "plasticity_score": round(float(components.get("plasticity_score", 0.65)), 4),
            "surprise_score": round(float(components.get("surprise_score", 0.60)), 4),
            "speed_score": round(float(components.get("speed_score", 0.80)), 4),
            "econ_score": round(float(components.get("econ_score", 0.70)), 4),
        }
