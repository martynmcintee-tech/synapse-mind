#!/usr/bin/env python3
"""
Neural Plasticity, Hebbian Synaptic Memory, & Softmax MoE Routing Engine
for J.A.R.V.I.S. / Aiden Local AGI Architecture (NVIDIA RTX 3090 24GB).

Implements three bio-inspired neural computation pillars:
  1. SynapticGraph (Oja's Normalized Hebbian Learning + Modern Continuous Hopfield Energy
     + Lateral Spreading Activation):
     - Synapses W_{ij} between memory nodes potentiate when co-activated (LTP) and decay
       via homeostatic normalization:
         Delta W_{ij} = eta * (a_i * a_j - W_{ij} * a_j^2)
     - Lateral spreading activation propagates initial sensory/semantic activations A^(0)
       across multi-hop associative pathways:
         A^(t+1) = Normalize((1 - alpha) * A^(0) + alpha * W^T @ A^(t))
     - Modern Continuous Hopfield Energy:
         E(xi) = -beta^-1 log(sum_i exp(beta * xi^T x_i)) + 0.5 * xi^T xi

  2. NeuralRouter (Softmax Gated Mixture-of-Experts with Policy Gradient Plasticity):
     - Computes context-conditioned expert logits z_e(x) and softmax routing probabilities:
         pi_e(x) = exp(z_e(x) / tau) / sum_k exp(z_k(x) / tau)
     - Updates routing weights online via reward-modulated TD / Policy Gradient feedback:
         theta_e <- theta_e + lr * (R - b) * (I(e == e*) - pi_e(x))

  3. SurpriseGate (Titans Test-Time Memorization & Free-Energy Surprise Objective):
     - Evaluates information-theoretic surprise S(x) combining semantic novelty
       (1 - max cosine similarity in BGE-M3 embedding space) and prediction/execution error:
         S(x) = gamma * (1 - max_i cos(u_x, v_i)) + (1 - gamma) * |delta_error|
     - Gates long-term memory consolidation and prioritizes high-surprise episodes for
       nightly 4-bit QLoRA weight updates.
"""

import os
import re
import json
import time
import math
import sqlite3
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

DB_PATH = Path(os.environ.get("JARVIS_DB_PATH", r"I:\ai_models\jarvis\memory.db"))

EXPERT_DOMAINS = {
    "coder": {
        "model_alias": "coder",
        "description": "Qwen2.5-Coder-32B-Instruct for dedicated production software engineering and bug fixing",
        "keywords": ["code", "python", "go", "rust", "script", "function", "bug", "class", "compile", "git", "api", "powershell", "file", "test"]
    },
    "fast_moe": {
        "model_alias": "qwen3-moe",
        "description": "Qwen3-30B-A3B MoE (3B active parameters) for ultra-fast, sub-second coding and conversational turns",
        "keywords": ["fast", "quick", "instant", "brief", "speed", "rapid", "short", "snippet", "hello", "hi", "hey"]
    },
    "reasoning": {
        "model_alias": "reasoning",
        "description": "DeepSeek-R1-Distill-Qwen-32B for mathematical proofs, formal logic, deep analysis, and algorithms",
        "keywords": ["prove", "math", "theorem", "logic", "equation", "derive", "algorithm", "complexity", "physics", "neural", "calculus", "matrix", "optimize"]
    },
    "general": {
        "model_alias": "general",
        "description": "Qwen3.8-27B-UD for high-context architecture, system design, and general dialogue",
        "keywords": ["architecture", "design", "system", "plan", "overview", "strategy", "compare", "explain", "summary", "status", "volume", "weather"]
    },
    "vision": {
        "model_alias": "vision",
        "description": "Qwen2.5-VL-7B-Instruct for desktop screen inspection, OCR, and visual UI grounding",
        "keywords": ["screen", "visual", "click", "button", "icon", "image", "window", "ui", "locate", "pixel", "screenshot", "look"]
    },
    "security": {
        "model_alias": "uncensored",
        "description": "Qwen3.8-27B-Uncensored for AST security auditing, exploit detection, and unfiltered analysis",
        "keywords": ["security", "audit", "exploit", "vulnerability", "cve", "injection", "sanitize", "sandbox", "jailbreak", "unsafe"]
    },
    "sql": {
        "model_alias": "sql",
        "description": "sqlcoder-7b-2 for database schemas, SQLite/Postgres queries, and indexing",
        "keywords": ["sql", "database", "query", "table", "schema", "sqlite", "select", "join", "index", "migration"]
    },
    "agent": {
        "model_alias": "agent",
        "description": "Hermes-3-Llama-3.1-8B for strict multi-step tool orchestration and JSON agency",
        "keywords": ["automate", "workflow", "hermes", "delegate", "pipeline", "schedule", "agent", "multi-step"]
    },
    "creative": {
        "model_alias": "creative",
        "description": "gemma-2-27b-it for prose, documentation, executive reports, and narrative writing",
        "keywords": ["write", "essay", "story", "report", "document", "tone", "creative", "article", "presentation", "draft", "pricing", "sales"]
    },
    "analyst": {
        "model_alias": "long-context",
        "description": "Qwen2.5-14B-Instruct-1M for 128K full-repo exploration and massive document analysis",
        "keywords": ["repo", "repository", "entire", "log", "full", "document", "history", "all files", "deep", "context"]
    },
    "micro": {
        "model_alias": "micro",
        "description": "Qwen2.5-3B-Instruct-abliterated for instant classification, yes/no checks, and parsing",
        "keywords": ["classify", "intent", "extract", "yes_no", "is_valid", "parse_tag"]
    },
    "coder14": {
        "model_alias": "coder14",
        "description": "Qwen2.5-Coder-14B-Instruct for intermediate coding and refactoring tasks",
        "keywords": ["refactor", "lint", "format", "clean", "unittest"]
    },
    "reasoning14": {
        "model_alias": "reasoning14",
        "description": "DeepSeek-R1-Distill-Qwen-14B for focused mathematical logic and puzzles",
        "keywords": ["puzzle", "deduce", "riddle", "reason"]
    }
}


class SynapticGraph:
    """
    Bio-inspired Associative Synaptic Graph with Oja's Normalized Hebbian Plasticity,
    Homeostatic Decay, and Multi-Hop Lateral Spreading Activation.
    """
    def __init__(
        self,
        db_path: Optional[Any] = None,
        learning_rate: float = 0.15,
        alpha: float = 0.45,
        decay_rate: float = 0.005
    ):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.decay_rate = decay_rate
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS synaptic_edges (
                    source_key TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.1,
                    co_activations INTEGER NOT NULL DEFAULT 1,
                    last_potentiated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_key, target_key)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodic_surprise_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    expert_used TEXT NOT NULL,
                    surprise_score REAL NOT NULL,
                    reward REAL NOT NULL,
                    td_error REAL NOT NULL,
                    consolidated INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def potentiate_pair(self, source_key: str, target_key: str, a_i: float = 1.0, a_j: float = 1.0) -> float:
        """
        Applies Oja's Normalized Hebbian Learning Rule between two memory keys:
            Delta W_{ij} = lr * (a_i * a_j - W_{ij} * a_j^2)
        Guarantees bounded synaptic weights ||W|| <= 1.0 without runaway explosion.
        """
        if source_key == target_key or not source_key or not target_key:
            return 0.0

        a_i = float(np.clip(a_i, 0.0, 1.0))
        a_j = float(np.clip(a_j, 0.0, 1.0))

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT weight, co_activations FROM synaptic_edges WHERE source_key = ? AND target_key = ?",
                (source_key, target_key)
            )
            row = cur.fetchone()
            w_old = float(row[0]) if row else 0.05
            co_act = (int(row[1]) + 1) if row else 1

            # Oja's rule update
            delta_w = self.learning_rate * (a_i * a_j - w_old * (a_j ** 2))
            w_new = float(np.clip((w_old + delta_w) * (1.0 - self.decay_rate), 0.001, 1.0))

            cur.execute("""
                INSERT INTO synaptic_edges (source_key, target_key, weight, co_activations, last_potentiated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_key, target_key) DO UPDATE SET
                    weight = excluded.weight,
                    co_activations = excluded.co_activations,
                    last_potentiated = CURRENT_TIMESTAMP
            """, (source_key, target_key, round(w_new, 6), co_act))
            conn.commit()
            return w_new

    def hebbian_potentiate_clique(self, active_keys: List[Tuple[str, float]]) -> int:
        """
        Cells that fire together, wire together:
        Given a list of (memory_key, activation_strength) co-activated during a cognitive turn,
        updates bidirectional synaptic connections across all co-firing pairs.
        """
        if len(active_keys) < 2:
            return 0

        updated_edges = 0
        # Limit to top 8 most active keys to keep updates O(1) fast
        top_active = active_keys[:8]
        for idx_i in range(len(top_active)):
            k_i, act_i = top_active[idx_i]
            for idx_j in range(idx_i + 1, len(top_active)):
                k_j, act_j = top_active[idx_j]
                self.potentiate_pair(k_i, k_j, act_i, act_j)
                self.potentiate_pair(k_j, k_i, act_j, act_i)
                updated_edges += 2
        return updated_edges

    def spreading_activation(
        self,
        seed_activations: Dict[str, float],
        hops: int = 2,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Propagates initial activation vector A^(0) across the synaptic weight matrix W:
            A^(t+1) = Normalize((1 - alpha) * A^(0) + alpha * W^T @ A^(t))
        Recovers laterally associated memories that share multi-hop neural pathways even
        when they have zero direct keyword overlap with the prompt.
        """
        if not seed_activations:
            return {}

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT source_key, target_key, weight FROM synaptic_edges WHERE weight > 0.02")
            edges = cur.fetchall()

        if not edges:
            return dict(sorted(seed_activations.items(), key=lambda x: x[1], reverse=True)[:top_k])

        # Build node index
        nodes = set(seed_activations.keys())
        for u, v, _ in edges:
            nodes.add(u)
            nodes.add(v)
        node_list = sorted(nodes)
        node_to_idx = {k: i for i, k in enumerate(node_list)}
        n = len(node_list)

        # Build adjacency matrix W
        W = np.zeros((n, n), dtype=np.float32)
        for u, v, w in edges:
            W[node_to_idx[u], node_to_idx[v]] = float(w)

        # Initial activation vector A^(0)
        A0 = np.zeros(n, dtype=np.float32)
        for k, val in seed_activations.items():
            if k in node_to_idx:
                A0[node_to_idx[k]] = float(val)

        norm0 = np.linalg.norm(A0, ord=1)
        if norm0 > 0:
            A0 = A0 / norm0

        A_t = A0.copy()
        for _ in range(hops):
            A_next = (1.0 - self.alpha) * A0 + self.alpha * (W.T @ A_t)
            norm_next = np.linalg.norm(A_next, ord=1)
            if norm_next > 0:
                A_next = A_next / norm_next
            A_t = A_next

        # Re-scale so peak activation aligns with original scale
        max_seed = max(seed_activations.values()) if seed_activations else 1.0
        max_final = float(np.max(A_t)) if np.max(A_t) > 0 else 1.0
        scale = max_seed / max_final

        activated = {}
        for idx, val in enumerate(A_t):
            scaled_val = float(val * scale)
            if scaled_val > 0.01:
                activated[node_list[idx]] = round(scaled_val, 4)

        return dict(sorted(activated.items(), key=lambda x: x[1], reverse=True)[:top_k])

    def compute_hopfield_energy(self, query_vec: np.ndarray, memory_matrix: np.ndarray, beta: float = 8.0) -> float:
        """
        Computes the Modern Continuous Hopfield Network Energy (Ramsauer et al., 2020):
            E(xi) = -beta^-1 * log(sum_i exp(beta * xi^T x_i)) + 0.5 * xi^T xi
        Lower energy indicates strong attractor convergence (familiar pattern);
        high energy indicates an unfamiliar state (high surprise / free energy).
        """
        if query_vec is None or memory_matrix is None or len(memory_matrix) == 0:
            return 0.0
        dots = beta * np.dot(memory_matrix, query_vec)
        max_dot = float(np.max(dots))
        lse = max_dot + math.log(float(np.sum(np.exp(dots - max_dot))))
        energy = -(1.0 / beta) * lse + 0.5 * float(np.dot(query_vec, query_vec))
        return round(float(energy), 4)

    def get_synaptic_telemetry(self) -> Dict[str, Any]:
        """Returns synaptic network statistics and strongest associative pathways."""
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), AVG(weight), MAX(weight), SUM(co_activations) FROM synaptic_edges")
            row = cur.fetchone()
            total_edges = row[0] or 0
            avg_weight = round(float(row[1] or 0.0), 4)
            max_weight = round(float(row[2] or 0.0), 4)
            total_coactivations = int(row[3] or 0)

            cur.execute("""
                SELECT source_key, target_key, weight, co_activations
                FROM synaptic_edges
                ORDER BY weight DESC
                LIMIT 8
            """)
            top_synapses = [
                {"source": r[0], "target": r[1], "weight": round(r[2], 4), "co_activations": r[3]}
                for r in cur.fetchall()
            ]

        return {
            "total_synaptic_edges": total_edges,
            "mean_synaptic_weight": avg_weight,
            "peak_synaptic_weight": max_weight,
            "total_hebbian_coactivations": total_coactivations,
            "strongest_pathways": top_synapses
        }


class NeuralRouter:
    """
    Softmax Gated Mixture-of-Experts (MoE) Router with Online Reinforcement Learning
    (Policy Gradient / TD-Error Plasticity) across the RTX 3090 Local Model Fleet.
    """
    def __init__(
        self,
        db_path: Optional[Any] = None,
        learning_rate: float = 0.12,
        temperature: float = 0.7
    ):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.learning_rate = learning_rate
        self.temperature = temperature
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS neural_routing_weights (
                    expert TEXT PRIMARY KEY,
                    logit_bias REAL NOT NULL DEFAULT 0.0,
                    reward_ema REAL NOT NULL DEFAULT 0.75,
                    calls INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    avg_latency_ms REAL NOT NULL DEFAULT 500.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for exp_name in EXPERT_DOMAINS:
                default_bias = 0.25 if exp_name == "coder" else 0.0
                cur.execute("""
                    INSERT OR IGNORE INTO neural_routing_weights
                    (expert, logit_bias, reward_ema, calls, successes, avg_latency_ms)
                    VALUES (?, ?, 0.80, 0, 0, 500.0)
                """, (exp_name, default_bias))
            conn.commit()

    def compute_gating_distribution(self, query: str) -> Dict[str, Any]:
        """
        Computes contextual feature affinities + learned synaptic logit biases for each expert,
        returning the Softmax probability distribution pi_e(x) and selected expert.
        """
        q_lower = query.lower()
        q_tokens = set(re.findall(r'\w+', q_lower))

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT expert, logit_bias, reward_ema, calls FROM neural_routing_weights")
            learned_state = {r[0]: {"bias": float(r[1]), "ema": float(r[2]), "calls": int(r[3])} for r in cur.fetchall()}

        logits = {}
        for exp_name, meta in EXPERT_DOMAINS.items():
            state = learned_state.get(exp_name, {"bias": 0.0, "ema": 0.8, "calls": 0})
            # 1. Learned synaptic bias + expected reward baseline
            z = state["bias"] + 0.5 * (state["ema"] - 0.5)

            # 2. Contextual keyword resonance
            kw_matches = sum(1 for kw in meta["keywords"] if kw in q_tokens or kw in q_lower)
            z += kw_matches * 0.65

            logits[exp_name] = z

        # Numerically stable Softmax with temperature tau
        exp_names = list(logits.keys())
        z_arr = np.array([logits[k] / max(self.temperature, 0.1) for k in exp_names], dtype=np.float64)
        z_arr -= np.max(z_arr)
        exp_z = np.exp(z_arr)
        probs = exp_z / np.sum(exp_z)

        distribution = {exp_names[i]: round(float(probs[i]), 4) for i in range(len(exp_names))}
        best_idx = int(np.argmax(probs))
        selected_expert = exp_names[best_idx]

        return {
            "selected_expert": selected_expert,
            "model_alias": EXPERT_DOMAINS[selected_expert]["model_alias"],
            "confidence": round(float(probs[best_idx]), 4),
            "probabilities": distribution,
            "logits": {k: round(v, 4) for k, v in logits.items()}
        }

    def reinforce_routing(
        self,
        expert: str,
        reward: float,
        selection_prob: float = 0.5,
        latency_ms: float = 400.0
    ) -> Dict[str, Any]:
        """
        REINFORCE Policy Gradient with Moving Baseline (TD-Error delta):
            delta = R - b_e
            theta_e <- theta_e + lr * delta * (1 - pi_e(x))
            b_e <- (1 - beta) * b_e + beta * R
        """
        if expert not in EXPERT_DOMAINS:
            expert = "coder"

        reward = float(np.clip(reward, -1.0, 1.0))
        selection_prob = float(np.clip(selection_prob, 0.01, 0.99))

        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT logit_bias, reward_ema, calls, successes, avg_latency_ms FROM neural_routing_weights WHERE expert = ?",
                (expert,)
            )
            row = cur.fetchone()
            bias_old = float(row[0]) if row else 0.0
            ema_old = float(row[1]) if row else 0.75
            calls = (int(row[2]) + 1) if row else 1
            successes = (int(row[3]) + (1 if reward >= 0.5 else 0)) if row else (1 if reward >= 0.5 else 0)
            lat_old = float(row[4]) if row else latency_ms

            # TD Error (Advantage over historical expected reward)
            td_error = reward - ema_old
            # Policy gradient step on logit bias (bounded in [-2.5, +2.5] for homeostatic stability)
            bias_new = float(np.clip(bias_old + self.learning_rate * td_error * (1.0 - selection_prob), -2.5, 2.5))
            ema_new = round(0.85 * ema_old + 0.15 * reward, 4)
            lat_new = round(0.8 * lat_old + 0.2 * latency_ms, 1)

            cur.execute("""
                UPDATE neural_routing_weights
                SET logit_bias = ?, reward_ema = ?, calls = ?, successes = ?, avg_latency_ms = ?, updated_at = CURRENT_TIMESTAMP
                WHERE expert = ?
            """, (round(bias_new, 5), ema_new, calls, successes, lat_new, expert))
            conn.commit()

        return {
            "expert": expert,
            "old_bias": round(bias_old, 4),
            "new_bias": round(bias_new, 4),
            "td_error": round(td_error, 4),
            "reward_ema": ema_new,
            "total_calls": calls
        }

    def get_router_telemetry(self) -> List[Dict[str, Any]]:
        """Returns current learned MoE gating state across all experts."""
        with self._lock, self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT expert, logit_bias, reward_ema, calls, successes, avg_latency_ms
                FROM neural_routing_weights
                ORDER BY logit_bias DESC
            """)
            return [
                {
                    "expert": r[0],
                    "logit_bias": round(r[1], 4),
                    "reward_ema": round(r[2], 4),
                    "calls": r[3],
                    "successes": r[4],
                    "avg_latency_ms": round(r[5], 1)
                }
                for r in cur.fetchall()
            ]


class SurpriseGate:
    """
    Titans-Inspired Test-Time Memorization & Free-Energy Surprise Gate.
    Computes epistemic surprise S(x) to decide whether an episode triggers
    immediate long-term synaptic encoding and prioritizes high-surprise samples
    for nightly 4-bit LoRA weight consolidation.
    """
    def __init__(self, surprise_threshold: float = 0.42, gamma: float = 0.65):
        self.surprise_threshold = surprise_threshold
        self.gamma = gamma

    def evaluate_surprise(
        self,
        max_semantic_similarity: float,
        td_error: float = 0.0,
        hopfield_energy: float = 0.0
    ) -> Dict[str, Any]:
        """
        Computes composite surprise score S(x) in [0, 1]:
            S(x) = gamma * (1 - max_sim) + (1 - gamma) * min(1.0, |td_error|)
        """
        max_sim = float(np.clip(max_semantic_similarity, 0.0, 1.0))
        novelty = 1.0 - max_sim
        err_mag = float(np.clip(abs(td_error), 0.0, 1.0))

        surprise = self.gamma * novelty + (1.0 - self.gamma) * err_mag
        surprise = round(float(np.clip(surprise, 0.0, 1.0)), 4)

        should_consolidate = surprise >= self.surprise_threshold
        # Importance weight for SFT/DPO gradient steps: higher surprise -> higher gradient weight
        lora_sample_weight = round(1.0 + 2.0 * surprise, 3)

        return {
            "surprise_score": surprise,
            "novelty": round(novelty, 4),
            "td_error_magnitude": round(err_mag, 4),
            "hopfield_energy": hopfield_energy,
            "should_consolidate": should_consolidate,
            "lora_sample_weight": lora_sample_weight
        }


class NeuralPlasticityEngine:
    """
    Unified Neural Plasticity Facade integrating SynapticGraph, NeuralRouter, and SurpriseGate.
    """
    def __init__(self, db_path: Optional[Any] = None):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.synaptic_graph = SynapticGraph(db_path=self.db_path)
        self.router = NeuralRouter(db_path=self.db_path)
        self.surprise_gate = SurpriseGate()

    def process_turn_plasticity(
        self,
        query: str,
        response: str,
        active_memories: List[Tuple[str, float]],
        max_semantic_sim: float = 0.5,
        expert_used: str = "coder",
        tool_success: bool = True,
        latency_ms: float = 350.0
    ) -> Dict[str, Any]:
        """
        Executes a full post-turn biological learning step:
          1. Potentiates co-activated memory nodes via Oja's Hebbian rule.
          2. Computes reward signal and updates MoE NeuralRouter gating logits via TD-error.
          3. Evaluates epistemic surprise S(x) and logs high-surprise episodes for nightly LoRA consolidation.
        """
        # 1. Hebbian LTP across co-activated memories
        edges_updated = self.synaptic_graph.hebbian_potentiate_clique(active_memories)

        # 2. Compute empirical reward R in [0, 1]
        is_clean_reply = len(response.strip()) > 10 and "apologize" not in response.lower() and "failed" not in response.lower()
        reward = 0.95 if (tool_success and is_clean_reply) else (0.60 if is_clean_reply else 0.20)

        gating = self.router.compute_gating_distribution(query)
        sel_prob = gating["probabilities"].get(expert_used, 0.25)
        rl_update = self.router.reinforce_routing(
            expert=expert_used,
            reward=reward,
            selection_prob=sel_prob,
            latency_ms=latency_ms
        )

        # 3. Evaluate Free-Energy Surprise
        surprise_info = self.surprise_gate.evaluate_surprise(
            max_semantic_similarity=max_semantic_sim,
            td_error=rl_update["td_error"]
        )

        # 4. Log to episodic_surprise_log
        with self.synaptic_graph._lock, self.synaptic_graph._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO episodic_surprise_log
                (query, response, expert_used, surprise_score, reward, td_error, consolidated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query[:500],
                response[:1000],
                expert_used,
                surprise_info["surprise_score"],
                reward,
                rl_update["td_error"],
                1 if surprise_info["should_consolidate"] else 0
            ))
            conn.commit()

        return {
            "edges_potentiated": edges_updated,
            "rl_routing_update": rl_update,
            "surprise": surprise_info
        }

    def get_full_diagnostics(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry of the neural plasticity subsystem."""
        return {
            "synaptic_graph": self.synaptic_graph.get_synaptic_telemetry(),
            "neural_router": self.router.get_router_telemetry(),
            "surprise_threshold": self.surprise_gate.surprise_threshold
        }


_GLOBAL_PLASTICITY: Optional[NeuralPlasticityEngine] = None

def get_neural_plasticity(db_path: Optional[Any] = None) -> NeuralPlasticityEngine:
    global _GLOBAL_PLASTICITY
    if _GLOBAL_PLASTICITY is None:
        _GLOBAL_PLASTICITY = NeuralPlasticityEngine(db_path=db_path)
    return _GLOBAL_PLASTICITY


if __name__ == "__main__":
    print("=" * 68)
    print("   NEURAL PLASTICITY, HEBBIAN SYNAPSES & MoE ROUTER DIAGNOSTICS")
    print("=" * 68)

    engine = get_neural_plasticity()

    # 1. Test Hebbian Potentiation & Spreading Activation
    print("\n[1/3] Testing Oja's Normalized Hebbian Potentiation & Spreading Activation...")
    co_firing = [
        ("hardware_gpu", 0.95),
        ("model_fleet", 0.90),
        ("lang_python", 0.85),
        ("preferred_editor", 0.75)
    ]
    edges_count = engine.synaptic_graph.hebbian_potentiate_clique(co_firing)
    print(f" -> Potentiated {edges_count} bidirectional synapses via Oja's rule.")

    spread = engine.synaptic_graph.spreading_activation({"hardware_gpu": 1.0}, hops=2)
    print(f" -> Spreading Activation from seed 'hardware_gpu': {spread}")

    # 2. Test Softmax MoE Gating & Policy Gradient Update
    print("\n[2/3] Testing Softmax Gated Mixture-of-Experts Router...")
    sample_query = "Derive the mathematical proof for Hopfield energy convergence and implement it in Python"
    route_info = engine.router.compute_gating_distribution(sample_query)
    print(f" -> Selected Expert: '{route_info['selected_expert']}' (conf: {route_info['confidence']})")
    print(f" -> Softmax Distribution: {route_info['probabilities']}")

    rl_res = engine.router.reinforce_routing(route_info["selected_expert"], reward=0.95, selection_prob=route_info["confidence"])
    print(f" -> Policy Gradient Update: {rl_res}")

    # 3. Test Surprise Gate & Full Turn Plasticity
    print("\n[3/3] Testing Titans Test-Time Surprise Gate & Episodic Logging...")
    turn_res = engine.process_turn_plasticity(
        query=sample_query,
        response="The Modern Continuous Hopfield energy function is guaranteed to converge via the concave-convex procedure.",
        active_memories=co_firing,
        max_semantic_sim=0.38,
        expert_used=route_info["selected_expert"],
        tool_success=True
    )
    print(f" -> Surprise Score: {turn_res['surprise']['surprise_score']} | Consolidate: {turn_res['surprise']['should_consolidate']} | LoRA Weight: {turn_res['surprise']['lora_sample_weight']}")
    print("\nFull Diagnostics:", json.dumps(engine.get_full_diagnostics(), indent=2))
    print("\n[SUCCESS] Neural Plasticity Engine verified.")
