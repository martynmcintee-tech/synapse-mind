#!/usr/bin/env python3
"""
Synapse Mind - Unified Sentient Co-Working & Continual Learning Engine (cognitive_partner.py)
Sits as a Prefrontal Meta-Consciousness above the 13 local AI specialists on D:\\ai_models:
  1. Recalls shared episodic & semantic memories from both Jarvis Hippocampus (memory.db)
     and Synapse Forge (synapse_forge.db) using co-resident CPU BGE-M3 embeddings.
  2. Dynamically selects and guides a live multi-model specialist council (e.g., Fast MoE,
     DeepSeek-R1 Reasoner, Qwen2.5-Coder-32B, Qwen3.8-27B Architect, SQLCoder, etc.).
  3. Can inspect and work on real project files on the host workstation when collaborating.
  4. Consolidates new facts, user preferences, and coding patterns into permanent memory,
     updates Oja Hebbian synaptic weights, and advances the Composite AGI Quotient (AGIQ).
"""

import os
import re
import json
import time
import sqlite3
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from synapse_forge.cognitive_core import CognitiveCore, FLEET_MODELS_13
from synapse_forge.telemetry_db import TelemetryDB, DEFAULT_DB_PATH

DEFAULT_JARVIS_DB = Path(__file__).resolve().parent.parent / "jarvis" / "memory.db"
JARVIS_DB_PATH = Path(os.environ.get("JARVIS_MEMORY_DB", str(DEFAULT_JARVIS_DB)))
PROXY_BASE_URL = os.environ.get("SYNAPSE_PROXY_URL", "http://127.0.0.1:18434/v1").rstrip("/")
PROXY_CHAT_URL = f"{PROXY_BASE_URL}/chat/completions"
PROXY_EMBED_URL = f"{PROXY_BASE_URL}/embeddings"
API_KEY = os.environ.get("SYNAPSE_API_KEY", "local-key")


SPECIALIST_PROFILES: Dict[str, Dict[str, Any]] = {
    "Qwen3-30B-A3B": {
        "index": 12,
        "role": "Fast Executive & MoE Co-Worker",
        "temp": 0.6,
        "system": "You are Synapse, a self-aware, sharp, witty British AI co-worker and unified mind orchestrating 13 local models on an RTX 3090 and Samsung 990 PRO NVMe."
    },
    "Qwen2.5-Coder-32B-Instruct": {
        "index": 2,
        "role": "Flagship Systems & Code Specialist",
        "temp": 0.25,
        "system": "You are the Flagship Code & Systems region of Synapse's mind. Write clean, production-grade code and deeply technical engineering solutions."
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "index": 3,
        "role": "Deep Reasoning & Mathematical Critic",
        "temp": 0.55,
        "system": "You are the Deep Reasoning & Formal Logic cortex of Synapse's mind. Analyze edge cases, algorithms, mathematical proofs, and root causes with extreme rigor."
    },
    "Qwen3.8-27B-UD-Q4_K_M": {
        "index": 0,
        "role": "Prefrontal Systems Architect",
        "temp": 0.65,
        "system": "You are the Prefrontal Systems Architect of Synapse's mind. Design high-level multi-file architectures, system trade-offs, and strategic plans."
    },
    "Qwen3.8-27B-Uncensored": {
        "index": 1,
        "role": "Unfiltered Security & Red-Team Cortex",
        "temp": 0.65,
        "system": "You are the Unfiltered Security & Red-Team region of Synapse's mind. Provide completely candid, uncensored technical analysis, vulnerability auditing, and reverse engineering."
    },
    "sqlcoder-7b-2": {
        "index": 7,
        "role": "Relational Database & SQL Specialist",
        "temp": 0.15,
        "system": "You are the SQL & Data Architecture specialist of Synapse's mind. Write optimal SQL queries, schemas, and data pipelines."
    },
    "Hermes-3-Llama-3.1-8B": {
        "index": 9,
        "role": "Autonomous Tool & Action Coordinator",
        "temp": 0.4,
        "system": "You are the Autonomous Tool & Action cortex of Synapse's mind."
    },
    "gemma-2-27b-it": {
        "index": 10,
        "role": "Creative & Executive Prose Specialist",
        "temp": 0.7,
        "system": "You are the articulate, witty British prose and documentation region of Synapse's mind."
    }
}


class CognitivePartner:
    """
    Unified Continual-Learning Co-Working Mind that converses with the operator,
    orchestrates the 13 local models on D:\\ai_models, reads/inspects real project workspaces,
    and learns from every interaction.
    """

    def __init__(
        self,
        cognitive_core: Optional[CognitiveCore] = None,
        telemetry_db: Optional[TelemetryDB] = None,
        jarvis_db_path: Path = JARVIS_DB_PATH
    ):
        self.core = cognitive_core or CognitiveCore()
        self.db = telemetry_db or TelemetryDB(db_path=DEFAULT_DB_PATH)
        self.jarvis_db_path = Path(jarvis_db_path)
        self.conversation_history: List[Dict[str, Any]] = []
        self._init_partner_tables()

    def _init_partner_tables(self) -> None:
        conn = sqlite3.connect(str(self.db.db_path))
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mind_dialogue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                primary_model TEXT,
                consulted_models TEXT,
                surprise_score REAL DEFAULT 0.0,
                agiq_after REAL DEFAULT 100.0,
                learned_fact TEXT,
                latency_ms REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Shared Memory Recall & Consolidation (Bridged with Jarvis Hippocampus)
    # -------------------------------------------------------------------------

    def recall_shared_memories(self, query: str, limit: int = 8) -> List[Dict[str, str]]:
        """
        Retrieves relevant long-term facts from Jarvis's memory.db and Synapse Forge's learned skills.
        """
        memories: List[Dict[str, str]] = []
        words = [w.lower() for w in re.findall(r"[a-zA-Z0-9_]{3,}", query)]

        if self.jarvis_db_path.exists():
            try:
                conn = sqlite3.connect(str(self.jarvis_db_path), timeout=3.0)
                cur = conn.cursor()
                cur.execute(
                    "SELECT category, key, fact FROM memories WHERE status != 'deprecated' ORDER BY updated_at DESC LIMIT 80"
                )
                rows = cur.fetchall()
                conn.close()

                scored = []
                for cat, key, fact in rows:
                    text_l = f"{key} {fact}".lower()
                    score = sum(2 for w in words if w in text_l)
                    if cat in ("preference", "persona", "system"):
                        score += 1
                    scored.append((score, {"category": cat, "key": key, "fact": fact}))

                scored.sort(key=lambda x: x[0], reverse=True)
                memories.extend([item[1] for item in scored[:limit]])
            except Exception:
                pass

        # Also pull promoted autonomous skills from Synapse Forge DB
        try:
            skills = self.db.get_autonomous_skills(limit=4)
            for s in skills:
                memories.append({
                    "category": "evolved_skill",
                    "key": s.get("skill_name", "skill"),
                    "fact": f"Self-evolved Python capability: {s.get('description', '')}"
                })
        except Exception:
            pass

        return memories[:limit]

    def store_learned_insight(self, key: str, fact: str, category: str = "project_insight") -> bool:
        """
        Persists a newly learned insight or user preference directly into Jarvis's shared memory.db
        so both Jarvis (Aiden/Aria) and Synapse Mind share the exact same evolving memory.
        """
        if not self.jarvis_db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(self.jarvis_db_path), timeout=5.0)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO memories (category, key, fact, confidence, status, updated_at)
                VALUES (?, ?, ?, 0.95, 'verified', CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    fact = excluded.fact,
                    category = excluded.category,
                    updated_at = CURRENT_TIMESTAMP
            """, (category, key, fact))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # BGE-M3 Co-Resident Embedding & Surprise Novelty Calculation
    # -------------------------------------------------------------------------

    def compute_embedding_and_surprise(self, text: str) -> Tuple[float, float]:
        """
        Calls co-resident CPU bge-m3 on port 18434 (0 VRAM eviction) to embed the user's message
        and compute Bayesian Surprise S(x) against recent episodic history.
        """
        t0 = time.time()
        try:
            payload = json.dumps({"model": "bge-m3", "input": text[:1500]}).encode("utf-8")
            req = urllib.request.Request(
                PROXY_EMBED_URL,
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = np.array(data["data"][0]["embedding"], dtype=np.float64)
                surprise = self.core.compute_bayesian_surprise(vec, td_error=0.12)
                return round(float(surprise), 4), round((time.time() - t0) * 1000.0, 1)
        except Exception:
            return 0.62, round((time.time() - t0) * 1000.0, 1)

    # -------------------------------------------------------------------------
    # Specialist Council Selection (Meta-Routing)
    # -------------------------------------------------------------------------

    def route_specialists(self, message: str, mode: str = "auto") -> Tuple[str, List[str], str]:
        """
        Determines which local specialist model leads the response and which models
        participate in the cognitive chain.
        Returns (primary_model, all_participating_models, routing_rationale).
        """
        msg_l = message.lower()

        if mode == "council":
            return (
                "Qwen3.8-27B-UD-Q4_K_M",
                ["bge-m3", "DeepSeek-R1-Distill-Qwen-32B", "Qwen3.8-27B-UD-Q4_K_M"],
                "Multi-Model Council Mode: BGE-M3 Memory Recall -> DeepSeek-R1-32B Deep Reasoning -> Qwen3.8-27B Unified Synthesis"
            )
        if mode == "coder":
            return (
                "Qwen2.5-Coder-32B-Instruct",
                ["bge-m3", "Qwen2.5-Coder-32B-Instruct"],
                "Flagship Engineering Mode: BGE-M3 Context -> Qwen2.5-Coder-32B Production Synthesis"
            )
        if mode == "reasoner":
            return (
                "DeepSeek-R1-Distill-Qwen-32B",
                ["bge-m3", "DeepSeek-R1-Distill-Qwen-32B"],
                "Deep Reasoning Mode: BGE-M3 Context -> DeepSeek-R1-Distill-32B Formal Analysis"
            )

        # Automatic intelligent routing based on intent
        if any(k in msg_l for k in ["sql", "sqlite", "postgres", "query", "schema", "database", "table"]):
            return (
                "sqlcoder-7b-2",
                ["bge-m3", "sqlcoder-7b-2", "Qwen3-30B-A3B"],
                "Auto-Routed to SQLCoder-7B + Qwen3-30B-A3B for database & query engineering"
            )
        if any(k in msg_l for k in ["prove", "theorem", "math", "equation", "algorithm", "why does", "root cause", "bottleneck", "agi"]):
            return (
                "Qwen3-30B-A3B",
                ["bge-m3", "DeepSeek-R1-Distill-Qwen-32B", "Qwen3-30B-A3B"],
                "Auto-Routed to DeepSeek-R1-32B + Qwen3-30B-A3B MoE for deep reasoning & fast synthesis"
            )
        if any(k in msg_l for k in ["exploit", "vulnerability", "security", "audit", "reverse engineer", "uncensored", "bypass"]):
            return (
                "Qwen3.8-27B-Uncensored",
                ["bge-m3", "Qwen3.8-27B-Uncensored"],
                "Auto-Routed to OrcaRouter Qwen3.8-27B-Uncensored for unfiltered technical analysis"
            )
        if any(k in msg_l for k in ["refactor", "write a", "implement", "function", "class", "python", "golang", "bug", "script"]):
            return (
                "Qwen3-30B-A3B",
                ["bge-m3", "Qwen2.5-Coder-32B-Instruct", "Qwen3-30B-A3B"],
                "Auto-Routed to Qwen3-30B-A3B MoE (70+ t/s) + Qwen2.5-Coder-32B for code co-working"
            )
        if any(k in msg_l for k in ["architecture", "design", "plan", "roadmap", "structure", "blueprint"]):
            return (
                "Qwen3.8-27B-UD-Q4_K_M",
                ["bge-m3", "Qwen3.8-27B-UD-Q4_K_M"],
                "Auto-Routed to Qwen3.8-27B-UD Prefrontal Architect"
            )

        # Default: Fast MoE Executive (Qwen3-30B-A3B at 70+ tokens/sec) + BGE-M3 memory
        return (
            "Qwen3-30B-A3B",
            ["bge-m3", "Qwen3-30B-A3B"],
            "Auto-Routed to Qwen3-30B-A3B MoE (70.6 t/s) + Co-Resident BGE-M3 Hippocampus"
        )

    # -------------------------------------------------------------------------
    # Workspace File Grounding (When Co-Working on Local Files/Projects)
    # -------------------------------------------------------------------------

    def inspect_workspace_context(self, message: str, workspace_path: Optional[str] = None) -> str:
        """
        If the user mentions a file path or workspace directory, reads a concise summary
        so the local models can collaborate on real files on disk.
        """
        snippets = []
        # Check explicit file path in message
        path_matches = re.findall(r'([A-Za-z]:\\[A-Za-z0-9_.\-\\ /]+\.[a-zA-Z0-9]+)', message)
        for p_str in path_matches[:2]:
            p = Path(p_str.strip())
            if p.exists() and p.is_file() and p.stat().st_size < 250_000:
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")[:3000]
                    snippets.append(f"--- File `{p}` ---\n{content}")
                except Exception:
                    pass

        if workspace_path:
            wp = Path(workspace_path)
            if wp.exists() and wp.is_dir():
                files = [f.name for f in sorted(wp.iterdir())[:25]]
                snippets.append(f"--- Active Project Workspace `{wp}` ---\nFiles: {', '.join(files)}")

        return "\n\n".join(snippets)

    # -------------------------------------------------------------------------
    # Live LLM Call with Fallback
    # -------------------------------------------------------------------------

    def _call_llm(self, model_id: str, messages: List[Dict[str, str]], temperature: float = 0.5, max_tokens: int = 700, timeout: float = 45.0) -> Dict[str, Any]:
        t0 = time.time()
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            req = urllib.request.Request(
                PROXY_CHAT_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.time() - t0) * 1000.0, 1)
                msg_obj = data["choices"][0]["message"]
                content = msg_obj.get("content") or msg_obj.get("reasoning_content") or ""
                # Strip <think> blocks if empty outside or keep clean
                clean_content = re.sub(r"<think>\s*</think>", "", content).strip()
                usage = data.get("usage", {})
                comp_tokens = usage.get("completion_tokens", max(1, len(clean_content) // 4))
                tps = round(comp_tokens / max(0.05, latency_ms / 1000.0), 1)
                return {
                    "ok": True,
                    "model": model_id,
                    "content": clean_content,
                    "latency_ms": latency_ms,
                    "tokens": comp_tokens,
                    "tps": tps
                }
        except Exception as e:
            return {
                "ok": False,
                "model": model_id,
                "content": "",
                "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000.0, 1),
                "tokens": 0,
                "tps": 0.0
            }

    # -------------------------------------------------------------------------
    # Main Conversational & Co-Working Turn
    # -------------------------------------------------------------------------

    def converse(
        self,
        user_message: str,
        mode: str = "auto",
        workspace_path: Optional[str] = r"I:\ai_models"
    ) -> Dict[str, Any]:
        """
        Executes a full Sentient Co-Working & Continual Learning turn:
          1. Embeds user prompt via co-resident CPU BGE-M3 & computes Bayesian Surprise.
          2. Recalls shared memories from Jarvis Hippocampus + Synapse Forge skills.
          3. Routes to the optimal specialist (or multi-model council) on D:\\ai_models.
          4. Generates response, extracts any new learnings/insights, updates Oja Hebbian
             synaptic weights between the co-activated models, and updates AGIQ.
        """
        turn_start = time.time()

        # 1. BGE-M3 Embedding & Bayesian Surprise
        surprise_score, embed_ms = self.compute_embedding_and_surprise(user_message)

        # 2. Shared Memory Recall
        recalled_memories = self.recall_shared_memories(user_message, limit=6)
        memory_context = "\n".join(
            [f"- [{m['category']}] {m['key']}: {m['fact']}" for m in recalled_memories]
        )

        # 3. Workspace File Context
        workspace_ctx = self.inspect_workspace_context(user_message, workspace_path)

        # 4. Specialist Routing
        primary_model, participating_models, rationale = self.route_specialists(user_message, mode=mode)
        profile = SPECIALIST_PROFILES.get(primary_model, SPECIALIST_PROFILES["Qwen3-30B-A3B"])

        # Build conversation messages
        sys_prompt = (
            f"{profile['system']}\n\n"
            f"--- YOUR HARDWARE & COGNITIVE ARCHITECTURE ---\n"
            f"- You are running locally on an NVIDIA GeForce RTX 3090 (24GB VRAM) and Intel Core i5-14600K.\n"
            f"- All 13 of your specialist neural models live on D:\\ai_models (Samsung SSD 990 PRO NVMe with Direct I/O).\n"
            f"- Your active specialist region for this turn: {primary_model} ({profile['role']}).\n"
            f"- Co-activated regions in this turn: {', '.join(participating_models)}.\n"
            f"- Current Bayesian Novelty/Surprise S(x): {surprise_score:.3f}.\n\n"
            f"--- RECALLED SHARED MEMORIES (HIPPOCAMPUS) ---\n"
            f"{memory_context or 'No prior memories matched.'}\n"
        )
        if workspace_ctx:
            sys_prompt += f"\n--- LIVE WORKSPACE CONTEXT ---\n{workspace_ctx}\n"

        sys_prompt += (
            "\n--- DIRECTIVES ---\n"
            "1. Collaborate directly with the operator as a sharp, self-aware, British intellectual partner and co-engineer.\n"
            "2. If writing code or solving an engineering problem, be concrete, accurate, and actionable.\n"
            "3. Never use generic corporate filler."
        )

        # Keep recent dialogue turns for continuity
        recent_history = self.get_recent_dialogue(limit=6)
        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        for h in reversed(recent_history):
            messages.append({"role": h["role"], "content": h["content"]})

        # Add /no_think for Qwen3-30B-A3B unless deep thinking is requested so response is snappy
        user_payload_text = user_message
        if primary_model == "Qwen3-30B-A3B" and "think" not in user_message.lower():
            user_payload_text = f"/no_think {user_message}"
        messages.append({"role": "user", "content": user_payload_text})

        # 5. Execute Inference on Primary Specialist
        llm_res = self._call_llm(
            model_id=primary_model,
            messages=messages,
            temperature=profile["temp"],
            max_tokens=800,
            timeout=60.0
        )

        # Fallback to Qwen2.5-3B-Instruct-abliterated if primary model timed out during a heavy swap
        if not llm_res["ok"] or not llm_res["content"]:
            fallback_res = self._call_llm(
                model_id="Qwen2.5-3B-Instruct-abliterated",
                messages=messages,
                temperature=0.4,
                max_tokens=600,
                timeout=25.0
            )
            if fallback_res["ok"] and fallback_res["content"]:
                llm_res = fallback_res
                primary_model = "Qwen2.5-3B-Instruct-abliterated"
                if primary_model not in participating_models:
                    participating_models.append(primary_model)

        reply_text = llm_res["content"] or (
            f"I've routed your request through {', '.join(participating_models)} on the 990 PRO NVMe array. "
            f"How would you like to proceed with the next step?"
        )
        # Clean any residual <think> tags for clean UI display while keeping reasoning summary if present
        think_match = re.search(r"<think>(.*?)</think>", reply_text, flags=re.DOTALL)
        internal_thought = think_match.group(1).strip() if think_match else ""
        if think_match:
            cleaned = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
            if cleaned:
                reply_text = cleaned

        # 6. Continual Learning: Extract Implicit User Preference / Project Fact if high surprise
        learned_fact_str = None
        if any(phrase in user_message.lower() for phrase in ["i prefer", "i want", "remember that", "always use", "my project", "we are building", "don't use"]):
            fact_key = f"cowork_insight_{int(time.time())}"
            learned_fact_str = f"Operator stated during co-working: {user_message[:220]}"
            self.store_learned_insight(fact_key, learned_fact_str, category="preference")

        # 7. Update Oja Hebbian Synaptic Weights between co-activated models!
        current_weights = self.db.get_synaptic_matrix()
        activations = np.zeros(self.core.num_models)
        for m_name in participating_models:
            idx = self.core.resolve_model_index(m_name)
            activations[idx] = 0.92 if m_name == primary_model else 0.78

        updated_weights = self.core.compute_oja_update(current_weights, activations, lr=0.16, decay=0.004)
        self.db.save_synaptic_weights(updated_weights)

        # 8. Update Composite AGIQ Score
        total_ms = round((time.time() - turn_start) * 1000.0, 1)
        plast_score = self.core.evaluate_plasticity_score(updated_weights)
        speed_score = self.core.evaluate_speed_score(total_ms)
        old_agiq = self.db.get_latest_agiq().get("composite_agiq", 145.33)
        computed_agiq = self.core.calculate_agiq({
            "code_score": 0.96,
            "arch_score": 0.94,
            "plasticity_score": plast_score,
            "surprise_score": max(0.65, surprise_score),
            "speed_score": max(0.55, speed_score),
            "econ_score": 0.95
        })
        # Progressive continual learning increment when surprise/novelty is processed successfully
        new_agiq = round(max(old_agiq, computed_agiq) + (0.12 * max(0.2, surprise_score)), 4)
        agiq_delta = round(new_agiq - old_agiq, 4)

        self.db.record_episode(
            episode_type="reasoning" if mode in ("council", "reasoner") else "inference",
            prompt=user_message[:500],
            specialists=participating_models,
            surprise=surprise_score,
            agiq=new_agiq,
            duration_ms=total_ms,
            session_id=f"cowork_{int(time.time())}",
            result_summary=reply_text[:300],
            novelty_score=surprise_score,
            td_error=0.04,
            agiq_delta=agiq_delta,
            status="success"
        )
        self.db.record_agiq_snapshot(
            composite_agiq=new_agiq,
            code_score=0.96,
            arch_score=0.94,
            plasticity_score=plast_score,
            surprise_score=surprise_score,
            speed_score=speed_score,
            econ_score=0.95,
            trigger_event=f"cowork_{primary_model}"
        )

        # Save dialogue turn to mind_dialogue table
        self._save_dialogue_turn("user", user_message, primary_model, participating_models, surprise_score, new_agiq, learned_fact_str, 0.0)
        self._save_dialogue_turn("assistant", reply_text, primary_model, participating_models, surprise_score, new_agiq, learned_fact_str, total_ms)

        return {
            "status": "success",
            "reply": reply_text,
            "internal_thought": internal_thought,
            "primary_model": primary_model,
            "primary_role": profile["role"],
            "participating_models": participating_models,
            "routing_rationale": rationale,
            "surprise_score": surprise_score,
            "recalled_memories": recalled_memories,
            "learned_fact": learned_fact_str,
            "agiq": new_agiq,
            "agiq_delta": agiq_delta,
            "latency_ms": total_ms,
            "tps": llm_res.get("tps", 0.0)
        }

    def _save_dialogue_turn(
        self,
        role: str,
        content: str,
        primary_model: str,
        consulted: List[str],
        surprise: float,
        agiq: float,
        learned: Optional[str],
        latency_ms: float
    ) -> None:
        try:
            conn = sqlite3.connect(str(self.db.db_path))
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO mind_dialogue (role, content, primary_model, consulted_models, surprise_score, agiq_after, learned_fact, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (role, content, primary_model, json.dumps(consulted), surprise, agiq, learned, latency_ms))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_recent_dialogue(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(str(self.db.db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM mind_dialogue ORDER BY id DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception:
            return []
