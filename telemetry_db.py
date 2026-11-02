#!/usr/bin/env python3
"""
Synapse Forge - Telemetry Database Engine (R2)
Manages the persistent SQLite WAL database (synapse_forge.db) with 6 relational tables:
  1. cognitive_episodes
  2. model_contributions
  3. synaptic_weights
  4. autonomous_skills
  5. commercial_products
  6. agiq_history

Thread-safe, WAL enabled, foreign keys enforced, with complete CRUD and telemetry extraction.
"""

import os
import json
import sqlite3
import threading
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from synapse_forge.cognitive_core import FLEET_MODELS_13

DEFAULT_DB_PATH = Path(os.environ.get("SYNAPSE_DB_PATH", str(Path(__file__).resolve().parent / "synapse_forge.db")))


class TelemetryDB:
    """
    SQLite Relational Telemetry Store for Synapse Forge.
    Enforces WAL mode, foreign keys, and indexes for real-time telemetry and provenance.
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """
        Returns an open SQLite connection configured with WAL mode, foreign keys, and Row factory.
        Caller is responsible for closing the connection.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextlib.contextmanager
    def session(self):
        """
        Context manager ensuring transactional commit, rollback on error, and guaranteed close.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """
        Initializes all 6 database tables, constraints, indexes, and baseline synaptic weights.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()

            # Table 1: Cognitive Episodes
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_type TEXT NOT NULL CHECK(episode_type IN ('synthesis', 'evolution', 'inference', 'reasoning', 'benchmark')),
                session_id TEXT,
                task_prompt TEXT NOT NULL,
                selected_specialists TEXT NOT NULL,
                result_summary TEXT,
                surprise_score REAL NOT NULL DEFAULT 0.0,
                novelty_score REAL NOT NULL DEFAULT 0.0,
                td_error REAL NOT NULL DEFAULT 0.0,
                hopfield_energy REAL NOT NULL DEFAULT 0.0,
                agiq_delta REAL NOT NULL DEFAULT 0.0,
                composite_agiq REAL NOT NULL DEFAULT 100.0,
                duration_ms REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'success' CHECK(status IN ('success', 'failure', 'rejected')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Table 2: Model Contributions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL REFERENCES cognitive_episodes(id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                role TEXT NOT NULL,
                tokens_prompt INTEGER NOT NULL DEFAULT 0,
                tokens_completion INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                throughput_tps REAL NOT NULL DEFAULT 0.0,
                reward REAL NOT NULL DEFAULT 0.8,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Table 3: Synaptic Weights Matrix
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS synaptic_weights (
                source_model TEXT NOT NULL,
                target_model TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.100000,
                co_activations INTEGER NOT NULL DEFAULT 0,
                last_potentiated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_model, target_model)
            );
            """)

            # Table 4: Autonomous Skills & Tools
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('code_generation', 'refactoring', 'testing', 'security_audit', 'commercial_copy', 'ui_synthesis')),
                code_snippet TEXT NOT NULL,
                ast_signature TEXT,
                validation_score REAL NOT NULL DEFAULT 1.0,
                usage_count INTEGER NOT NULL DEFAULT 0,
                success_rate REAL NOT NULL DEFAULT 1.0,
                promoted_from_sandbox INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Table 5: Commercial Software Products
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS commercial_products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                tagline TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'standard' CHECK(tier IN ('starter', 'pro', 'enterprise')),
                price_usd REAL NOT NULL DEFAULT 49.0,
                directory_path TEXT NOT NULL,
                source_files TEXT NOT NULL,
                docs_content TEXT,
                landing_copy TEXT,
                agiq_at_creation REAL NOT NULL,
                provenance_json TEXT,
                status TEXT NOT NULL DEFAULT 'ready' CHECK(status IN ('synthesizing', 'ready', 'published', 'archived')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Table 6: AGIQ Evolution History
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agiq_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                composite_agiq REAL NOT NULL,
                code_score REAL NOT NULL,
                arch_score REAL NOT NULL,
                plasticity_score REAL NOT NULL,
                surprise_score REAL NOT NULL,
                speed_score REAL NOT NULL,
                econ_score REAL NOT NULL,
                trigger_event TEXT NOT NULL
            );
            """)

            # Indexes for low-latency queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_type ON cognitive_episodes(episode_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_created ON cognitive_episodes(created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contributions_model ON model_contributions(model_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contributions_episode ON model_contributions(episode_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON autonomous_skills(skill_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_uuid ON commercial_products(product_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_weights_source ON synaptic_weights(source_model);")

            # Seed initial synaptic matrix if empty
            cursor.execute("SELECT COUNT(*) as count FROM synaptic_weights;")
            row = cursor.fetchone()
            if row["count"] == 0:
                models = FLEET_MODELS_13
                initial_records = []
                for src in models:
                    for tgt in models:
                        w = 0.0 if src == tgt else 0.100000
                        initial_records.append((src, tgt, w, 0))
                cursor.executemany("""
                INSERT OR IGNORE INTO synaptic_weights (source_model, target_model, weight, co_activations)
                VALUES (?, ?, ?, ?);
                """, initial_records)

            # Seed initial AGIQ snapshot if empty
            cursor.execute("SELECT COUNT(*) as count FROM agiq_history;")
            if cursor.fetchone()["count"] == 0:
                cursor.execute("""
                INSERT INTO agiq_history (
                    composite_agiq, code_score, arch_score, plasticity_score,
                    surprise_score, speed_score, econ_score, trigger_event
                ) VALUES (100.0, 0.75, 0.70, 0.65, 0.60, 0.80, 0.70, 'genesis_baseline');
                """)

    # -------------------------------------------------------------------------
    # Cognitive Episodes & Contributions CRUD
    # -------------------------------------------------------------------------

    def record_episode(
        self,
        episode_type: str,
        prompt: str,
        specialists: Union[List[str], str],
        surprise: float,
        agiq: float,
        duration_ms: float,
        session_id: Optional[str] = None,
        result_summary: Optional[str] = None,
        novelty_score: float = 0.0,
        td_error: float = 0.0,
        hopfield_energy: float = 0.0,
        agiq_delta: float = 0.0,
        status: str = "success"
    ) -> int:
        """
        Records a cognitive episode and returns the generated episode ID.
        """
        spec_json = json.dumps(specialists) if isinstance(specialists, list) else str(specialists)
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO cognitive_episodes (
                episode_type, session_id, task_prompt, selected_specialists,
                result_summary, surprise_score, novelty_score, td_error,
                hopfield_energy, agiq_delta, composite_agiq, duration_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                episode_type, session_id, prompt, spec_json,
                result_summary, float(surprise), float(novelty_score), float(td_error),
                float(hopfield_energy), float(agiq_delta), float(agiq), float(duration_ms), status
            ))
            return int(cursor.lastrowid)

    def record_model_contribution(
        self,
        episode_id: int,
        model_id: str,
        role: str,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: float = 0.0,
        throughput_tps: float = 0.0,
        reward: float = 0.8
    ) -> int:
        """
        Records granular inference and contribution metrics for a model within an episode.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO model_contributions (
                episode_id, model_id, role, tokens_prompt, tokens_completion,
                latency_ms, throughput_tps, reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                episode_id, model_id, role, int(tokens_prompt), int(tokens_completion),
                float(latency_ms), float(throughput_tps), float(reward)
            ))
            return int(cursor.lastrowid)

    def get_recent_episodes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves recent cognitive episodes formatted as dicts.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM cognitive_episodes ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                try:
                    d["selected_specialists"] = json.loads(d["selected_specialists"])
                except Exception:
                    pass
                results.append(d)
            return results

    # -------------------------------------------------------------------------
    # Synaptic Weights CRUD
    # -------------------------------------------------------------------------

    def save_synaptic_weights(
        self,
        matrix_13x13: np.ndarray,
        model_ids: Optional[List[str]] = None
    ) -> None:
        """
        Persists a 13x13 synaptic weight matrix to the database.
        """
        models = model_ids if model_ids and len(model_ids) == 13 else FLEET_MODELS_13
        w = np.array(matrix_13x13, dtype=np.float64)
        records = []
        for i, src in enumerate(models):
            for j, tgt in enumerate(models):
                val = 0.0 if src == tgt else float(w[i, j])
                records.append((val, src, tgt))

        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
            UPDATE synaptic_weights
            SET weight = ?,
                co_activations = co_activations + (CASE WHEN ? > 0.001 THEN 1 ELSE 0 END),
                last_potentiated = CURRENT_TIMESTAMP
            WHERE source_model = ? AND target_model = ?;
            """, [(val, val, src, tgt) for val, src, tgt in records])

    def get_synaptic_matrix(self, model_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Reconstructs and returns the 13x13 synaptic weight matrix from the database.
        """
        models = model_ids if model_ids and len(model_ids) == 13 else FLEET_MODELS_13
        n = len(models)
        mat = np.zeros((n, n), dtype=np.float64)
        model_index = {m: i for i, m in enumerate(models)}

        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_model, target_model, weight FROM synaptic_weights;")
            rows = cursor.fetchall()
            for r in rows:
                src, tgt, w = r["source_model"], r["target_model"], r["weight"]
                if src in model_index and tgt in model_index:
                    mat[model_index[src], model_index[tgt]] = float(w)

        np.fill_diagonal(mat, 0.0)
        return mat

    def get_synaptic_edges(self, min_weight: float = 0.01) -> List[Dict[str, Any]]:
        """
        Retrieves synaptic edges for neural topology canvas visualization.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT source_model, target_model, weight, co_activations
            FROM synaptic_weights
            WHERE source_model != target_model AND weight >= ?
            ORDER BY weight DESC;
            """, (min_weight,))
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Autonomous Skills CRUD
    # -------------------------------------------------------------------------

    def record_autonomous_skill(
        self,
        skill_name: str,
        description: str,
        category: str,
        code_snippet: str,
        ast_signature: Optional[str] = None,
        validation_score: float = 1.0,
        promoted_from_sandbox: int = 1
    ) -> int:
        """
        Promotes and records an autonomous self-evolved skill or tool.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO autonomous_skills (
                skill_name, description, category, code_snippet,
                ast_signature, validation_score, promoted_from_sandbox, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(skill_name) DO UPDATE SET
                description = excluded.description,
                category = excluded.category,
                code_snippet = excluded.code_snippet,
                ast_signature = excluded.ast_signature,
                validation_score = excluded.validation_score,
                usage_count = autonomous_skills.usage_count + 1,
                updated_at = CURRENT_TIMESTAMP;
            """, (
                skill_name, description, category, code_snippet,
                ast_signature, float(validation_score), int(promoted_from_sandbox)
            ))
            if cursor.lastrowid and cursor.lastrowid > 0:
                return int(cursor.lastrowid)
            cursor.execute("SELECT id FROM autonomous_skills WHERE skill_name = ?;", (skill_name,))
            row = cursor.fetchone()
            return int(row["id"]) if row else 1

    def get_autonomous_skills(
        self,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetches promoted autonomous skills.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("""
                SELECT * FROM autonomous_skills WHERE category = ? ORDER BY id DESC LIMIT ?;
                """, (category, limit))
            else:
                cursor.execute("""
                SELECT * FROM autonomous_skills ORDER BY id DESC LIMIT ?;
                """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Commercial Products CRUD
    # -------------------------------------------------------------------------

    def record_commercial_product(
        self,
        product_uuid: str,
        name: str,
        tagline: str,
        tier: str,
        price_usd: float,
        directory_path: str,
        source_files: Union[List[Dict[str, Any]], str],
        docs_content: str,
        landing_copy: str,
        agiq_at_creation: float,
        provenance_json: Optional[Union[Dict[str, Any], str]] = None,
        status: str = "ready"
    ) -> int:
        """
        Records a synthesized commercial product release package.
        """
        src_json = json.dumps(source_files) if isinstance(source_files, list) else str(source_files)
        prov_json = json.dumps(provenance_json) if isinstance(provenance_json, dict) else provenance_json

        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO commercial_products (
                product_uuid, name, tagline, tier, price_usd, directory_path,
                source_files, docs_content, landing_copy, agiq_at_creation,
                provenance_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_uuid) DO UPDATE SET
                name = excluded.name,
                tagline = excluded.tagline,
                tier = excluded.tier,
                price_usd = excluded.price_usd,
                directory_path = excluded.directory_path,
                source_files = excluded.source_files,
                docs_content = excluded.docs_content,
                landing_copy = excluded.landing_copy,
                agiq_at_creation = excluded.agiq_at_creation,
                provenance_json = excluded.provenance_json,
                status = excluded.status;
            """, (
                product_uuid, name, tagline, tier, float(price_usd), str(directory_path),
                src_json, docs_content, landing_copy, float(agiq_at_creation),
                prov_json, status
            ))
            if cursor.lastrowid and cursor.lastrowid > 0:
                return int(cursor.lastrowid)
            cursor.execute("SELECT id FROM commercial_products WHERE product_uuid = ?;", (product_uuid,))
            row = cursor.fetchone()
            return int(row["id"]) if row else 1

    def get_commercial_products(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Lists commercial products available for distribution.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM commercial_products ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                try:
                    d["source_files"] = json.loads(d["source_files"])
                except Exception:
                    pass
                try:
                    if d.get("provenance_json"):
                        d["provenance_json"] = json.loads(d["provenance_json"])
                except Exception:
                    pass
                results.append(d)
            return results

    def get_product_by_uuid(self, product_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single product package by its unique identifier.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM commercial_products WHERE product_uuid = ?;
            """, (product_uuid,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["source_files"] = json.loads(d["source_files"])
            except Exception:
                pass
            return d

    # -------------------------------------------------------------------------
    # AGIQ Evolution History CRUD
    # -------------------------------------------------------------------------

    def record_agiq_snapshot(
        self,
        composite_agiq: float,
        code_score: float,
        arch_score: float,
        plasticity_score: float,
        surprise_score: float,
        speed_score: float,
        econ_score: float,
        trigger_event: str
    ) -> int:
        """
        Records a timestamped snapshot of the multi-dimensional AGIQ metric.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO agiq_history (
                composite_agiq, code_score, arch_score, plasticity_score,
                surprise_score, speed_score, econ_score, trigger_event
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                float(composite_agiq), float(code_score), float(arch_score),
                float(plasticity_score), float(surprise_score), float(speed_score),
                float(econ_score), trigger_event
            ))
            return int(cursor.lastrowid)

    def get_latest_agiq(self) -> Dict[str, Any]:
        """
        Returns the latest recorded AGIQ entry.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM agiq_history ORDER BY id DESC LIMIT 1;
            """)
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {
                "composite_agiq": 100.0,
                "code_score": 0.75,
                "arch_score": 0.70,
                "plasticity_score": 0.65,
                "surprise_score": 0.60,
                "speed_score": 0.80,
                "econ_score": 0.70,
                "trigger_event": "default_baseline"
            }

    def get_agiq_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Returns AGIQ historical trajectory.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM agiq_history ORDER BY id ASC LIMIT ?;
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Returns aggregate counts and metrics across all 6 tables.
        """
        with self._lock, self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cognitive_episodes;")
            episodes_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM autonomous_skills;")
            skills_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM commercial_products;")
            products_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM model_contributions;")
            contributions_count = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(weight) FROM synaptic_weights WHERE source_model != target_model;")
            avg_weight = cursor.fetchone()[0] or 0.10

            cursor.execute("SELECT * FROM agiq_history ORDER BY id DESC LIMIT 1;")
            row = cursor.fetchone()
            latest_agiq = dict(row) if row else {
                "composite_agiq": 100.0,
                "code_score": 0.75,
                "arch_score": 0.70,
                "plasticity_score": 0.65,
                "surprise_score": 0.60,
                "speed_score": 0.80,
                "econ_score": 0.70,
                "trigger_event": "default_baseline"
            }

            return {
                "episodes_count": episodes_count,
                "skills_count": skills_count,
                "products_count": products_count,
                "contributions_count": contributions_count,
                "average_synaptic_weight": round(float(avg_weight), 4),
                "latest_agiq": latest_agiq
            }
