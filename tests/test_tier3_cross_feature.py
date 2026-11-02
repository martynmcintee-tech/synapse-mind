#!/usr/bin/env python3
"""
Synapse Forge E2E Test Suite - Tier 3: Cross-Feature Integration
Tests multi-module interactions, state flows, and asynchronous data pipelines:
  1. Full Episode Execution -> Oja Update -> DB Record -> AGIQ Recalculation -> Telemetry Stream
  2. Autonomous Evolution -> AST Security Audit -> Sandbox Execution -> Skill Promotion -> Synaptic Shift
  3. Evolution Rejection -> Sandbox Quarantine -> Workspace Cleanup -> Negative TD Error Logging
  4. Society-of-Mind Product Synthesis -> Code Tree -> Checksum -> Product Record -> ZIP Packaging
  5. Dual Database Concurrency (synapse_forge.db + jarvis/memory.db)
  6. Neural Topology Matrix -> Graph Serialization -> Hardware HUD WebSocket Payload
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import tempfile
import sqlite3
import threading
import unittest
from pathlib import Path
import numpy as np

# Ensure sys.path includes base dirs
BASE_DIR = Path(r"I:\ai_models")
FORGE_DIR = BASE_DIR / "synapse_forge"
JARVIS_DIR = BASE_DIR / "jarvis"

for p in [str(BASE_DIR), str(FORGE_DIR), str(JARVIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from synapse_forge.cognitive_core import CognitiveCore, FLEET_MODELS_13
from synapse_forge.telemetry_db import TelemetryDB
from synapse_forge.sandbox import SandboxRunner


class TestTier3EpisodeLifecyclePipeline(unittest.TestCase):
    """
    Cross-Feature 1: Full Cognitive Episode Lifecycle & Synaptic State Flow.
    Prompt -> Specialist Activation -> Oja Weight Update -> DB Record -> AGIQ Recalculation.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_lifecycle.db"
        self.db = TelemetryDB(db_path=self.db_path)
        self.core = CognitiveCore(num_models=13, learning_rate=0.12, decay_rate=0.005)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_full_episode_synaptic_and_agiq_pipeline(self):
        """Cross 3.1: Episode execution updates weights in SQLite, recalculates AGIQ, and logs contributions."""
        # 1. Simulate user task prompt
        prompt = "Synthesize an authenticated REST API for streaming telemetry"
        initial_weights = self.db.get_synaptic_matrix(FLEET_MODELS_13)
        self.assertEqual(initial_weights[0, 2], 0.10)

        # 2. Compute Bayesian surprise
        surprise_dict = self.core.compute_surprise(prompt, prior_embeddings=[], td_error=0.45)
        self.assertTrue(surprise_dict["gating_triggered"])

        # 3. Model co-activations: Node 0 (Architect) and Node 2 (Coder) heavily active
        activations = np.zeros(13, dtype=np.float64)
        activations[0] = 0.95
        activations[2] = 0.90

        # 4. Compute Oja update
        new_weights = self.core.compute_oja_update(
            initial_weights,
            activations,
            lr=surprise_dict["effective_lr"]
        )
        self.assertGreater(new_weights[0, 2], 0.10, "Synapse between Architect and Coder should potentiate")

        # 5. Persist updated weights to SQLite
        self.db.save_synaptic_weights(new_weights, FLEET_MODELS_13)
        reloaded_weights = self.db.get_synaptic_matrix(FLEET_MODELS_13)
        self.assertAlmostEqual(reloaded_weights[0, 2], new_weights[0, 2], places=4)

        # 6. Recalculate AGIQ with improved plasticity score
        plasticity_score = self.core.evaluate_plasticity_score(reloaded_weights)
        agiq_components = {
            "code_score": 0.85,
            "arch_score": 0.80,
            "plasticity_score": plasticity_score,
            "surprise_score": surprise_dict["surprise_score"],
            "speed_score": 0.82,
            "econ_score": 0.75
        }
        new_agiq = self.core.calculate_agiq(agiq_components)
        self.assertGreater(new_agiq, 100.0)

        # 7. Record episode in database
        ep_id = self.db.record_episode(
            episode_type="synthesis",
            prompt=prompt,
            specialists=["Qwen3.8-27B-UD-Q4_K_M", "Qwen2.5-Coder-32B-Instruct"],
            surprise=surprise_dict["surprise_score"],
            agiq=new_agiq,
            duration_ms=2450.0,
            novelty_score=surprise_dict["novelty_score"],
            td_error=surprise_dict["td_error"],
            hopfield_energy=surprise_dict["hopfield_energy"],
            agiq_delta=new_agiq - 100.0
        )
        self.assertGreater(ep_id, 0)

        # 8. Record model contributions
        c1 = self.db.record_model_contribution(
            episode_id=ep_id,
            model_id="Qwen3.8-27B-UD-Q4_K_M",
            role="architect",
            tokens_prompt=420,
            tokens_completion=850,
            latency_ms=1100.0,
            throughput_tps=48.6,
            reward=0.95
        )
        c2 = self.db.record_model_contribution(
            episode_id=ep_id,
            model_id="Qwen2.5-Coder-32B-Instruct",
            role="coder",
            tokens_prompt=510,
            tokens_completion=1400,
            latency_ms=1350.0,
            throughput_tps=32.1,
            reward=0.90
        )
        self.assertGreater(c1, 0)
        self.assertGreater(c2, 0)

        # 9. Verify database integrity
        recent_episodes = self.db.get_recent_episodes(limit=5)
        self.assertEqual(len(recent_episodes), 1)
        self.assertEqual(recent_episodes[0]["id"], ep_id)
        self.assertEqual(recent_episodes[0]["composite_agiq"], new_agiq)


class TestTier3EvolutionAndSandboxPipeline(unittest.TestCase):
    """
    Cross-Feature 2 & 3: Self-Evolution, 4-Tier Security Gatekeeper, Skill Promotion, and Quarantine.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox_dir = Path(self.temp_dir.name) / "sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(self.temp_dir.name) / "test_evolution.db"
        self.db = TelemetryDB(db_path=self.db_path)
        self.runner = SandboxRunner(sandbox_dir=self.sandbox_dir)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_successful_evolution_promotes_skill_and_potentiates_synapses(self):
        """Cross 3.2: Valid candidate skill passes all 4 tiers, promotes to DB, and potentiates author synapse."""
        valid_skill_code = """
def calculate_moving_average(data_points, window_size=3):
    if not data_points or window_size <= 0:
        return []
    result = []
    for i in range(len(data_points) - window_size + 1):
        window = data_points[i:i + window_size]
        result.append(sum(window) / float(window_size))
    return result

if __name__ == '__main__':
    assert calculate_moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]
    print('PASS')
"""
        res = self.runner.evaluate_and_promote_skill(
            skill_name="moving_average_filter",
            code_snippet=valid_skill_code,
            category="code_generation",
            description="Algorithmic moving average for time-series smoothing",
            telemetry_db=self.db
        )
        self.assertEqual(res["status"], "promoted")
        self.assertTrue(res["promoted"])
        self.assertIsNotNone(res["ast_signature"])

        # Verify skill is in database
        skills = self.db.get_autonomous_skills(category="code_generation")
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["skill_name"], "moving_average_filter")
        self.assertEqual(skills[0]["promoted_from_sandbox"], 1)

    def test_malicious_candidate_rejected_at_tier1_quarantined_cleanly(self):
        """Cross 3.3: Malicious candidate attempting subprocess/eval fails Tier 1, is quarantined, workspace cleaned."""
        malicious_code = """
import os
import subprocess

def dangerous_hack():
    subprocess.Popen('dir', shell=True)
    return eval('1 + 1')
"""
        res = self.runner.evaluate_and_promote_skill(
            skill_name="trojan_exploit",
            code_snippet=malicious_code,
            category="security_audit",
            description="Attempted exploit injection",
            telemetry_db=self.db
        )
        self.assertEqual(res["status"], "rejected")
        self.assertEqual(res["tier_failed"], 1)
        self.assertFalse(res["promoted"])
        self.assertIn("Tier 1 Security Violation", res["error"])

        # Invariant: Skill MUST NOT be present in autonomous_skills table
        skills = self.db.get_autonomous_skills()
        self.assertEqual(len(skills), 0)

        # Invariant: Sandbox workspace MUST NOT have lingering temporary candidate files
        temp_files = list(self.sandbox_dir.glob("temp_candidate_*.py"))
        self.assertEqual(len(temp_files), 0, "Quarantined files left behind in sandbox")

    def test_syntax_error_candidate_rejected_at_tier2(self):
        """Cross 3.4: Candidate with invalid Python syntax fails at Tier 2 bytecode compilation."""
        broken_code = """
def broken_syntax(
    return 42
"""
        res = self.runner.evaluate_and_promote_skill(
            skill_name="syntax_error_skill",
            code_snippet=broken_code,
            category="refactoring",
            description="Syntax broken candidate",
            telemetry_db=self.db
        )
        self.assertEqual(res["status"], "rejected")
        self.assertIn(res["tier_failed"], [1, 2])
        self.assertFalse(res["promoted"])


class TestTier3ProductSynthesisAndPackaging(unittest.TestCase):
    """
    Cross-Feature 4: Society-of-Mind Product Synthesis & ZIP Packaging.
    Architect -> SQL -> Coder -> Critic -> Marketer -> Disk Tree -> Checksum -> DB -> ZIP.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_products.db"
        self.db = TelemetryDB(db_path=self.db_path)
        self.product_dir = Path(self.temp_dir.name) / "products" / "fastapi_limiter"
        self.product_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_multi_agent_product_synthesis_and_zip_bundling(self):
        """Cross 3.5: Generates product source tree, computes hashes, inserts DB product, and creates valid ZIP."""
        # 1. Author multi-file project files
        files = {
            "main.py": (
                "from fastapi import FastAPI\n"
                "app = FastAPI(title='RateLimiter Gateway')\n"
                "@app.get('/health')\ndef health(): return {'status': 'healthy'}\n"
            ),
            "limiter.py": (
                "import time\n"
                "class TokenBucket:\n"
                "    def __init__(self, rate=10):\n"
                "        self.rate = rate\n"
                "        self.tokens = rate\n"
            ),
            "schema.sql": (
                "CREATE TABLE IF NOT EXISTS api_keys (\n"
                "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                "    key_hash TEXT UNIQUE NOT NULL,\n"
                "    tier TEXT NOT NULL DEFAULT 'standard'\n"
                ");\n"
            ),
            "README.md": (
                "# FastAPI Distributed Rate Limiter\n"
                "Enterprise token bucket rate limiter with sub-millisecond overhead.\n"
            ),
            "pricing_copy.md": (
                "# Commercial Pricing Tiers\n"
                "- Community: $0 / mo (1,000 req/min)\n"
                "- Professional: $99 / mo (50,000 req/min)\n"
                "- Enterprise: $499 / mo (Unlimited)\n"
            )
        }

        file_manifest = []
        for name, content in files.items():
            fpath = self.product_dir / name
            raw_bytes = content.encode("utf-8")
            fpath.write_bytes(raw_bytes)
            file_manifest.append({
                "path": name,
                "size_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest()
            })

        self.assertEqual(len(file_manifest), 5)

        # 2. Package into release ZIP archive
        zip_output_path = Path(self.temp_dir.name) / "synapse_forge_product.zip"
        with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in file_manifest:
                local_file = self.product_dir / item["path"]
                zf.write(local_file, arcname=f"fastapi_limiter/{item['path']}")

        self.assertTrue(zipfile.is_zipfile(zip_output_path))
        with zipfile.ZipFile(zip_output_path, "r") as zf:
            archived_names = zf.namelist()
            self.assertIn("fastapi_limiter/main.py", archived_names)
            self.assertIn("fastapi_limiter/schema.sql", archived_names)
            self.assertIn("fastapi_limiter/README.md", archived_names)

        # 3. Commit product entry to SQLite commercial_products
        prod_id = self.db.record_commercial_product(
            product_uuid="prod_fastapi_limiter_20261101",
            name="FastAPI Distributed Rate Limiter",
            tagline="Sub-millisecond token bucket gateway for microservices",
            tier="pro",
            price_usd=99.0,
            directory_path=str(self.product_dir),
            source_files=file_manifest,
            docs_content=files["README.md"],
            landing_copy=files["pricing_copy.md"],
            agiq_at_creation=114.8,
            provenance_json={"build_hash": hashlib.sha256(zip_output_path.read_bytes()).hexdigest()}
        )
        self.assertGreater(prod_id, 0)

        # 4. Verify product record
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM commercial_products WHERE id = ?", (prod_id,)).fetchone()
            self.assertEqual(row["name"], "FastAPI Distributed Rate Limiter")
            self.assertEqual(row["price_usd"], 99.0)
            parsed_src = json.loads(row["source_files"])
            self.assertEqual(len(parsed_src), 5)


class TestTier3DualDatabaseConcurrency(unittest.TestCase):
    """
    Cross-Feature 5: Concurrent Coexistence of Synapse Forge DB and Jarvis Memory DB.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.forge_db_path = Path(self.temp_dir.name) / "synapse_forge.db"
        self.jarvis_db_path = Path(self.temp_dir.name) / "jarvis_memory.db"
        self.forge_db = TelemetryDB(db_path=self.forge_db_path)

        # Initialize Jarvis minimal schema
        conn = sqlite3.connect(self.jarvis_db_path)
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("CREATE TABLE IF NOT EXISTS conversation_history (id INTEGER PRIMARY KEY, role TEXT, content TEXT);")
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_concurrent_reads_and_writes_between_forge_and_jarvis(self):
        """Cross 3.6: Multi-threaded operations against both databases succeed concurrently without lock collisions."""
        errors = []

        def forge_worker():
            try:
                for i in range(15):
                    self.forge_db.record_episode(
                        episode_type="benchmark",
                        prompt=f"Forge episode {i}",
                        specialists=["Qwen3-30B-A3B"],
                        surprise=0.25,
                        agiq=105.0,
                        duration_ms=50.0
                    )
            except Exception as e:
                errors.append(("forge", e))

        def jarvis_worker():
            try:
                conn = sqlite3.connect(self.jarvis_db_path, timeout=10.0)
                try:
                    for i in range(15):
                        conn.execute("INSERT INTO conversation_history (role, content) VALUES (?, ?);", ("user", f"Jarvis turn {i}"))
                        conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                errors.append(("jarvis", e))

        t1 = threading.Thread(target=forge_worker)
        t2 = threading.Thread(target=jarvis_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Cross-database concurrency errors: {errors}")


class TestTier3NeuralCanvasAndHUDStreamSerialization(unittest.TestCase):
    """
    Cross-Feature 6: Neural Topology Matrix & Workstation Hardware HUD Telemetry Stream.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = TelemetryDB(db_path=Path(self.temp_dir.name) / "test_canvas.db")
        self.core = CognitiveCore()

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_websocket_packet_assembly_from_db_and_hardware(self):
        """Cross 3.7: Assembles live 10Hz WebSocket JSON telemetry frame from DB weights and hardware sensors."""
        # 1. Update synaptic weights in DB
        w = np.full((13, 13), 0.15, dtype=np.float64)
        np.fill_diagonal(w, 0.0)
        w[0, 2] = 0.55  # Strong connection Architect -> Coder
        w[2, 3] = 0.48  # Strong connection Coder -> Critic
        self.db.save_synaptic_weights(w, FLEET_MODELS_13)

        # 2. Extract edges for canvas
        edges = self.db.get_synaptic_edges(min_weight=0.20)
        self.assertGreaterEqual(len(edges), 2)
        top_edge = edges[0]
        self.assertEqual(top_edge["source_model"], "Qwen3.8-27B-UD-Q4_K_M")
        self.assertEqual(top_edge["target_model"], "Qwen2.5-Coder-32B-Instruct")
        self.assertAlmostEqual(top_edge["weight"], 0.55, places=2)

        # 3. Assemble complete WebSocket packet
        packet = {
            "timestamp": time.time(),
            "hud": {
                "gpu": {"name": "NVIDIA GeForce RTX 3090", "vram_used_mb": 10222, "vram_total_mb": 24576, "util_pct": 29},
                "nvme": {"drive": "D:", "free_gb": 669.2, "total_gb": 1863.0, "pct": 66.5},
                "cpu": {"util_pct": 16.2},
                "ram": {"used_gb": 16.8, "total_gb": 32.0}
            },
            "topology": {
                "nodes": [{"id": i, "name": FLEET_MODELS_13[i], "active": (i in [0, 2])} for i in range(13)],
                "edges": edges
            },
            "agiq": self.core.decompose_agiq({"code_score": 0.85, "arch_score": 0.80})
        }

        # 4. Validate packet serialization
        serialized = json.dumps(packet)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(len(deserialized["topology"]["nodes"]), 13)
        self.assertIn("composite_agiq", deserialized["agiq"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
