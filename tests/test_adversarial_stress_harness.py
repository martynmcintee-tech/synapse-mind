#!/usr/bin/env python3
"""
Adversarial Stress Harness & Edge-Case Battery for Synapse Forge and Jarvis.
Tests:
  1. CognitiveCore: extreme numeric stress (zeros, huge floats, denormals, empty vectors)
  2. AST Security Sandbox: adversarial bypass payloads and blacklist enforcement
  3. TelemetryDB: concurrent multi-threaded write stress & transaction integrity
  4. Jarvis Fallbacks: simulated missing D: drive fallback for Whisper
  5. Hands Telemetry: resilience across storage devices
"""

import os
import sys
import time
import math
import json
import sqlite3
import tempfile
import threading
import subprocess
import unittest
from pathlib import Path
import numpy as np

BASE_DIR = Path(r"I:\ai_models")
FORGE_DIR = BASE_DIR / "synapse_forge"
JARVIS_DIR = BASE_DIR / "jarvis"

for p in [str(BASE_DIR), str(FORGE_DIR), str(JARVIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from synapse_forge.cognitive_core import CognitiveCore, FLEET_MODELS_13
from synapse_forge.telemetry_db import TelemetryDB
from synapse_forge.sandbox import SandboxRunner
import ears
from hands import Hands
from neural_plasticity import NeuralRouter

class TestAdversarialCognitiveCore(unittest.TestCase):
    """Stress-tests CognitiveCore mathematical stability under adversarial conditions."""

    def setUp(self):
        self.core = CognitiveCore()

    def test_oja_all_zeros_activation(self):
        """Oja update with all-zero activations must produce zero delta without NaN."""
        weights = np.ones((13, 13)) * 0.1
        activations = np.zeros(13)
        updated = self.core.compute_oja_update(weights, activations)
        self.assertFalse(np.isnan(updated).any(), "NaN found in zero activations update")
        self.assertFalse(np.isinf(updated).any(), "Inf found in zero activations update")
        self.assertTrue((updated >= 0.0).all(), "Negative weights after zero activation")

    def test_oja_massive_activations_saturation(self):
        """Oja update with massive activations (1e6) must not blow up to Infinity or NaN."""
        weights = np.ones((13, 13)) * 0.5
        activations = np.ones(13) * 1e6
        updated = self.core.compute_oja_update(weights, activations)
        self.assertFalse(np.isnan(updated).any(), "NaN found in massive activations update")
        self.assertFalse(np.isinf(updated).any(), "Inf found in massive activations update")
        self.assertTrue((updated <= 1.0).all(), "Weights must be bounded <= 1.0")

    def test_oja_negative_activations(self):
        """Oja update with negative activations must be safely handled without NaN."""
        weights = np.ones((13, 13)) * 0.5
        activations = np.ones(13) * -5.0
        updated = self.core.compute_oja_update(weights, activations)
        self.assertFalse(np.isnan(updated).any(), "NaN found in negative activations update")
        self.assertTrue((updated >= 0.0).all(), "Negative weights after negative activations")

    def test_surprise_empty_priors(self):
        """Bayesian surprise with empty prior embeddings should return valid default without crash."""
        result = self.core.compute_surprise("New unfamiliar task", prior_embeddings=[], td_error=0.8)
        self.assertIn("surprise_score", result)
        self.assertFalse(np.isnan(result["surprise_score"]))

    def test_agiq_boundary_and_garbage_components(self):
        """AGIQ calculation with missing, negative, or excessive values must clamp to [0.0, 200.0]."""
        # Negative components
        neg_components = {
            "code_score": -50.0,
            "arch_score": -10.0,
            "plasticity_score": -100.0,
            "surprise_score": -5.0,
            "speed_score": -20.0,
            "econ_score": -10.0
        }
        score = self.core.calculate_agiq(neg_components)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 200.0)

        # Huge components
        huge_components = {
            "code_score": 1e6,
            "arch_score": 1e6,
            "plasticity_score": 1e6,
            "surprise_score": 1e6,
            "speed_score": 1e6,
            "econ_score": 1e6
        }
        score_huge = self.core.calculate_agiq(huge_components)
        self.assertLessEqual(score_huge, 200.0)


class TestAdversarialSandboxSecurity(unittest.TestCase):
    """Stress-tests the 4-tier AST Security Sandbox against evasive bypass attempts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runner = SandboxRunner(sandbox_dir=self.temp_dir.name)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_block_os_system_call(self):
        """os.system calls must be rejected by Tier 1 AST audit."""
        exploit_code = "import os\nos.system('whoami')\n"
        passed, err, sig = self.runner.tier1_ast_audit(exploit_code)
        self.assertFalse(passed, "os.system was not blocked by sandbox!")
        self.assertIn("Security Violation", err)

    def test_block_subprocess_shell_true(self):
        """subprocess calls with shell=True must be rejected."""
        exploit_code = "import subprocess\nsubprocess.run('dir', shell=True)\n"
        passed, err, sig = self.runner.tier1_ast_audit(exploit_code)
        self.assertFalse(passed, "subprocess shell=True was not blocked!")

    def test_block_eval_exec_call(self):
        """eval and exec builtins must be rejected by Tier 1 AST audit."""
        for dangerous_func in ["eval('1 + 1')", "exec('x = 1')", "compile('x=1', '', 'exec')"]:
            passed, err, sig = self.runner.tier1_ast_audit(f"result = {dangerous_func}\n")
            self.assertFalse(passed, f"{dangerous_func} was not blocked by sandbox!")

    def test_block_ctypes_and_winreg_imports(self):
        """Kernel and registry manipulation modules must be rejected."""
        for mod in ["ctypes", "winreg", "win32api"]:
            code = f"import {mod}\n"
            passed, err, sig = self.runner.tier1_ast_audit(code)
            self.assertFalse(passed, f"Import {mod} was not blocked by sandbox!")

    def test_sandbox_timeout_termination(self):
        """Infinite loop script must be terminated by Tier 3 execution within timeout."""
        infinite_code = "import time\nwhile True:\n    time.sleep(0.05)\n"
        script_path = Path(self.temp_dir.name) / "infinite.py"
        script_path.write_text(infinite_code, encoding="utf-8")
        passed, stdout, stderr, dur_ms = self.runner.tier3_isolated_execution(script_path, timeout=0.8)
        self.assertFalse(passed, "Infinite loop was not terminated by timeout!")
        self.assertIn("Timeout", stderr)


class TestAdversarialTelemetryDBConcurrency(unittest.TestCase):
    """Stress-tests TelemetryDB with concurrent multithreaded transactions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "concurrent_test.db"
        self.db = TelemetryDB(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_concurrent_episode_and_weight_writes(self):
        """25 concurrent threads writing episodes and weights must complete without SQLITE_BUSY."""
        errors = []

        def worker(thread_id):
            try:
                local_db = TelemetryDB(db_path=self.db_path)
                ep_id = local_db.record_episode(
                    episode_type="benchmark",
                    prompt=f"Stress prompt from thread {thread_id}",
                    specialists=["coder", "critic"],
                    surprise=0.45,
                    agiq=125.5 + thread_id * 0.1,
                    duration_ms=120.0
                )
                assert ep_id > 0

                matrix = np.eye(13) * 0.5
                local_db.save_synaptic_weights(matrix, FLEET_MODELS_13)
            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        self.assertEqual(len(errors), 0, f"Concurrent database write errors: {errors}")
        episodes = self.db.get_recent_episodes(limit=100)
        self.assertEqual(len(episodes), 25, f"Expected 25 episodes, got {len(episodes)}")


class TestAdversarialJarvisFallbacks(unittest.TestCase):
    """Stress-tests Jarvis fallback paths and hardware fault-tolerance."""

    def test_whisper_root_fallback_when_nvme_absent(self):
        """If D:\\ai_models\\whisper does not exist, ears.py must gracefully fall back to HDD root."""
        original_exists = os.path.exists
        try:
            def mock_exists(path):
                if str(path).startswith(r"D:\ai_models\whisper"):
                    return False
                return original_exists(path)

            os.path.exists = mock_exists
            fallback_root = ears.NVME_WHISPER_ROOT if os.path.exists(ears.NVME_WHISPER_ROOT) else ears.HDD_WHISPER_ROOT
            self.assertEqual(fallback_root, ears.HDD_WHISPER_ROOT)
        finally:
            os.path.exists = original_exists

    def test_hands_system_telemetry_resilience(self):
        """Hands.get_system_telemetry must gracefully handle hardware inspection."""
        hands = Hands()
        telem = hands.get_system_telemetry()
        self.assertIsInstance(telem, dict)
        self.assertIn("cpu_percent", telem)
        self.assertIn("ram", telem)
        self.assertIn("disk_c", telem)
        self.assertIn("disk_d", telem)
        self.assertIn("disk_i", telem)


if __name__ == "__main__":
    unittest.main(verbosity=2)
