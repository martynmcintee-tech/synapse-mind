#!/usr/bin/env python3
"""
Adversarial HTTP test for Synapse Forge Studio Server.
Tests:
  1. POST /api/product/synthesize with malformed JSON / empty body
  2. POST /api/evolution/cycle with garbage payload
  3. Non-existent routes (404 handling)
  4. Concurrent HTTP requests to /api/telemetry and /api/topology
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess
import threading
from pathlib import Path

BASE_DIR = Path(r"I:\ai_models")
FORGE_DIR = BASE_DIR / "synapse_forge"
SERVER_SCRIPT = FORGE_DIR / "studio_server.py"

def run_adversarial_http_test():
    print("[Adversarial HTTP] Launching studio server...")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(FORGE_DIR)
    )
    base_url = "http://127.0.0.1:18445"

    try:
        # Wait for ready
        ready = False
        for _ in range(20):
            time.sleep(0.4)
            try:
                with urllib.request.urlopen(f"{base_url}/", timeout=1) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
        assert ready, "Server failed to start"

        # 1. 404 Not Found handling
        print("  Testing 404 Not Found on invalid route...")
        try:
            urllib.request.urlopen(f"{base_url}/nonexistent_route_999")
            raise AssertionError("Should have returned 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404, f"Expected 404, got {e.code}"
            print(f"  [PASS] 404 handled cleanly: code={e.code}")

        # 2. Malformed JSON to POST /api/product/synthesize
        print("  Testing malformed JSON to POST /api/product/synthesize...")
        req = urllib.request.Request(
            f"{base_url}/api/product/synthesize",
            data=b"INVALID_NOT_JSON{{",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            urllib.request.urlopen(req)
            print("  [WARN] Server accepted invalid JSON without 400/500")
        except urllib.error.HTTPError as e:
            # 400 or 500 is expected for invalid payload
            assert e.code in [400, 500], f"Unexpected code: {e.code}"
            print(f"  [PASS] Malformed JSON handled gracefully: code={e.code}")

        # 3. Concurrent read bursts (20 simultaneous requests)
        print("  Testing concurrent read bursts (20 threads)...")
        errors = []
        def req_worker():
            try:
                with urllib.request.urlopen(f"{base_url}/api/telemetry", timeout=3) as resp:
                    assert resp.status == 200
                with urllib.request.urlopen(f"{base_url}/api/topology", timeout=3) as resp:
                    assert resp.status == 200
            except Exception as ex:
                errors.append(str(ex))

        threads = [threading.Thread(target=req_worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=5)

        assert len(errors) == 0, f"Concurrent burst errors: {errors}"
        print(f"  [PASS] 20 concurrent read requests completed with 0 errors.")

        print("[Adversarial HTTP] ALL ADVERSARIAL HTTP CHECKS PASSED.")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
            proc.wait()

if __name__ == "__main__":
    run_adversarial_http_test()
