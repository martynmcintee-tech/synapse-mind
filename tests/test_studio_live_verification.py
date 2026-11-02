#!/usr/bin/env python3
"""
Empirical verification script for Synapse Forge Studio Server.
Tests startup and queries:
  - GET /
  - GET /store
  - GET /api/telemetry
  - GET /api/topology
  - GET /api/agiq
  - GET /api/provenance
  - GET /api/bundle/export
"""

import os
import sys
import time
import json
import urllib.request
import subprocess
from pathlib import Path

BASE_DIR = Path(r"I:\ai_models")
FORGE_DIR = BASE_DIR / "synapse_forge"
SERVER_SCRIPT = FORGE_DIR / "studio_server.py"

def run_live_verification():
    print("[Verification] Starting studio_server.py on http://127.0.0.1:18445...")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(FORGE_DIR)
    )

    endpoints = [
        ("/", "GET / (Interactive Studio HTML)", "text/html"),
        ("/store", "GET /store (Commercial Storefront HTML)", "text/html"),
        ("/api/telemetry", "GET /api/telemetry (Hardware HUD JSON)", "application/json"),
        ("/api/topology", "GET /api/topology (13-Node Topology JSON)", "application/json"),
        ("/api/agiq", "GET /api/agiq (AGIQ Score JSON)", "application/json"),
        ("/api/provenance", "GET /api/provenance (Provenance JSON)", "application/json"),
        ("/api/bundle/export", "GET /api/bundle/export (ZIP release export)", "application/zip"),
    ]

    results = {}
    base_url = "http://127.0.0.1:18445"

    try:
        # Wait for server to bind
        ready = False
        for attempt in range(20):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{base_url}/", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
        
        if not ready:
            stdout, stderr = proc.communicate(timeout=2)
            raise RuntimeError(f"Server failed to start in time. Stderr: {stderr.decode('utf-8', errors='replace')}")

        print("[Verification] Server is ready on port 18445. Testing endpoints...")

        for path, description, expected_ct in endpoints:
            url = f"{base_url}{path}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
                ct = resp.headers.get("Content-Type", "")
                data = resp.read()
                results[path] = {
                    "description": description,
                    "status": status,
                    "content_type": ct,
                    "bytes_received": len(data)
                }
                print(f"  [PASS] {description} -> Status: {status}, Content-Type: {ct}, Size: {len(data)} bytes")

                if "application/json" in ct:
                    json_data = json.loads(data.decode("utf-8"))
                    results[path]["json_keys"] = list(json_data.keys()) if isinstance(json_data, dict) else len(json_data)
                    print(f"         JSON structure verified: keys={results[path]['json_keys']}")
                elif "text/html" in ct:
                    html_snippet = data.decode("utf-8", errors="replace")[:100].replace("\n", " ")
                    print(f"         HTML snippet: {html_snippet}")

        # Assert mandatory endpoints
        assert results["/"]["status"] == 200, "GET / failed"
        assert results["/store"]["status"] == 200, "GET /store failed"
        assert results["/api/telemetry"]["status"] == 200, "GET /api/telemetry failed"
        assert results["/api/topology"]["status"] == 200, "GET /api/topology failed"

        # Validate topology node count
        topo_data = json.loads(urllib.request.urlopen(f"{base_url}/api/topology").read().decode("utf-8"))
        nodes = topo_data.get("nodes", [])
        assert len(nodes) == 13, f"Expected 13 topology nodes, got {len(nodes)}"
        print(f"[Verification] Verified exactly {len(nodes)} topology nodes in /api/topology.")

        # Validate telemetry structure
        telem_data = json.loads(urllib.request.urlopen(f"{base_url}/api/telemetry").read().decode("utf-8"))
        assert "gpu" in telem_data, "Missing 'gpu' in /api/telemetry"
        assert "nvme" in telem_data, "Missing 'nvme' in /api/telemetry"
        assert "cpu_ram" in telem_data, "Missing 'cpu_ram' in /api/telemetry"
        print(f"[Verification] Verified hardware telemetry fields: GPU={telem_data['gpu'].get('name')}, NVMe={telem_data['nvme'].get('drive')}")

        print("[Verification] ALL STUDIO SERVER ENDPOINTS VERIFIED SUCCESSFULLY (100% PASS).")
        return results

    finally:
        print("[Verification] Shutting down studio server process...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("[Verification] Studio server shut down cleanly.")

if __name__ == "__main__":
    run_live_verification()
