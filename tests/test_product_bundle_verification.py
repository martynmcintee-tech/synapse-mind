#!/usr/bin/env python3
"""
Empirical verification for Product Synthesis & ZIP Bundle Export.
Validates:
  1. synapse_hyperscale_gateway directory contents (app.py, core.py, README.md, landing_page.html, schema.sql, manifest.json, test_suite.py)
  2. Syntactic validity of code files via py_compile
  3. SQL schema validity in SQLite
  4. ZIP release archive validity in releases directory
  5. End-to-end extraction and content integrity
"""

import os
import sys
import json
import zipfile
import sqlite3
import tempfile
import py_compile
from pathlib import Path

BASE_DIR = Path(r"I:\ai_models")
FORGE_DIR = BASE_DIR / "synapse_forge"
PRODUCTS_DIR = FORGE_DIR / "products"
RELEASES_DIR = FORGE_DIR / "releases"
PRODUCT_DIR = PRODUCTS_DIR / "synapse_hyperscale_gateway"

def verify_product_bundle():
    print(f"[Product Verification] Verifying product bundle at {PRODUCT_DIR}...")
    assert PRODUCT_DIR.exists(), f"Product dir {PRODUCT_DIR} does not exist"

    # 1. Verify required files
    required_files = [
        "manifest.json",
        "README.md",
        "landing_page.html",
        "schema.sql",
        "core.py",
        "app.py",
        "test_suite.py"
    ]
    for rf in required_files:
        fpath = PRODUCT_DIR / rf
        assert fpath.exists(), f"Missing required file: {rf}"
        content = fpath.read_text(encoding="utf-8")
        assert len(content) > 50, f"File {rf} is suspiciously small ({len(content)} chars)"
        print(f"  [PASS] File exists: {rf} ({len(content)} chars)")

    # 2. Verify Python syntax
    for py_file in ["core.py", "app.py", "test_suite.py"]:
        p = PRODUCT_DIR / py_file
        py_compile.compile(str(p), doraise=True)
        print(f"  [PASS] Python syntax clean: {py_file}")

    # 3. Verify SQL schema
    schema_path = PRODUCT_DIR / "schema.sql"
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name
    try:
        conn = sqlite3.connect(temp_db_path)
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert len(tables) > 0, "No tables created by schema.sql"
        print(f"  [PASS] SQL schema valid. Created tables: {tables}")
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

    # 4. Verify manifest.json
    manifest_path = PRODUCT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "product_title" in manifest or "product_name" in manifest or "name" in manifest, "Missing product title in manifest"
    print(f"  [PASS] Manifest JSON valid. Product: {manifest.get('product_title') or manifest.get('product_name') or manifest.get('name')}")

    # 5. Verify ZIP archive in releases directory
    zip_candidates = list(RELEASES_DIR.glob("*synapse_hyperscale_gateway*.zip"))
    assert len(zip_candidates) > 0, "No ZIP bundle found for synapse_hyperscale_gateway in releases/"
    zip_path = zip_candidates[0]
    print(f"[Product Verification] Verifying ZIP archive: {zip_path.name} ({zip_path.stat().st_size} bytes)...")
    assert zipfile.is_zipfile(zip_path), f"File {zip_path} is not a valid zip archive"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"  [PASS] ZIP contents ({len(namelist)} items): {namelist}")
        # Verify archive includes core deliverables
        names_lower = [n.lower() for n in namelist]
        assert any("manifest.json" in n for n in names_lower), "Missing manifest.json in ZIP"
        assert any("readme" in n for n in names_lower), "Missing README in ZIP"
        assert any("landing_page" in n or ".html" in n for n in names_lower), "Missing landing page in ZIP"
        assert any(".py" in n for n in names_lower), "Missing Python code in ZIP"
        
        # Test integrity check (testzip returns None if all CRC32 match)
        corrupt = zf.testzip()
        assert corrupt is None, f"Corrupted file in ZIP: {corrupt}"
        print(f"  [PASS] CRC32 checksums verified for all files in {zip_path.name}")

    print("[Product Verification] ALL PRODUCT SYNTHESIS & ZIP BUNDLE CHECKS PASSED (100% PASS).")

if __name__ == "__main__":
    verify_product_bundle()
