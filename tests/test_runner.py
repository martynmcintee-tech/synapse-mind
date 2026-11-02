#!/usr/bin/env python3
"""
Synapse Forge - Unified 4-Tier E2E Test Suite Runner
Orchestrates and executes the complete 4-tier opaque-box test framework:
  Tier 1: Feature Coverage (test_tier1_features.py)
  Tier 2: Boundary & Corner Cases (test_tier2_boundaries.py)
  Tier 3: Cross-Feature Integration (test_tier3_cross_feature.py)
  Tier 4: Real-World Commercial Scenarios (test_tier4_real_world.py)

CLI Options:
  python test_runner.py --all
  python test_runner.py --tier 1
  python test_runner.py --tier 2
  python test_runner.py --tier 3
  python test_runner.py --tier 4
  python test_runner.py --verbose
  python test_runner.py --fail-fast

Provides ANSI terminal formatting, timing diagnostics, summary matrix, and strict exit codes.
"""

import os
import sys
import time
import argparse
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure paths
TESTS_DIR = Path(__file__).resolve().parent
FORGE_DIR = TESTS_DIR.parent
BASE_DIR = FORGE_DIR.parent
JARVIS_DIR = BASE_DIR / "jarvis"

# Reconfigure stdout/stderr for UTF-8 on Windows with replace fallback
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

for p in [str(BASE_DIR), str(FORGE_DIR), str(JARVIS_DIR), str(TESTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class FormattedTestResult(unittest.TestResult):
    """
    Custom TestResult capturing timing, outcome badges, and detailed failure diagnostics.
    """

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.test_records: List[Dict[str, Any]] = []
        self._current_start_time = 0.0

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self._current_start_time = time.perf_counter()

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        elapsed = time.perf_counter() - self._current_start_time
        doc = (test._testMethodDoc or "").strip() or test._testMethodName
        self.test_records.append({
            "name": test._testMethodName,
            "class": test.__class__.__name__,
            "doc": doc,
            "status": "PASS",
            "duration": elapsed,
            "error": None
        })
        if self.verbose:
            print(f"  {GREEN}[PASS]{RESET} {test._testMethodName} ({elapsed:.3f}s) - {DIM}{doc}{RESET}")

    def addFailure(self, test: unittest.TestCase, err: Tuple) -> None:
        super().addFailure(test, err)
        elapsed = time.perf_counter() - self._current_start_time
        doc = (test._testMethodDoc or "").strip() or test._testMethodName
        err_msg = self._exc_info_to_string(err, test)
        self.test_records.append({
            "name": test._testMethodName,
            "class": test.__class__.__name__,
            "doc": doc,
            "status": "FAIL",
            "duration": elapsed,
            "error": err_msg
        })
        print(f"  {RED}[FAIL]{RESET} {test._testMethodName} ({elapsed:.3f}s)")
        print(f"{DIM}{err_msg}{RESET}")

    def addError(self, test: unittest.TestCase, err: Tuple) -> None:
        super().addError(test, err)
        elapsed = time.perf_counter() - self._current_start_time
        doc = (test._testMethodDoc or "").strip() or test._testMethodName
        err_msg = self._exc_info_to_string(err, test)
        self.test_records.append({
            "name": test._testMethodName,
            "class": test.__class__.__name__,
            "doc": doc,
            "status": "ERROR",
            "duration": elapsed,
            "error": err_msg
        })
        print(f"  {RED}[ERROR]{RESET} {test._testMethodName} ({elapsed:.3f}s)")
        print(f"{DIM}{err_msg}{RESET}")

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        elapsed = time.perf_counter() - self._current_start_time
        doc = (test._testMethodDoc or "").strip() or test._testMethodName
        self.test_records.append({
            "name": test._testMethodName,
            "class": test.__class__.__name__,
            "doc": doc,
            "status": "SKIP",
            "duration": elapsed,
            "error": reason
        })
        if self.verbose:
            print(f"  {YELLOW}[SKIP]{RESET} {test._testMethodName} - {reason}")


def print_banner():
    banner = f"""
{CYAN}{BOLD}======================================================================
     SYNAPSE FORGE -- 4-TIER E2E AUTONOMOUS AGI TEST FRAMEWORK
======================================================================{RESET}
  {DIM}Autonomous Software Factory & Evolving AGI Studio | RTX 3090 Direct I/O{RESET}
"""
    print(banner)


def run_tier(tier_num: int, verbose: bool = False, fail_fast: bool = False) -> Dict[str, Any]:
    """
    Loads and runs test module for the specified tier.
    """
    tier_modules = {
        1: ("Tier 1: Feature Coverage", "test_tier1_features"),
        2: ("Tier 2: Boundary & Invariance", "test_tier2_boundaries"),
        3: ("Tier 3: Cross-Feature Integration", "test_tier3_cross_feature"),
        4: ("Tier 4: Real-World Scenarios", "test_tier4_real_world")
    }

    tier_title, module_name = tier_modules[tier_num]
    print(f"\n{BOLD}{MAGENTA}>> Running {tier_title}{RESET} ({module_name}.py)")
    print(f"{DIM}{'-' * 70}{RESET}")

    start_time = time.perf_counter()
    loader = unittest.TestLoader()

    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception as e:
        print(f"{RED}Failed to load module {module_name}: {e}{RESET}")
        return {
            "tier": tier_num,
            "title": tier_title,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "duration": 0.0,
            "success": False
        }

    result = FormattedTestResult(verbose=verbose)
    if fail_fast:
        result.failfast = True

    suite.run(result)
    tier_duration = time.perf_counter() - start_time

    passed = len([r for r in result.test_records if r["status"] == "PASS"])
    failed = len([r for r in result.test_records if r["status"] == "FAIL"])
    errors = len([r for r in result.test_records if r["status"] == "ERROR"])
    skipped = len([r for r in result.test_records if r["status"] == "SKIP"])
    total = len(result.test_records)
    is_success = (failed == 0 and errors == 0)

    status_badge = f"{GREEN}PASSED{RESET}" if is_success else f"{RED}FAILED{RESET}"
    print(f"{DIM}{'-' * 70}{RESET}")
    print(f"  {status_badge}: {passed}/{total} tests passed ({tier_duration:.2f}s)")

    return {
        "tier": tier_num,
        "title": tier_title,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "duration": tier_duration,
        "success": is_success
    }


def print_summary_table(results: List[Dict[str, Any]]):
    """
    Renders an ASCII summary table across all executed tiers.
    """
    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{BOLD}                        TEST EXECUTION SUMMARY                        {RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{BOLD}{'Tier / Test Suite':<35} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Time (s)':<10}{RESET}")
    print(f"{DIM}{'-' * 70}{RESET}")

    grand_total = 0
    grand_passed = 0
    grand_failed = 0
    grand_errors = 0
    grand_duration = 0.0

    for r in results:
        status_color = GREEN if r["success"] else RED
        print(f"{status_color}{r['title']:<35}{RESET} {r['total']:<8} {GREEN}{r['passed']:<8}{RESET} {RED if r['failed'] or r['errors'] else DIM}{r['failed'] + r['errors']:<8}{RESET} {r['duration']:<10.2f}")
        grand_total += r["total"]
        grand_passed += r["passed"]
        grand_failed += r["failed"]
        grand_errors += r["errors"]
        grand_duration += r["duration"]

    print(f"{DIM}{'-' * 70}{RESET}")
    total_color = GREEN if (grand_failed == 0 and grand_errors == 0) else RED
    print(f"{BOLD}{'GRAND TOTAL':<35} {grand_total:<8} {GREEN}{grand_passed:<8}{RESET} {total_color}{grand_failed + grand_errors:<8}{RESET} {grand_duration:<10.2f}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")

    if grand_failed == 0 and grand_errors == 0 and grand_total > 0:
        print(f"\n  {GREEN}{BOLD}[OK] ALL {grand_total} TEST CASES PASSED WITH 100% SPECIFICATION FIDELITY!{RESET}\n")
    else:
        print(f"\n  {RED}{BOLD}[X] TEST SUITE FAILED: {grand_failed} failures, {grand_errors} errors.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Synapse Forge E2E Test Suite Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific test tier (1-4)")
    parser.add_argument("--all", action="store_true", help="Run all 4 test tiers (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose per-test output")
    parser.add_argument("--fail-fast", "-f", action="store_true", help="Stop execution on first failure")

    args = parser.parse_args()
    print_banner()

    tiers_to_run = [args.tier] if args.tier else [1, 2, 3, 4]

    results = []
    overall_success = True

    for t in tiers_to_run:
        res = run_tier(t, verbose=args.verbose, fail_fast=args.fail_fast)
        results.append(res)
        if not res["success"]:
            overall_success = False
            if args.fail_fast:
                break

    print_summary_table(results)
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
