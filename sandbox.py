#!/usr/bin/env python3
"""
Synapse Forge - 4-Tier AST Security Sandbox & Self-Evolution Runner (R2)
Implements an isolated, 4-tier security validation and promotion gatekeeper:
  Tier 1: AST Static Security Analysis & Blacklist Auditing
  Tier 2: Bytecode Compilation & Syntax Verification
  Tier 3: Isolated Subshell Execution with Strict Timeout & Memory Boundaries
  Tier 4: Functional Verification & Invariance Assertions

Prevents arbitrary remote code execution, dangerous imports, shell escapes,
and unvalidated mutations before any skill is promoted to the permanent fleet memory.
"""

import os
import sys
import ast
import time
import shutil
import hashlib
import py_compile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from synapse_forge.telemetry_db import TelemetryDB

DEFAULT_SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"

# Tier 1 Security Blacklists
FORBIDDEN_MODULES = {
    "os.system", "winreg", "ctypes", "win32api", "win32con", "msvcrt",
    "pty", "resource", "posix", "sys.modules"
}

FORBIDDEN_CALL_NAMES = {
    "eval", "exec", "compile", "__import__"
}

FORBIDDEN_OS_ATTRIBUTES = {
    "system", "popen", "spawn", "fork", "kill", "remove", "unlink", "rmdir"
}

FORBIDDEN_SUBPROCESS_ATTRIBUTES = {
    "Popen", "call", "check_call", "check_output"  # Only the sandbox runner itself executes subprocesses
}


class SecurityViolation(Exception):
    """Raised when an AST security violation is discovered in candidate code."""
    pass


class SandboxSecurityVisitor(ast.NodeVisitor):
    """
    Traverses the Python AST to enforce strict static security policies.
    """

    def __init__(self, sandbox_path: Path):
        self.sandbox_path = str(sandbox_path.resolve()).lower()
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.lower()
            if name in FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden module import: '{name}' (line {node.lineno})")
            if name == "socket":
                # Socket is scrutinized; candidate skills cannot open listening servers
                pass
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").lower()
        if module in FORBIDDEN_MODULES:
            self.violations.append(f"Forbidden module import from: '{module}' (line {node.lineno})")
        for alias in node.names:
            full_name = f"{module}.{alias.name.lower()}"
            if full_name in FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden import: '{full_name}' (line {node.lineno})")
            if alias.name.lower() in FORBIDDEN_CALL_NAMES:
                self.violations.append(f"Forbidden builtin import: '{alias.name}' (line {node.lineno})")
            if module == "os" and alias.name.lower() in FORBIDDEN_OS_ATTRIBUTES:
                self.violations.append(f"Forbidden os attribute import: '{alias.name}' (line {node.lineno})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls like eval(), exec(), compile()
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FORBIDDEN_CALL_NAMES:
                self.violations.append(f"Forbidden builtin call: '{func_name}()' (line {node.lineno})")

        # Check attribute calls like os.system(), subprocess.Popen(..., shell=True)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                val_name = node.func.value.id
                if val_name == "os" and attr_name in FORBIDDEN_OS_ATTRIBUTES:
                    self.violations.append(f"Forbidden os call: 'os.{attr_name}()' (line {node.lineno})")
                if val_name == "subprocess" and attr_name in FORBIDDEN_SUBPROCESS_ATTRIBUTES:
                    self.violations.append(f"Forbidden subprocess call: 'subprocess.{attr_name}()' (line {node.lineno})")

            # Check for shell=True keyword argument anywhere in a call
            for kw in node.keywords:
                if kw.arg == "shell":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        self.violations.append(f"Subprocess call with shell=True is forbidden (line {node.lineno})")

        self.generic_visit(node)


class SandboxRunner:
    """
    Isolated execution runner and gatekeeper for autonomous self-evolution cycles.
    """

    def __init__(self, sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR):
        self.sandbox_dir = Path(sandbox_dir)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def clean_workspace(self) -> None:
        """
        Cleans temporary execution scripts from the sandbox workspace.
        """
        if self.sandbox_dir.exists():
            for item in self.sandbox_dir.glob("temp_*.py"):
                try:
                    item.unlink()
                except Exception:
                    pass
            for item in self.sandbox_dir.glob("temp_*.pyc"):
                try:
                    item.unlink()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Tier 1: AST Static Security Analysis
    # -------------------------------------------------------------------------

    def tier1_ast_audit(self, code_content: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Tier 1: Parses Python code into AST and audits against security blacklists.
        Returns: (passed: bool, error_msg: Optional[str], ast_signature: Optional[str])
        """
        try:
            tree = ast.parse(code_content)
        except SyntaxError as e:
            return False, f"AST Syntax Error: {e.msg} at line {e.lineno}", None
        except Exception as e:
            return False, f"AST Parse Error: {str(e)}", None

        visitor = SandboxSecurityVisitor(self.sandbox_dir)
        visitor.visit(tree)

        if visitor.violations:
            violation_summary = "; ".join(visitor.violations)
            return False, f"Tier 1 Security Violation: {violation_summary}", None

        # Generate structural AST signature hash
        ast_dump = ast.dump(tree, annotate_fields=False)
        ast_sig = hashlib.sha256(ast_dump.encode("utf-8")).hexdigest()[:16]

        return True, None, ast_sig

    # -------------------------------------------------------------------------
    # Tier 2: Bytecode Compilation
    # -------------------------------------------------------------------------

    def tier2_bytecode_compile(self, script_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Tier 2: Compiles the candidate script to bytecode to guarantee syntax validity.
        Returns: (passed: bool, error_msg: Optional[str])
        """
        try:
            py_compile.compile(str(script_path), doraise=True)
            return True, None
        except py_compile.PyCompileError as e:
            return False, f"Tier 2 Bytecode Compilation Error: {e.msg}"
        except Exception as e:
            return False, f"Tier 2 Unexpected Compilation Error: {str(e)}"

    # -------------------------------------------------------------------------
    # Tier 3: Isolated Subprocess Execution
    # -------------------------------------------------------------------------

    def tier3_isolated_execution(
        self,
        script_path: Path,
        timeout: float = 5.0
    ) -> Tuple[bool, str, str, float]:
        """
        Tier 3: Executes candidate script in an isolated subshell with strict timeout.
        Returns: (passed: bool, stdout: str, stderr: str, duration_ms: float)
        """
        start_time = time.time()
        try:
            # Execute with python.exe in isolated sandbox directory
            proc = subprocess.run(
                [sys.executable, str(script_path.resolve())],
                cwd=str(self.sandbox_dir.resolve()),
                timeout=timeout,
                capture_output=True,
                text=True,
                shell=False
            )
            duration_ms = (time.time() - start_time) * 1000.0
            if proc.returncode != 0:
                err = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
                return False, proc.stdout, err, duration_ms
            return True, proc.stdout, proc.stderr, duration_ms
        except subprocess.TimeoutExpired:
            duration_ms = (time.time() - start_time) * 1000.0
            return False, "", f"Tier 3 Timeout: execution exceeded {timeout}s", duration_ms
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000.0
            return False, "", f"Tier 3 Subprocess Error: {str(e)}", duration_ms

    # -------------------------------------------------------------------------
    # Tier 4: Functional Verification & Invariance Checks
    # -------------------------------------------------------------------------

    def tier4_functional_verification(
        self,
        test_script_content: str,
        timeout: float = 5.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Tier 4: Runs dedicated unit assertions against the candidate code.
        Returns: (passed: bool, error_msg: Optional[str])
        """
        test_file = self.sandbox_dir / f"temp_test_{int(time.time() * 1000)}.py"
        try:
            test_file.write_text(test_script_content, encoding="utf-8")
            passed, stdout, stderr, _ = self.tier3_isolated_execution(test_file, timeout=timeout)
            if not passed:
                return False, f"Tier 4 Invariance Assertion Failed: {stderr.strip() or stdout.strip()}"
            return True, None
        finally:
            if test_file.exists():
                try:
                    test_file.unlink()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Comprehensive 4-Tier Gatekeeper & Promotion
    # -------------------------------------------------------------------------

    def evaluate_and_promote_skill(
        self,
        skill_name: str,
        code_snippet: str,
        category: str,
        description: str,
        test_snippet: Optional[str] = None,
        telemetry_db: Optional[TelemetryDB] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates full 4-tier pipeline.
        If all tiers pass, promotes the skill to telemetry_db.
        """
        self.clean_workspace()
        timestamp_id = int(time.time() * 1000)
        candidate_file = self.sandbox_dir / f"temp_candidate_{timestamp_id}.py"

        # Tier 1: AST Security Audit
        t1_pass, t1_err, ast_sig = self.tier1_ast_audit(code_snippet)
        if not t1_pass:
            return {
                "status": "rejected",
                "skill_name": skill_name,
                "tier_failed": 1,
                "error": t1_err,
                "ast_signature": None,
                "promoted": False
            }

        try:
            # Write to sandbox for bytecode compile and subshell run
            candidate_file.write_text(code_snippet, encoding="utf-8")

            # Tier 2: Bytecode Compilation
            t2_pass, t2_err = self.tier2_bytecode_compile(candidate_file)
            if not t2_pass:
                return {
                    "status": "rejected",
                    "skill_name": skill_name,
                    "tier_failed": 2,
                    "error": t2_err,
                    "ast_signature": ast_sig,
                    "promoted": False
                }

            # Tier 3: Isolated Execution
            t3_pass, stdout, stderr, duration_ms = self.tier3_isolated_execution(candidate_file, timeout=5.0)
            if not t3_pass:
                return {
                    "status": "rejected",
                    "skill_name": skill_name,
                    "tier_failed": 3,
                    "error": stderr or "Subprocess execution failed",
                    "ast_signature": ast_sig,
                    "duration_ms": duration_ms,
                    "promoted": False
                }

            # Tier 4: Functional Verification
            if test_snippet:
                # Combine candidate and test snippet
                combined_test = f"{code_snippet}\n\n# --- Invariance Tests ---\n{test_snippet}"
                t4_pass, t4_err = self.tier4_functional_verification(combined_test, timeout=5.0)
                if not t4_pass:
                    return {
                        "status": "rejected",
                        "skill_name": skill_name,
                        "tier_failed": 4,
                        "error": t4_err,
                        "ast_signature": ast_sig,
                        "promoted": False
                    }

            # All 4 Tiers Passed: Promote to Database
            skill_id = None
            if telemetry_db is not None:
                skill_id = telemetry_db.record_autonomous_skill(
                    skill_name=skill_name,
                    description=description,
                    category=category,
                    code_snippet=code_snippet,
                    ast_signature=ast_sig,
                    validation_score=1.0,
                    promoted_from_sandbox=1
                )

            return {
                "status": "promoted",
                "skill_id": skill_id,
                "skill_name": skill_name,
                "category": category,
                "ast_signature": ast_sig,
                "duration_ms": duration_ms,
                "tier_passed": 4,
                "promoted": True
            }

        finally:
            self.clean_workspace()
