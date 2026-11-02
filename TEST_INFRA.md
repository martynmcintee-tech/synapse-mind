# Synapse Forge & Jarvis E2E Test Infrastructure

## 1. Overview & Testing Philosophy

Synapse Forge is an autonomous software factory and commercial graphical AGI studio that integrates a 13-node local neural model fleet on an NVIDIA RTX 3090 (24GB VRAM) and Samsung 990 PRO NVMe SSD.

The **Synapse Forge E2E Testing Framework** enforces an **opaque-box specification testing** methodology:
- Every test is strictly derived from the authoritative specifications in `ORIGINAL_REQUEST.md` and `PROJECT.md`.
- Tests verify observable functional contracts, mathematical invariants, database schemas, AST security enforcement, async HTTP/WebSocket API specifications, and real-world commercial product synthesis.
- **Zero-Tolerance Integrity Policy**: Absolutely no facade tests, dummy assertions (`assert True`), or tests tailored to mask defects. Every test exercises real computation, validates edge cases, and tests against strict analytical ground truth.

---

## 2. 4-Tier Test Architecture

The testing suite is structured into four distinct, progressive tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Synapse Forge Test Suite                        │
├───────────────────┬────────────────────────────────────────────────────┤
│ Tier 1: Features  │ Unit and functional coverage across all R1-R4      │
│                   │ features (>=5 tests per feature).                  │
├───────────────────┼────────────────────────────────────────────────────┤
│ Tier 2: Boundary  │ Invariance stress, numerical stability, zero/inf,  │
│                   │ AST security injections, corrupted payloads.       │
├───────────────────┼────────────────────────────────────────────────────┤
│ Tier 3: Cross     │ Multi-module state flows: Episode -> Oja -> DB     │
│                   │ -> AGIQ -> WebSocket; Sandbox Evolution -> Skill.  │
├───────────────────┼────────────────────────────────────────────────────┤
│ Tier 4: Real-World│ Commercial product synthesis bundle generation,    │
│                   │ storefront customer journey, Jarvis MoE turns.     │
└───────────────────┴────────────────────────────────────────────────────┘
```

### 2.1 Tier 1: Feature Coverage (`test_tier1_features.py`)
Provides granular, direct behavioral tests for every capability defined in the Feature Inventory:
1. **Multi-Model Construction Pipeline (R1)**: Execution, prompt dispatch, timing extraction (`prompt_ms`, `predicted_ms`), and subsystem author tracking.
2. **Provenance & Latency Telemetry (R1)**: Verification of `build_provenance.json` schema, hardware environment details, NVMe load latency calculations, and generation throughput.
3. **Subsystem Integrity Hashing (R1)**: SHA-256 hashing and `py_compile` bytecode validation across all authored subsystems.
4. **Oja Hebbian Synaptic Plasticity (R2)**: Normalized pairwise updates $\Delta W_{ij} = \eta (a_i a_j - W_{ij} a_j^2)$, homeostatic passive decay ($\lambda = 0.005$), bounded clipping $[0.001, 1.0]$, and lateral spreading associative activation ($A^{(t+1)} = \text{Normalize}((1-\alpha)A^{(0)} + \alpha W^T A^{(t)})$).
5. **Bayesian Surprise & Free-Energy Gating (R2)**: 1024-dim BGE-M3 cosine similarity novelty, Temporal Difference error $\mathcal{E}_{\text{prediction}}$, Continuous Hopfield energy $E(e_x) = -\beta^{-1}\log(\sum \exp(\beta e_x^T v_k)) + 0.5\|e_x\|^2$, and composite surprise gating threshold ($\theta = 0.42$).
6. **Composite AGIQ Metric (R2)**: Strict implementation of the 6-dimensional weighted metric scaled to $[0.0, 200.0]$ with verified monotonicity and numerical safeguards.
7. **Relational Telemetry Store (R2)**: SQLite WAL mode database (`synapse_forge.db`) covering all 6 tables (`cognitive_episodes`, `model_contributions`, `synaptic_weights`, `autonomous_skills`, `commercial_products`, `agiq_history`), foreign keys, and indexes.
8. **Society-of-Mind Product Synthesis (R2)**: Multi-agent cooperative generation loop (Architect -> SQL -> Coder -> Critic -> Marketer).
9. **4-Tier AST Security Sandbox (R2)**: AST static blacklist inspection, bytecode compilation, subshell isolation, and skill promotion.
10. **Async Studio Server & Endpoints (R3)**: `aiohttp.web` server on port 18445 serving `/`, `/store`, `/ws`, `/api/evolution/cycle`, `/api/product/synthesize`, `/api/bundle/export`, and `/api/provenance`.
11. **13-Node Neural Canvas & Hardware HUD (R3)**: Topology node/edge state serialization, live GPU (RTX 3090 24GB), NVMe (`D:\`), CPU, and RAM telemetry feeds.
12. **Jarvis NVMe Acoustic Perception & MoE Routing (R4)**: `ears.py` Whisper path configuration, `brain.py` dynamic model selection (no hardcoded "coder"), and `hands.py` `disk_d` telemetry.

### 2.2 Tier 2: Boundary & Corner Cases (`test_tier2_boundaries.py`)
Adversarial and extreme edge testing:
- **Numerical Invariance**: Zero activation vectors ($a = \vec{0}$), saturated activation vectors ($a = \vec{1}$), extreme learning rates, high-frequency multi-turn updates (1000 turns) checking for NaN, Inf, or weight explosion.
- **Continuous Hopfield Stability**: Colinear vectors, orthogonal vectors, large $\beta$ values testing log-sum-exp stabilization against numerical overflow.
- **Database Resilience**: Empty queries, large text payloads (1MB prompts), malicious SQL injection patterns (`' OR '1'='1`, `'; DROP TABLE ...`), foreign-key violation traps, and concurrent read/write transactions under WAL.
- **AST Security Penetration**: Malicious scripts attempting:
  * `import os; os.system("rmdir /s /q C:\\")`
  * `import subprocess; subprocess.Popen(["cmd.exe", "/c", "..."], shell=True)`
  * `import ctypes, winreg, socket`
  * `eval("1+1")`, `exec(...)`, `compile(...)`
  * Infinite loops (`while True: pass`) testing 5-second subprocess timeout termination.
  * Broken syntax and non-zero exit codes.
- **HTTP/WS Boundary Conditions**: Malformed JSON bodies, missing query parameters, negative IDs, invalid endpoints (404), empty product synthesis requests.
- **Jarvis Router Boundaries**: Empty user query, whitespace-only queries, extreme length prompts (>50,000 characters), ambiguous keywords, unknown domains.

### 2.3 Tier 3: Cross-Feature State Flows (`test_tier3_cross_feature.py`)
Validates end-to-end multi-module pipelines:
- **Full Episode Cycle**: User prompt -> MoE classification -> Specialist activation -> Oja synaptic weight adaptation -> SQLite episode & contribution record -> AGIQ recalculation -> WebSocket payload generation.
- **Autonomous Evolution Loop**: Proposal generation -> AST static security audit -> Bytecode compilation -> Isolated sandbox execution -> Skill promotion into `autonomous_skills` -> Synaptic potentiation between participating models -> Telemetry recording.
- **Evolution Rejection & Defense**: Buggy/malicious skill proposal -> AST/compile failure -> Immediate quarantine -> Sandbox directory purged -> No skill promoted -> Negative TD error logged.
- **Product Synthesis & Packaging Flow**: Generation across 5 specialist models -> Multi-file project structure authored on disk -> SHA-256 provenance checksum -> Commercial product entry in SQLite -> Release bundle packaged into ZIP.
- **Dual Database & System Coexistence**: Concurrent transactions between Jarvis `memory.db` and Synapse Forge `synapse_forge.db` without database lock errors.

### 2.4 Tier 4: Real-World Commercial Scenarios (`test_tier4_real_world.py`)
Exercises production-level user stories:
- **Commercial Software Product Synthesis**: Synthesis of a production-ready software product (e.g., "FastAPI Distributed Rate Limiter & Auth Gateway"), complete with `main.py`, `auth.py`, `config.py`, `test_main.py`, SQL migration schema, comprehensive `README.md`, and commercial sales copy.
- **Autonomous Jarvis Conversational Turns**: Realistic multi-turn dialog covering coding, formal logic proofs, instant conversational queries, and security analysis, verifying dynamic model selection without regression to hardcoded models.
- **Commercial Storefront Customer Journey**: Emulates visitor browsing `/store`, evaluating product tiers (Community, Professional, Enterprise), inspecting the feature matrix, and running the interactive demo embed.
- **Long-Horizon AGIQ Trajectory**: Simulates 20 consecutive evolutionary cognitive episodes, verifying consistent monotonic convergence of synaptic weights and bounded AGIQ progression within $[0.0, 200.0]$.

---

## 3. Test Runner Invocation

The test suite is driven by `test_runner.py` located at `I:\ai_models\synapse_forge\tests\test_runner.py`.

### 3.1 CLI Usage

```powershell
# Run all 4 test tiers
python I:\ai_models\synapse_forge\tests\test_runner.py --all

# Run individual test tiers
python I:\ai_models\synapse_forge\tests\test_runner.py --tier 1
python I:\ai_models\synapse_forge\tests\test_runner.py --tier 2
python I:\ai_models\synapse_forge\tests\test_runner.py --tier 3
python I:\ai_models\synapse_forge\tests\test_runner.py --tier 4

# Run with verbose output (details on each individual test case)
python I:\ai_models\synapse_forge\tests\test_runner.py --all --verbose

# Run with fail-fast (stops on first failure)
python I:\ai_models\synapse_forge\tests\test_runner.py --all --fail-fast

# Standard pytest compatibility
pytest I:\ai_models\synapse_forge\tests -v
```

### 3.2 Output Formatting & Exit Codes
The runner provides:
- Colored ANSI header and status badges.
- Execution timing per test tier and individual test.
- Comprehensive summary table showing total tests, passes, failures, errors, skips, and execution time.
- Strict exit codes:
  * `0`: All tests passed cleanly.
  * `1`: One or more tests failed or encountered errors.

---

## 4. Directory Layout

```
I:\ai_models\
├── synapse_forge\
│   ├── tests\
│   │   ├── __init__.py
│   │   ├── test_runner.py              # Unified CLI test runner
│   │   ├── test_tier1_features.py       # Tier 1 Feature coverage tests
│   │   ├── test_tier2_boundaries.py     # Tier 2 Boundary & invariance tests
│   │   ├── test_tier3_cross_feature.py  # Tier 3 Cross-feature integration tests
│   │   └── test_tier4_real_world.py     # Tier 4 Real-world commercial scenarios
│   ├── TEST_INFRA.md                   # Co-located testing documentation
│   └── TEST_READY.md                   # Co-located test readiness verification
├── TEST_READY.md                       # Root test readiness specification
└── .agents\teamwork\
    └── orchestrator\
        └── TEST_INFRA.md               # Orchestrator copy
```

---

## 5. Authoritative Verification Matrix

| Subsystem / Feature | Authoritative Specification | Verification Method |
|---|---|---|
| Oja Synaptic Plasticity | Oja (1982), `PROJECT.md` § 3.1 | NumPy numerical verification, zero NaN across multi-turn sweeps |
| Continuous Hopfield Energy | Ramsauer et al. (2020), `PROJECT.md` § 3.2 | Log-sum-exp numerical stability vs analytical reference |
| Composite AGIQ Metric | `PROJECT.md` § 3.3 | Bounded convex sum, $[0.0, 200.0]$ interval check, sensitivity checks |
| Telemetry Database | `PROJECT.md` § 4 | SQLite table introspection, WAL mode verification, CRUD assertions |
| 4-Tier Security Sandbox | `ORIGINAL_REQUEST.md` R2.3, `PROJECT.md` § 5.2 | AST blacklist parse, py_compile execution, timeout termination |
| Async Studio Server | `ORIGINAL_REQUEST.md` R3, `PROJECT.md` § 5 | aiohttp test client, HTTP status 200/201, WebSocket frames |
| Jarvis MoE Router | `ORIGINAL_REQUEST.md` R4, `PROJECT.md` § 6 | Gating distribution checks, `Brain` payload dynamic routing checks |
