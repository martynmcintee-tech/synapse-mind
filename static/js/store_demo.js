/**
 * Synapse Forge - Interactive Storefront Simulator (store_demo.js)
 * Implements:
 *   - Multi-Model product synthesis interactive simulation embed
 *   - 5-stage pipeline stepping (Architect -> SQL -> Coder -> Critic -> Marketer)
 *   - Code, documentation, and commercial copy tab previewer
 *   - Monthly vs Perpetual pricing tier switcher with dynamic calculation
 * 100% self-contained - zero external dependencies.
 */

(function () {
  'use strict';

  // Demo Product Presets
  const PRESETS = {
    cache_go: {
      name: "Synapse Lock-Free Memory Cache (Go)",
      stages: [
        { model: "Qwen3.8-27B-UD", desc: "Partitioned Sharded Map with Atomic Pointer Swaps" },
        { model: "sqlcoder-7b-2", desc: "SQLite Write-Ahead Snapshot & Recovery Schema" },
        { model: "Qwen2.5-Coder-32B", desc: "Go 1.23 sync.Pool & Atomic CAS Ring Buffer" },
        { model: "DeepSeek-R1-Distill-32B", desc: "Formal Lock-Freedom & Deadlock-Free Proof" },
        { model: "gemma-2-27b-it", desc: "Enterprise Sales Landing Copy & Benchmark Docs" }
      ],
      code: `package main

import (
\t"fmt"
\t"sync/atomic"
\t"unsafe"
)

// SynapseCache represents a sub-microsecond lock-free cache shard.
type CacheEntry struct {
\tKey   string
\tValue unsafe.Pointer
\tTTL   int64
}

type SynapseCache struct {
\tentries unsafe.Pointer // *[]*CacheEntry
\tcount   int64
}

func NewSynapseCache(capacity int) *SynapseCache {
\tslice := make([]*CacheEntry, capacity)
\treturn &SynapseCache{
\t\tentries: unsafe.Pointer(&slice),
\t}
}

func (c *SynapseCache) Store(idx int, val unsafe.Pointer) {
\tslice := (*[]*CacheEntry)(atomic.LoadPointer(&c.entries))
\tentry := &CacheEntry{Value: val}
\tatomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&(*slice)[idx])), unsafe.Pointer(entry))
\tatomic.AddInt64(&c.count, 1)
}

func main() {
\tcache := NewSynapseCache(1024)
\tfmt.Println("Synapse Lock-Free Cache Online. Latency: <140ns")
}`,
      docs: `# Synapse Lock-Free Cache (Go)
High-performance, zero-allocation lock-free in-memory cache synthesized autonomously by Synapse Forge.

## Architectural Highlights
- **Sub-Microsecond Latency**: Direct CAS atomic pointer swaps without mutex locks.
- **Hardware Optimized**: Built specifically for modern multi-core NVMe architectures.
- **Formally Verified**: Mathematical proofs conducted by DeepSeek-R1-Distill-32B confirming zero race conditions.

## Benchmarks
| Concurrency | Throughput | 99th % Latency |
|:---|:---|:---|
| 100 Goroutines | 14.8M ops/sec | 0.12 μs |
| 1000 Goroutines| 12.1M ops/sec | 0.28 μs |`,
      copy: `<div class="saas-highlight">
  <h3>Stop Bleeding Cloud Redis Fees.</h3>
  <p>Run your mission-critical caching layer directly in-process with zero network hops and zero locking overhead.</p>
  <ul>
    <li>14M+ operations/sec per core</li>
    <li>Zero runtime allocations</li>
    <li>Embedded SQLite WAL persistence fallback</li>
  </ul>
  <a href="#pricing" class="btn-cta">Deploy Now for $99</a>
</div>`,
      pricing: `{
  "product_id": "synapse-cache-go-pro",
  "version": "1.0.4",
  "tier": "professional",
  "price_usd": 99.0,
  "license": "Commercial Redistribution",
  "included_assets": [
    "source/cache.go",
    "docs/README.md",
    "benchmarks/bench_test.go",
    "sales/landing_copy.html"
  ]
}`
    },

    webhook_py: {
      name: "Stripe Billing Webhook Handler (Python)",
      stages: [
        { model: "Qwen3.8-27B-UD", desc: "Idempotent Async Ingestion Architecture" },
        { model: "sqlcoder-7b-2", desc: "ACID Audit Trail Ledger Schema" },
        { model: "Qwen2.5-Coder-32B", desc: "FastAPI + Ed25519 Cryptographic Verification" },
        { model: "DeepSeek-R1-Distill-32B", desc: "Timing-Attack Resistance & Nonce Validation" },
        { model: "gemma-2-27b-it", desc: "Developer Integration Manual & Pricing Deck" }
      ],
      code: `import hmac
import hashlib
import time
from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="Synapse Secure Stripe Webhook Ingestion Engine")

SIGNING_SECRET = "whsec_synapse_forge_cryptographic_key"

@app.post("/api/v1/stripe/webhook")
async def stripe_webhook_handler(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")
    
    # Constant-time cryptographic signature verification
    timestamp, sig = parse_sig_header(sig_header)
    if abs(time.time() - float(timestamp)) > 300:
        raise HTTPException(status_code=400, detail="Signature timestamp expired")
        
    expected_sig = hmac.new(
        SIGNING_SECRET.encode(),
        f"{timestamp}.".encode() + payload,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_sig, sig):
        raise HTTPException(status_code=403, detail="Invalid cryptographic signature")
        
    return {"status": "accepted", "idempotency_key": hashlib.sha256(payload).hexdigest()}`,
      docs: `# Synapse Stripe Billing Engine
Autonomous production webhook router with zero-loss idempotency guarantees.

## Security Audit
- Verified constant-time HMAC validation against side-channel timing attacks.
- Built-in nonce replay prevention window (5 minutes max clock skew).
- Native SQLite WAL audit trail.`,
      copy: `<div class="saas-highlight">
  <h3>Never Lose a Paying Customer to Webhook Drops.</h3>
  <p>The enterprise-grade billing webhook engine with cryptographic verification and zero-downtime ledger replay.</p>
  <a href="#pricing" class="btn-cta">Acquire Commercial Rights - $79</a>
</div>`,
      pricing: `{
  "product_id": "synapse-stripe-engine-pro",
  "version": "2.1.0",
  "tier": "professional",
  "price_usd": 79.0,
  "license": "Commercial Perpetual"
}`
    },

    vector_db: {
      name: "High-Throughput Vector Kernel (C/Python)",
      stages: [
        { model: "Qwen3.8-27B-UD", desc: "SIMD AVX-512 Dot-Product Architecture" },
        { model: "sqlcoder-7b-2", desc: "HNSW Graph Index Topology Tables" },
        { model: "Qwen2.5-Coder-32B", desc: "C/CUDA Vector Cosine Affinity Kernel" },
        { model: "DeepSeek-R1-Distill-32B", desc: "Numerical Precision Bounds [1e-7 Epsilon]" },
        { model: "gemma-2-27b-it", desc: "AI Infrastructure Technical Whitepaper" }
      ],
      code: `#include <immintrin.h>
#include <math.h>

// AVX-512 Accelerated Cosine Similarity Kernel (1024-dim BGE-M3 Embeddings)
float avx512_cosine_affinity(const float* a, const float* b, int dim) {
    __m512 sum_dot = _mm512_setzero_ps();
    __m512 sum_a2 = _mm512_setzero_ps();
    __m512 sum_b2 = _mm512_setzero_ps();

    for (int i = 0; i < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum_dot = _mm512_fmadd_ps(va, vb, sum_dot);
        sum_a2 = _mm512_fmadd_ps(va, va, sum_a2);
        sum_b2 = _mm512_fmadd_ps(vb, vb, sum_b2);
    }

    float dot = _mm512_reduce_add_ps(sum_dot);
    float norm_a = sqrtf(_mm512_reduce_add_ps(sum_a2));
    float norm_b = sqrtf(_mm512_reduce_add_ps(sum_b2));

    if (norm_a < 1e-7f || norm_b < 1e-7f) return 0.0f;
    return dot / (norm_a * norm_b);
}`,
      docs: `# AVX-512 High-Throughput Vector Similarity Engine
Accelerated embedding comparison engine delivering 4.5M similarity checks per second per CPU core.`,
      copy: `<div class="saas-highlight">
  <h3>High-Density Vector Search on Commodity Hardware.</h3>
  <p>Bypass expensive cloud vector databases with raw hardware-accelerated SIMD cosine search.</p>
</div>`,
      pricing: `{
  "product_id": "synapse-vector-simd-pro",
  "version": "1.0.0",
  "tier": "enterprise",
  "price_usd": 149.0
}`
    }
  };

  class StorefrontDemo {
    constructor() {
      this.presetSelect = document.getElementById('demoPresetSelect');
      this.runBtn = document.getElementById('demoRunBtn');
      this.progressBar = document.getElementById('demoProgressBar');
      this.contentPane = document.getElementById('demoContentPane');
      this.stepCards = document.querySelectorAll('.step-card');
      this.tabBtns = document.querySelectorAll('.demo-tab-btn');
      
      this.activePresetKey = 'cache_go';
      this.activeTab = 'code';
      this.isSimulating = false;

      this.initEvents();
      this.initPricingToggle();
      this.renderCurrentTab();
    }

    initEvents() {
      if (this.presetSelect) {
        this.presetSelect.addEventListener('change', (e) => {
          this.activePresetKey = e.target.value;
          this.resetStages();
          this.renderCurrentTab();
        });
      }

      if (this.runBtn) {
        this.runBtn.addEventListener('click', () => {
          this.startSimulation();
        });
      }

      this.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          this.tabBtns.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this.activeTab = btn.dataset.tab;
          this.renderCurrentTab();
        });
      });
    }

    resetStages() {
      this.stepCards.forEach(c => {
        c.classList.remove('active', 'completed');
      });
      if (this.progressBar) {
        this.progressBar.style.width = '0%';
      }
    }

    startSimulation() {
      if (this.isSimulating) return;
      this.isSimulating = true;
      this.runBtn.disabled = true;
      this.runBtn.textContent = 'Synthesizing...';
      this.resetStages();

      const preset = PRESETS[this.activePresetKey];
      let currentStep = 0;
      const totalSteps = 5;

      const stepInterval = setInterval(() => {
        if (currentStep > 0) {
          this.stepCards[currentStep - 1].classList.remove('active');
          this.stepCards[currentStep - 1].classList.add('completed');
        }

        if (currentStep < totalSteps) {
          this.stepCards[currentStep].classList.add('active');
          const pct = Math.round(((currentStep + 1) / totalSteps) * 100);
          if (this.progressBar) this.progressBar.style.width = `${pct}%`;
          currentStep++;
        } else {
          clearInterval(stepInterval);
          this.isSimulating = false;
          this.runBtn.disabled = false;
          this.runBtn.innerHTML = '&#9654; Synthesize Product';
          this.renderCurrentTab();
        }
      }, 500);
    }

    renderCurrentTab() {
      if (!this.contentPane) return;
      const preset = PRESETS[this.activePresetKey];
      if (!preset) return;

      let content = '';
      if (this.activeTab === 'code') {
        content = `<pre class="demo-code-block">${this.escapeHtml(preset.code)}</pre>`;
      } else if (this.activeTab === 'docs') {
        content = `<pre class="demo-code-block">${this.escapeHtml(preset.docs)}</pre>`;
      } else if (this.activeTab === 'copy') {
        content = `<div style="background:#060911;padding:16px;border-radius:8px;border:1px solid rgba(0,240,255,0.2);">${preset.copy}</div>`;
      } else if (this.activeTab === 'pricing') {
        content = `<pre class="demo-code-block">${this.escapeHtml(preset.pricing)}</pre>`;
      }

      this.contentPane.innerHTML = content;
    }

    escapeHtml(str) {
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    initPricingToggle() {
      const toggleSwitch = document.getElementById('billingToggle');
      const monthlyLabel = document.getElementById('monthlyLabel');
      const perpetualLabel = document.getElementById('perpetualLabel');
      
      const proPrice = document.getElementById('pricePro');
      const proPeriod = document.getElementById('periodPro');
      const entPrice = document.getElementById('priceEnt');
      const entPeriod = document.getElementById('periodEnt');

      let isPerpetual = false;

      const updatePricing = () => {
        if (isPerpetual) {
          toggleSwitch.classList.add('perpetual');
          monthlyLabel.classList.remove('active');
          perpetualLabel.classList.add('active');
          if (proPrice) proPrice.textContent = '799';
          if (proPeriod) proPeriod.textContent = '/ perpetual';
          if (entPrice) entPrice.textContent = '3,499';
          if (entPeriod) entPeriod.textContent = '/ perpetual';
        } else {
          toggleSwitch.classList.remove('perpetual');
          monthlyLabel.classList.add('active');
          perpetualLabel.classList.remove('active');
          if (proPrice) proPrice.textContent = '99';
          if (proPeriod) proPeriod.textContent = '/ mo';
          if (entPrice) entPrice.textContent = '499';
          if (entPeriod) entPeriod.textContent = '/ mo';
        }
      };

      if (toggleSwitch) {
        toggleSwitch.addEventListener('click', () => {
          isPerpetual = !isPerpetual;
          updatePricing();
        });
      }
      if (monthlyLabel) {
        monthlyLabel.addEventListener('click', () => {
          isPerpetual = false;
          updatePricing();
        });
      }
      if (perpetualLabel) {
        perpetualLabel.addEventListener('click', () => {
          isPerpetual = true;
          updatePricing();
        });
      }
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    new StorefrontDemo();
  });
})();
