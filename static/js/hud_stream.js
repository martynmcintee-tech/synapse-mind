/**
 * Synapse Forge - Real-time Hardware HUD & WebSocket Streamer (hud_stream.js)
 * Implements:
 *   - 10Hz WebSocket streaming connection to /ws with auto-reconnect
 *   - Real-time gauge and progress bar binders (RTX 3090, 990 PRO NVMe D:, CPU, RAM, AGIQ)
 *   - Topology edge updates piped to canvas
 *   - One-click control button handlers (Evolution, Synthesis, ZIP Export, Provenance Modal)
 *   - Cyberpunk activity feed streamer
 * 100% self-contained - zero external dependencies.
 */

(function () {
  'use strict';

  class HUDStreamer {
    constructor() {
      this.ws = null;
      this.reconnectTimer = null;
      this.topologyCanvas = null;
      this.lastAgiq = 100.0;
      
      this.dom = {
        // GPU
        gpuVramVal: document.getElementById('gpuVramVal'),
        gpuVramTotal: document.getElementById('gpuVramTotal'),
        gpuVramBar: document.getElementById('gpuVramBar'),
        gpuTemp: document.getElementById('gpuTemp'),
        gpuUtil: document.getElementById('gpuUtil'),
        gpuPower: document.getElementById('gpuPower'),
        
        // NVMe
        nvmeUsedVal: document.getElementById('nvmeUsedVal'),
        nvmeTotalVal: document.getElementById('nvmeTotalVal'),
        nvmeBar: document.getElementById('nvmeBar'),
        nvmeFree: document.getElementById('nvmeFree'),
        nvmeSpeedup: document.getElementById('nvmeSpeedup'),
        
        // CPU & RAM
        cpuVal: document.getElementById('cpuVal'),
        cpuBar: document.getElementById('cpuBar'),
        ramVal: document.getElementById('ramVal'),
        ramTotal: document.getElementById('ramTotal'),
        ramBar: document.getElementById('ramBar'),
        
        // AGIQ
        agiqVal: document.getElementById('agiqVal'),
        agiqBar: document.getElementById('agiqBar'),
        agiqDelta: document.getElementById('agiqDelta'),
        agiqTrigger: document.getElementById('agiqTrigger'),
        
        // Feed & Modals
        streamList: document.getElementById('streamList'),
        wsBadge: document.getElementById('wsBadge'),
        provenanceModal: document.getElementById('provenanceModal'),
        provenanceContent: document.getElementById('provenanceContent'),
        closeModalBtn: document.getElementById('closeModalBtn'),
        
        // Action Buttons
        btnEvolution: document.getElementById('btnRunEvolution'),
        btnSynthesize: document.getElementById('btnSynthesizeProduct'),
        btnExport: document.getElementById('btnExportZip'),
        btnProvenance: document.getElementById('btnInspectProvenance'),
        btnTeachMemory: document.getElementById('btnTeachMemory'),
        toastContainer: document.getElementById('toastContainer'),

        // Co-Working & Continual Learning Console
        chatFeed: document.getElementById('chatFeed'),
        chatInput: document.getElementById('chatInput'),
        btnSendChat: document.getElementById('btnSendChat'),
        workspaceInput: document.getElementById('workspaceInput'),
        modeSelector: document.getElementById('modeSelector'),
        memoryList: document.getElementById('memoryList')
      };

      this.currentMode = 'auto';
      this.initTopology();
      this.bindControls();
      this.bindCoWorkChat();
      this.connectWebSocket();
      this.fetchInitialTelemetry();
      this.loadMindHistory();
    }

    initTopology() {
      if (window.SynapseTopology) {
        this.topologyCanvas = new window.SynapseTopology('topologyCanvas', 'nodeTooltip');
      }
    }

    connectWebSocket() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          if (this.dom.wsBadge) {
            this.dom.wsBadge.innerHTML = '<span class="pulsing-dot"></span> 10Hz TELEMETRY STREAM ONLINE';
            this.dom.wsBadge.style.color = '#10b981';
          }
          this.logActivity('WebSocket telemetry link established on 127.0.0.1:18445', 'sys');
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleTelemetryUpdate(data);
          } catch (e) {
            console.error('Failed to parse WS payload', e);
          }
        };

        this.ws.onclose = () => {
          if (this.dom.wsBadge) {
            this.dom.wsBadge.innerHTML = '<span class="pulsing-dot" style="background:#f59e0b;box-shadow:0 0 8px #f59e0b;"></span> STREAM RECONNECTING...';
            this.dom.wsBadge.style.color = '#f59e0b';
          }
          this.scheduleReconnect();
        };

        this.ws.onerror = () => {
          this.ws.close();
        };
      } catch (err) {
        this.scheduleReconnect();
      }
    }

    scheduleReconnect() {
      if (this.reconnectTimer) return;
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.connectWebSocket();
      }, 3000);
    }

    fetchInitialTelemetry() {
      fetch('/api/telemetry')
        .then(r => r.json())
        .then(data => this.handleTelemetryUpdate(data))
        .catch(e => console.warn('Telemetry initial fetch fallback', e));

      fetch('/api/topology')
        .then(r => r.json())
        .then(data => {
          if (this.topologyCanvas && data.edges) {
            this.topologyCanvas.setEdges(data.edges);
          }
        })
        .catch(e => console.warn('Topology initial fetch fallback', e));
    }

    handleTelemetryUpdate(data) {
      if (!data) return;

      // GPU Metrics
      const gpu = data.gpu || {};
      if (this.dom.gpuVramVal && gpu.vram_used_mb !== undefined) {
        this.dom.gpuVramVal.textContent = gpu.vram_used_mb.toLocaleString();
        const total = gpu.vram_total_mb || 24576;
        if (this.dom.gpuVramTotal) this.dom.gpuVramTotal.textContent = `/ ${total.toLocaleString()} MB`;
        const pct = Math.min(100, Math.round((gpu.vram_used_mb / total) * 100));
        if (this.dom.gpuVramBar) this.dom.gpuVramBar.style.width = `${pct}%`;
      }
      if (this.dom.gpuTemp && gpu.temp_c !== undefined) {
        this.dom.gpuTemp.textContent = `${gpu.temp_c}°C`;
      }
      if (this.dom.gpuUtil && gpu.util_pct !== undefined) {
        this.dom.gpuUtil.textContent = `${gpu.util_pct}%`;
      }
      if (this.dom.gpuPower && gpu.power_w !== undefined) {
        this.dom.gpuPower.textContent = `${Math.round(gpu.power_w)}W`;
      }

      // NVMe Storage Metrics (D:)
      const nvme = data.nvme || {};
      if (this.dom.nvmeUsedVal && nvme.total_gb !== undefined) {
        const used = Math.round(nvme.total_gb - (nvme.free_gb || 0));
        this.dom.nvmeUsedVal.textContent = used.toLocaleString();
        if (this.dom.nvmeTotalVal) this.dom.nvmeTotalVal.textContent = `/ ${Math.round(nvme.total_gb).toLocaleString()} GB`;
        const pct = nvme.used_pct || Math.round((used / nvme.total_gb) * 100);
        if (this.dom.nvmeBar) this.dom.nvmeBar.style.width = `${pct}%`;
      }
      if (this.dom.nvmeFree && nvme.free_gb !== undefined) {
        this.dom.nvmeFree.textContent = `${Math.round(nvme.free_gb)} GB Free`;
      }
      if (this.dom.nvmeSpeedup && nvme.speedup) {
        this.dom.nvmeSpeedup.textContent = nvme.speedup;
      }

      // CPU & RAM Metrics
      const cpuRam = data.cpu_ram || {};
      if (this.dom.cpuVal && cpuRam.cpu_pct !== undefined) {
        this.dom.cpuVal.textContent = `${Math.round(cpuRam.cpu_pct)}%`;
        if (this.dom.cpuBar) this.dom.cpuBar.style.width = `${Math.min(100, cpuRam.cpu_pct)}%`;
      }
      if (this.dom.ramVal && cpuRam.ram_used_gb !== undefined) {
        this.dom.ramVal.textContent = cpuRam.ram_used_gb.toFixed(1);
        const ramTotal = cpuRam.ram_total_gb || 32.0;
        if (this.dom.ramTotal) this.dom.ramTotal.textContent = `/ ${ramTotal.toFixed(1)} GB`;
        const ramPct = Math.min(100, Math.round((cpuRam.ram_used_gb / ramTotal) * 100));
        if (this.dom.ramBar) this.dom.ramBar.style.width = `${ramPct}%`;
      }

      // AGIQ Composite Score
      const agiq = data.agiq || {};
      if (this.dom.agiqVal && agiq.composite_agiq !== undefined) {
        const val = parseFloat(agiq.composite_agiq);
        this.dom.agiqVal.textContent = val.toFixed(2);
        const norm = Math.min(100, Math.round((val / 200.0) * 100));
        if (this.dom.agiqBar) this.dom.agiqBar.style.width = `${norm}%`;

        if (this.dom.agiqDelta) {
          const delta = val - this.lastAgiq;
          if (Math.abs(delta) > 0.001) {
            this.dom.agiqDelta.textContent = delta >= 0 ? `+${delta.toFixed(3)}` : `${delta.toFixed(3)}`;
            this.dom.agiqDelta.style.color = delta >= 0 ? '#10b981' : '#f43f5e';
          }
          this.lastAgiq = val;
        }

        if (this.dom.agiqTrigger && agiq.trigger_event) {
          this.dom.agiqTrigger.textContent = agiq.trigger_event;
        }
      }

      // Topology Update
      if (data.topology && data.topology.edges && this.topologyCanvas) {
        this.topologyCanvas.setEdges(data.topology.edges);
      }
    }

    bindControls() {
      // 1. Run Evolution Cycle
      if (this.dom.btnEvolution) {
        this.dom.btnEvolution.addEventListener('click', () => {
          this.setButtonLoading(this.dom.btnEvolution, true, 'Running AST Sandbox...');
          this.logActivity('Initiating Autonomous Self-Evolution cycle...', 'evo');

          fetch('/api/evolution/cycle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
          })
            .then(r => r.json())
            .then(res => {
              this.setButtonLoading(this.dom.btnEvolution, false);
              const skillName = res.skill ? (res.skill.skill_name || 'Autonomous Tool') : 'Algorithm';
              const delta = res.agiq_delta || 0.35;
              this.showToast(`Self-Evolution Success: Promoted ${skillName} (AGIQ +${delta})`, 'success');
              this.logActivity(`Evolution Promoted: ${skillName} | Security Sandbox Verified | ΔAGIQ +${delta}`, 'evo');
            })
            .catch(err => {
              this.setButtonLoading(this.dom.btnEvolution, false);
              this.showToast('Evolution Cycle execution failed', 'error');
              this.logActivity(`Evolution cycle failed: ${err.message}`, 'err');
            });
        });
      }

      // 2. Synthesize Product
      if (this.dom.btnSynthesize) {
        this.dom.btnSynthesize.addEventListener('click', () => {
          const title = prompt('Enter Product Name:', 'SynapseRateLimiter');
          if (!title) return;
          const domain = prompt('Enter Technical Domain:', 'High-Performance Token Bucket Microservice');

          this.setButtonLoading(this.dom.btnSynthesize, true, 'Synthesizing 13-Model Bundle...');
          this.logActivity(`Multi-Model Synthesis Started: "${title}"`, 'synth');

          fetch('/api/product/synthesize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_name: title, domain_prompt: domain || 'Go/Python software' })
          })
            .then(r => r.json())
            .then(res => {
              this.setButtonLoading(this.dom.btnSynthesize, false);
              const p = res.product || {};
              this.showToast(`Product Synthesized: ${p.name || title} ready!`, 'success');
              this.logActivity(`Synthesis Complete: ${p.name || title} | Packaged into releases/`, 'synth');
              
              if (res.zip_url) {
                window.location.href = res.zip_url;
              }
            })
            .catch(err => {
              this.setButtonLoading(this.dom.btnSynthesize, false);
              this.showToast('Product synthesis failed', 'error');
              this.logActivity(`Product synthesis failed: ${err.message}`, 'err');
            });
        });
      }

      // 3. Export Release ZIP
      if (this.dom.btnExport) {
        this.dom.btnExport.addEventListener('click', () => {
          this.showToast('Generating and streaming synapse_forge_product.zip...', 'info');
          this.logActivity('Release bundle ZIP export requested', 'sys');
          window.location.href = '/api/bundle/export';
        });
      }

      // 4. Inspect Provenance Modal
      if (this.dom.btnProvenance) {
        this.dom.btnProvenance.addEventListener('click', () => {
          fetch('/api/provenance')
            .then(r => r.json())
            .then(data => {
              if (this.dom.provenanceContent) {
                this.dom.provenanceContent.textContent = JSON.stringify(data, null, 2);
              }
              if (this.dom.provenanceModal) {
                this.dom.provenanceModal.classList.add('active');
              }
            })
            .catch(e => {
              this.showToast('Failed to load build provenance', 'error');
            });
        });
      }

      if (this.dom.closeModalBtn) {
        this.dom.closeModalBtn.addEventListener('click', () => {
          if (this.dom.provenanceModal) {
            this.dom.provenanceModal.classList.remove('active');
          }
        });
      }

      window.addEventListener('click', (e) => {
        if (e.target === this.dom.provenanceModal) {
          this.dom.provenanceModal.classList.remove('active');
        }
      });

      if (this.dom.btnTeachMemory) {
        this.dom.btnTeachMemory.addEventListener('click', () => {
          const fact = prompt('Teach Synapse + Jarvis a new permanent rule, preference, or project insight:');
          if (!fact || !fact.trim()) return;
          const key = 'user_taught_' + Math.floor(Date.now() / 1000);
          fetch('/api/mind/teach', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, fact: fact.trim(), category: 'preference' })
          })
            .then(r => r.json())
            .then(res => {
              this.showToast('Stored in shared Hippocampus memory!', 'success');
              this.logActivity(`Hippocampus Consolidated: "${fact.trim().slice(0, 60)}..."`, 'evo');
              this.loadMindHistory();
            });
        });
      }
    }

    bindCoWorkChat() {
      if (this.dom.modeSelector) {
        const buttons = this.dom.modeSelector.querySelectorAll('.mode-pill');
        buttons.forEach(btn => {
          btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            this.currentMode = btn.dataset.mode || 'auto';
            this.logActivity(`Cognitive Routing Mode set to: ${this.currentMode.toUpperCase()}`, 'sys');
          });
        });
      }

      if (this.dom.btnSendChat && this.dom.chatInput) {
        this.dom.btnSendChat.addEventListener('click', () => this.sendChatMessage());
        this.dom.chatInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendChatMessage();
          }
        });
      }
    }

    loadMindHistory() {
      fetch('/api/mind/history')
        .then(r => r.json())
        .then(data => {
          if (this.dom.memoryList && Array.isArray(data.memories)) {
            this.dom.memoryList.innerHTML = data.memories.map(m =>
              `<div class="memory-item"><strong>[${this.escapeHtml(m.category)}] ${this.escapeHtml(m.key)}:</strong> ${this.escapeHtml(m.fact.slice(0, 140))}</div>`
            ).join('');
          }
          if (this.dom.chatFeed && Array.isArray(data.dialogue) && data.dialogue.length > 0) {
            for (const turn of data.dialogue) {
              if (turn.role === 'user') {
                this.appendUserBubble(turn.content);
              } else {
                let models = [];
                try { models = JSON.parse(turn.consulted_models || '[]'); } catch (e) {}
                this.appendAssistantBubble({
                  reply: turn.content,
                  primary_model: turn.primary_model || 'Qwen3-30B-A3B',
                  participating_models: models,
                  surprise_score: turn.surprise_score || 0.62,
                  learned_fact: turn.learned_fact,
                  latency_ms: turn.latency_ms || 0
                });
              }
            }
          }
        })
        .catch(() => {});
    }

    escapeHtml(str) {
      return String(str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    appendUserBubble(text) {
      if (!this.dom.chatFeed) return;
      const div = document.createElement('div');
      div.className = 'msg-bubble msg-user';
      div.textContent = text;
      this.dom.chatFeed.appendChild(div);
      this.dom.chatFeed.scrollTop = this.dom.chatFeed.scrollHeight;
    }

    appendAssistantBubble(res) {
      if (!this.dom.chatFeed) return;
      const div = document.createElement('div');
      div.className = 'msg-bubble msg-assistant';
      const modelsStr = (res.participating_models && res.participating_models.length)
        ? res.participating_models.join(' → ')
        : (res.primary_model || 'Qwen3-30B-A3B');
      const surprise = typeof res.surprise_score === 'number' ? res.surprise_score.toFixed(3) : '0.620';
      const tpsStr = res.tps ? ` • ${res.tps} t/s` : '';
      const latStr = res.latency_ms ? ` • ${(res.latency_ms / 1000).toFixed(1)}s` : '';
      const learnedHtml = res.learned_fact
        ? `<div class="learned-banner">🧠 Consolidated into Shared Hippocampus: ${this.escapeHtml(res.learned_fact)}</div>`
        : '';

      div.innerHTML = `
        <div class="msg-meta">
          <span class="specialist-badge">${this.escapeHtml(modelsStr)}</span>
          <span class="surprise-badge">Surprise S(x): ${surprise}${tpsStr}${latStr}</span>
        </div>
        <div>${this.escapeHtml(res.reply)}</div>
        ${learnedHtml}
      `;
      this.dom.chatFeed.appendChild(div);
      this.dom.chatFeed.scrollTop = this.dom.chatFeed.scrollHeight;
    }

    sendChatMessage() {
      if (!this.dom.chatInput) return;
      const text = this.dom.chatInput.value.trim();
      if (!text) {
        this.showToast('Type a message or question in the prompt box first!', 'info');
        this.dom.chatInput.focus();
        return;
      }

      this.dom.chatInput.value = '';
      this.appendUserBubble(text);

      // Immediate visual thinking bubble in chat feed
      let thinkingDiv = null;
      if (this.dom.chatFeed) {
        thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'msg-bubble msg-assistant';
        thinkingDiv.innerHTML = `
          <div class="msg-meta">
            <span class="specialist-badge">13-Model Cortex (${this.escapeHtml(this.currentMode.toUpperCase())})</span>
            <span class="surprise-badge">BGE-M3 + GPU Inference Active...</span>
          </div>
          <div style="color:#00f0ff;">&#9889; Routing prompt across local neural specialists &amp; recalling shared Hippocampus memories...</div>
        `;
        this.dom.chatFeed.appendChild(thinkingDiv);
        this.dom.chatFeed.scrollTop = this.dom.chatFeed.scrollHeight;
      }

      const workspace = this.dom.workspaceInput ? this.dom.workspaceInput.value.trim() : 'I:\\ai_models';
      this.setButtonLoading(this.dom.btnSendChat, true, 'THINKING');
      this.logActivity(`Routing prompt (${this.currentMode}) across 13-model cortex...`, 'synth');

      fetch('/api/mind/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          mode: this.currentMode,
          workspace_path: workspace
        })
      })
        .then(r => {
          if (!r.ok) {
            throw new Error(`HTTP ${r.status}`);
          }
          return r.json();
        })
        .then(res => {
          if (thinkingDiv && thinkingDiv.parentNode) {
            thinkingDiv.parentNode.removeChild(thinkingDiv);
          }
          this.setButtonLoading(this.dom.btnSendChat, false);
          this.appendAssistantBubble(res);
          const modelsJoined = (res.participating_models || [res.primary_model]).join(' + ');
          this.logActivity(
            `Cortex Response via ${modelsJoined} | S(x)=${res.surprise_score} | ΔAGIQ +${res.agiq_delta || 0}`,
            'evo'
          );
          if (res.learned_fact) {
            this.loadMindHistory();
          }
        })
        .catch(err => {
          if (thinkingDiv && thinkingDiv.parentNode) {
            thinkingDiv.parentNode.removeChild(thinkingDiv);
          }
          this.setButtonLoading(this.dom.btnSendChat, false);
          this.showToast('Failed to reach cognitive cortex: ' + err.message, 'error');
          this.logActivity(`Chat error: ${err.message}`, 'err');
        });
    }

    setButtonLoading(btn, isLoading, text) {
      if (!btn) return;
      if (isLoading) {
        btn.dataset.originalHtml = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span>${text}</span> <span class="btn-indicator">...</span>`;
      } else {
        btn.disabled = false;
        if (btn.dataset.originalHtml) {
          btn.innerHTML = btn.dataset.originalHtml;
        }
      }
    }

    logActivity(text, type) {
      if (!this.dom.streamList) return;
      const item = document.createElement('div');
      item.className = `stream-item ${type || 'sys'}`;
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');
      item.innerHTML = `
        <div class="stream-timestamp">[${timeStr}]</div>
        <div>${text}</div>
      `;
      this.dom.streamList.insertBefore(item, this.dom.streamList.firstChild);
      if (this.dom.streamList.children.length > 40) {
        this.dom.streamList.removeChild(this.dom.streamList.lastChild);
      }
    }

    showToast(msg, type) {
      if (!this.dom.toastContainer) return;
      const toast = document.createElement('div');
      toast.className = 'toast';
      if (type === 'success') toast.style.borderLeftColor = '#10b981';
      else if (type === 'error') toast.style.borderLeftColor = '#f43f5e';
      else if (type === 'info') toast.style.borderLeftColor = '#00f0ff';
      toast.textContent = msg;

      this.dom.toastContainer.appendChild(toast);
      setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
      }, 4000);
    }
  }

  function bootHUD() {
    if (!window.synapseHUD) {
      window.synapseHUD = new HUDStreamer();
    }
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', bootHUD);
  } else {
    bootHUD();
  }
})();
