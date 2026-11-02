/**
 * Synapse Forge - 13-Node Neural Topology Canvas (canvas_topology.js)
 * Implements:
 *   - Radial multi-ring 13-node specialist model layout
 *   - Dynamic Oja synaptic connections with weight-scaled lines
 *   - Animated particle pulses propagating between active models
 *   - Interactive hover collision detection with glassmorphism tooltip
 *   - HiDPI Retina canvas scaling at 60fps
 * 100% self-contained - zero external dependencies.
 */

(function () {
  'use strict';

  // Canonical 13 Fleet Models Specification
  const FLEET_SPECS = [
    { index: 0, id: 'N1', name: 'Qwen3.8-27B-UD', alias: 'architect', domain: 'High-Context Architecture', precision: 'Q4_K_M', vram: '17.4 GB', params: '27B', color: '#6366f1', layer: 'Executive Nucleus' },
    { index: 1, id: 'N2', name: 'Qwen3.8-27B-Uncensored', alias: 'security', domain: 'AST Security & Exploit Audit', precision: 'Q4_K_M', vram: '17.4 GB', params: '27B', color: '#f59e0b', layer: 'Safety & Audit' },
    { index: 2, id: 'N3', name: 'Qwen2.5-Coder-32B', alias: 'coder', domain: 'Flagship Code Synthesis', precision: 'Q4_K_M', vram: '18.5 GB', params: '32B', color: '#38bdf8', layer: 'Synthesis Core' },
    { index: 3, id: 'N4', name: 'DeepSeek-R1-Distill-32B', alias: 'critic', domain: 'Deep Reasoning & Math Critic', precision: 'Q4_K_M', vram: '18.5 GB', params: '32B', color: '#f43f5e', layer: 'Cognitive Core' },
    { index: 4, id: 'N5', name: 'Qwen2.5-Coder-14B', alias: 'coder14', domain: 'Fast Mid-Tier Code Engine', precision: 'Q5_K_M', vram: '10.5 GB', params: '14B', color: '#38bdf8', layer: 'Synthesis Core' },
    { index: 5, id: 'N6', name: 'DeepSeek-R1-Distill-14B', alias: 'reasoning14', domain: 'Fast Logical Inference', precision: 'Q5_K_M', vram: '10.5 GB', params: '14B', color: '#f43f5e', layer: 'Cognitive Core' },
    { index: 6, id: 'N7', name: 'Qwen2.5-VL-7B', alias: 'vision', domain: 'Multimodal UI & Vision', precision: 'Q5_K_M', vram: '5.4 GB', params: '7B', color: '#a855f7', layer: 'Perception' },
    { index: 7, id: 'N8', name: 'sqlcoder-7b-2', alias: 'sql', domain: 'Relational Database & SQL', precision: 'Q5_K_M', vram: '4.8 GB', params: '7B', color: '#06b6d4', layer: 'Data Persistence' },
    { index: 8, id: 'N9', name: 'Qwen2.5-Coder-14B-Unc', alias: 'coder-unc', domain: 'Unrestricted Coder', precision: 'Q5_K_M', vram: '10.5 GB', params: '14B', color: '#f59e0b', layer: 'Safety & Audit' },
    { index: 9, id: 'N10', name: 'Hermes-3-Llama-3.1-8B', alias: 'agent', domain: 'Tool Agency & Strict JSON', precision: 'Q8_0', vram: '8.5 GB', params: '8B', color: '#10b981', layer: 'Action Orchestration' },
    { index: 10, id: 'N11', name: 'gemma-2-27b-it', alias: 'creative', domain: 'Commercial Copy & Docs', precision: 'Q4_K_M', vram: '15.5 GB', params: '27B', color: '#ec4899', layer: 'Commercialization' },
    { index: 11, id: 'N12', name: 'bge-m3', alias: 'memory', domain: '1024-dim Embedding & Surprise', precision: 'Q8_0', vram: '0.6 GB', params: '567M', color: '#8b5cf6', layer: 'Memory Nucleus' },
    { index: 12, id: 'N13', name: 'Qwen3-30B-A3B', alias: 'moe', domain: 'Hyper-Fast MoE Coding', precision: 'Q4_K_M', vram: '17.3 GB', params: '30B (3B act)', color: '#00f0ff', layer: 'Fast Execution' }
  ];

  class TopologyCanvas {
    constructor(canvasId, tooltipId) {
      this.canvas = document.getElementById(canvasId);
      if (!this.canvas) return;
      this.ctx = this.canvas.getContext('2d');
      this.tooltip = document.getElementById(tooltipId);
      
      this.nodes = [];
      this.edges = [];
      this.particles = [];
      this.hoveredNode = null;
      this.mouse = { x: -1000, y: -1000 };
      this.time = 0;
      
      this.initNodes();
      this.initParticles();
      this.bindEvents();
      this.resize();
      this.animate();
    }

    initNodes() {
      this.nodes = FLEET_SPECS.map(spec => ({
        ...spec,
        x: 0,
        y: 0,
        radius: 20,
        targetRadius: 20,
        activity: 0.1,
        pulsePhase: Math.random() * Math.PI * 2
      }));
    }

    initParticles() {
      // 35 animated particles traveling along synaptic pathways
      this.particles = [];
      for (let i = 0; i < 35; i++) {
        this.particles.push({
          srcIdx: Math.floor(Math.random() * 13),
          tgtIdx: Math.floor(Math.random() * 13),
          progress: Math.random(),
          speed: 0.004 + Math.random() * 0.008,
          size: 1.5 + Math.random() * 2.0,
          color: '#00f0ff'
        });
      }
    }

    setEdges(edges) {
      if (!Array.isArray(edges)) return;
      this.edges = edges.map(e => {
        let src = typeof e.src === 'number' ? e.src : this.resolveIndex(e.source_model);
        let tgt = typeof e.tgt === 'number' ? e.tgt : this.resolveIndex(e.target_model);
        let weight = parseFloat(e.weight) || 0.1;
        return { src, tgt, weight, co_activations: e.co_activations || 0 };
      }).filter(e => e.src >= 0 && e.tgt >= 0 && e.src !== e.tgt);
    }

    resolveIndex(nameOrAlias) {
      if (!nameOrAlias) return -1;
      const lower = String(nameOrAlias).toLowerCase();
      for (let n of this.nodes) {
        if (n.name.toLowerCase() === lower || n.alias.toLowerCase() === lower || lower.includes(n.alias.toLowerCase())) {
          return n.index;
        }
      }
      return -1;
    }

    updateNodePositions() {
      const w = this.canvas.width / (this.dpr || 1);
      const h = this.canvas.height / (this.dpr || 1);
      const cx = w / 2;
      const cy = h / 2;
      const minDim = Math.min(w, h);

      // Radial Ring Layout:
      // Hub (Center): N11 (bge-m3 memory) & N0 (architect)
      this.nodes[11].x = cx - minDim * 0.07;
      this.nodes[11].y = cy;
      this.nodes[11].radius = 24;

      this.nodes[0].x = cx + minDim * 0.07;
      this.nodes[0].y = cy;
      this.nodes[0].radius = 24;

      // Inner Ring (radius = 0.24 * minDim, 5 nodes: coder, critic, agent, sql, moe)
      const innerIndices = [2, 3, 9, 7, 12];
      const innerRadius = minDim * 0.25;
      innerIndices.forEach((idx, i) => {
        const angle = (i / innerIndices.length) * Math.PI * 2 - Math.PI / 2;
        this.nodes[idx].x = cx + Math.cos(angle) * innerRadius;
        this.nodes[idx].y = cy + Math.sin(angle) * innerRadius;
        this.nodes[idx].radius = 20;
      });

      // Outer Ring (radius = 0.40 * minDim, 6 nodes: security, coder14, reasoning14, vision, unc-coder, creative)
      const outerIndices = [1, 4, 5, 6, 8, 10];
      const outerRadius = minDim * 0.41;
      outerIndices.forEach((idx, i) => {
        const angle = (i / outerIndices.length) * Math.PI * 2 - Math.PI / 3;
        this.nodes[idx].x = cx + Math.cos(angle) * outerRadius;
        this.nodes[idx].y = cy + Math.sin(angle) * outerRadius;
        this.nodes[idx].radius = 18;
      });
    }

    bindEvents() {
      window.addEventListener('resize', () => this.resize());
      
      this.canvas.addEventListener('mousemove', (e) => {
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = e.clientX - rect.left;
        this.mouse.y = e.clientY - rect.top;
        this.checkHover();
      });

      this.canvas.addEventListener('mouseleave', () => {
        this.mouse.x = -1000;
        this.mouse.y = -1000;
        this.hoveredNode = null;
        if (this.tooltip) {
          this.tooltip.style.opacity = '0';
        }
      });

      this.canvas.addEventListener('click', () => {
        if (this.hoveredNode) {
          // Trigger particle burst from hovered node
          for (let p of this.particles) {
            if (Math.random() > 0.4) {
              p.srcIdx = this.hoveredNode.index;
              p.progress = 0;
            }
          }
          this.hoveredNode.activity = 1.0;
        }
      });
    }

    checkHover() {
      let found = null;
      for (let node of this.nodes) {
        const dx = this.mouse.x - node.x;
        const dy = this.mouse.y - node.y;
        if (Math.sqrt(dx * dx + dy * dy) <= node.radius + 6) {
          found = node;
          break;
        }
      }

      this.hoveredNode = found;

      if (this.tooltip && found) {
        this.tooltip.innerHTML = `
          <div class="tooltip-title">${found.name}</div>
          <div class="tooltip-domain">${found.domain}</div>
          <div class="tooltip-meta">
            <span>Layer: ${found.layer}</span>
            <span>Params: ${found.params} (${found.precision})</span>
            <span>VRAM: ${found.vram}</span>
            <span>Status: ACTIVE (DIO NVMe)</span>
          </div>
        `;
        
        let tx = found.x + 24;
        let ty = found.y - 30;
        const maxW = this.canvas.width / (this.dpr || 1);
        if (tx + 200 > maxW) tx = found.x - 210;
        if (ty < 10) ty = 10;
        
        this.tooltip.style.left = `${tx}px`;
        this.tooltip.style.top = `${ty}px`;
        this.tooltip.style.opacity = '1';
      } else if (this.tooltip) {
        this.tooltip.style.opacity = '0';
      }
    }

    resize() {
      this.dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      const w = rect.width || 800;
      const h = rect.height || 520;
      this.canvas.width = w * this.dpr;
      this.canvas.height = h * this.dpr;
      this.ctx.scale(this.dpr, this.dpr);
      this.updateNodePositions();
    }

    drawEdges() {
      // If edges not yet populated from backend, synthesize base interconnected topology
      const activeEdges = this.edges.length > 0 ? this.edges : this.getDefaultEdges();
      
      for (let edge of activeEdges) {
        const src = this.nodes[edge.src];
        const tgt = this.nodes[edge.tgt];
        if (!src || !tgt) continue;

        const isHoverConnected = this.hoveredNode && 
          (this.hoveredNode.index === edge.src || this.hoveredNode.index === edge.tgt);

        const w = Math.max(0.05, Math.min(1.0, edge.weight || 0.1));
        const lineWidth = isHoverConnected ? (2.5 + w * 6.0) : (0.8 + w * 3.5);

        this.ctx.beginPath();
        this.ctx.moveTo(src.x, src.y);
        this.ctx.lineTo(tgt.x, tgt.y);

        if (isHoverConnected) {
          this.ctx.strokeStyle = `rgba(0, 240, 255, ${0.45 + w * 0.5})`;
          this.ctx.shadowColor = '#00f0ff';
          this.ctx.shadowBlur = 10;
        } else if (this.hoveredNode) {
          this.ctx.strokeStyle = `rgba(99, 102, 241, ${0.08 + w * 0.15})`;
          this.ctx.shadowBlur = 0;
        } else {
          this.ctx.strokeStyle = `rgba(99, 102, 241, ${0.15 + w * 0.35})`;
          this.ctx.shadowColor = 'rgba(0, 240, 255, 0.2)';
          this.ctx.shadowBlur = 4;
        }

        this.ctx.lineWidth = lineWidth;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
      }
    }

    getDefaultEdges() {
      const edges = [];
      // Connect hub (N11, N0) to inner models
      [2, 3, 7, 9, 12].forEach(i => {
        edges.push({ src: 11, tgt: i, weight: 0.65 });
        edges.push({ src: 0, tgt: i, weight: 0.72 });
      });
      // Connect coder (N2) to critic (N3), sql (N7), agent (N9)
      edges.push({ src: 2, tgt: 3, weight: 0.85 });
      edges.push({ src: 2, tgt: 7, weight: 0.58 });
      edges.push({ src: 2, tgt: 9, weight: 0.62 });
      // Connect creative (N10) to agent and architect
      edges.push({ src: 10, tgt: 0, weight: 0.45 });
      edges.push({ src: 10, tgt: 9, weight: 0.52 });
      // Connect security (N1, N8) to coder and critic
      edges.push({ src: 1, tgt: 2, weight: 0.55 });
      edges.push({ src: 8, tgt: 2, weight: 0.48 });
      // Connect fast MoE (N12) to coder and vision (N6)
      edges.push({ src: 12, tgt: 2, weight: 0.78 });
      edges.push({ src: 6, tgt: 0, weight: 0.40 });
      return edges;
    }

    drawParticles() {
      for (let p of this.particles) {
        p.progress += p.speed;
        if (p.progress >= 1.0) {
          p.progress = 0;
          p.srcIdx = Math.floor(Math.random() * 13);
          p.tgtIdx = Math.floor(Math.random() * 13);
          while (p.tgtIdx === p.srcIdx) {
            p.tgtIdx = Math.floor(Math.random() * 13);
          }
        }

        const src = this.nodes[p.srcIdx];
        const tgt = this.nodes[p.tgtIdx];
        if (!src || !tgt) continue;

        const curX = src.x + (tgt.x - src.x) * p.progress;
        const curY = src.y + (tgt.y - src.y) * p.progress;

        this.ctx.beginPath();
        this.ctx.arc(curX, curY, p.size, 0, Math.PI * 2);
        this.ctx.fillStyle = p.color;
        this.ctx.shadowColor = p.color;
        this.ctx.shadowBlur = 8;
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
      }
    }

    drawNodes() {
      for (let node of this.nodes) {
        const isHovered = this.hoveredNode && this.hoveredNode.index === node.index;
        const currentRadius = isHovered ? node.radius + 4 : node.radius;

        // Outer Glow Halo
        const glowRadius = currentRadius + 8 + Math.sin(this.time * 2 + node.pulsePhase) * 3;
        const gradient = this.ctx.createRadialGradient(node.x, node.y, currentRadius * 0.6, node.x, node.y, glowRadius);
        gradient.addColorStop(0, node.color);
        gradient.addColorStop(0.7, 'rgba(0, 240, 255, 0.2)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();

        // Node Body Circle
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, currentRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = '#0d1424';
        this.ctx.fill();

        // Node Border Ring
        this.ctx.lineWidth = isHovered ? 2.8 : 1.6;
        this.ctx.strokeStyle = isHovered ? '#ffffff' : node.color;
        this.ctx.shadowColor = node.color;
        this.ctx.shadowBlur = isHovered ? 16 : 6;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;

        // Node Label (Short ID e.g. N1, N3)
        this.ctx.font = 'bold 11px -apple-system, sans-serif';
        this.ctx.fillStyle = '#ffffff';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(node.id, node.x, node.y);

        // Subtitle below node
        this.ctx.font = '9px ui-monospace, monospace';
        this.ctx.fillStyle = isHovered ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
        this.ctx.fillText(node.alias, node.x, node.y + currentRadius + 11);
      }
    }

    animate() {
      this.time += 0.02;
      const w = this.canvas.width / (this.dpr || 1);
      const h = this.canvas.height / (this.dpr || 1);
      
      this.ctx.clearRect(0, 0, w, h);

      this.drawEdges();
      this.drawParticles();
      this.drawNodes();

      requestAnimationFrame(() => this.animate());
    }
  }

  // Expose globally
  window.SynapseTopology = TopologyCanvas;
})();
