/**
 * GitHub Copilot Multi-Agent Real-Time Flow Visualizer Application
 */

class AgentFlowVisualizer {
  constructor() {
    this.topology = { nodes: [], edges: [] };
    this.nodesState = {};
    this.selectedAgentId = null;
    this.activePackets = []; // Array of active traveling particles
    this.eventHistory = [];
    this.startTime = null;
    this.timerInterval = null;
    this.totalTokens = 0;
    this.speedMultiplier = 1.0;
    this.isPaused = false;
    this.isRunning = false;

    // Canvas Pan & Zoom
    this.camera = { x: 0, y: 0, scale: 1.0 };
    this.isPanning = false;
    this.panStart = { x: 0, y: 0 };
    this.draggedNode = null;

    // DOM References
    this.viewport = document.getElementById('canvasViewport');
    this.svgLayer = document.getElementById('svgGraphLayer');
    this.edgesContainer = document.getElementById('edgesContainer');
    this.particlesContainer = document.getElementById('particlesContainer');
    this.nodesContainer = document.getElementById('nodesContainer');
    this.edgeLabelsContainer = document.getElementById('edgeLabelsContainer');
    
    // UI Elements
    this.hudActiveState = document.getElementById('hudActiveState');
    this.hudActiveTask = document.getElementById('hudActiveTask');
    this.hudTokensCount = document.getElementById('hudTokensCount');
    this.hudElapsedTime = document.getElementById('hudElapsedTime');
    
    this.inspectorDrawer = document.getElementById('inspectorDrawer');
    this.hoverHud = document.getElementById('hoverTooltipHud');
    this.globalLogs = document.getElementById('globalLiveLogStream');

    this.ws = null;
    this.init();
  }

  async init() {
    this.setupEventListeners();
    this.connectWebSocket();
    this.startAnimationLoop();
    lucide.createIcons();
  }

  // --- WebSocket & Real-Time Sync ---
  connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const statusBadge = document.getElementById('connectionStatus');
    const statusLabel = document.getElementById('connLabel');

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        statusBadge.className = 'flex items-center gap-1.5 text-xs text-emerald-400 font-medium bg-emerald-950/40 border border-emerald-800/40 px-2.5 py-1.5 rounded-lg';
        statusLabel.textContent = 'Live Connected';
        this.addGlobalLog('System', 'Connected to real-time agent server.', 'success');
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          this.handleServerEvent(msg);
        } catch (e) {
          console.error('Error parsing WS message', e);
        }
      };

      this.ws.onclose = () => {
        statusBadge.className = 'flex items-center gap-1.5 text-xs text-amber-400 font-medium bg-amber-950/40 border border-amber-800/40 px-2.5 py-1.5 rounded-lg';
        statusLabel.textContent = 'Offline (Simulator Mode)';
        setTimeout(() => this.connectWebSocket(), 3000);
      };

      this.ws.onerror = () => {
        statusBadge.className = 'flex items-center gap-1.5 text-xs text-amber-400 font-medium bg-amber-950/40 border border-amber-800/40 px-2.5 py-1.5 rounded-lg';
        statusLabel.textContent = 'Simulator Ready';
      };
    } catch (e) {
      this.fallbackFetchTopology();
    }
  }

  async fallbackFetchTopology() {
    try {
      const res = await fetch('/api/agents');
      const topo = await res.json();
      this.setTopology(topo);
    } catch (e) {
      console.warn('Using default fallback topology', e);
    }
  }

  handleServerEvent(event) {
    const { type, data } = event;

    if (type === 'init') {
      this.setTopology(data.topology);
      if (data.nodes_state) {
        this.nodesState = data.nodes_state;
        this.updateAllNodesVisuals();
      }
      return;
    }

    if (type === 'topology_updated') {
      this.setTopology(data.topology);
      return;
    }

    if (type === 'scenario_started') {
      this.isRunning = true;
      this.startTimer();
      document.getElementById('playBtnLabel').textContent = 'Running...';
      this.addGlobalLog('Scenario', `Started scenario: ${data.scenario_id}`, 'info');
    } else if (type === 'scenario_completed') {
      this.isRunning = false;
      this.stopTimer();
      document.getElementById('playBtnLabel').textContent = 'Start Flow';
      this.addGlobalLog('Scenario', `Completed scenario execution.`, 'success');
      this.updateHudStats();
    } else if (type === 'flow_reset') {
      this.resetVisuals();
      this.addGlobalLog('System', `Flow state reset.`, 'info');
    } else if (type === 'agent_status_change') {
      this.updateAgentStatus(data);
    } else if (type === 'data_transfer') {
      this.spawnDataPacket(data);
    } else if (type === 'speed_changed') {
      this.speedMultiplier = data.speed;
    }
  }

  setTopology(topology) {
    this.topology = topology;
    document.getElementById('agentsCountBadge').textContent = `(${topology.nodes.length} agents)`;
    
    // Initialize nodesState
    topology.nodes.forEach(node => {
      if (!this.nodesState[node.id]) {
        this.nodesState[node.id] = {
          id: node.id,
          status: 'idle',
          color_state: 'slate',
          active_task: 'Standby',
          last_input: null,
          last_output: null,
          error_details: null,
          logs: [],
          tokens_used: 0,
          duration_ms: 0
        };
      }
    });

    this.renderGraph();
    this.fitView();
  }

  // --- Graph Rendering & DOM Construction ---
  renderGraph() {
    this.nodesContainer.innerHTML = '';
    this.edgesContainer.innerHTML = '';
    this.edgeLabelsContainer.innerHTML = '';

    // Render Nodes
    this.topology.nodes.forEach(node => {
      const nodeEl = document.createElement('div');
      nodeEl.id = `node-${node.id}`;
      nodeEl.className = `agent-node ${node.is_orchestrator ? 'is-orchestrator' : ''}`;
      nodeEl.style.left = `${node.x}px`;
      nodeEl.style.top = `${node.y}px`;

      const iconName = node.icon || (node.is_orchestrator ? 'cpu' : 'bot');
      const modelShort = (node.model || 'gpt-4o').replace('whisper-', 'w-').replace('pyannote-', 'py-');

      nodeEl.innerHTML = `
        <div class="processing-indicator hidden" id="indicator-${node.id}"></div>
        <div style="padding:12px;position:relative;">
          <!-- Header -->
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="width:24px;height:24px;border-radius:3px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;color:#666;">
                <i data-lucide="${iconName}" style="width:14px;height:14px;"></i>
              </div>
              <div>
                <div style="font-weight:600;font-size:12px;color:#e5e5e5;line-height:1.2;">${node.name}</div>
                <div style="font-size:10px;color:#666;font-family:monospace;">${modelShort}</div>
              </div>
            </div>
            <span id="badge-${node.id}" style="padding:2px 6px;font-size:9px;font-weight:600;text-transform:uppercase;border-radius:3px;background:#1a1a1a;color:#666;">IDLE</span>
          </div>

          <!-- Role -->
          <div style="font-size:10px;color:#999;line-height:1.4;margin-bottom:8px;">
            ${node.role || 'Specialized Subagent'}
          </div>

          <!-- Task Bar -->
          <div style="padding-top:8px;border-top:1px solid #1a1a1a;display:flex;align-items:center;justify-content:space-between;font-size:9px;">
            <span style="color:#666;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" id="task-preview-${node.id}">Standby</span>
            <span style="font-family:monospace;color:#666;" id="duration-${node.id}">0ms</span>
          </div>
        </div>
      `;

      // Hover Event for Floating HUD
      nodeEl.addEventListener('mouseenter', (e) => this.showHoverHud(node.id, e));
      nodeEl.addEventListener('mousemove', (e) => this.moveHoverHud(e));
      nodeEl.addEventListener('mouseleave', () => this.hideHoverHud());

      // Click Event for Deep-Dive Inspector
      nodeEl.addEventListener('click', (e) => {
        e.stopPropagation();
        this.openInspector(node.id);
      });

      this.nodesContainer.appendChild(nodeEl);
    });

    // Render Edges
    this.renderEdges();
    lucide.createIcons();
    this.updateTransform();
  }

  renderEdges() {
    this.edgesContainer.innerHTML = '';
    this.edgeLabelsContainer.innerHTML = '';

    this.topology.edges.forEach(edge => {
      const srcNode = this.topology.nodes.find(n => n.id === edge.source);
      const tgtNode = this.topology.nodes.find(n => n.id === edge.target);
      if (!srcNode || !tgtNode) return;

      const srcCenter = this.getNodeCenter(srcNode);
      const tgtCenter = this.getNodeCenter(tgtNode);

      const pathData = this.calculateCurvedPath(srcCenter, tgtCenter);

      const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      pathEl.id = `edge-path-${edge.id}`;
      pathEl.setAttribute('d', pathData);
      pathEl.setAttribute('class', 'flow-edge-path');
      pathEl.setAttribute('marker-end', 'url(#arrowhead)');
      this.edgesContainer.appendChild(pathEl);

      // Edge Label Pill at midpoint
      const midPoint = this.getBezierMidpoint(srcCenter, tgtCenter);
      const labelEl = document.createElement('div');
      labelEl.id = `edge-label-${edge.id}`;
      labelEl.className = 'edge-label-pill';
      labelEl.style.left = `${midPoint.x}px`;
      labelEl.style.top = `${midPoint.y}px`;
      labelEl.textContent = edge.label || 'data';
      
      labelEl.addEventListener('click', (e) => {
        e.stopPropagation();
        this.openEdgePayloadModal(edge, { label: edge.label, sample: "Payload stream" });
      });

      this.edgeLabelsContainer.appendChild(labelEl);
    });
  }

  getNodeCenter(node) {
    const width = node.is_orchestrator ? 260 : 220;
    const height = 110;
    return {
      x: node.x + width / 2,
      y: node.y + height / 2
    };
  }

  calculateCurvedPath(src, tgt) {
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const cx1 = src.x + dx * 0.2;
    const cy1 = src.y + dy * 0.6;
    const cx2 = src.x + dx * 0.8;
    const cy2 = src.y + dy * 0.4;
    return `M ${src.x} ${src.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${tgt.x} ${tgt.y}`;
  }

  getBezierMidpoint(src, tgt) {
    return {
      x: (src.x + tgt.x) / 2,
      y: (src.y + tgt.y) / 2
    };
  }

  // --- Real-Time Agent State Updates & Color Themes ---
  updateAgentStatus(data) {
    const { agent_id, status, color_state, active_task, output_payload, input_payload, error, duration_ms, tokens, log } = data;
    
    if (!this.nodesState[agent_id]) return;
    const state = this.nodesState[agent_id];
    state.status = status;
    state.color_state = color_state;

    if (active_task) state.active_task = active_task;
    if (input_payload) state.last_input = input_payload;
    if (output_payload) state.last_output = output_payload;
    if (error) state.error_details = { message: error, data };
    if (duration_ms) state.duration_ms = duration_ms;
    if (tokens) {
      state.tokens_used += tokens;
      this.totalTokens += tokens;
    }
    if (log) {
      state.logs.push({ time: new Date().toLocaleTimeString(), text: log, level: color_state });
      this.addGlobalLog(agent_id, log, color_state);
    }

    this.applyNodeVisualState(agent_id, color_state, status, active_task, duration_ms);

    // If this node is currently open in the Inspector, update it live!
    if (this.selectedAgentId === agent_id) {
      this.populateInspector(agent_id);
    }

    this.updateHudStats();
  }

  applyNodeVisualState(nodeId, colorState, status, activeTask, durationMs) {
    const nodeEl = document.getElementById(`node-${nodeId}`);
    const badgeEl = document.getElementById(`badge-${nodeId}`);
    const indicatorEl = document.getElementById(`indicator-${nodeId}`);
    const taskEl = document.getElementById(`task-preview-${nodeId}`);
    const durEl = document.getElementById(`duration-${nodeId}`);

    if (!nodeEl) return;

    // Reset base classes
    nodeEl.classList.remove('state-amber', 'state-green', 'state-red');

    if (colorState === 'amber' || status === 'processing') {
      nodeEl.classList.add('state-amber');
      badgeEl.style.cssText = 'padding:2px 6px;font-size:9px;font-weight:600;text-transform:uppercase;border-radius:3px;background:#f59e0b;color:#000;';
      badgeEl.textContent = 'PROCESSING';
      indicatorEl.classList.remove('hidden');
      if (activeTask) taskEl.textContent = activeTask;
      taskEl.style.color = '#f59e0b';
    } else if (colorState === 'green' || status === 'passed') {
      nodeEl.classList.add('state-green');
      badgeEl.style.cssText = 'padding:2px 6px;font-size:9px;font-weight:600;text-transform:uppercase;border-radius:3px;background:#10b981;color:#000;';
      badgeEl.textContent = 'PASSED';
      indicatorEl.classList.add('hidden');
      taskEl.textContent = 'Completed';
      taskEl.style.color = '#10b981';
      if (durationMs) durEl.textContent = `${durationMs}ms`;
    } else if (colorState === 'red' || status === 'failed') {
      nodeEl.classList.add('state-red');
      badgeEl.style.cssText = 'padding:2px 6px;font-size:9px;font-weight:600;text-transform:uppercase;border-radius:3px;background:#ef4444;color:#fff;';
      badgeEl.textContent = 'FAILED';
      indicatorEl.classList.add('hidden');
      taskEl.textContent = 'Error';
      taskEl.style.color = '#ef4444';
    } else {
      badgeEl.style.cssText = 'padding:2px 6px;font-size:9px;font-weight:600;text-transform:uppercase;border-radius:3px;background:#1a1a1a;color:#666;';
      badgeEl.textContent = 'IDLE';
      indicatorEl.classList.add('hidden');
      taskEl.textContent = 'Standby';
      taskEl.style.color = '#666';
    }
  }

  updateAllNodesVisuals() {
    Object.keys(this.nodesState).forEach(aid => {
      const s = this.nodesState[aid];
      this.applyNodeVisualState(aid, s.color_state, s.status, s.active_task, s.duration_ms);
    });
  }

  // --- Real-Time Data Passing Particle Animation ---
  spawnDataPacket(transferData) {
    const { source, target, label, payload, duration_ms } = transferData;
    const srcNode = this.topology.nodes.find(n => n.id === source);
    const tgtNode = this.topology.nodes.find(n => n.id === target);
    if (!srcNode || !tgtNode) return;

    const srcCenter = this.getNodeCenter(srcNode);
    const tgtCenter = this.getNodeCenter(tgtNode);

    // Active Edge highlight
    const edge = this.topology.edges.find(e => e.source === source && e.target === target);
    if (edge) {
      const edgePath = document.getElementById(`edge-path-${edge.id}`);
      const edgeLabel = document.getElementById(`edge-label-${edge.id}`);
      if (edgePath) {
        edgePath.classList.add('active-amber');
        edgePath.setAttribute('marker-end', 'url(#arrowhead-amber)');
      }
      if (edgeLabel) {
        edgeLabel.classList.add('active-amber');
        edgeLabel.textContent = `⚡ ${label || 'Transferring...'}`;
      }

      setTimeout(() => {
        if (edgePath) {
          edgePath.classList.remove('active-amber');
          edgePath.setAttribute('marker-end', 'url(#arrowhead)');
        }
        if (edgeLabel) {
          edgeLabel.classList.remove('active-amber');
          edgeLabel.textContent = edge.label || 'data';
        }
      }, duration_ms || 800);
    }

    // Create moving SVG particle circle
    const particle = {
      id: `pkt-${Date.now()}-${Math.random()}`,
      src: srcCenter,
      tgt: tgtCenter,
      progress: 0,
      duration: duration_ms || 800,
      startTime: performance.now(),
      payload: payload,
      label: label,
      source: source,
      target: target
    };

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.id = particle.id;
    circle.setAttribute('r', '6');
    circle.setAttribute('class', 'data-packet-particle amber');
    circle.style.cursor = 'pointer';

    circle.addEventListener('click', (e) => {
      e.stopPropagation();
      this.openEdgePayloadModal({ source, target, label }, payload);
    });

    this.particlesContainer.appendChild(circle);
    particle.element = circle;
    this.activePackets.push(particle);

    this.addGlobalLog(source, `Passed data to [${target}]: ${label || 'Payload'} (${JSON.stringify(payload).length} B)`, 'info');
  }

  startAnimationLoop() {
    const animate = (currentTime) => {
      for (let i = this.activePackets.length - 1; i >= 0; i--) {
        const p = this.activePackets[i];
        const elapsed = currentTime - p.startTime;
        p.progress = Math.min(elapsed / (p.duration / this.speedMultiplier), 1.0);

        // Bezier position interpolation
        const pos = this.computeBezierPoint(p.src, p.tgt, p.progress);
        p.element.setAttribute('cx', pos.x);
        p.element.setAttribute('cy', pos.y);

        if (p.progress >= 1.0) {
          if (p.element.parentNode) {
            p.element.parentNode.removeChild(p.element);
          }
          this.activePackets.splice(i, 1);
        }
      }

      requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }

  computeBezierPoint(src, tgt, t) {
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const cx1 = src.x + dx * 0.2;
    const cy1 = src.y + dy * 0.6;
    const cx2 = src.x + dx * 0.8;
    const cy2 = src.y + dy * 0.4;

    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * t;

    const x = uuu * src.x + 3 * uu * t * cx1 + 3 * u * tt * cx2 + ttt * tgt.x;
    const y = uuu * src.y + 3 * uu * t * cy1 + 3 * u * tt * cy2 + ttt * tgt.y;

    return { x, y };
  }

  // --- Hover HUD Tooltip ---
  showHoverHud(agentId, e) {
    const node = this.topology.nodes.find(n => n.id === agentId);
    const state = this.nodesState[agentId] || {};
    if (!node) return;

    document.getElementById('hudTooltipName').textContent = node.name;
    document.getElementById('hudTooltipRole').textContent = node.role || 'Subagent';
    document.getElementById('hudTooltipTask').textContent = state.active_task || 'Idle';
    document.getElementById('hudTooltipModel').textContent = node.model || 'gpt-4o';
    document.getElementById('hudTooltipTokens').textContent = `${state.tokens_used || 0} tok`;

    const badge = document.getElementById('hudTooltipBadge');
    badge.textContent = (state.status || 'IDLE').toUpperCase();
    if (state.status === 'processing') badge.className = 'px-2 py-0.5 text-[9px] font-bold uppercase rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/40';
    else if (state.status === 'passed') badge.className = 'px-2 py-0.5 text-[9px] font-bold uppercase rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40';
    else if (state.status === 'failed') badge.className = 'px-2 py-0.5 text-[9px] font-bold uppercase rounded-full bg-red-500/20 text-red-400 border border-red-500/40';
    else badge.className = 'px-2 py-0.5 text-[9px] font-bold uppercase rounded-full bg-slate-800 text-slate-400';

    this.hoverHud.classList.remove('hidden');
    this.moveHoverHud(e);
  }

  moveHoverHud(e) {
    const x = e.clientX + 16;
    const y = e.clientY + 16;
    this.hoverHud.style.left = `${x}px`;
    this.hoverHud.style.top = `${y}px`;
  }

  hideHoverHud() {
    this.hoverHud.classList.add('hidden');
  }

  // --- Inspector Drawer (Deep Metadata & Payloads) ---
  openInspector(agentId) {
    this.selectedAgentId = agentId;
    this.populateInspector(agentId);
    this.inspectorDrawer.classList.remove('translate-x-full');
  }

  closeInspector() {
    this.selectedAgentId = null;
    this.inspectorDrawer.classList.add('translate-x-full');
  }

  populateInspector(agentId) {
    const node = this.topology.nodes.find(n => n.id === agentId);
    const state = this.nodesState[agentId] || {};
    if (!node) return;

    document.getElementById('inspectorTitle').textContent = node.name;
    document.getElementById('inspectorSubtitle').textContent = node.role || 'Agent Specification';
    document.getElementById('inspectorAgentId').textContent = `@${node.id}`;
    document.getElementById('inspectorModel').textContent = node.model || 'gpt-4o';
    document.getElementById('inspectorFile').textContent = node.filename || `${node.id}.agent.md`;

    // State Badge
    const stateBadge = document.getElementById('inspectorStateBadge');
    stateBadge.textContent = (state.status || 'IDLE').toUpperCase();
    if (state.status === 'processing') stateBadge.className = 'px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/40';
    else if (state.status === 'passed') stateBadge.className = 'px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40';
    else if (state.status === 'failed') stateBadge.className = 'px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-red-500/20 text-red-400 border border-red-500/40';
    else stateBadge.className = 'px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-slate-700 text-slate-300';

    document.getElementById('inspectorActiveTaskText').textContent = state.active_task || 'Standby';

    // Tools List
    const toolsContainer = document.getElementById('inspectorToolsList');
    toolsContainer.innerHTML = '';
    const tools = node.tools || [];
    if (tools.length === 0) {
      toolsContainer.innerHTML = '<span class="text-slate-500 italic text-[11px]">No specific tools declared.</span>';
    } else {
      tools.forEach(t => {
        const pill = document.createElement('span');
        pill.className = 'px-2 py-0.5 bg-indigo-950/40 border border-indigo-800/40 text-indigo-300 rounded font-mono text-[10px]';
        pill.textContent = t;
        toolsContainer.appendChild(pill);
      });
    }

    // Markdown Instructions
    document.getElementById('inspectorMarkdownBody').textContent = node.body_markdown || 'No instructions specified.';

    // Payloads
    document.getElementById('inspectorInputPayload').textContent = JSON.stringify(state.last_input || { note: "No input received yet" }, null, 2);
    document.getElementById('inspectorOutputPayload').textContent = JSON.stringify(state.last_output || { note: "No output produced yet" }, null, 2);

    // Logs Console
    const logsBox = document.getElementById('inspectorLogsConsole');
    logsBox.innerHTML = '';
    const logs = state.logs || [];
    document.getElementById('logsCountBadge').textContent = `${logs.length} logs`;
    if (logs.length === 0) {
      logsBox.innerHTML = '<div class="text-slate-500 italic">No execution events recorded yet.</div>';
    } else {
      logs.forEach(l => {
        const row = document.createElement('div');
        row.className = `flex gap-2 ${l.level === 'amber' ? 'text-amber-300' : l.level === 'green' ? 'text-emerald-300' : l.level === 'red' ? 'text-red-300' : 'text-slate-300'}`;
        row.innerHTML = `<span class="text-slate-500 shrink-0">[${l.time}]</span><span>${l.text}</span>`;
        logsBox.appendChild(row);
      });
    }

    // Telemetry
    document.getElementById('telemetryDuration').textContent = `${state.duration_ms || 0} ms`;
    document.getElementById('telemetryTokens').textContent = `${state.tokens_used || 0}`;
    const conf = state.status === 'failed' ? 0.42 : 0.96;
    document.getElementById('telemetryConfidenceBar').style.width = `${conf * 100}%`;
    document.getElementById('telemetryConfidenceBar').className = state.status === 'failed' ? 'bg-red-500 h-2 rounded-full' : 'bg-emerald-500 h-2 rounded-full';
    document.getElementById('telemetryConfidenceText').textContent = conf.toFixed(2);
  }

  // --- Edge Data Modal ---
  openEdgePayloadModal(edge, payload) {
    document.getElementById('edgeModalSrc').textContent = `@${edge.source}`;
    document.getElementById('edgeModalTgt').textContent = `@${edge.target}`;
    document.getElementById('edgeModalTime').textContent = new Date().toLocaleTimeString();
    document.getElementById('edgeModalJson').textContent = JSON.stringify(payload, null, 2);
    document.getElementById('edgeDataModal').classList.remove('hidden');
  }

  // --- Canvas Pan & Zoom Engine ---
  updateTransform() {
    const transform = `translate(${this.camera.x}px, ${this.camera.y}px) scale(${this.camera.scale})`;
    this.nodesContainer.style.transform = transform;
    this.edgeLabelsContainer.style.transform = transform;
    // SVG inner groups get the transform (not the SVG element itself which must cover full viewport)
    this.edgesContainer.setAttribute('transform', `translate(${this.camera.x}, ${this.camera.y}) scale(${this.camera.scale})`);
    const pg = document.getElementById('particlesContainer');
    if (pg) pg.setAttribute('transform', `translate(${this.camera.x}, ${this.camera.y}) scale(${this.camera.scale})`);
  }

  fitView() {
    const performFit = () => {
      if (this.topology.nodes.length === 0) return;
      const rect = this.viewport.getBoundingClientRect();
      const vpW = rect.width || window.innerWidth;
      const vpH = rect.height || (window.innerHeight - 220); // header(64) + footer(144) + buffer

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      this.topology.nodes.forEach(n => {
        const w = n.is_orchestrator ? 260 : 220;
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + w);
        maxY = Math.max(maxY, n.y + 115);
      });

      const graphW = maxX - minX + 100;
      const graphH = maxY - minY + 100;

      const scaleX = vpW / graphW;
      const scaleY = vpH / graphH;
      const scale = Math.min(Math.max(Math.min(scaleX, scaleY) * 0.85, 0.4), 1.3);

      this.camera.scale = scale;
      this.camera.x = (vpW - graphW * scale) / 2 - minX * scale + 50 * scale;
      this.camera.y = (vpH - graphH * scale) / 2 - minY * scale + 30 * scale;

      this.updateTransform();
    };

    // Defer one frame to ensure layout is computed
    requestAnimationFrame(() => requestAnimationFrame(performFit));
  }

  // --- Controls & Scenario Triggers ---
  setupEventListeners() {
    // Play / Run
    document.getElementById('playBtn').addEventListener('click', () => {
      const scenarioId = document.getElementById('scenarioSelect').value;
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'run_scenario', scenario_id: scenarioId, speed: this.speedMultiplier }));
      } else {
        fetch('/api/scenarios/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario_id: scenarioId, speed: this.speedMultiplier })
        });
      }
    });

    // Pause
    document.getElementById('pauseBtn').addEventListener('click', () => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'pause' }));
      } else {
        fetch('/api/scenarios/pause', { method: 'POST' });
      }
    });

    // Reset
    document.getElementById('resetBtn').addEventListener('click', () => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'reset' }));
      } else {
        fetch('/api/scenarios/reset', { method: 'POST' });
      }
    });

    // Chaos Failure Injection (optional - only if button exists)
    const chaosBtn = document.getElementById('chaosBtn');
    if (chaosBtn) {
      chaosBtn.addEventListener('click', () => {
        const activeNode = Object.values(this.nodesState).find(n => n.status === 'processing') || { id: 'asr_engine' };
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: 'chaos_fail_node', node_id: activeNode.id }));
        }
        this.addGlobalLog('Chaos', `Synthetic failure injected into @${activeNode.id}!`, 'red');
      });
    }

    // Reload Agents (optional - only if button exists)
    const reloadAgentsBtn = document.getElementById('reloadAgentsBtn');
    if (reloadAgentsBtn) {
      reloadAgentsBtn.addEventListener('click', () => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: 'refresh_agents' }));
        } else {
          this.fallbackFetchTopology();
        }
      });
    }

    // Speed Selector Buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.speed-btn').forEach(b => {
          b.className = 'speed-btn px-2 py-1 rounded hover:text-white transition';
        });
        btn.className = 'speed-btn px-2 py-1 rounded bg-indigo-600 text-white shadow';
        this.speedMultiplier = parseFloat(btn.dataset.speed);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: 'set_speed', speed: this.speedMultiplier }));
        }
      });
    });

    // Pan & Zoom on Canvas Viewport
    this.viewport.addEventListener('mousedown', (e) => {
      if (e.target.closest('.agent-node') || e.target.closest('.edge-label-pill') || e.target.closest('button')) return;
      this.isPanning = true;
      this.panStart = { x: e.clientX - this.camera.x, y: e.clientY - this.camera.y };
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isPanning) {
        this.camera.x = e.clientX - this.panStart.x;
        this.camera.y = e.clientY - this.panStart.y;
        this.updateTransform();
      }
    });

    window.addEventListener('mouseup', () => {
      this.isPanning = false;
    });

    this.viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      const newScale = Math.min(Math.max(this.camera.scale * zoomFactor, 0.3), 2.5);

      const rect = this.viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      this.camera.x = mouseX - (mouseX - this.camera.x) * (newScale / this.camera.scale);
      this.camera.y = mouseY - (mouseY - this.camera.y) * (newScale / this.camera.scale);
      this.camera.scale = newScale;

      this.updateTransform();
    }, { passive: false });

    // Zoom Buttons
    document.getElementById('zoomInBtn').addEventListener('click', () => {
      this.camera.scale = Math.min(this.camera.scale * 1.2, 2.5);
      this.updateTransform();
    });
    document.getElementById('zoomOutBtn').addEventListener('click', () => {
      this.camera.scale = Math.max(this.camera.scale * 0.8, 0.3);
      this.updateTransform();
    });
    document.getElementById('fitViewBtn').addEventListener('click', () => this.fitView());
    
    const resetZoomBtn = document.getElementById('resetZoomBtn');
    if (resetZoomBtn) {
      resetZoomBtn.addEventListener('click', () => {
        this.camera = { x: 0, y: 0, scale: 1.0 };
        this.updateTransform();
      });
    }

    // Inspector Tabs
    document.querySelectorAll('.inspector-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.inspector-tab').forEach(t => {
          t.className = 'inspector-tab px-3 py-2.5 text-slate-400 hover:text-slate-200 transition';
        });
        tab.className = 'inspector-tab active px-3 py-2.5 text-indigo-400 border-b-2 border-indigo-500 transition';

        const targetTab = tab.dataset.tab;
        document.getElementById('tabContentOverview').classList.toggle('hidden', targetTab !== 'overview');
        document.getElementById('tabContentPayloads').classList.toggle('hidden', targetTab !== 'payloads');
        document.getElementById('tabContentLogs').classList.toggle('hidden', targetTab !== 'logs');
        document.getElementById('tabContentTelemetry').classList.toggle('hidden', targetTab !== 'telemetry');
      });
    });

    document.getElementById('closeInspectorBtn').addEventListener('click', () => this.closeInspector());
    document.getElementById('closeEdgeModalBtn').addEventListener('click', () => {
      document.getElementById('edgeDataModal').classList.add('hidden');
    });

    // Copy JSON Buttons
    document.querySelectorAll('.copy-json-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.dataset.target;
        const text = document.getElementById(targetId).textContent;
        navigator.clipboard.writeText(text);
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy JSON', 1500);
      });
    });

    document.getElementById('copyEdgeJsonBtn').addEventListener('click', () => {
      const text = document.getElementById('edgeModalJson').textContent;
      navigator.clipboard.writeText(text);
      document.getElementById('copyEdgeJsonBtn').textContent = 'Copied!';
      setTimeout(() => document.getElementById('copyEdgeJsonBtn').textContent = 'Copy Payload', 1500);
    });

    document.getElementById('clearLogsBtn').addEventListener('click', () => {
      this.globalLogs.innerHTML = '';
    });

    const exportTraceBtn = document.getElementById('exportTraceBtn');
    if (exportTraceBtn) {
      exportTraceBtn.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify({
          timestamp: new Date().toISOString(),
          nodes: this.nodesState,
          events: this.eventHistory
        }, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `multi-agent-trace-${Date.now()}.json`;
        a.click();
      });
    }

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        document.getElementById('pauseBtn').click();
      } else if (e.code === 'KeyR') {
        document.getElementById('resetBtn').click();
      }
    });
  }

  // --- HUD & Global Stream Helpers ---
  startTimer() {
    this.startTime = Date.now();
    this.totalTokens = 0;
    if (this.timerInterval) clearInterval(this.timerInterval);
    this.timerInterval = setInterval(() => {
      if (this.startTime && !this.isPaused) {
        const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
        this.hudElapsedTime.textContent = `${elapsed}s`;
      }
    }, 100);
  }

  stopTimer() {
    if (this.timerInterval) clearInterval(this.timerInterval);
  }

  updateHudStats() {
    const activeNodes = Object.values(this.nodesState).filter(n => n.status === 'processing');
    if (activeNodes.length > 0) {
      this.hudActiveState.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span> Processing (${activeNodes.length})`;
      this.hudActiveTask.textContent = activeNodes[0].active_task || 'Active';
    } else {
      const hasFailed = Object.values(this.nodesState).some(n => n.status === 'failed');
      if (hasFailed) {
        this.hudActiveState.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-400"></span> Error Alert`;
        this.hudActiveTask.textContent = 'Anomaly Detected';
      } else {
        const allPassed = Object.values(this.nodesState).some(n => n.status === 'passed');
        this.hudActiveState.innerHTML = allPassed ? `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Completed` : `<span class="w-2 h-2 rounded-full bg-slate-500"></span> Idle`;
        this.hudActiveTask.textContent = allPassed ? 'Pipeline Finished' : 'Ready for run';
      }
    }
    this.hudTokensCount.textContent = `${this.totalTokens} tokens`;
  }

  addGlobalLog(source, message, level = 'info') {
    const time = new Date().toLocaleTimeString();
    this.eventHistory.push({ time, source, message, level });
    
    const row = document.createElement('div');
    const colorClass = level === 'amber' ? 'text-amber-300' : level === 'green' || level === 'success' ? 'text-emerald-300' : level === 'red' || level === 'error' ? 'text-red-400' : 'text-slate-300';

    row.className = `flex items-center gap-2 ${colorClass}`;
    row.innerHTML = `<span class="text-slate-600 shrink-0">[${time}]</span><span class="font-bold text-indigo-400">[@${source}]</span><span>${message}</span>`;
    
    this.globalLogs.appendChild(row);
    this.globalLogs.scrollTop = this.globalLogs.scrollHeight;
  }

  resetVisuals() {
    Object.keys(this.nodesState).forEach(aid => {
      this.nodesState[aid] = {
        id: aid,
        status: 'idle',
        color_state: 'slate',
        active_task: 'Standby',
        last_input: null,
        last_output: null,
        error_details: null,
        logs: [],
        tokens_used: 0,
        duration_ms: 0
      };
    });
    this.updateAllNodesVisuals();
    this.stopTimer();
    this.hudElapsedTime.textContent = '0.0s';
    this.totalTokens = 0;
    this.updateHudStats();
    if (this.selectedAgentId) this.populateInspector(this.selectedAgentId);
  }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  window.flowVisualizer = new AgentFlowVisualizer();
});
