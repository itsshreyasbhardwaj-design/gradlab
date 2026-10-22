/* Gradient Microscope — renders GradLab execution traces.
 *
 * This file draws what it is given. It performs no mathematics of its own: every
 * value, gradient, shape and derivative rule shown on screen is read straight
 * out of a trace JSON produced by gradlab.inspection. If a field is absent, the
 * UI says so rather than filling the gap.
 */
'use strict';

const SCHEMA_SUPPORTED = '1.0';

const state = {
  index: null,
  trace: null,
  view: 'graph',
  technical: false,
  step: 0,
  layout: null,
  capabilities: { allow_expressions: false },
};

/* ------------------------------------------------------------------ helpers */

const el = (id) => document.getElementById(id);

function fmtNum(v, digits = 6) {
  if (typeof v === 'string') return v;              // "NaN" / "Infinity"
  if (v === null || v === undefined) return '—';
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(3);
  return Number(v.toPrecision(digits)).toString();
}

function fmtValues(desc, limit = 12) {
  if (!desc || !desc.present) return '—';
  const vals = desc.values || [];
  const shown = vals.slice(0, limit).map((v) => fmtNum(v)).join(', ');
  const more = vals.length > limit || desc.truncated;
  const tailCount = desc.size - Math.min(limit, vals.length);
  if (desc.shape && desc.shape.length === 0) return shown;
  return `[${shown}${more ? `, … (+${tailCount} more)` : ''}]`;
}

function shapeLabel(shape) {
  if (!shape || shape.length === 0) return 'scalar';
  return `(${shape.join('×')})`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function statsRow(desc) {
  if (!desc || !desc.present || !desc.stats) return '';
  const s = desc.stats;
  let html = `min ${fmtNum(s.min)} · max ${fmtNum(s.max)} · mean ${fmtNum(s.mean)} · ‖·‖₂ ${fmtNum(s.l2_norm)}`;
  if (s.num_nonfinite > 0) {
    html += ` <span class="tag warn">${s.num_nonfinite} non-finite</span>`;
  }
  return `<div class="hint">${html}</div>`;
}

/* --------------------------------------------------------------- data loading */

async function boot() {
  try {
    const res = await fetch('./api/capabilities');
    if (res.ok) state.capabilities = await res.json();
  } catch { /* served statically: no API, which is fine */ }

  if (state.capabilities.allow_expressions) {
    el('custom-panel').hidden = false;
    const lim = state.capabilities.limits || {};
    el('custom-hint').textContent =
      `Allowlisted operations only. Up to ${lim.max_nodes || '?'} operations, `
      + `${lim.max_variables || '?'} variables, ${lim.max_tensor_elements || '?'} elements each.`;
  }

  try {
    const res = await fetch('./traces/index.json');
    if (!res.ok) throw new Error('no index');
    state.index = await res.json();
  } catch {
    el('empty-state').hidden = false;
    return;
  }

  el('version-badge').textContent =
    `GradLab ${state.index.gradlab_version} · schema ${state.index.schema_version}`;
  if (state.index.schema_version !== SCHEMA_SUPPORTED) {
    el('version-badge').classList.add('warn');
    el('version-badge').title =
      `This page understands schema ${SCHEMA_SUPPORTED}; the traces are ${state.index.schema_version}. Regenerate them.`;
  }

  buildSidebar();
  if (state.index.traces.length) await loadTrace(state.index.traces[0]);
}

const GROUPS = {
  expression: 'Expressions',
  training_step: 'One optimiser step',
  training_run: 'A whole training run',
  lr_comparison: 'Learning rates',
};

function buildSidebar() {
  const list = el('trace-list');
  list.innerHTML = '';
  const byKind = {};
  for (const entry of state.index.traces) (byKind[entry.kind] ||= []).push(entry);

  for (const [kind, label] of Object.entries(GROUPS)) {
    if (!byKind[kind]) continue;
    const head = document.createElement('li');
    head.className = 'group';
    head.textContent = label;
    list.appendChild(head);
    for (const entry of byKind[kind]) {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.innerHTML = `${escapeHtml(entry.label)}<span class="meta">${
        entry.nodes !== undefined ? `${entry.nodes} nodes` :
        entry.val_accuracy !== undefined ? `val acc ${entry.val_accuracy.toFixed(4)}` : ''
      }</span>`;
      btn.dataset.slug = entry.slug;
      btn.addEventListener('click', () => loadTrace(entry));
      li.appendChild(btn);
      list.appendChild(li);
    }
  }
}

async function loadTrace(entry) {
  const res = await fetch(`./traces/${entry.file}`);
  state.trace = await res.json();
  state.step = 0;
  showTrace();
  for (const btn of document.querySelectorAll('#trace-list button')) {
    btn.setAttribute('aria-current', String(btn.dataset.slug === entry.slug));
  }
}

/* ------------------------------------------------------------------ chrome */

function showTrace() {
  const t = state.trace;
  el('empty-state').hidden = true;
  el('trace-view').hidden = false;
  el('trace-title').textContent = t.label || t.slug || 'trace';
  el('trace-expression').textContent = t.expression_text || '';
  el('trace-note').textContent = t.teaching_note || '';

  const available = {
    graph: Boolean(t.nodes && t.nodes.length),
    forward: Boolean(t.forward_steps && t.forward_steps.length),
    backward: Boolean(t.backward_steps && t.backward_steps.length),
    params: Boolean(t.parameter_updates && t.parameter_updates.length),
    dynamics: Boolean(t.train_loss || t.runs),
  };
  for (const tab of document.querySelectorAll('.tab')) {
    const v = tab.dataset.view;
    tab.disabled = !available[v];
    tab.onclick = () => selectView(v);
  }
  if (!available[state.view]) {
    state.view = Object.keys(available).find((k) => available[k]) || 'graph';
  }
  selectView(state.view);
}

function selectView(view) {
  state.view = view;
  state.step = 0;
  for (const tab of document.querySelectorAll('.tab')) {
    tab.setAttribute('aria-selected', String(tab.dataset.view === view));
  }
  for (const name of ['graph', 'forward', 'backward', 'params', 'dynamics']) {
    el(`view-${name}`).hidden = name !== view;
  }
  ({ graph: renderGraphView, forward: renderForward, backward: renderBackward,
     params: renderParams, dynamics: renderDynamics }[view])();
}

/* ------------------------------------------------- View A: computation graph */

const NODE_W = 128;
const NODE_H = 46;
const GAP_X = 74;
const GAP_Y = 20;

function computeLayout(trace) {
  const nodes = new Map();
  for (const leaf of trace.leaves || []) {
    nodes.set(leaf.id, {
      id: leaf.id, kind: leaf.is_parameter ? 'param' : 'leaf',
      title: leaf.name, shape: leaf.shape, layer: 0, data: leaf,
    });
  }
  for (const node of trace.nodes || []) {
    nodes.set(node.id, {
      id: node.id, kind: 'op', title: node.op, shape: node.output.shape,
      layer: 0, data: node,
    });
  }
  // trace.nodes is in forward (seq) order, so one pass settles every layer.
  for (const node of trace.nodes || []) {
    const sources = (trace.edges || []).filter((e) => e.to === node.id);
    const target = nodes.get(node.id);
    target.layer = 1 + Math.max(0, ...sources.map((e) => (nodes.get(e.from)?.layer ?? 0)));
  }

  const byLayer = new Map();
  for (const node of nodes.values()) {
    if (!byLayer.has(node.layer)) byLayer.set(node.layer, []);
    byLayer.get(node.layer).push(node);
  }
  let maxRows = 0;
  for (const column of byLayer.values()) maxRows = Math.max(maxRows, column.length);

  const height = Math.max(240, maxRows * (NODE_H + GAP_Y) + 50);
  for (const [layer, column] of byLayer) {
    const total = column.length * (NODE_H + GAP_Y) - GAP_Y;
    column.forEach((node, i) => {
      node.x = 26 + layer * (NODE_W + GAP_X);
      node.y = (height - total) / 2 + i * (NODE_H + GAP_Y);
    });
  }
  const width = 52 + (Math.max(...byLayer.keys()) + 1) * (NODE_W + GAP_X);
  return { nodes, edges: trace.edges || [], width, height };
}

function renderGraphView() {
  const trace = state.trace;
  state.layout = computeLayout(trace);
  drawGraph();
  el('graph-inspector').innerHTML = '<p class="placeholder">Select a node to inspect it.</p>';
}

const GRAPH_SVG_FOR_VIEW = { graph: 'graph-svg', forward: 'forward-svg', backward: 'backward-svg' };

function drawGraph(activeNodeId, flowingEdges) {
  const { nodes, edges, width, height } = state.layout;
  const svg = el(GRAPH_SVG_FOR_VIEW[state.view] || 'graph-svg');
  if (!svg) return;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  // Scale down to fit the pane when the graph is wider than it, but never scale
  // a small graph up — oversized nodes look broken and read as a rendering bug.
  const available = (svg.parentElement?.clientWidth || width) - 4;
  const scale = Math.min(1, available / width);
  svg.setAttribute('width', Math.round(width * scale));
  svg.setAttribute('height', Math.round(height * scale));

  const flowing = new Set(flowingEdges || []);
  let markup = '';

  for (const edge of edges) {
    const from = nodes.get(edge.from);
    const to = nodes.get(edge.to);
    if (!from || !to) continue;
    const x1 = from.x + NODE_W;
    const y1 = from.y + NODE_H / 2;
    const x2 = to.x;
    const y2 = to.y + NODE_H / 2;
    const mid = (x1 + x2) / 2;
    const isActive = flowing.has(`${edge.from}->${edge.to}`);
    markup += `<path class="gedge${isActive ? ' active' : ''}" `
      + `d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}"/>`;
    if (edge.shape && edge.shape.length) {
      markup += `<text class="edge-label" x="${mid}" y="${(y1 + y2) / 2 - 4}" `
        + `text-anchor="middle">${edge.shape.join('×')}</text>`;
    }
  }

  for (const node of nodes.values()) {
    const classes = ['gnode', node.kind];
    if (node.id === activeNodeId) classes.push('active');
    const label = node.kind === 'op'
      ? node.title
      : node.title + (node.kind === 'param' ? ' ⚙' : '');
    markup += `<g class="${classes.join(' ')}" data-id="${node.id}">`
      + `<rect x="${node.x}" y="${node.y}" width="${NODE_W}" height="${NODE_H}"/>`
      + `<text x="${node.x + NODE_W / 2}" y="${node.y + 20}" text-anchor="middle">`
      + `${escapeHtml(label)}</text>`
      + `<text class="shape-label" x="${node.x + NODE_W / 2}" y="${node.y + 34}" `
      + `text-anchor="middle">${shapeLabel(node.shape)}</text></g>`;
  }

  svg.innerHTML = markup;
  if (state.view === 'graph') {
    for (const g of svg.querySelectorAll('.gnode')) {
      g.addEventListener('click', () => inspectNode(g.dataset.id));
    }
  }
}

function inspectNode(id) {
  const node = state.layout.nodes.get(id);
  if (!node) return;
  drawGraph(id);
  const panel = el('graph-inspector');

  if (node.kind !== 'op') {
    const leaf = node.data;
    panel.innerHTML = `<h3>${escapeHtml(leaf.name)}</h3>
      <p class="hint">${leaf.is_parameter ? 'A learnable parameter' : 'A leaf tensor'} —
      nothing produced it, so it is where a gradient comes to rest.</p>
      <dl class="kv">
        <dt>shape</dt><dd>${shapeLabel(leaf.shape)}</dd>
        <dt>dtype</dt><dd>${leaf.dtype}</dd>
        <dt>requires_grad</dt><dd>${leaf.requires_grad}</dd>
        <dt>is_leaf</dt><dd>${leaf.is_leaf}</dd>
      </dl>
      <h3>value</h3><div class="values">${fmtValues(leaf.value, 24)}</div>
      ${statsRow(leaf.value)}`;
    return;
  }

  const n = node.data;
  const inputs = n.inputs.map((t) => `<dt>${escapeHtml(t.name)}</dt>`
    + `<dd>${shapeLabel(t.shape)} ${t.is_leaf ? '<span class="tag">leaf</span>' : ''}</dd>`).join('');
  const metaRows = Object.entries(n.meta || {})
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(JSON.stringify(v))}</dd>`).join('');

  panel.innerHTML = `<h3>${escapeHtml(n.op)} <span class="tag">seq ${n.seq}</span></h3>
    <p class="expression">${escapeHtml(n.expression)}</p>
    <div class="rule">${escapeHtml(n.backward_rule)}</div>
    <h3>inputs</h3><dl class="kv">${inputs}</dl>
    <h3>output</h3>
    <dl class="kv"><dt>shape</dt><dd>${shapeLabel(n.output.shape)}</dd>
      <dt>dtype</dt><dd>${n.output.dtype}</dd></dl>
    <div class="values">${n.output.value.present
      ? fmtValues(n.output.value, 24)
      : `<span class="hint">${escapeHtml(n.output.value.reason || 'not retained')}</span>`}</div>
    ${statsRow(n.output.value)}
    ${n.num_broadcast_inputs > 0
      ? `<p class="hint"><span class="tag">broadcast</span> ${n.num_broadcast_inputs} input(s)
         were stretched to the output shape, so their gradients get summed back down.</p>` : ''}
    ${metaRows ? `<h3>saved for backward</h3><dl class="kv">${metaRows}</dl>` : ''}`;
}

/* ------------------------------------------------- shared stepper machinery */

function renderStepper(container, count, onSelect, labeller) {
  const parts = [`<button class="nav" data-step="prev">◀ prev</button>`];
  for (let i = 0; i < count; i += 1) {
    parts.push(`<button data-step="${i}" aria-current="${i === state.step}">${labeller(i)}</button>`);
  }
  parts.push(`<button class="nav" data-step="next">next ▶</button>`);
  container.innerHTML = parts.join('');
  for (const btn of container.querySelectorAll('button')) {
    btn.addEventListener('click', () => {
      const v = btn.dataset.step;
      if (v === 'prev') state.step = Math.max(0, state.step - 1);
      else if (v === 'next') state.step = Math.min(count - 1, state.step + 1);
      else state.step = Number(v);
      onSelect();
    });
  }
}

/* ------------------------------------------------- View B: forward propagation */

function renderForward() {
  const steps = state.trace.forward_steps || [];
  if (!steps.length) return;
  state.layout = state.layout || computeLayout(state.trace);
  renderStepper(el('forward-stepper'), steps.length, renderForward, (i) => `${i + 1}`);

  const step = steps[state.step];
  const node = (state.trace.nodes || []).find((n) => n.id === step.node);
  const inputs = (node ? node.inputs : []).map((t) => `
    <div class="term"><span class="lbl">${escapeHtml(t.name)} ${shapeLabel(t.shape)}</span>
      <span class="values">${fmtValues(t.value, 8)}</span></div>`).join('<span class="op">,</span>');

  el('forward-detail').innerHTML = `
    <div class="card">
      <h3>Step ${step.step} of ${steps.length} — ${escapeHtml(step.op)}</h3>
      <p class="expression">${escapeHtml(step.expression)}</p>
      <div class="chain">${inputs}
        <span class="op">→</span>
        <div class="term"><span class="lbl">output ${shapeLabel(step.output.shape)}</span>
          <span class="values">${node && node.output.value.present
            ? fmtValues(node.output.value, 8)
            : '<span class="hint">not retained</span>'}</span></div>
      </div>
      ${node ? statsRow(node.output.value) : ''}
      <div class="beginner-only">
        <p class="note">This is one operation of the forward pass. GradLab computed the
        output above and, at the same time, recorded this step as a node in the graph —
        including whatever it needs to remember in order to differentiate it later.</p>
      </div>
      <div class="technical-only">
        <div class="rule">backward rule registered here: ${escapeHtml(step.backward_rule)}</div>
        ${node && Object.keys(node.meta || {}).length
          ? `<p class="hint">saved context: ${escapeHtml(JSON.stringify(node.meta))}</p>` : ''}
      </div>
    </div>`;

  drawGraph(step.node, (state.trace.edges || [])
    .filter((e) => e.to === step.node).map((e) => `${e.from}->${e.to}`));
}

/* --------------------------------------------------- View C: backpropagation */

function renderBackward() {
  const steps = state.trace.backward_steps || [];
  if (!steps.length) return;
  state.layout = state.layout || computeLayout(state.trace);
  renderStepper(el('backward-stepper'), steps.length, renderBackward, (i) => `${i + 1}`);

  const step = steps[state.step];
  const seed = state.trace.seed;

  const contributions = step.contributions.map((c) => {
    if (!c.received_gradient) {
      return `<tr><td>${escapeHtml(c.name)}</td><td class="num">${shapeLabel(c.shape)}</td>
        <td colspan="2" class="hint">${escapeHtml(c.why_none || 'no gradient')}</td></tr>`;
    }
    const tags = [
      c.is_leaf ? '<span class="tag">leaf</span>' : '',
      c.broadcast_reduced ? '<span class="tag">summed over broadcast axes</span>' : '',
    ].join('');
    const local = c.local_derivative
      ? `<td class="num values">${fmtValues(c.local_derivative, 6)}</td>`
      : '<td class="hint">—</td>';
    return `<tr><td>${escapeHtml(c.name)}${tags}</td>
      <td class="num">${shapeLabel(c.shape)}</td>
      ${local}
      <td class="num values grad">${fmtValues(c.gradient, 6)}</td></tr>`;
  }).join('');

  const notes = step.contributions
    .filter((c) => c.broadcast_note || c.shape_note || c.local_derivative_note)
    .map((c) => `<p class="hint"><strong>${escapeHtml(c.name)}:</strong> ${
      escapeHtml(c.broadcast_note || c.shape_note || '')}${
      c.local_derivative_note ? ` ${escapeHtml(c.local_derivative_note)}` : ''}</p>`).join('');

  el('backward-detail').innerHTML = `
    ${state.step === 0 && seed ? `
    <div class="card">
      <h3>The seed</h3>
      <div class="chain"><div class="term"><span class="lbl">dL/dL</span>
        <span class="values grad">${fmtValues(seed.value, 8)}</span></div></div>
      <p class="note">${escapeHtml(seed.explanation)}</p>
    </div>` : ''}
    <div class="card">
      <h3>Backward step ${step.step} of ${steps.length} — ${escapeHtml(step.op)}</h3>
      <div class="chain">
        <div class="term"><span class="lbl">upstream gradient ${shapeLabel(step.output_shape)}</span>
          <span class="values grad">${fmtValues(step.upstream_gradient, 8)}</span></div>
        <span class="op">×</span>
        <div class="term"><span class="lbl">local derivative of ${escapeHtml(step.op)}</span>
          <span class="values">${escapeHtml(step.backward_rule)}</span></div>
        <span class="op">=</span>
        <div class="term"><span class="lbl">gradient for each input</span>
          <span class="values grad">see table</span></div>
      </div>
      <table>
        <thead><tr><th>input</th><th>shape</th><th>local derivative</th><th>gradient received</th></tr></thead>
        <tbody>${contributions}</tbody>
      </table>
      ${notes}
      <div class="beginner-only">
        <p class="note">The chain rule in one line: whatever gradient arrived at this
        operation's output gets multiplied by how much that output changes per unit change
        in each input. The product is what each input receives. If an input is used in more
        than one place, the contributions from every place are added together.</p>
      </div>
    </div>
    ${state.step === steps.length - 1 ? renderLeafGradients() : ''}`;

  drawGraph(step.node, (state.trace.edges || [])
    .filter((e) => e.to === step.node).map((e) => `${e.from}->${e.to}`));
}

function renderLeafGradients() {
  const leaves = state.trace.leaf_gradients || [];
  if (!leaves.length) return '';
  const rows = leaves.map((l) => `<tr><td>${escapeHtml(l.name)}</td>
    <td class="num">${shapeLabel(l.shape)}</td>
    <td class="num values grad">${fmtValues(l.gradient, 12)}</td></tr>`).join('');
  return `<div class="card"><h3>Final accumulated gradients</h3>
    <table><thead><tr><th>leaf</th><th>shape</th><th>d(output)/d(leaf)</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="note">These are the values sitting in <code>.grad</code> once the pass
    finishes. An optimiser reads exactly this.</p></div>`;
}

/* ------------------------------------------------- View D: parameter updates */

function renderParams() {
  const updates = state.trace.parameter_updates || [];
  const opt = state.trace.optimizer || {};
  const rows = updates.map((u) => {
    const before = u.before.values || [];
    const after = u.after.values || [];
    const grad = u.gradient.present ? (u.gradient.values || []) : [];
    const delta = u.delta.present ? (u.delta.values || []) : [];
    return `<tr>
      <td>${escapeHtml(u.name)}<span class="meta"> ${shapeLabel(u.shape)}</span></td>
      <td class="num">${fmtNum(before[0])}</td>
      <td class="num grad">${grad.length ? fmtNum(grad[0]) : '—'}</td>
      <td class="num">${delta.length ? fmtNum(delta[0]) : '—'}</td>
      <td class="num">${fmtNum(after[0])}</td>
      <td class="num">${fmtNum(u.learning_rate)}</td>
    </tr>`;
  }).join('');

  const moments = updates.filter((u) => u.state_after && u.state_after.bias_correction);
  const momentCards = moments.slice(0, 4).map((u) => {
    const s = u.state_after;
    const bc = s.bias_correction;
    return `<div class="card">
      <h3>${escapeHtml(u.name)} — Adam state</h3>
      <dl class="kv">
        <dt>m (1st moment)</dt><dd>${fmtValues(s.m, 6)}</dd>
        <dt>v (2nd moment)</dt><dd>${fmtValues(s.v, 6)}</dd>
        <dt>t</dt><dd>${bc.t}</dd>
        <dt>1 − β₁ᵗ</dt><dd>${fmtNum(bc.one_minus_beta1_t)}</dd>
        <dt>1 − β₂ᵗ</dt><dd>${fmtNum(bc.one_minus_beta2_t)}</dd>
      </dl>
      <p class="note">${escapeHtml(bc.note)}</p>
    </div>`;
  }).join('');

  el('params-detail').innerHTML = `
    <div class="card">
      <h3>${escapeHtml(opt.type || 'optimiser')} — step ${opt.step_count ?? '?'}</h3>
      <div class="rule">${escapeHtml(opt.update_rule || '')}</div>
      <p class="hint">${escapeHtml(opt.repr || '')}</p>
      <p class="note">One row per parameter, showing the first element of each tensor.
      The <em>update</em> column is literally <code>after − before</code> as measured
      on the real tensors, not a recomputation of the rule.</p>
      <table>
        <thead><tr><th>parameter</th><th>before</th><th>gradient</th><th>update</th>
          <th>after</th><th>lr</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${state.trace.loss ? `<div class="card"><h3>Loss at this step</h3>
      <div class="values">${fmtValues(state.trace.loss.value)}</div>
      <p class="hint">criterion: ${escapeHtml(state.trace.loss.criterion)}</p></div>` : ''}
    <div class="grid-2">${momentCards}</div>`;
}

/* ------------------------------------------------- View E: training dynamics */

const PALETTE = ['#6ea8fe', '#ffa94d', '#51cf66', '#f783ac', '#9775fa', '#ffd43b', '#4dd4c4'];

function lineChart(series, opts = {}) {
  const W = opts.width || 560;
  const H = opts.height || 220;
  const pad = { l: 52, r: 12, t: 10, b: 30 };
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  if (!all.length) return '<p class="hint">no finite values to plot</p>';
  const log = opts.log && all.every((v) => v > 0);
  const tx = (v) => (log ? Math.log10(v) : v);
  let lo = Math.min(...all.map(tx));
  let hi = Math.max(...all.map(tx));
  if (hi - lo < 1e-12) { hi = lo + 1; }
  const maxLen = Math.max(...series.map((s) => s.values.length));

  const px = (i, n) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - pad.l - pad.r));
  const py = (v) => pad.t + (1 - (tx(v) - lo) / (hi - lo)) * (H - pad.t - pad.b);

  let g = '';
  for (let k = 0; k <= 4; k += 1) {
    const frac = k / 4;
    const y = pad.t + frac * (H - pad.t - pad.b);
    const val = lo + (1 - frac) * (hi - lo);
    g += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="#2a2f3d"/>`;
    g += `<text x="${pad.l - 6}" y="${y + 3.5}" text-anchor="end" fill="#9aa1b1" `
      + `font-size="9.5" font-family="ui-monospace,monospace">`
      + `${log ? `1e${val.toFixed(1)}` : fmtNum(val, 3)}</text>`;
  }
  series.forEach((s, si) => {
    const pts = s.values
      .map((v, i) => (Number.isFinite(v) ? `${px(i, s.values.length).toFixed(1)},${py(v).toFixed(1)}` : null))
      .filter(Boolean).join(' ');
    g += `<polyline fill="none" stroke="${s.color || PALETTE[si % PALETTE.length]}" `
      + `stroke-width="1.8" points="${pts}"/>`;
  });
  if (opts.markX !== undefined && opts.markX !== null) {
    const x = px(opts.markX, maxLen);
    g += `<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H - pad.b}" stroke="#ffa94d" `
      + `stroke-dasharray="3 3"/>`;
  }
  g += `<text x="${pad.l}" y="${H - 8}" fill="#9aa1b1" font-size="9.5">1</text>`;
  g += `<text x="${W - pad.r}" y="${H - 8}" text-anchor="end" fill="#9aa1b1" font-size="9.5">`
    + `${maxLen}</text>`;
  g += `<text x="${(W) / 2}" y="${H - 8}" text-anchor="middle" fill="#9aa1b1" font-size="9.5">`
    + `${escapeHtml(opts.xLabel || 'epoch')}</text>`;

  const legend = series.map((s, si) => `<span><i class="swatch" style="background:${
    s.color || PALETTE[si % PALETTE.length]}"></i>${escapeHtml(s.label)}</span>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}">${g}</svg><div class="legend">${legend}</div>`;
}

function surfaceChart(surface, points) {
  const W = 300;
  const H = 300;
  const res = surface.resolution;
  const scores = surface.scores;
  const cell = W / res;
  let g = '';
  for (let r = 0; r < res; r += 1) {
    for (let c = 0; c < res; c += 1) {
      const v = scores[r][c];
      const p = 1 / (1 + Math.exp(-v));
      const col = p > 0.5
        ? `rgba(110,168,254,${(p - 0.5) * 1.4})`
        : `rgba(255,169,77,${(0.5 - p) * 1.4})`;
      g += `<rect x="${(c * cell).toFixed(2)}" y="${((res - 1 - r) * cell).toFixed(2)}" `
        + `width="${cell + 0.6}" height="${cell + 0.6}" fill="${col}"/>`;
    }
  }
  const xs = surface.x;
  const ys = surface.y;
  const sx = (v) => ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * W;
  const sy = (v) => H - ((v - ys[0]) / (ys[ys.length - 1] - ys[0])) * H;
  if (points) {
    points.X.forEach((pt, i) => {
      const cls = points.y[i];
      g += `<circle cx="${sx(pt[0]).toFixed(1)}" cy="${sy(pt[1]).toFixed(1)}" r="2.3" `
        + `fill="${cls > 0.5 ? '#6ea8fe' : '#ffa94d'}" stroke="#0f1117" stroke-width=".6"/>`;
    });
  }
  return `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

function histogramChart(hist) {
  const W = 320;
  const H = 150;
  const counts = hist.counts;
  const max = Math.max(...counts, 1);
  const bw = W / counts.length;
  let g = '';
  counts.forEach((n, i) => {
    const h = (n / max) * (H - 20);
    g += `<rect x="${(i * bw).toFixed(1)}" y="${(H - 16 - h).toFixed(1)}" `
      + `width="${(bw - 1).toFixed(1)}" height="${h.toFixed(1)}" fill="#6ea8fe" opacity=".8"/>`;
  });
  g += `<text x="0" y="${H - 3}" fill="#9aa1b1" font-size="9.5">${fmtNum(hist.bin_edges[0])}</text>`;
  g += `<text x="${W}" y="${H - 3}" text-anchor="end" fill="#9aa1b1" font-size="9.5">`
    + `${fmtNum(hist.bin_edges[hist.bin_edges.length - 1])}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

function renderDynamics() {
  const t = state.trace;
  const target = el('dynamics-detail');

  if (t.runs) { renderLrComparison(target, t); return; }

  const cards = [];
  cards.push(`<div class="chart"><h3>Loss</h3>${lineChart([
    { label: 'train', values: t.train_loss || [] },
    { label: 'validation', values: t.val_loss || [] },
  ], { log: true, markX: t.config && t.val_loss && t.val_loss.length
      ? t.val_loss.indexOf(Math.min(...t.val_loss)) : null })}
    <p class="hint">Log scale. The dashed line marks the lowest validation loss —
    past it, the training curve keeps falling while validation does not.</p></div>`);

  if (t.val_metric && t.val_metric.length) {
    cards.push(`<div class="chart"><h3>Accuracy</h3>${lineChart([
      { label: 'train', values: t.train_metric || [] },
      { label: 'validation', values: t.val_metric || [] },
    ], {})}</div>`);
  }
  if (t.grad_norm && t.grad_norm.length) {
    cards.push(`<div class="chart"><h3>Gradient norm ‖g‖₂</h3>${lineChart([
      { label: '‖g‖₂', values: t.grad_norm, color: PALETTE[1] },
    ], { log: true })}
      <p class="hint">Growing by orders of magnitude means divergence; collapsing early
      means the signal is not reaching the first layers.</p></div>`);
  }
  if (t.learning_rate && new Set(t.learning_rate).size > 1) {
    cards.push(`<div class="chart"><h3>Learning rate</h3>${lineChart([
      { label: 'lr', values: t.learning_rate, color: PALETTE[2] },
    ], {})}</div>`);
  }

  let extras = '';
  if (t.decision_surfaces && t.decision_surfaces.length) {
    const panels = t.decision_surfaces.map((s) => `<div class="chart">
      <h3>Epoch ${s.epoch}</h3>${surfaceChart(s.surface, t.points && t.points.val)}</div>`).join('');
    extras += `<h3 style="margin-top:1rem">Decision boundary over training</h3>
      <div class="grid-2">${panels}</div>
      <p class="hint">Blue and orange are the model's two predicted regions; dots are the
      validation points. Each panel is a real forward pass over a grid at that epoch.</p>`;
  }
  if (t.weight_histograms && t.weight_histograms.length) {
    const first = t.weight_histograms[0];
    const last = t.weight_histograms[t.weight_histograms.length - 1];
    extras += `<h3 style="margin-top:1rem">Weight distribution</h3><div class="grid-2">
      <div class="chart"><h3>Epoch ${first.epoch} — ‖w‖₂ ${fmtNum(first.l2_norm)}</h3>
        ${histogramChart(first)}</div>
      <div class="chart"><h3>Epoch ${last.epoch} — ‖w‖₂ ${fmtNum(last.l2_norm)}</h3>
        ${histogramChart(last)}</div></div>`;
  }

  const final = t.final
    ? `<div class="card"><h3>Final measured accuracy</h3>
       <p class="values">train ${fmtNum(t.final.train_accuracy)} ·
       validation ${fmtNum(t.final.val_accuracy)}</p></div>` : '';

  target.innerHTML = final + `<div class="grid-2">${cards.join('')}</div>` + extras
    + (t.warnings && t.warnings.length
      ? `<div class="card"><h3>Warnings recorded during the run</h3>${
        t.warnings.map((w) => `<p class="hint">${escapeHtml(w)}</p>`).join('')}</div>` : '');
}

function renderLrComparison(target, t) {
  const series = t.runs.filter((r) => r.train_loss.length)
    .map((r, i) => ({ label: `lr = ${r.lr}`, values: r.train_loss, color: PALETTE[i % PALETTE.length] }));
  const rows = t.runs.map((r) => `<tr${r.stopped ? ' class="highlight"' : ''}>
    <td class="num">${r.lr}</td>
    <td class="num">${r.stopped ? '—' : fmtNum(r.final_loss)}</td>
    <td class="num">${r.stopped ? '—' : fmtNum(r.max_loss)}</td>
    <td class="num">${r.accuracy === null || r.accuracy === undefined ? '—' : fmtNum(r.accuracy)}</td>
    <td>${r.stopped ? `<span class="tag warn">stopped: ${escapeHtml(r.error || '')}</span>`
      : (r.warnings && r.warnings.length ? '<span class="tag warn">unstable</span>'
        : '<span class="tag ok">ran</span>')}</td></tr>`).join('');

  const d = t.divergence_demo;
  const demo = d ? `<div class="chart">
    <h3>A loss that really does blow up</h3>
    ${d.batch_loss && d.batch_loss.length
      ? lineChart([{ label: 'batch loss', values: d.batch_loss, color: PALETTE[3] }],
        { log: true, xLabel: 'batch' })
      : '<p class="hint">no finite batches recorded</p>'}
    <p class="hint">${escapeHtml(d.config || '')}</p>
    <p class="note">${escapeHtml(d.note || '')}</p>
    ${d.stopped ? `<p class="hint"><span class="tag warn">GradLab stopped the run</span>
      ${escapeHtml(d.error)}</p>` : ''}
  </div>` : '';

  target.innerHTML = `
    <div class="chart"><h3>Training loss at each learning rate</h3>
      ${lineChart(series, { log: true })}</div>
    <div class="card"><h3>Measured outcome per learning rate</h3>
      <table><thead><tr><th>lr</th><th>final loss</th><th>worst loss</th>
        <th>accuracy</th><th>status</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${demo}`;
}

/* ----------------------------------------------------- custom expressions */

el('custom-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const expression = el('custom-input').value.trim();
  const raw = el('custom-values').value.trim();
  const values = {};
  for (const piece of raw.split(/[,;]/).map((s) => s.trim()).filter(Boolean)) {
    const idx = piece.indexOf('=');
    if (idx < 0) continue;
    try {
      values[piece.slice(0, idx).trim()] = JSON.parse(piece.slice(idx + 1).trim());
    } catch {
      el('custom-hint').textContent = `Could not read a number from "${piece}".`;
      el('custom-hint').className = 'hint error';
      return;
    }
  }
  const hint = el('custom-hint');
  hint.textContent = 'differentiating…';
  hint.className = 'hint';
  try {
    const res = await fetch('./api/trace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression, values }),
    });
    const payload = await res.json();
    if (!res.ok) {
      hint.textContent = payload.error || 'rejected';
      hint.className = 'hint error';
      return;
    }
    hint.textContent = `${payload.num_nodes} nodes, depth ${payload.depth}`;
    state.trace = payload;
    state.step = 0;
    state.layout = null;
    showTrace();
    for (const btn of document.querySelectorAll('#trace-list button')) {
      btn.setAttribute('aria-current', 'false');
    }
  } catch (err) {
    hint.textContent = String(err);
    hint.className = 'hint error';
  }
});

/* --------------------------------------------------------------- mode toggle */

el('technical-mode').addEventListener('change', (event) => {
  state.technical = event.target.checked;
  document.body.classList.toggle('technical', state.technical);
  document.body.classList.toggle('beginner', !state.technical);
});
document.body.classList.add('beginner');

document.addEventListener('keydown', (event) => {
  if (!['forward', 'backward'].includes(state.view)) return;
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    const steps = state.view === 'forward'
      ? (state.trace.forward_steps || []) : (state.trace.backward_steps || []);
    state.step = Math.min(steps.length - 1,
      Math.max(0, state.step + (event.key === 'ArrowRight' ? 1 : -1)));
    (state.view === 'forward' ? renderForward : renderBackward)();
    event.preventDefault();
  }
});

boot();
