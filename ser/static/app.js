/* Local sound-event UI. Every widget is populated from kernel JSON responses;
   no label is ever derived from a file or folder name. */

const state = {
  recordings: [],
  current: null,
  waveform: null,
  inference: null,
  frameProbs: null,
  cueEntries: [],
  annotations: [],
  legacyAnnotations: [],
  cuts: [],
  selectedUid: null,
  selectedWindow: null,
  audioEl: null,
};

const CLASS_COLORS = {
  knock: "#e07a5f",
  bell: "#81b29a",
  chime: "#f2cc8f",
  machine: "#5c7cfa",
  unknown: "#9d9d9d",
  background: "#cccccc",
};

const AUDIT_LABELS = {
  audited: "试听审定",
  pending_review: "待审定",
  unclaimed: "待认领（分轨移走）",
};

async function api(path, options) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status}: ${text}`);
  }
  return resp.json();
}

function postJson(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

function fmt(t) {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
}

async function init() {
  await refreshHealth();
  await refreshRecordings();
  await refreshFeedback();
  if (window.PRESELECTED_RECORDING) {
    await openRecording(window.PRESELECTED_RECORDING);
  }
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const el = document.getElementById("health");
    el.textContent = h.ready
      ? "本地链路就绪 · 无远端识别调用"
      : "数据或模型缺失，请先运行 prepare_data / train_model";
    el.className = h.ready ? "health ok" : "health warn";
  } catch (err) {
    document.getElementById("health").textContent = "服务未就绪";
  }
}

async function refreshRecordings() {
  try {
    state.recordings = await api("/api/recordings");
  } catch (err) {
    document.getElementById("recording-list").innerHTML =
      `<li class="error">${err.message}</li>`;
    return;
  }
  const ul = document.getElementById("recording-list");
  ul.innerHTML = "";
  for (const rec of state.recordings) {
    const li = document.createElement("li");
    li.className = "rec-item" +
      (state.current === rec.recording_id ? " active" : "");
    li.innerHTML = `
      <div class="rec-id" title="${rec.recording_id}">${rec.recording_id}</div>
      <div class="rec-meta">${rec.split} · ${rec.condition} · ${rec.duration.toFixed(1)}s
        · 目标 ${rec.n_events} · 未知 ${rec.n_unknown}</div>`;
    li.onclick = () => openRecording(rec.recording_id);
    ul.appendChild(li);
  }
}

async function refreshFeedback() {
  try {
    const data = await api("/api/feedback");
    const sum = document.getElementById("feedback-summary");
    sum.textContent = `共 ${data.summary.count} 条 · ${JSON.stringify(data.summary.by_reason)}`;
    const ul = document.getElementById("feedback-list");
    ul.innerHTML = "";
    for (const row of data.records.slice(-8).reverse()) {
      const li = document.createElement("li");
      li.textContent = `${row.recording_id.slice(0, 8)}… ${row.reason}` +
        (row.expected_class ? ` → ${row.expected_class}` : "");
      ul.appendChild(li);
    }
  } catch (err) {
    /* feedback file may not exist yet */
  }
}

async function openRecording(recordingId) {
  state.current = recordingId;
  state.selectedUid = null;
  state.selectedWindow = null;
  await refreshRecordings();
  const content = document.getElementById("content");
  content.innerHTML = `
    <div class="rec-header">
      <h2>${recordingId}</h2>
      <div class="controls">
        <button id="btn-infer">重新运行内核推理</button>
        <button id="btn-verify">用分轨核对提示单</button>
      </div>
    </div>
    <div class="player-row">
      <audio id="audio" controls preload="auto"
        src="/api/recordings/${recordingId}/audio/mix"></audio>
    </div>
    <div class="canvas-wrap">
      <canvas id="wave" width="1100" height="140"></canvas>
      <canvas id="events" width="1100" height="170"></canvas>
      <div id="time-label" class="time-label">0:00.00</div>
    </div>
    <div class="panels">
      <section class="panel">
        <h3>事件与配音提示单（可编辑）</h3>
        <div class="toolbar">
          <label>裁剪 [s] <input id="cut-start" type="number" step="0.05" style="width:70px"></label>
          <label>到 <input id="cut-end" type="number" step="0.05" style="width:70px"></label>
          <button id="btn-cut">剪掉中间段（合并标记）</button>
          <button id="btn-restore">补回最近裁剪（拆回标记）</button>
        </div>
        <table class="cue-table">
          <thead><tr>
            <th>选择</th><th>事件ID</th><th>类别</th><th>开始</th><th>结束</th>
            <th>置信度</th><th>可能次数</th><th>分轨身份</th><th>状态</th><th>操作</th>
          </tr></thead>
          <tbody id="cue-body"></tbody>
        </table>
      </section>
      <section class="panel">
        <h3>证据：模型真正分析的输入</h3>
        <div id="evidence" class="evidence">
          <p class="hint">在上方时间轴或表格选择事件 / 分析窗，查看该 1 秒 mel 特征窗口、各类概率与阈值。</p>
        </div>
      </section>
    </div>
    <section class="panel">
      <h3>标注资料与试听审定（决定样本是否进入训练）</h3>
      <div class="toolbar">
        <button id="btn-annotations">刷新审定状态</button>
        <span class="hint">仅“试听审定”且分轨在位的区间进入单类训练；待审定/待认领区间被排除。</span>
      </div>
      <table class="cue-table">
        <thead><tr>
          <th>事件ID</th><th>标签</th><th>发生位置</th><th>时长</th>
          <th>审定状态</th><th>身份</th><th>试听操作</th>
        </tr></thead>
        <tbody id="annotation-body"></tbody>
      </table>
      <div id="legacy-box" class="verify-box muted">旧标签映射加载中…</div>
    </section>
    <section class="panel">
      <h3>帧级概率与分析窗网格</h3>
      <canvas id="probs" width="1100" height="160"></canvas>
      <div class="legend" id="prob-legend"></div>
    </section>
    <section class="panel">
      <h3>核对与失败反馈</h3>
      <div id="verify-box" class="verify-box muted">尚未核对。</div>
    </section>`;

  document.getElementById("btn-infer").onclick = async () => {
    state.inference = await postJson(
      `/api/recordings/${recordingId}/inference?force=1`);
    state.cueEntries = state.inference.cue_entries;
    state.annotations = state.inference.annotations || [];
    state.legacyAnnotations = state.inference.legacy_annotations || [];
    await loadFrameProbs();
    renderAll();
  };
  document.getElementById("btn-verify").onclick = runVerify;
  document.getElementById("btn-cut").onclick = runCut;
  document.getElementById("btn-restore").onclick = runRestore;
  document.getElementById("btn-annotations").onclick = refreshAnnotations;

  state.audioEl = document.getElementById("audio");
  state.waveform = await api(
    `/api/recordings/${recordingId}/waveform`);
  state.inference = await api(
    `/api/recordings/${recordingId}/inference`);
  state.cueEntries = state.inference.cue_entries;
  state.annotations = state.inference.annotations || [];
  state.legacyAnnotations = state.inference.legacy_annotations || [];
  const cue = await api(`/api/cuesheet/${recordingId}`);
  state.cuts = cue.cuts;
  await loadFrameProbs();
  renderAll();
}

async function loadFrameProbs() {
  state.frameProbs = await api(
    `/api/recordings/${state.current}/frame-probs`);
}

function xForTime(t) {
  const width = 1100;
  return (t / state.waveform.duration) * width;
}

function renderAll() {
  drawWaveform();
  drawEvents();
  drawProbs();
  renderCueTable();
  renderAnnotations();
  renderLegend();
  bindCanvasClicks();
}

function drawWaveform() {
  const cv = document.getElementById("wave");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const peaks = state.waveform.peaks;
  ctx.strokeStyle = "#495057";
  ctx.beginPath();
  for (let i = 0; i < peaks.length; i++) {
    const x = (i / peaks.length) * cv.width;
    const h = peaks[i] * (cv.height - 10);
    ctx.moveTo(x, cv.height / 2 - h / 2);
    ctx.lineTo(x, cv.height / 2 + h / 2);
  }
  ctx.stroke();
  // analysis window grid
  ctx.strokeStyle = "rgba(0,0,0,0.18)";
  for (const t of state.inference.window_times) {
    const x = xForTime(t);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cv.height);
    ctx.stroke();
  }
}

function drawEvents() {
  const cv = document.getElementById("events");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const rows = state.cueEntries;
  rows.forEach((row, i) => {
    const x0 = xForTime(row.start);
    const x1 = xForTime(row.end);
    const y = 12 + (i % 6) * 26;
    ctx.fillStyle = CLASS_COLORS[row.cls] || "#888";
    if (state.selectedUid === row.event_uid) {
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 2;
    } else {
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 1;
    }
    ctx.beginPath();
    ctx.roundRect(x0, y, Math.max(3, x1 - x0), 20, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#111";
    ctx.font = "11px sans-serif";
    ctx.fillText(
      `${row.cls}${row.possible_count > 1 ? " ×" + row.possible_count : ""}`,
      x0 + 4, y + 14);
  });
}

function drawProbs() {
  const cv = document.getElementById("probs");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const fp = state.frameProbs.frame_probs;
  const th = state.frameProbs.thresholds;
  const classes = window.TARGET_CLASSES;
  const n = fp.length;
  classes.forEach((cls, c) => {
    ctx.strokeStyle = CLASS_COLORS[cls];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / n) * cv.width;
      const y = cv.height - fp[i][c] * (cv.height - 20);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // threshold line
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = CLASS_COLORS[cls];
    const ty = cv.height - th[c] * (cv.height - 20);
    ctx.beginPath();
    ctx.moveTo(0, ty);
    ctx.lineTo(cv.width, ty);
    ctx.stroke();
    ctx.setLineDash([]);
  });
  if (state.selectedWindow !== null) {
    const t0 = state.inference.window_times[state.selectedWindow];
    const t1 = t0 + 1.0;
    ctx.fillStyle = "rgba(0,0,0,0.08)";
    ctx.fillRect(xForTime(t0), 0, xForTime(t1) - xForTime(t0), cv.height);
  }
}

function renderLegend() {
  const el = document.getElementById("prob-legend");
  el.innerHTML = window.TARGET_CLASSES.map(
    c => `<span class="chip" style="background:${CLASS_COLORS[c]}">${c}</span>`
  ).join("") + `<span class="hint">虚线为该类校准阈值；阴影为当前证据窗</span>`;
}

function renderAnnotations() {
  const body = document.getElementById("annotation-body");
  if (!body) return;
  body.innerHTML = "";
  const rows = state.annotations.filter(a =>
    window.TARGET_CLASSES.includes(a.cls));
  for (const row of rows) {
    const tr = document.createElement("tr");
    const status = row.audit_status || "pending_review";
    if (status !== "audited") tr.className = "annotation-held";
    const identity = row.identity_uncertain
      ? "身份不确定"
      : (row.source_event_id || row.source || "—");
    tr.innerHTML = `
      <td>${row.event_id.slice(0, 10)}</td>
      <td><span class="chip" style="background:${CLASS_COLORS[row.cls]}">${row.cls}</span>
        ${row.ring_count > 1 ? "×" + row.ring_count : ""}</td>
      <td>${fmt(row.start)} – ${fmt(row.end)}</td>
      <td>${(row.end - row.start).toFixed(2)}s</td>
      <td>${AUDIT_LABELS[status] || status}</td>
      <td>${identity}</td>
      <td class="row-actions">
        <button class="ann-play">试听</button>
        <button class="ann-approve" ${status === "audited" ? "disabled" : ""}>审定通过</button>
        <button class="ann-hold" ${status === "pending_review" ? "disabled" : ""}>置为待审定</button>
      </td>`;
    tr.querySelector(".ann-play").onclick = () => {
      state.audioEl.currentTime = row.start;
      state.audioEl.play();
      setTimeout(() => state.audioEl.pause(),
        (row.end - row.start) * 1000 + 100);
    };
    tr.querySelector(".ann-approve").onclick = () =>
      submitAudit(row.event_id, "audited");
    tr.querySelector(".ann-hold").onclick = () =>
      submitAudit(row.event_id, "pending_review",
        "operator returned interval to listening review");
    body.appendChild(tr);
  }
  const legacyBox = document.getElementById("legacy-box");
  if (legacyBox) {
    if (!state.legacyAnnotations.length) {
      legacyBox.textContent = "该录音无旧标签迁移记录。";
    } else {
      legacyBox.classList.remove("muted");
      legacyBox.innerHTML = state.legacyAnnotations.map(g => `
        <div>旧标签 <strong>${g.legacy_class}</strong>
          （${g.scope === "whole_segment" ? "整段" : "单次事件"}）
          ${fmt(g.start)} – ${fmt(g.end)} →
          候选新类：${g.candidate_classes.join(" / ")}
          ${g.note ? "｜" + g.note : ""}</div>`).join("");
    }
  }
}

async function submitAudit(eventId, status, note) {
  await postJson(`/api/recordings/${state.current}/audit`, {
    event_id: eventId, status, note: note || null,
  });
  await refreshAnnotations();
}

async function refreshAnnotations() {
  const inf = await api(`/api/recordings/${state.current}/inference`);
  state.annotations = inf.annotations || [];
  state.legacyAnnotations = inf.legacy_annotations || [];
  renderAnnotations();
}

function renderCueTable() {
  const body = document.getElementById("cue-body");
  body.innerHTML = "";
  state.cueEntries.forEach((row) => {
    const tr = document.createElement("tr");
    if (state.selectedUid === row.event_uid) tr.className = "selected";
    const identity = row.source_event_id
      ? `${row.source_match}:${row.source_event_id.slice(0, 6)}`
      : (row.source_match === "count_only" ? "仅事件数（混音歧义）" : "未匹配");
    tr.innerHTML = `
      <td><input type="radio" name="cue-select"
        ${state.selectedUid === row.event_uid ? "checked" : ""}></td>
      <td>${row.event_uid}</td>
      <td><select class="relabel">
        ${window.TARGET_CLASSES.map(c =>
          `<option ${c === row.cls ? "selected" : ""}>${c}</option>`).join("")}
        <option ${row.cls === "unknown" ? "selected" : ""}>unknown</option>
      </select></td>
      <td>${fmt(row.start)}</td>
      <td>${fmt(row.end)}</td>
      <td>${row.confidence ?? ""}</td>
      <td>${row.possible_count ?? 1}</td>
      <td>${identity}</td>
      <td>${row.cue_status || ""}</td>
      <td class="row-actions">
        <button class="act-play">播放</button>
        <button class="act-split">拆分</button>
        <button class="act-fail">报失败</button>
        <button class="act-delete">删除</button>
      </td>`;
    tr.querySelector("input").onchange = () => selectEvent(row.event_uid);
    tr.querySelector(".relabel").onchange = async (e) => {
      await postJson(`/api/cuesheet/${state.current}/relabel`, {
        uid: row.event_uid, new_class: e.target.value,
      });
      await reloadCue();
    };
    tr.querySelector(".act-play").onclick = () => {
      state.audioEl.currentTime = row.start;
      state.audioEl.play();
      setTimeout(() => state.audioEl.pause(),
        (row.end - row.start) * 1000 + 100);
    };
    tr.querySelector(".act-split").onclick = async () => {
      const at = Number(prompt("拆分时间点（秒）",
        ((row.start + row.end) / 2).toFixed(2)));
      if (!Number.isNaN(at)) {
        await postJson(`/api/cuesheet/${state.current}/split`,
          { uid: row.event_uid, at });
        await reloadCue();
      }
    };
    tr.querySelector(".act-fail").onclick = () => reportFailure(row);
    tr.querySelector(".act-delete").onclick = async () => {
      await postJson(`/api/cuesheet/${state.current}/delete`,
        { uid: row.event_uid });
      await reloadCue();
    };
    body.appendChild(tr);
  });
}

async function reloadCue() {
  const cue = await api(`/api/cuesheet/${state.current}`);
  state.cueEntries = cue.entries;
  state.cuts = cue.cuts;
  renderEvents();
  renderCueTable();
}

function bindCanvasClicks() {
  const cv = document.getElementById("events");
  cv.onclick = (e) => {
    const rect = cv.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (cv.width / rect.width);
    const t = (x / cv.width) * state.waveform.duration;
    const hit = state.cueEntries.find(
      row => t >= row.start && t <= row.end);
    if (hit) selectEvent(hit.event_uid);
  };
  const pcv = document.getElementById("probs");
  pcv.onclick = (e) => {
    const rect = pcv.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (pcv.width / rect.width);
    const t = (x / pcv.width) * state.waveform.duration;
    let best = 0;
    let bestDist = Infinity;
    state.inference.window_times.forEach((wt, i) => {
      const center = wt + 0.5;
      if (Math.abs(center - t) < bestDist) {
        bestDist = Math.abs(center - t);
        best = i;
      }
    });
    selectWindow(best);
  };
}

async function selectEvent(uid) {
  state.selectedUid = uid;
  const row = state.cueEntries.find(r => r.event_uid === uid);
  if (row) {
    let best = 0;
    let bestDist = Infinity;
    state.inference.window_times.forEach((wt, i) => {
      const d = Math.abs(wt + 0.5 - (row.start + row.end) / 2);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    await selectWindow(best);
    state.audioEl.currentTime = row.start;
  }
  renderEvents();
  renderCueTable();
}

async function selectWindow(index) {
  state.selectedWindow = index;
  drawProbs();
  const data = await api(
    `/api/recordings/${state.current}/evidence/${index}`);
  renderEvidence(data);
}

function renderEvidence(data) {
  const el = document.getElementById("evidence");
  const probRows = Object.entries(data.class_probabilities).map(
    ([cls, p]) => {
      const on = data.decisions[cls];
      return `<tr><td><span class="chip" style="background:${CLASS_COLORS[cls]}">${cls}</span></td>
        <td>${p.toFixed(3)}</td><td>${data.thresholds[cls].toFixed(3)}</td>
        <td>${on ? "超过阈值（保留）" : "未超过（拒绝）"}</td></tr>`;
    }).join("");
  const gt = data.ground_truth_in_window.map(
    g => `${g.class} [${g.start.toFixed(2)}–${g.end.toFixed(2)}]`
  ).join("，") || "仅背景 / 未知（默认不归属任何目标类）";
  el.innerHTML = `
    <div class="ev-meta">
      分析窗 #${data.window_index}：${fmt(data.window_start)} – ${fmt(data.window_end)}
      （${data.n_mels} 个 mel 频带 × 50 帧，采样率 ${data.sample_rate} Hz）
      <button id="btn-play-window">播放该 1 秒输入</button>
    </div>
    <canvas id="mel" width="520" height="260"></canvas>
    <table class="prob-table">
      <thead><tr><th>类别</th><th>概率</th><th>阈值</th><th>裁决</th></tr></thead>
      <tbody>${probRows}</tbody>
    </table>
    <div class="gt-box"><strong>该窗标注：</strong>${gt}</div>
    <div class="gt-box"><strong>关联事件：</strong>${
      data.predicted_events.map(e =>
        `${e.cls}[${e.start.toFixed(2)}–${e.end.toFixed(2)}]`).join("，") || "无"}
    </div>`;
  drawMel(data.mel_patch);
  document.getElementById("btn-play-window").onclick = () => {
    state.audioEl.currentTime = data.window_start;
    state.audioEl.play();
    setTimeout(() => state.audioEl.pause(), 1050);
  };
}

function drawMel(matrix) {
  const cv = document.getElementById("mel");
  const ctx = cv.getContext("2d");
  const rows = matrix.length;
  const cols = matrix[0].length;
  let lo = Infinity, hi = -Infinity;
  for (const row of matrix) for (const v of row) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const cw = cv.width / cols;
  const ch = cv.height / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const norm = (matrix[r][c] - lo) / Math.max(1e-6, hi - lo);
      const shade = Math.floor(255 * (1 - norm));
      ctx.fillStyle = `rgb(${shade},${Math.floor(shade * 0.8)},${Math.floor(255 * (1 - norm) * 0.7 + shade * 0.3)})`;
      // flip y so low frequencies are at the bottom
      ctx.fillRect(c * cw, cv.height - (r + 1) * ch, cw + 0.5, ch + 0.5);
    }
  }
}

async function reportFailure(row) {
  const reason = prompt(
    "失败类型（wrong_class / boundary / false_alarm / unknown_not_rejected）",
    "wrong_class");
  if (!reason) return;
  const expected = prompt("期望类别（可留空，unknown 表示应拒绝）", "") || null;
  await postJson(`/api/cuesheet/${state.current}/failure`, {
    uid: row.event_uid, reason, expected_class: expected,
    window_indices: row.windows || [],
  });
  await reloadCue();
  await refreshFeedback();
  await runVerify(true);
}

async function runVerify(silent) {
  const report = await api(`/api/cuesheet/${state.current}/verify`);
  const box = document.getElementById("verify-box");
  box.classList.remove("muted");
  box.innerHTML = `
    <div>身份与边界全部一致：<strong>${report.all_identities_exact ? "是" : "否"}</strong></div>
    <div>匹配 ${report.matched.length} · 未匹配提示 ${report.unmatched_cues.length}
      · 缺失声源 ${report.missing_sources.length}</div>
    <pre>${JSON.stringify(report.matched, null, 2)}</pre>`;
}

async function runCut() {
  const start = Number(document.getElementById("cut-start").value);
  const end = Number(document.getElementById("cut-end").value);
  if (Number.isNaN(start) || Number.isNaN(end) || end <= start) {
    alert("请输入合法的起止秒数");
    return;
  }
  await postJson(`/api/cuesheet/${state.current}/cut`, { start, end });
  await reloadCue();
}

async function runRestore() {
  if (!state.cuts.length) {
    alert("没有可补回的裁剪");
    return;
  }
  await postJson(`/api/cuesheet/${state.current}/restore`, {});
  await reloadCue();
}

init().catch(err => {
  document.getElementById("content").innerHTML =
    `<div class="empty-state error">${err.message}</div>`;
});
