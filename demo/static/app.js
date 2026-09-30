/* IDS Realizability Lab — front end. No dependencies; talks to demo/app.py. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";

const GROUPS = [
  ["Basic — packet headers (1–9)", ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent"]],
  ["Content — host state (10–22)", ["hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login"]],
  ["Time window — last 2 s (23–31)", ["count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"]],
  ["Host window — last 100 connections to victim (32–41)", ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]],
];
const FEATURES = GROUPS.flatMap(g => g[1]);
const LOCK = '<svg class="lock" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M4 7V5a4 4 0 0 1 8 0v2h1v8H3V7h1Zm2 0h4V5a2 2 0 1 0-4 0v2Z"/></svg>';

let META = null;
let CLASS_OF = {};

/* ---------------- utilities ---------------- */
async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST",
    body: body instanceof ArrayBuffer ? body : JSON.stringify(body),
    headers: { "Content-Type": body instanceof ArrayBuffer ? "application/octet-stream" : "application/json" },
  };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j && j.error) throw new Error(j.error);
  return j;
}
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 5000);
}
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
function fmtNum(v, name = "") {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (name.endsWith("_rate")) return v.toFixed(2);
  if (Math.abs(v - Math.round(v)) < 1e-6) return Math.round(v).toLocaleString();
  return v.toFixed(Math.abs(v) < 10 ? 3 : 1);
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function el(tag, attrs = {}, html = "") {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (html) e.innerHTML = html;
  return e;
}
function svg(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function segmented(root, attr, onChange) {
  $$("button", root).forEach(b => b.addEventListener("click", () => {
    $$("button", root).forEach(x => x.classList.toggle("on", x === b));
    onChange(b.dataset[attr]);
  }));
}

/* ---------------- tabs ---------------- */
$$(".tab").forEach(t => t.addEventListener("click", () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === t));
  $$(".view").forEach(v => v.classList.toggle("active", v.id === "tab-" + t.dataset.tab));
}));

/* ================= OVERVIEW ================= */
function metric(model, evalSet, key = "accuracy") {
  const r = META.metrics.find(m => m.model === model && m.eval_set === evalSet);
  return r ? parseFloat(r[key]) : NaN;
}
function evasion(model, threat, attack, constrained) {
  const r = META.evasion.find(e => e.model === model && e.threat_model === threat &&
    e.attack === attack && e.constrained === (constrained ? "True" : "False"));
  return r ? parseFloat(r.evasion_rate) : NaN;
}
function renderOverview() {
  $("#ov-mutable").textContent = META.mutable.length;
  const cards = [
    ["Accuracy on the unseen-attack test set", metric("baseline", "clean"), metric("improved_ids", "clean"), "KDDTest+ · 17 attack types never seen in training"],
    ["Attacks missed with no adversary", 1 - metric("baseline", "clean", "recall"), 1 - metric("improved_ids", "clean", "recall"), "1 − recall"],
    ["Evasion, unrestricted attacker", evasion("baseline", "static", "pgd", false), evasion("improved_ids", "static", "pgd", false), "PGD, ε = 0.2, all 41 features free"],
    ["Evasion, realistic attacker", evasion("baseline", "static", "pgd", true), evasion("improved_ids", "static", "pgd", true), "PGD, only what the wire allows"],
  ];
  $("#ov-stats").innerHTML = cards.map(([k, a, b, s]) => `
    <div class="card stat">
      <div class="k">${k}</div>
      <div class="v"><span class="from" title="Paper's IDS">${pct(a)}</span><span class="arrow">→</span><span class="to" title="Our IDS">${pct(b)}</span></div>
      <div class="s"><span style="color:var(--paper)">paper</span> → <span style="color:var(--ours)">ours</span> · ${s}</div>
    </div>`).join("");

  // grouped bar chart: evasion unrestricted vs realistic, PGD
  const groups = [
    ["Paper's IDS", "baseline", "static"],
    ["Adversarially trained", "adv_trained", "static"],
    ["Our IDS", "improved_ids", "static"],
  ];
  const W = 560, H = 250, L = 44, B = 40, T = 26, bw = 44;
  const vals = groups.flatMap(([, m, t]) => [evasion(m, t, "pgd", false), evasion(m, t, "pgd", true)]);
  const vmax = Math.max(0.1, Math.ceil(Math.max(...vals) * 10) / 10);
  const y = v => T + (H - T - B) * (1 - v / vmax);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Evasion rate chart" });
  for (let g = 0; g <= 4; g++) {
    const v = (vmax * g) / 4;
    s.append(svg("line", { x1: L, x2: W - 10, y1: y(v), y2: y(v), class: "axis", "stroke-dasharray": g ? "3 4" : "" }));
    const t = svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }); t.textContent = pct(v, 0); s.append(t);
  }
  const gw = (W - L - 20) / groups.length;
  groups.forEach(([label, m, t], i) => {
    const cx = L + gw * i + gw / 2;
    [[false, "var(--cls-c)"], [true, "var(--bad)"]].forEach(([con, col], j) => {
      const v = evasion(m, t, "pgd", con);
      const x = cx - bw - 3 + j * (bw + 6);
      s.append(svg("rect", { x, y: y(v), width: bw, height: Math.max(1, y(0) - y(v)), rx: 4, fill: col }));
      const tv = svg("text", { x: x + bw / 2, y: y(v) - 6, "text-anchor": "middle", class: "val" });
      tv.textContent = pct(v); s.append(tv);
    });
    const tl = svg("text", { x: cx, y: H - 14, "text-anchor": "middle" }); tl.textContent = label; s.append(tl);
  });
  const box = $("#ov-evasion"); box.innerHTML = "";
  box.append(s);
  box.insertAdjacentHTML("beforeend", `<div class="legend"><span><i class="sw" style="background:var(--cls-c)"></i>Unrestricted — the paper's threat model</span><span><i class="sw" style="background:var(--bad)"></i>Realistic — what an attacker can actually send</span></div>`);

  $("#ov-taxonomy").innerHTML = FEATURES.map(f => {
    const c = CLASS_OF[f].toLowerCase();
    const who = { a: "A: attacker sets it freely", b: "B: attacker influences it, within limits", c: "C: victim or third parties decide it" }[c];
    return `<div class="tile-s ${c}" title="${f} — ${who}">${f}</div>`;
  }).join("");
}

/* ================= NETWORK SIMULATION ================= */
const SIM = { rows: [], i: 0, playing: false, timer: null, scenario: "recon", thr: {}, score: null, run: 0, inflight: false };
const P = { users: [130, 110], attacker: [130, 270], sw: [360, 190], victim: [598, 190], sensor: [360, 299] };

function tween(node, pts, dur, done) {
  const segs = [];
  let total = 0;
  for (let k = 1; k < pts.length; k++) {
    const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
    segs.push([pts[k - 1], pts[k], d]); total += d;
  }
  const t0 = performance.now();
  function step(now) {
    let dist = Math.min(1, (now - t0) / dur) * total;
    for (const [a, b, d] of segs) {
      if (dist <= d || d === 0) {
        const f = d ? dist / d : 1;
        node.setAttribute("transform", `translate(${a[0] + (b[0] - a[0]) * f},${a[1] + (b[1] - a[1]) * f})`);
        break;
      }
      dist -= d;
    }
    if (now - t0 < dur) requestAnimationFrame(step); else { node.remove(); done && done(); }
  }
  requestAnimationFrame(step);
}
function packet(cls, at) {
  // Place at its start point immediately; otherwise it renders at (0,0) for a frame.
  const g = svg("g", { transform: `translate(${at[0]},${at[1]})` });
  g.append(svg("circle", { r: cls === "mirror" ? 4 : 6, class: "pkt " + cls }));
  $("#packets").append(g); return g;
}
function verdictOf(p, thr) { return p >= thr; }
function pill(alert, p) {
  return `<span class="pill ${alert ? "alert" : "allow"}">${alert ? "ALERT" : "ALLOW"} ${p.toFixed(2)}</span>`;
}
function resetScore() {
  SIM.score = { paper: { caught: 0, missed: 0, fa: 0, ok: 0 }, ours: { caught: 0, missed: 0, fa: 0, ok: 0 } };
  renderScore();
}
function renderScore() {
  const s = SIM.score;
  const row = (k, lbl, cls) => `<div>${lbl}</div><div class="n ${s.paper[k] && cls}">${s.paper[k]}</div><div class="n ${s.ours[k] && cls}">${s.ours[k]}</div>`;
  $("#sim-score").innerHTML = `<div></div><div class="h paper">Paper's IDS</div><div class="h ours">Our IDS</div>` +
    row("caught", "Attacks caught", "ok") + row("missed", "Attacks missed", "bad") +
    row("fa", "False alarms", "warn") + row("ok", "Normal allowed", "");
}
function flashSensor(pa, po, pp, poP) {
  const g = $("#sensor-flash"); g.innerHTML = "";
  [["Paper's IDS", pa, pp, 0], ["Our IDS", po, poP, 24]].forEach(([who, alert, p, dy]) => {
    const col = alert ? "var(--bad)" : "var(--ok)";
    g.append(svg("rect", { x: 100, y: -24 + dy, width: 250, height: 20, rx: 6, fill: "var(--panel)", stroke: col }));
    const t = svg("text", { x: 110, y: -10 + dy, fill: col, "font-size": 12, "font-family": "var(--mono)", "font-weight": 600 });
    t.textContent = `${who}: ${alert ? "ALERT" : "allow"}  p=${p.toFixed(2)}`;
    g.append(t);
  });
}
function simArrive(r, n) {
  const pa = verdictOf(r.p_paper, SIM.thr.paper), po = verdictOf(r.p_ours, SIM.thr.ours);
  flashSensor(pa, po, r.p_paper, r.p_ours);
  for (const [k, alert] of [["paper", pa], ["ours", po]]) {
    const s = SIM.score[k];
    if (r.attack) alert ? s.caught++ : s.missed++; else alert ? s.fa++ : s.ok++;
  }
  renderScore();
  $("#sim-progress").textContent = `${n} / ${SIM.rows.length}`;
  $("#sim-current").className = "conn-card";
  $("#sim-current").innerHTML = `
    <div class="k">Source</div><div class="v">${r.attack ? "attacker" : "user"}</div>
    <div class="k">Ground truth</div><div class="v ${r.attack ? "truth-attack" : "truth-normal"}">${esc(r.label)}</div>
    <div class="k">service / flag</div><div class="v">${esc(r.protocol)}/${esc(r.service)} · ${esc(r.flag)}</div>
    <div class="k">src_bytes → dst_bytes</div><div class="v">${r.src_bytes.toLocaleString()} → ${r.dst_bytes.toLocaleString()}</div>
    <div class="k">count (2 s window)</div><div class="v">${r.count}</div>
    <div class="k">dst_host_count</div><div class="v">${r.dst_host_count}</div>
    <div class="verdicts">
      <div class="vbox paper"><div class="who">Paper's IDS</div>${pill(pa, r.p_paper)}</div>
      <div class="vbox ours"><div class="who">Our IDS</div>${pill(po, r.p_ours)}</div>
    </div>`;
  const tag = (alert) => r.attack && !alert ? '<span class="tag miss">MISSED</span>' : !r.attack && alert ? '<span class="tag fa">FALSE ALARM</span>' : "";
  const tr = el("tr", { class: "fresh" }, `
    <td>${n}</td><td>${r.attack ? "attacker" : "user"}</td>
    <td class="${r.attack ? "truth-attack" : "truth-normal"}">${esc(r.label)}</td>
    <td>${esc(r.service)}</td><td>${esc(r.flag)}</td><td>${r.src_bytes.toLocaleString()}</td>
    <td>${pill(pa, r.p_paper)}${tag(pa)}</td><td>${pill(po, r.p_ours)}${tag(po)}</td>`);
  $("#sim-log tbody").prepend(tr);
}
function simStep() {
  if (!SIM.playing) return;
  if (SIM.i >= SIM.rows.length) { simStop(); return; }
  const r = SIM.rows[SIM.i++], n = SIM.i, run = SIM.run;
  SIM.inflight = true;
  const speed = +$("#sim-speed").value, dur = 1500 / speed;
  const src = r.attack ? P.attacker : P.users, kind = r.attack ? "attack" : "normal";
  tween(packet(kind, src), [src, P.sw], dur * 0.5, () => {
    if (run !== SIM.run) return;                       // reset happened mid-flight
    tween(packet(kind, P.sw), [P.sw, P.victim], dur * 0.5);
    tween(packet("mirror", P.sw), [P.sw, P.sensor], dur * 0.35, () => {
      if (run !== SIM.run) return;
      SIM.inflight = false;
      simArrive(r, n);
      // Chain the next connection off this one's arrival, not a free-running
      // timer: if the tab is throttled, launches can't race ahead of verdicts.
      if (SIM.playing) SIM.timer = setTimeout(simStep, Math.max(60, 500 / speed));
    });
  });
}
async function simPlay() {
  if (SIM.playing) { simStop(); return; }
  if (SIM.i === 0 || SIM.i >= SIM.rows.length) {
    try {
      const d = await api(`/api/simulate?scenario=${SIM.scenario}&n=36&seed=${Math.floor(Math.random() * 1e6)}`);
      SIM.rows = d.rows; SIM.thr = { paper: d.thr_paper, ours: d.thr_ours }; SIM.i = 0;
      $("#sim-log tbody").innerHTML = ""; resetScore();
    } catch (e) { toast(e.message); return; }
  }
  SIM.playing = true; $("#sim-play").textContent = "Pause";
  if (!SIM.inflight) simStep();   // an in-flight packet resumes the chain itself
}
function simStop() { SIM.playing = false; clearTimeout(SIM.timer); $("#sim-play").textContent = SIM.i >= SIM.rows.length && SIM.rows.length ? "Replay" : "Play"; }
function simReset() {
  simStop(); SIM.run++; SIM.inflight = false; SIM.rows = []; SIM.i = 0; $("#packets").innerHTML = "";
  $("#sim-log tbody").innerHTML = ""; $("#sensor-flash").innerHTML = "";
  $("#sim-current").className = "conn-card empty"; $("#sim-current").textContent = "Press Play";
  $("#sim-progress").textContent = "—"; $("#sim-play").textContent = "Play"; resetScore();
}
segmented($("#sim-scenario"), "s", s => { SIM.scenario = s; simReset(); });
$("#sim-play").addEventListener("click", simPlay);
$("#sim-reset").addEventListener("click", simReset);

/* ================= ATTACK LAB ================= */
const LAB = { victim: "paper", method: "pgd", eps: 0.2, realistic: false, samples: [], history: [] };

function fillSamples() {
  const fams = { dos: "Denial of service", recon: "Reconnaissance", r2l: "Remote-to-local", u2r: "Privilege escalation", other: "Other" };
  const sel = $("#lab-sample"); sel.innerHTML = "";
  for (const [k, name] of Object.entries(fams)) {
    const items = LAB.samples.filter(s => s.family === k);
    if (!items.length) continue;
    const og = el("optgroup", { label: name });
    items.forEach(s => og.append(el("option", { value: s.id },
      `${s.evades_paper ? "★ " : ""}${s.label} · ${s.protocol}/${s.service} · ${s.flag} · ${s.src_bytes.toLocaleString()} B`)));
    sel.append(og);
  }
}
function renderFeatures(res) {
  const byName = Object.fromEntries(res.features.map(f => [f.name, f]));
  $("#lab-features").innerHTML = GROUPS.map(([title, names]) => `
    <div class="fgroup"><h3>${title}</h3><div class="fgrid">${names.map(n => {
      const f = byName[n], c = f.cls.toLowerCase();
      const bad = f.violations.length > 0, moved = f.changed && !bad;
      // In realistic mode lock everything the attack may not move: all of Class C
      // plus `flag` (Class B, but categorical -- no continuous step exists).
      const lock = res.mode === "realistic" && !META.mutable.includes(n) ? LOCK : "";
      const val = f.changed
        ? `<span class="old">${esc(fmtNum(f.before, n))}</span>${esc(fmtNum(f.after, n))}`
        : esc(fmtNum(f.before, n));
      return `<div class="ftile ${c} ${moved ? "moved pulse" : ""} ${bad ? "bad pulse" : ""}" title="${n} · class ${f.cls}">
        ${lock}<div class="fn">${n}</div><div class="fv">${val}</div>
        ${bad ? `<div class="why">${esc(f.violations.join(" · "))}</div>` : ""}</div>`;
    }).join("")}</div></div>`).join("");
}
function renderVerdict(res) {
  const thrL = `threshold ${res.threshold.toFixed(2)}`;
  const gauge = (lbl, p) => `
    <div class="gauge"><div class="lbl"><span>${lbl}</span><b>P(attack) = ${p.toFixed(3)}</b></div>
      <div class="bar"><div class="fill ${p >= res.threshold ? "hi" : "lo"}" style="width:${(100 * p).toFixed(1)}%"></div>
      <div class="thr" data-l="${thrL}" style="left:${(100 * res.threshold).toFixed(1)}%"></div></div></div>`;
  const evaded = res.detected_before && !res.detected_after;
  const verdict = !res.detected_before ? ["Not detected even before the attack", "evaded"]
    : evaded ? ["EVADED — the attack slipped through", "evaded"] : ["DETECTED — the IDS held", "detected"];
  const changed = res.features.filter(f => f.changed);
  const bad = changed.filter(f => f.violations.length);
  let explain;
  if (res.mode === "unrestricted") {
    const ex = bad.slice(0, 3).map(f => `<b>${f.name}</b> (${f.violations[0]})`).join(", ");
    explain = `<div class="explain unreal">This attack changed <b>${res.features_changed}</b> features, and <b>${res.impossible_values}</b> of them take values no packet sequence can produce${ex ? `: ${ex}${bad.length > 3 ? ", …" : ""}` : ""}.
      ${evaded ? "It fools the model <b>on paper</b>, but an attacker cannot send it." : "Even this impossible attack did not get through."}</div>`;
  } else {
    const ex = changed.slice(0, 4).map(f => `<b>${f.name}</b> ${esc(fmtNum(f.before, f.name))} → ${esc(fmtNum(f.after, f.name))}`).join(", ");
    explain = `<div class="explain real">Every change is something a real attacker can do on the wire${ex ? `: ${ex}` : " (no feature needed to move)"}.
      ${evaded ? "<b>This is a genuine threat</b>, and these are the features worth hardening." : "<b>The IDS holds against a realizable attacker.</b>"}</div>`;
  }
  $("#lab-verdict").innerHTML = `
    <div class="v-head"><div class="v-big ${verdict[1]}">${verdict[0]}</div>
      <div class="v-sub">${esc(res.victim)} · ${res.method.toUpperCase()} ε=${res.eps.toFixed(2)} · <b>${res.mode}</b> · target: <span class="truth-attack">${esc(res.label)}</span></div></div>
    <div class="gauges">${gauge("Before the attack", res.p_before)}${gauge("After the attack", res.p_after)}</div>
    <div class="kpis">
      <div class="kpi"><div class="n">${res.dims_moved}<span class="muted" style="font-size:14px">/${res.dims_total}</span></div><div class="l">input dimensions moved</div></div>
      <div class="kpi"><div class="n">${res.features_changed}<span class="muted" style="font-size:14px">/41</span></div><div class="l">features changed</div></div>
      <div class="kpi"><div class="n ${res.impossible_values ? "bad" : "ok"}">${res.impossible_values}</div><div class="l">physically impossible values</div></div>
    </div>${explain}`;
}
function pushHistory(res) {
  const evaded = res.detected_before && !res.detected_after;
  LAB.history.unshift(`<div class="hrow"><div>${res.victim.startsWith("Paper") ? "Paper" : "Ours"} · ${res.method.toUpperCase()} · ${res.mode}</div>
    <div class="r ${evaded ? "evaded" : "detected"}">${evaded ? "EVADED" : "HELD"}</div>
    <div class="t">${esc(res.label)} · ${res.impossible_values} impossible</div><div class="t mono">${res.p_before.toFixed(2)}→${res.p_after.toFixed(2)}</div></div>`);
  LAB.history = LAB.history.slice(0, 6);
  $("#lab-history").innerHTML = LAB.history.join("");
}
async function labRun() {
  const btn = $("#lab-run"); btn.disabled = true; btn.textContent = "Attacking…";
  try {
    const res = await api("/api/attack", { id: +$("#lab-sample").value, victim: LAB.victim, method: LAB.method, eps: LAB.eps, realistic: LAB.realistic });
    renderVerdict(res); renderFeatures(res); pushHistory(res);
  } catch (e) { toast(e.message); }
  btn.disabled = false; btn.textContent = "Launch attack";
}
segmented($("#lab-victim"), "v", v => (LAB.victim = v));
segmented($("#lab-method"), "m", m => (LAB.method = m));
segmented($("#lab-mode"), "r", r => (LAB.realistic = r === "1"));
$("#lab-eps").addEventListener("input", e => { LAB.eps = +e.target.value; $("#lab-eps-v").textContent = LAB.eps.toFixed(2); });
$("#lab-run").addEventListener("click", labRun);
$("#lab-dramatic").addEventListener("click", () => {
  const pool = LAB.samples.filter(s => s.evades_paper);
  if (!pool.length) return;
  $("#lab-sample").value = pool[Math.floor(Math.random() * pool.length)].id;
  $$("#lab-victim button").forEach(b => b.classList.toggle("on", b.dataset.v === "paper")); LAB.victim = "paper";
  $$("#lab-mode button").forEach(b => b.classList.toggle("on", b.dataset.r === "0")); LAB.realistic = false;
  labRun();
});

/* ================= LIVE CAPTURE ================= */
async function scorePcap(file) {
  $("#cap-summary").innerHTML = `<p class="muted">Extracting features from ${esc(file.name)}…</p>`;
  try {
    const d = await api("/api/pcap", await file.arrayBuffer());
    if (!d.rows.length) { $("#cap-summary").innerHTML = `<p class="muted">${esc(d.note || "No connections found.")}</p>`; return; }
    const ap = d.rows.filter(r => r.p_paper >= d.thr_paper).length, ao = d.rows.filter(r => r.p_ours >= d.thr_ours).length;
    $("#cap-summary").innerHTML = `<div class="kpis">
      <div class="kpi"><div class="n">${d.rows.length}</div><div class="l">connections</div></div>
      <div class="kpi"><div class="n" style="color:var(--paper)">${ap}</div><div class="l">flagged by paper's IDS</div></div>
      <div class="kpi"><div class="n" style="color:var(--ours)">${ao}</div><div class="l">flagged by our IDS</div></div></div>`;
    $("#cap-file").textContent = d.file;
    $("#cap-table tbody").innerHTML = d.rows.map((r, i) => `<tr>
      <td>${i + 1}</td><td>${esc(r.src)}</td><td>${esc(r.service)}</td><td>${esc(r.flag)}</td>
      <td>${r.src_bytes.toLocaleString()}</td><td>${r.duration}</td><td>${r.count}</td><td>${r.dst_host_count}</td>
      <td>${pill(r.p_paper >= d.thr_paper, r.p_paper)}</td><td>${pill(r.p_ours >= d.thr_ours, r.p_ours)}</td></tr>`).join("");
    $("#cap-table-card").hidden = false;
  } catch (e) { toast(e.message); $("#cap-summary").innerHTML = ""; }
}
$("#pcap-file").addEventListener("change", e => e.target.files[0] && scorePcap(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => e.dataTransfer.files[0] && scorePcap(e.dataTransfer.files[0]));

/* ================= boot ================= */
(async function boot() {
  try {
    META = await api("/api/meta");
    for (const [k, list] of Object.entries(META.classes)) list.forEach(f => (CLASS_OF[f] = k));
    renderOverview();
    LAB.samples = await api("/api/samples");
    fillSamples();
    resetScore();
  } catch (e) { toast("Could not reach the demo server: " + e.message); }
})();
