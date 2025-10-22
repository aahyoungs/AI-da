const RISK_COLORS = { low: "#FFC107", middle: "#FF6F3C", high: "#B22222", none: "#E3E7F1" };
const qs = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const fmtDate = d => d.toISOString().slice(0, 10);

function currentRange() {
  const s = qs("#startDate")?.value || "";
  const e = qs("#endDate")?.value || "";
  const quick = qs("#quickRange")?.value || "all";
  const mode = (quick === "all") ? "all" : "range";
  return { start: s, end: e, mode };
}

function buildQuery() {
  const { start, end, mode } = currentRange();
  if (mode === "all") return "?mode=all";
  if (start && end) return `?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&mode=range`;
  return "?mode=range"; ƒ
}

function buildRangeParams() {
  const { start, end, mode } = currentRange();
  const p = new URLSearchParams();
  if (mode === "all") return "";
  if (start) p.set("start", start);
  if (end) p.set("end", end);
  return p.toString();
}

// ==============================
// 상단바
// ==============================
(function initTopbar() {
  const quick = qs("#quickRange");
  const s = qs("#startDate"), e = qs("#endDate");
  const applyBtn = qs("#applyRange");
  const helpIcon = qs("#helpIcon"), helpTooltip = qs("#helpTooltip");

  quick.addEventListener("change", () => {
    const today = new Date();
    const end = fmtDate(today);
    let start = end;
    const DAY = 24 * 60 * 60 * 1000;

    if (quick.value === "all") {
      s.value = ""; e.value = "";
      s.disabled = true; e.disabled = true;
    } else {
      s.disabled = false; e.disabled = false;
      if (quick.value === "24h") start = fmtDate(new Date(today.getTime() - DAY));
      else if (quick.value === "7d") start = fmtDate(new Date(today.getTime() - 7 * DAY));
      else if (quick.value === "30d") start = fmtDate(new Date(today.getTime() - 30 * DAY));
      if (quick.value !== "custom") { s.value = start; e.value = end; }
    }
  });


  // tooltip
  let tipVisible = false;
  helpIcon?.addEventListener("mouseenter", () => { tipVisible = true; helpTooltip.classList.remove("invisible", "opacity-0"); });
  helpIcon?.addEventListener("mouseleave", () => { tipVisible = false; helpTooltip.classList.add("invisible", "opacity-0"); });
  document.addEventListener("mousemove", ev => {
    if (!tipVisible) return;
    const off = 10;
    const tw = helpTooltip.offsetWidth || 150;
    const th = helpTooltip.offsetHeight || 60;
    let left = ev.pageX + off, top = ev.pageY + off;
    if (left + tw > window.innerWidth) left = ev.pageX - tw - off;
    if (top + th > window.innerHeight) top = ev.pageY - th - off;
    helpTooltip.style.left = `${left}px`; helpTooltip.style.top = `${top}px`;
  });

  applyBtn.addEventListener("click", async () => {
    await refreshAll();
  });

  window.updateTopbarMetrics = async function () {
    const q = buildQuery();
    const res = await fetch(`/api/metrics${q}`);
    if (!res.ok) return;
    const m = await res.json();
    const rb = qs("#riskBadge"), rs = qs("#riskScore"), rl = qs("#riskLevel"), ec = qs("#eventCount");
    rs.textContent = String(m.score ?? 0);
    rl.textContent = m.level || "none";
    rb.style.backgroundColor = RISK_COLORS[m.level] || RISK_COLORS.none;
    ec.textContent = `${m.count ?? 0}건`;
  };
})();

// ==============================
// Tabs (heatmap / flow)
// ==============================
(function initTabs() {
  const btns = qsa(".tab-btn");
  const panels = qsa(".tab-panel");

  function setActiveTab(targetId) {
    btns.forEach(btn => {
      const isActive = btn.getAttribute("data-tab-target") === targetId;
      btn.classList.toggle("bg-indigo-600", isActive);
      btn.classList.toggle("text-white", isActive);
      btn.classList.toggle("shadow-card", isActive);
      btn.classList.toggle("bg-white", !isActive);
      btn.classList.toggle("text-gray-700", !isActive);
      btn.classList.toggle("border", !isActive);
      btn.classList.toggle("border-gray-300", !isActive);
      btn.classList.toggle("hover:bg-gray-50", !isActive);
    });
    panels.forEach(panel => {
      if ("#" + panel.id === targetId) { panel.classList.remove("hidden"); panel.classList.add("block"); }
      else { panel.classList.add("hidden"); panel.classList.remove("block"); }
    });
  }

  const defaultTab = btns[0]?.getAttribute("data-tab-target") || "#tab-heatmap";
  setActiveTab(defaultTab);
  btns.forEach(btn => btn.addEventListener("click", async () => {
    const target = btn.getAttribute("data-tab-target");
    setActiveTab(target);
    if (target === "#tab-heatmap") await refreshHeatmap();
    if (target === "#tab-flow") await refreshFlow();
  }));
})();

// ==============================
// Heatmap
// ==============================
async function loadHeatmap(query = "") {
  const res = await fetch(`/api/heatmap${query}`);
  if (!res.ok) return;
  const data = await res.json();
  const { tactics, order, unknown_count } = data;

  const titleEl = qs("#heatmapTitle");
  if (titleEl) {
    const old = qs("#unknownLabel");
    if (old) old.remove();
    if (unknown_count && unknown_count > 0) {
      const span = document.createElement("span");
      span.id = "unknownLabel";
      span.className = "ml-2 text-xs text-gray-500";
      span.textContent = `Unknown 이벤트 수 : ${unknown_count}건`;
      titleEl.appendChild(span);
    }
  }

  const thead = qs("#heatmapHeader"); thead.innerHTML = "";
  order.forEach(tac => {
    const th = document.createElement("th");
    th.className = "p-2 text-center";
    th.textContent = tac;
    thead.appendChild(th);
  });

  const tbody = qs("#heatmapBody"); tbody.innerHTML = "";
  if (!order || order.length === 0) return;
  const maxRows = Math.max(...order.map(tac => (tactics[tac] || []).length));
  const pick = n => n >= 100 ? "#494CA2" : n >= 10 ? "#8186D5" : n >= 1 ? "#C6CBEF" : "#E3E7F1";

  for (let r = 0; r < maxRows; r++) {
    const tr = document.createElement("tr");
    order.forEach(tac => {
      const td = document.createElement("td");
      td.className = "p-1 rounded-xl align-top";
      td.style.backgroundClip = "content-box"
      td.style.padding = "3px";
      const tech = (tactics[tac] || [])[r];
      if (tech) {
        const div = document.createElement("div");
        div.className = "heat-cell cursor-pointer";
        td.style.backgroundColor = pick(tech.count);
        div.addEventListener("click", () => openTechniqueModal(tech));
        const inner = document.createElement("div");
        inner.className = "cell-inner";
        inner.innerHTML = `<div class="font-semibold text-[10px] sm:text-xs">${tech.technique}</div><div class="text-[10px]">${tech.count}</div>`;
        div.appendChild(inner);
        td.appendChild(div);
      } else {
        td.innerHTML = `<div class="heat-cell"></div>`;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}
function openTechniqueModal(tech) {
  const modal = qs("#heatmapModal");
  if (!modal) return;
  qs("#heatmapModalTitle").textContent = `${tech.tid || ""} ${tech.technique}`;
  qs("#heatmapModalBody").innerHTML = `
    <div class="text-sm space-y-2">
      <div><b>TID:</b> ${tech.tid || "-"}</div>
      <div><b>Description:</b><div class="mt-1 bg-gray-50 border rounded p-2 text-xs">${tech.description || "-"}</div></div>
      <div class="text-xs text-gray-600">Events: ${tech.count}건</div>
    </div>`;
  modal.classList.remove("hidden");
  modal.classList.add("block");
  qs("#heatmapModalClose").onclick = () => modal.classList.add("hidden");
}

async function loadMiniHeatmap(query = "") {
  const res = await fetch(`/api/heatmap${query}`);
  if (!res.ok) return;
  const data = await res.json();
  const { tactics, order } = data;
  const wrap = qs("#miniHeatmap .grid");
  if (!wrap) return;
  wrap.innerHTML = "";
  const pick = n => n >= 100 ? "#494CA2" : n >= 10 ? "#8186D5" : n >= 1 ? "#C6CBEF" : "#E3E7F1";
  order.forEach(tac => {
    const row = document.createElement("div");
    row.className = "flex items-center gap-2";
    const label = document.createElement("div");
    label.className = "w-6 text-xs font-semibold text-gray-600";
    label.textContent = tac.split(" ").map(x => x[0]).join("").slice(0, 1);
    row.appendChild(label);
    const col = document.createElement("div");
    col.className = "flex items-center flex-wrap gap-1";
    (tactics[tac] || []).forEach(tech => {
      const box = document.createElement("div");
      box.className = "w-4 h-4 rounded";
      box.style.backgroundColor = pick(tech.count);
      col.appendChild(box);
    });
    row.appendChild(col);
    wrap.appendChild(row);
  });
}

async function refreshHeatmap() {
  const q = buildQuery();
  await loadHeatmap(q);
  await loadMiniHeatmap(q);
  await updateTopbarMetrics();
}

// ==============================
// Flow
// ==============================
let FLOW_STATE = { currentSession: null, page: 1, pageSize: 10 };

async function refreshFlow() {
  const rangeQS = buildRangeParams();
  const url = `/api/sessions${rangeQS ? ("?" + rangeQS) : ""}`;
  const res = await fetch(url);
  const flowControls = qs("#flowControls");
  flowControls.innerHTML = "";
  if (!res.ok) {
    flowControls.innerHTML = `<div class="text-xs text-red-600">세션을 불러올 수 없습니다.</div>`;
    return;
  }
  const sessions = await res.json();

  sessions.forEach(s => {
    const btn = document.createElement("button");
    btn.className = "px-3 py-1 border rounded bg-white hover:bg-gray-50 text-xs";
    btn.textContent = `${s.label} (${s.count}건)`;
    btn.dataset.sid = s.id;

    btn.addEventListener("click", async () => {
      qsa("#flowControls button").forEach(b => {
        b.classList.remove(
          "bg-gray-200", "text-gray-800",
          "bg-indigo-100", "text-indigo-800"
        );
        b.classList.add("bg-white", "hover:bg-gray-50", "text-gray-700");
      });

      btn.classList.remove("bg-white", "hover:bg-gray-50", "text-gray-700");
      btn.classList.add("bg-gray-200", "text-gray-800");

      await openSession(s.id);
    });

    flowControls.appendChild(btn);
  });

  const singlesRes = await fetch(`/api/single_events${rangeQS ? ("?" + rangeQS) : ""}`);
  if (singlesRes.ok) {
    const singles = await singlesRes.json();
    const btn = document.createElement("button");
    btn.className = "px-3 py-1 border rounded border-indigo-600 bg-white hover:bg-indigo-50 text-xs ml-2";
    btn.textContent = `단일공격 조회 (${singles.count}건)`;
    btn.dataset.sid = "single";

    btn.addEventListener("click", () => {
      qsa("#flowControls button").forEach(b => {
        b.classList.remove(
          "bg-gray-200", "text-gray-800",
          "bg-indigo-100", "text-indigo-800"
        );
        b.classList.add("bg-white", "hover:bg-gray-50", "text-gray-700");
      });

      btn.classList.remove("bg-white", "hover:bg-gray-50", "text-gray-700");
      btn.classList.add("bg-indigo-100", "text-indigo-800");

      renderSingleEventsTable(singles.events || []);
    });

    flowControls.appendChild(btn);
  }


  if (sessions && sessions.length > 0) {
    await openSession(sessions[0].id);
  } else {
    qs("#flowCanvas").innerHTML = `<div class="text-xs text-gray-500 p-3">표시할 세션이 없습니다.</div>`;
    qs("#timeline").innerHTML = `<li class="text-gray-500">표시할 타임라인이 없습니다.</li>`;
    qs("#eventsTable tbody").innerHTML = "";
    qs("#pageInfo").textContent = "0 / 0";

    const ai2Panel = document.querySelector("#ai2Panel");
    if (ai2Panel) ai2Panel.classList.add("hidden");
  }

}

async function openSession(sid) {
  const rangeQS = buildRangeParams();
  const url = `/api/session/${sid}${rangeQS ? ("?" + rangeQS) : ""}`;
  const res = await fetch(url);
  if (!res.ok) { console.warn("세션 상세 실패:", res.status); return; }
  const session = await res.json();
  if (session.error) { console.warn(session.error); return; }

  FLOW_STATE.currentSession = session;
  renderGraph(session);

  populateEventsTable(session.events || []);
  qs("#pageInfo").textContent = `1 / 1`;

  renderTimeline(session.events || []);
  loadSessionEvents(sid);

  const ai2Panel = document.querySelector("#ai2Panel");
  if (ai2Panel) ai2Panel.classList.remove("hidden");
}

function populateEventsTable(events, page = 1, per_page = 10) {
  const tbody = qs("#eventsTable tbody");
  const pageInfo = qs("#pageInfo");
  const prev = qs("#prevPage");
  const next = qs("#nextPage");

  if (!events || events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11" class="text-center p-4 text-gray-500">표시할 이벤트가 없습니다.</td></tr>`;
    pageInfo.textContent = "0 / 0";
    return;
  }

  const total = events.length;
  const totalPages = Math.max(1, Math.ceil(total / per_page));
  const start = (page - 1) * per_page;
  const end = Math.min(start + per_page, total);
  const pageEvents = events.slice(start, end);

  tbody.innerHTML = "";
  pageEvents.forEach(ev => {
    const tr = document.createElement("tr");
    tr.className = "border-t hover:bg-gray-50 cursor-pointer";
    tr.dataset.raw = JSON.stringify(ev);
    tr.innerHTML = `
      <td class="p-2">${ev.timestamp || "-"}</td>
      <td class="p-2">${ev.src_ip || "-"}</td>
      <td class="p-2">${ev.src_port ?? "-"}</td>
      <td class="p-2">${ev.dst_ip || "-"}</td>
      <td class="p-2">${ev.dst_port ?? "-"}</td>
      <td class="p-2">${ev.protocol || "-"}</td>
      <td class="p-2">${ev.fwd_bytes ?? "-"}</td>
      <td class="p-2">${ev.bwd_bytes ?? "-"}</td>
      <td class="p-2">${ev.tactic || "-"}</td>
      <td class="p-2">${ev.tid || "-"}</td>
      <td class="p-2">${ev.technique || "-"}</td>`;
    tr.addEventListener("dblclick", () => openEventModal(JSON.parse(tr.dataset.raw)));
    tbody.appendChild(tr);
  });

  pageInfo.textContent = `${page} / ${totalPages}`;
  prev.onclick = () => { if (page > 1) populateEventsTable(events, page - 1, per_page); };
  next.onclick = () => { if (page < totalPages) populateEventsTable(events, page + 1, per_page); };
}

async function loadSessionEvents(sid) {
  try {
    const rangeQS = buildRangeParams();
    const url = `/api/session/${sid}${rangeQS ? ("?" + rangeQS) : ""}`;
    const res = await fetch(url);
    const ai2Panel = document.getElementById("ai2Panel");

    if (!res.ok) {
      console.warn("이벤트 목록 실패:", res.status);
      if (ai2Panel) ai2Panel.classList.add("hidden");
      return;
    }

    const session = await res.json();
    if (!session || !session.events || session.events.length === 0) {
      console.warn("세션 데이터 없음", session);
      if (ai2Panel) ai2Panel.classList.add("hidden");
      return;
    }

    FLOW_STATE.currentSession = session;

    renderTimeline(session.events, session.session_confidence || null);

    const timeline = document.querySelector("#timeline");
    if (
      !timeline ||
      timeline.textContent.includes("표시할 타임라인이 없습니다.") ||
      timeline.textContent.includes("이벤트가 없습니다.")
    ) {
      if (ai2Panel) ai2Panel.classList.add("hidden");
      return;
    }

    if (ai2Panel) ai2Panel.classList.remove("hidden");

    populateEventsTable(session.events);

    ai2Panel.innerHTML = ""; // 초기화

    const seenTids = new Set((session.events || []).map(e => e.tid).filter(Boolean));

    const ctx = (session.context_tids || []).filter(t => t && t !== "<UNK>");
    const ctxDisplay = ctx.length ? ctx.join(" → ") : "(none)";

    const rawPreds = session.next_tid_topk || [];
    const futurePreds = rawPreds.filter(p => p && p.tid && !seenTids.has(p.tid));

    let html = `<div class="mb-2 text-sm font-semibold text-gray-700">Next TID Prediction</div>`;
    html += `<div class="text-xs text-gray-500 mb-2">최근 시퀀스: ${ctxDisplay}</div>`;

    if (!futurePreds.length) {
      html += `<div class="text-xs text-gray-400 italic">예측된 다음 공격이 현재 세션에서 이미 발생했거나 예측 없음</div>`;
    } else {
      html += `<div class="space-y-2">`;
      futurePreds.forEach(p => {
        const pct = Math.round((p.prob || 0) * 100);
        const barColor = pct >= 70 ? "bg-red-500" : (pct >= 40 ? "bg-orange-500" : "bg-emerald-500");
        const tech = p.technique ? p.technique : "-";
        html += `
          <div class="flex items-center justify-between text-sm">
            <div class="flex flex-col">
              <div class="font-semibold">${p.tid} <span class="text-xs text-gray-500">(${tech})</span></div>
              <div class="text-xs text-gray-500">${pct}%</div>
            </div>
            <div class="w-36 ml-3">
              <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="${barColor} h-2 rounded-full" style="width:${pct}%;"></div>
              </div>
            </div>
          </div>
        `;
      });
      html += `</div>`;
    }

    ai2Panel.innerHTML = html;

  } catch (err) {
    console.error("loadSessionEvents error:", err);
    const ai2Panel = document.getElementById("ai2Panel");
    if (ai2Panel) ai2Panel.classList.add("hidden");
  }
}

function renderSingleEventsTable(events) {
  qs("#flowCanvas").innerHTML = `<div class="text-xs text-gray-600 p-3">단일 이벤트 목록 (그래프는 공격플로우만 표시됩니다)</div>`;
  qs("#timeline").innerHTML = `<li class="text-gray-500">단일공격은 타임라인이 표시되지 않습니다.</li>`;
  const ai2Panel = document.querySelector("#ai2Panel");
  if (ai2Panel) ai2Panel.classList.add("hidden");
  populateEventsTable(events);
}

function probToBadgeClass(prob) {
  // prob: 0.0 ~ 1.0
  const p = prob * 100;
  if (p >= 70) return "bg-red-50 text-red-600 border-red-100";
  if (p >= 40) return "bg-orange-50 text-orange-600 border-orange-100";
  return "bg-emerald-50 text-emerald-600 border-emerald-100";
}

function renderTimeline(events, sessionConf = null) {
  const ol = document.getElementById("timeline");
  const wrap = document.getElementById("timelineWrap");

  const prevSummary = document.getElementById("timelineSessionSummary");
  if (prevSummary) prevSummary.remove();

  ol.innerHTML = "";

  if (sessionConf && sessionConf.combined != null) {
    const combined = sessionConf.combined;
    const color =
      combined >= 80 ? "bg-red-500" : combined >= 60 ? "bg-orange-500" : "bg-emerald-500";

    const summaryDiv = document.createElement("div");
    summaryDiv.id = "timelineSessionSummary";
    summaryDiv.className = "mb-3";
    summaryDiv.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <span class="text-sm font-semibold text-gray-700">세션 일관성 (공격 흐름 확률)</span>
        <span class="text-sm font-bold">${combined.toFixed(1)}%</span>
      </div>
      <div class="w-full bg-gray-200 rounded-full h-2">
        <div class="${color} h-2 rounded-full transition-all duration-700 ease-out" style="width:${combined}%;"></div>
      </div>
    `;

    wrap.insertBefore(summaryDiv, wrap.firstChild);
  }

  if (!events || events.length === 0) {
    ol.innerHTML = `<li class="text-gray-500">이벤트가 없습니다.</li>`;
    return;
  }

  events.forEach((ev) => {
    const li = document.createElement("li");
    li.className = "relative border-l-2 border-gray-200 pl-3 pb-3 tl-item";

    const time = ev.timestamp || "-";
    const src = `${ev.src_ip || "?"}`;
    const dst = `${ev.dst_ip || "?"}`;
    const tid = ev.tid || "Unknown";
    const tech = ev.technique || "-";

    const s1 = ev.stage1_conf != null ? Math.round(ev.stage1_conf * 100) : null;
    const s2 = ev.stage2_conf != null ? Math.round(ev.stage2_conf * 100) : null;

    const color1 = s1 >= 80 ? "bg-red-500" : s1 >= 60 ? "bg-orange-500" : "bg-emerald-500";
    const color2 = s2 >= 80 ? "bg-red-500" : s2 >= 60 ? "bg-orange-500" : "bg-emerald-500";

    li.innerHTML = `
      <div class="flex justify-between mb-0.5">
        <span class="text-[11px] text-gray-500">${time}</span>
      </div>
      <div class="text-sm text-gray-600">${src} → ${dst}</div>
      <div class="font-semibold text-xs">${tid} <span class="text-gray-500">(${tech})</span></div>

      ${s1 != null
        ? `
      <div class="mt-1">
        <div class="flex justify-between text-[11px] text-gray-600">
          <span>공격 확률</span>
          <span class="font-semibold">${s1}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-1.5 mt-0.5">
          <div class="${color1} h-1.5 rounded-full transition-all duration-700 ease-out" style="width:${s1}%;"></div>
        </div>
      </div>`
        : ""
      }

      ${s2 != null
        ? `
      <div class="mt-1">
        <div class="flex justify-between text-[11px] text-gray-600">
          <span>TTP 신뢰도</span>
          <span class="font-semibold">${s2}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-1.5 mt-0.5">
          <div class="${color2} h-1.5 rounded-full transition-all duration-700 ease-out" style="width:${s2}%;"></div>
        </div>
      </div>`
        : ""
      }
    `;
    ol.appendChild(li);
  });

} function renderTimeline(events, sessionConf = null) {
  const ol = document.getElementById("timeline");
  const wrap = document.getElementById("timelineWrap");

  const prevSummary = document.getElementById("timelineSessionSummary");
  if (prevSummary) prevSummary.remove();

  ol.innerHTML = "";

  if (sessionConf && sessionConf.combined != null) {
    const combined = sessionConf.combined;
    const color =
      combined >= 80 ? "bg-red-500" : combined >= 60 ? "bg-orange-500" : "bg-emerald-500";

    const summaryDiv = document.createElement("div");
    summaryDiv.id = "timelineSessionSummary";
    summaryDiv.className = "mb-3";
    summaryDiv.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <span class="text-sm font-semibold text-gray-700">세션 일관성 (공격 흐름 확률)</span>
        <span class="text-sm font-bold">${combined.toFixed(1)}%</span>
      </div>
      <div class="w-full bg-gray-200 rounded-full h-2">
        <div class="${color} h-2 rounded-full transition-all duration-700 ease-out" style="width:${combined}%;"></div>
      </div>
    `;

    wrap.insertBefore(summaryDiv, wrap.firstChild);
  }

  if (!events || events.length === 0) {
    ol.innerHTML = `<li class="text-gray-500">이벤트가 없습니다.</li>`;
    return;
  }

  events.forEach((ev) => {
    const li = document.createElement("li");
    li.className = "relative border-l-2 border-gray-200 pl-3 pb-3 tl-item";

    const time = ev.timestamp || "-";
    const src = `${ev.src_ip || "?"}:${ev.src_port ?? "-"}`;
    const dst = `${ev.dst_ip || "?"}:${ev.dst_port ?? "-"}`;
    const tid = ev.tid || "Unknown";
    const tech = ev.technique || "-";

    const s1 = ev.stage1_conf != null ? Math.round(ev.stage1_conf * 100) : null;
    const s2 = ev.stage2_conf != null ? Math.round(ev.stage2_conf * 100) : null;

    const color1 = s1 >= 80 ? "bg-red-500" : s1 >= 60 ? "bg-orange-500" : "bg-emerald-500";
    const color2 = s2 >= 80 ? "bg-red-500" : s2 >= 60 ? "bg-orange-500" : "bg-emerald-500";

    li.innerHTML = `
      <div class="flex justify-between mb-0.5">
        <span class="text-[11px] text-gray-500">${time}</span>
      </div>
      <div class="text-xs text-gray-600">${src} → ${dst}</div>
      <div class="font-semibold text-sm">${tid} <span class="text-gray-500">(${tech})</span></div>

      ${s1 != null
        ? `
      <div class="mt-1">
        <div class="flex justify-between text-[11px] text-gray-600">
          <span>공격 확률</span>
          <span class="font-semibold">${s1}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-1.5 mt-0.5">
          <div class="${color1} h-1.5 rounded-full transition-all duration-700 ease-out" style="width:${s1}%;"></div>
        </div>
      </div>`
        : ""
      }

      ${s2 != null
        ? `
      <div class="mt-1">
        <div class="flex justify-between text-[11px] text-gray-600">
          <span>TTP 신뢰도</span>
          <span class="font-semibold">${s2}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-1.5 mt-0.5">
          <div class="${color2} h-1.5 rounded-full transition-all duration-700 ease-out" style="width:${s2}%;"></div>
        </div>
      </div>`
        : ""
      }
    `;
    ol.appendChild(li);
  });

}

function drawFutureConnector(events, ai2_summary) {
  const overlay = document.getElementById("timelineOverlay");
  const badgesLayer = document.getElementById("timelinePredBadges");
  const stage = document.getElementById("timelineStage");
  const wrap = document.getElementById("timelineWrap");

  overlay.innerHTML = "";
  badgesLayer.innerHTML = "";

  const stageRect = stage.getBoundingClientRect();
  overlay.setAttribute("width", String(stageRect.width));
  overlay.setAttribute("height", String(stageRect.height));
  overlay.setAttribute("viewBox", `0 0 ${stageRect.width} ${stageRect.height}`);

  const items = Array.from(document.querySelectorAll("#timeline .tl-item"));
  if (items.length === 0) return;


  const lastLi = items[items.length - 1];
  const lastRect = lastLi.getBoundingClientRect();
  const startX = lastRect.right - stageRect.left;
  const startY = lastRect.top + lastRect.height / 2 - stageRect.top;

  const preds = (ai2_summary && ai2_summary.next_tid_topk) ? ai2_summary.next_tid_topk : [];
  if (!preds || preds.length === 0) return;

  const top1 = preds[0];
  const prob = top1.prob ?? 0.0;
  const badgeClass = probToBadgeClass(prob);

  const badgeLeft = Math.min(stageRect.width - 10, startX + 24);
  const badgeTop = Math.max(8, startY - 18);

  const badge = document.createElement("div");
  badge.className = `absolute pointer-events-auto rounded-lg px-3 py-1 border text-xs ${badgeClass}`;
  badge.style.left = `${badgeLeft}px`;
  badge.style.top = `${badgeTop}px`;

  const top3_html = preds.map(p => `${p.tid} ${(p.prob * 100).toFixed(0)}%`).join(" · ");
  badge.innerHTML = `<div class="font-semibold">${top1.tid}</div><div class="text-[11px] text-gray-500">${top3_html}</div>`;

  badgesLayer.appendChild(badge);

  const badgeRect = badge.getBoundingClientRect();
  const endX = (badgeRect.left - stageRect.left);
  const endY = (badgeRect.top - stageRect.top) + (badgeRect.height / 2);

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(startX));
  line.setAttribute("y1", String(startY));
  line.setAttribute("x2", String(endX));
  line.setAttribute("y2", String(endY));
  line.setAttribute("stroke", "#94a3b8");
  line.setAttribute("stroke-width", "1.6");
  line.setAttribute("stroke-dasharray", "5,5");
  line.setAttribute("opacity", "0.95");
  overlay.appendChild(line);
}


function openEventModal(raw) {
  const modal = qs("#eventModal");
  if (!modal) return;
  qs("#eventModalTitle").textContent = "Raw 이벤트 상세";
  qs("#eventModalBody").innerHTML = `
    <div class="grid grid-cols-2 gap-4 text-sm mb-4">
      <div><span class="text-gray-500">timestamp</span><div class="font-medium">${raw.timestamp || "-"}</div></div>
      <div><span class="text-gray-500">src → dst</span><div class="font-medium">${raw.src_ip || "-"} → ${raw.dst_ip || "-"}</div></div>
      <div><span class="text-gray-500">tactic</span><div class="font-medium">${raw.tactic || "-"}</div></div>
      <div><span class="text-gray-500">technique</span><div class="font-medium">${raw.technique || "-"} (${raw.tid || "-"})</div></div>
    </div>
    <div class="border rounded-xl bg-gray-50 text-xs p-2 mb-3 overflow-auto max-h-[40vh]">
      <pre>${JSON.stringify(raw, null, 2)}</pre>
    </div>`;
  modal.classList.remove("hidden");
  qs("#eventModalClose").onclick = () => modal.classList.add("hidden");
}

function renderGraph(session) {
  const flowCanvas = qs("#flowCanvas");
  flowCanvas.innerHTML = "";
  const width = flowCanvas.clientWidth || 900;
  const height = flowCanvas.clientHeight || 520;

  const svg = d3.select(flowCanvas).append("svg").attr("width", width).attr("height", height).style("background", "#fff");
  const container = svg.append("g");

  let nodes = (session.nodes || []).map(n => ({ id: n.id, color: n.color || "#dc2626", count: n.count || 1 }));
  let links = (session.edges || []).map(e => ({ source: e.source, target: e.target, count: e.count || 1 }));

  const degIn = {}, degOut = {};
  links.forEach(l => { degOut[l.source] = (degOut[l.source] || 0) + 1; degIn[l.target] = (degIn[l.target] || 0) + 1; });
  const startNode = nodes.find(n => !degIn[n.id] && degOut[n.id])?.id || null;
  const endNode = nodes.find(n => !degOut[n.id] && degIn[n.id])?.id || null;

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(140).strength(0.8))
    .force("charge", d3.forceManyBody().strength(-700))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(40))
    .alphaDecay(0.04);

  let tooltip = d3.select(flowCanvas).append("div")
    .attr("id", "graphTooltip")
    .style("position", "absolute")
    .style("background", "rgba(0,0,0,0.75)")
    .style("color", "#fff")
    .style("padding", "6px 8px")
    .style("border-radius", "6px")
    .style("font-size", "12px")
    .style("pointer-events", "none")
    .style("opacity", 0);

  function showTip(ev, text) {
    tooltip.transition().duration(80).style("opacity", 1);
    tooltip.html(text)
      .style("left", (ev.pageX + 12) + "px")
      .style("top", (ev.pageY - 20) + "px");
  }
  function hideTip() { tooltip.transition().duration(120).style("opacity", 0); }

  const maxStroke = 8;
  const link = container.append("g").attr("stroke", "#9ca3af").attr("stroke-opacity", 0.8)
    .selectAll("line").data(links).join("line")
    .attr("stroke-width", d => Math.min(maxStroke, 1 + Math.log(d.count + 1) * 2))
    .on("mouseover", (ev, d) => showTip(ev, `Count: ${d.count}건`))
    .on("mouseout", hideTip);

  const nodeG = container.append("g").selectAll("g").data(nodes).join("g")
    .attr("class", "node-group")
    .style("cursor", "pointer")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on("mouseover", (ev, d) => showTip(ev, d.id))
    .on("mouseout", hideTip)
    .on("click", async (ev, d) => {
      if (!session || !session.id) return;
      const ai2Panel = document.querySelector("#ai2Panel");
      if (ai2Panel) ai2Panel.classList.remove("hidden");
      await updateTimeline(session.id, d.id);
      await loadSessionEvents(session.id, 1, FLOW_STATE.pageSize, d.id);
    });

  nodeG.append("circle")
    .attr("r", 22)
    .attr("fill", "#fff")
    .attr("stroke", d => d.color)
    .attr("stroke-width", 3);

  nodeG.append("text")
    .attr("dy", 4)
    .attr("text-anchor", "middle")
    .attr("font-size", 14)
    .attr("font-weight", "700")
    .text(d => {
      if (d.id === startNode) return "S";
      if (d.id === endNode) return "E";
      return "";
    });

  nodeG.append("text")
    .attr("dy", 42)
    .attr("text-anchor", "middle")
    .attr("font-size", 11)
    .attr("fill", "#111827")
    .text(d => d.id);

  simulation.on("tick", () => {
    link
      .attr("x1", d => (d.source.x))
      .attr("y1", d => (d.source.y))
      .attr("x2", d => (d.target.x))
      .attr("y2", d => (d.target.y));
    nodeG.attr("transform", d => `translate(${d.x},${d.y})`);
  });

  svg.call(d3.zoom().scaleExtent([0.5, 3]).on("zoom", ev => container.attr("transform", ev.transform)));
}

async function updateTimeline(sid, nodeId = null) {
  const session = FLOW_STATE.currentSession;
  if (!session) return;

  if (!session.events) {
    renderTimeline([]);
    return;
  }

  if (!nodeId) {
    renderTimeline(session.events.slice(0, 50));
  } else {
    const evs = session.events.filter(ev => {
      if (ev.src_ip === nodeId || ev.dst_ip === nodeId) return true;
      if (nodeId.includes("외")) {
        const base = nodeId.split(" 외 ")[0].trim();
        if (ev.src_ip === base || ev.dst_ip === base) return true;
      }
      return false;
    });
    renderTimeline(evs);
  }

  try {
    const res = await fetch(`/api/session/${sid}`);
    const data = await res.json();
    const panel = document.querySelector("#nextTidPanel");
    if (!panel) return;
    panel.innerHTML = "";

    if (!data.next_tid_topk || data.next_tid_topk.length === 0) {
      panel.innerHTML = `
        <p class="text-gray-400 italic text-[12px]">
          AI2 예측 데이터가 없습니다.
        </p>`;
      return;
    }

    const ctxHTML = (data.context_tids || [])
      .map(
        t =>
          `<span class="px-1.5 py-0.5 bg-gray-100 rounded text-[11px] font-mono">${t}</span>`
      )
      .join(" ");
    const ctxBlock = `
      <div class="mb-2 text-[11px] text-gray-500">
        <div class="mb-1">최근 ${data.context_T || 4}개 시퀀스</div>
        <div class="flex flex-wrap gap-1">${ctxHTML}</div>
      </div>
    `;
    panel.insertAdjacentHTML("beforeend", ctxBlock);

    data.next_tid_topk.forEach(p => {
      const prob = (p.prob * 100).toFixed(1);
      const color =
        prob >= 70 ? "bg-red-500" :
          prob >= 40 ? "bg-orange-400" :
            "bg-green-400";
      const card = `
        <div class="p-2 rounded-lg border border-gray-200 bg-gray-50 hover:bg-white transition shadow-sm">
          <div class="flex justify-between text-xs font-semibold text-gray-800 mb-1">
            <span>${p.tid}</span>
            <span>${prob}%</span>
          </div>
          <div class="text-[11px] text-gray-500 mb-1">${p.technique || ""}</div>
          <div class="w-full bg-gray-200 rounded-full h-1.5">
            <div class="${color} h-1.5 rounded-full" style="width:${prob}%"></div>
          </div>
        </div>`;
      panel.insertAdjacentHTML("beforeend", card);
    });
  } catch (e) {
    console.warn("[AI2] 예측 데이터 로드 실패:", e);
    const panel = document.querySelector("#nextTidPanel");
    if (panel)
      panel.innerHTML = `<p class="text-gray-400 italic text-[12px]">
        AI2 예측 데이터를 불러오는 중 오류가 발생했습니다.
      </p>`;
  }
}

async function refreshAll() {
  await Promise.all([refreshHeatmap(), updateTopbarMetrics()]);
  await refreshFlow();
}

(async function init() {
  const today = new Date();
  qs("#endDate").value = fmtDate(today);
  qs("#startDate").value = fmtDate(new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000));
  await refreshAll();
})();