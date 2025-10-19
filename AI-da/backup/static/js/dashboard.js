// ==============================
// Utilities
// ==============================
const RISK_COLORS = { low: "#FFC107", middle: "#FF6F3C", high: "#B22222" };
function qs(sel, root = document) { return root.querySelector(sel); }
function qsa(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }
function fmtDate(d) { return d.toISOString().slice(0, 10); }
function currentRange() {
  const s = qs("#startDate")?.value || "";
  const e = qs("#endDate")?.value || "";
  const quick = qs("#quickRange")?.value || "realtime";
  // quickRange 값으로 모드 판정
  const mode = (quick === "realtime") ? "live" : "range";
  return { start: s, end: e, mode };
}
function buildQuery() {
  const { start, end, mode } = currentRange();
  if (mode === "live") return "?mode=live";
  if (start && end) return `?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&mode=range`;
  return "?mode=range"; // 기본 safeguard
}

// ==============================
// Topbar behaviors (Risk / Help tooltip / Quick range)
// ==============================
(function initTopbar() {
  const riskBadge = qs("#riskBadge");
  const riskScore = qs("#riskScore");
  const riskLevel = qs("#riskLevel");
  const eventCount = qs("#eventCount");

  // 조회 버튼
  qs("#applyRange").addEventListener("click", async () => {
    await refreshAll(true);
  });

  // 빠른 범위 선택
  const quick = qs("#quickRange");
  const s = qs("#startDate"); const e = qs("#endDate");
  quick.addEventListener("change", () => {
    const today = new Date();
    const end = fmtDate(today);
    let start = end;
    const DAY = 24 * 60 * 60 * 1000;
    if (quick.value === "realtime") {
      // 실시간: 날짜 입력 비활성화(선택)
      s.value = ""; e.value = "";
      s.disabled = true; e.disabled = true;
    } else {
      s.disabled = false; e.disabled = false;
      if (quick.value === "24h") start = fmtDate(new Date(today.getTime() - 1 * DAY));
      else if (quick.value === "7d") start = fmtDate(new Date(today.getTime() - 7 * DAY));
      else if (quick.value === "30d") start = fmtDate(new Date(today.getTime() - 30 * DAY));
      // custom은 사용자가 직접 입력
      if (quick.value !== "custom") { s.value = start; e.value = end; }
    }
  });

  // Tooltip
  const helpIcon = qs("#helpIcon");
  const helpTooltip = qs("#helpTooltip");
  let tipVisible = false;
  helpIcon.addEventListener("mouseenter", () => {
    tipVisible = true; helpTooltip.classList.remove("invisible","opacity-0");
  });
  helpIcon.addEventListener("mouseleave", () => {
    tipVisible = false; helpTooltip.classList.add("invisible","opacity-0");
  });
  document.addEventListener("mousemove", (ev) => {
    if (!tipVisible) return;
    const off = 10;
    const tooltipWidth = helpTooltip.offsetWidth || 150;
    const tooltipHeight = helpTooltip.offsetHeight || 50;
    let left = ev.pageX + off, top = ev.pageY + off;
    if (left + tooltipWidth > window.innerWidth) left = ev.pageX - tooltipWidth - off;
    if (top + tooltipHeight > window.innerHeight) top = ev.pageY - tooltipHeight - off;
    helpTooltip.style.left = `${left}px`; helpTooltip.style.top = `${top}px`;
  });

  // 위험도/이벤트 수 업데이트 함수
  window.updateTopbarMetrics = async function() {
    const q = buildQuery();
    const res = await fetch(`/api/metrics${q}`);
    const m = await res.json();
    riskScore.textContent = String(m.score ?? 0);
    riskLevel.textContent = m.level || "low";
    riskBadge.style.backgroundColor = RISK_COLORS[m.level] || "#FFC107";
    eventCount.textContent = `${m.count ?? 0}건`;
  };
})();

// ==============================
// Tabs
// ==============================
// ==============================
// Tabs (Active style toggle)
// ==============================
(function initTabs() {
  const btns = qsa(".tab-btn");
  const panels = qsa(".tab-panel");

  // 함수: 활성 탭 스타일 변경
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
      if ("#" + panel.id === targetId) {
        panel.classList.remove("hidden");
        panel.classList.add("block");
      } else {
        panel.classList.add("hidden");
        panel.classList.remove("block");
      }
    });
  }

  // 초기 탭 상태 설정 (첫 번째 탭 활성화)
  const defaultTab = btns[0]?.getAttribute("data-tab-target") || "#tab-heatmap";
  setActiveTab(defaultTab);

  // 클릭 이벤트 설정
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab-target");
      setActiveTab(target);

      if (target === "#tab-heatmap") {
        computeHeatmapCellSize?.();
      }
    });
  });
})();


// ==============================
// Heatmap (build from TTPs.json via /api/heatmap)
// ==============================
function computeHeatmapCellSize() {
  // CSS 변수를 쓰므로 여기서는 생략 가능 (반응형은 custom.css 미디어쿼리 사용)
}

async function loadHeatmap(query = "") {
  const res = await fetch(`/api/heatmap${query}`);
  const data = await res.json();
  const { tactics, order } = data;

  // Header (tactic)
  const thead = qs("#heatmapHeader"); thead.innerHTML = "";
  order.forEach(tac => {
    const th = document.createElement("th");
    th.className = "p-2 text-center";
    th.textContent = tac;
    thead.appendChild(th);
  });

  // Body (vertical columns by tactic, rows = max techniques)
  const tbody = qs("#heatmapBody"); tbody.innerHTML = "";
  const maxRows = Math.max(...order.map(tac => tactics[tac].length));
  const pick = n => n >= 100 ? "#494CA2" : n >= 10 ? "#8186D5" : n >= 1 ? "#C6CBEF" : "#E3E7F1";

  for (let r = 0; r < maxRows; r++) {
    const tr = document.createElement("tr");
    order.forEach(tac => {
      const td = document.createElement("td");
      td.className = "p-1 align-top";
      const tech = tactics[tac][r];
      if (tech) {
        const div = document.createElement("div");
        div.className = "heat-cell cursor-pointer";
        div.style.backgroundColor = pick(tech.count);
        div.dataset.tech = tech.technique;
        div.dataset.tac = tac;
        div.dataset.count = tech.count;
        div.dataset.tid = tech.tid || "";
        div.dataset.description = tech.description || "";
        div.dataset.events = JSON.stringify(tech.events || []);
        // 내부 텍스트
        const inner = document.createElement("div");
        inner.className = "cell-inner";
        inner.innerHTML = `<div class="font-semibold text-[10px] sm:text-xs">${tech.technique}</div><div class="text-[10px]">${tech.count}</div>`;
        div.appendChild(inner);
        td.appendChild(div);
        // 둥근 모서리/간격은 CSS로 제어
      } else {
        td.innerHTML = `<div class="heat-cell"></div>`;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  computeHeatmapCellSize();
  await updateTopbarMetrics();
}

// Heatmap Modal
(function initHeatmapModal() {
  const modal = qs("#heatmapModal"); if (!modal) return;
  const title = qs("#heatmapModalTitle");
  const body = qs("#heatmapModalBody");
  const btnCsv = qs("#btnCsv");
  qs("#heatmapModalClose").addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", e => { if (e.target === modal) modal.classList.add("hidden"); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") modal.classList.add("hidden"); });

  document.addEventListener("click", async (e) => {
    const cell = e.target.closest(".heat-cell"); if (!cell) return;
    const { start, end, mode } = currentRange();
    const tech = cell.dataset.tech; const tac = cell.dataset.tac; const cnt = cell.dataset.count;
    const tid = cell.dataset.tid; const desc = cell.dataset.description;

    title.textContent = `${tac} / ${tech}`;
    const q = buildQuery();
    btnCsv.href = `/api/export_events${q}&tech=${encodeURIComponent(tech)}&tac=${encodeURIComponent(tac)}`;

    body.innerHTML = `
      <div class="space-y-3 text-sm">
        <div><span class="text-gray-500">TID</span><div class="font-medium">${tid || "-"}</div></div>
        <div><span class="text-gray-500">Description</span><div class="font-medium">${desc || "-"}</div></div>
        <div><span class="text-gray-500">이벤트 수</span><div class="font-medium">${cnt || "0"} 건</div></div>
      </div>
      <div class="mt-4 text-xs text-gray-500">※ CSV 다운로드를 눌러 해당 Technique 관련 이벤트를 내려받을 수 있습니다.</div>
    `;
    modal.classList.remove("hidden");
    modal.scrollTop = 0;
  });
})();

// ==============================
// Mini Heatmap (flow 좌측)
// ==============================
async function loadMiniHeatmap(query = "") {
  const res = await fetch(`/api/heatmap${query}`);
  const data = await res.json();
  const wrap = qs("#miniHeatmap .grid");
  wrap.innerHTML = "";

  const { tactics, order } = data;
  const pick = n => n >= 100 ? "#494CA2" : n >= 10 ? "#8186D5" : n >= 1 ? "#C6CBEF" : "#E3E7F1";

  order.forEach(tac => {
    const row = document.createElement("div");
    row.className = "flex items-center gap-2";
    const label = document.createElement("div");
    label.className = "w-6 text-xs font-semibold text-gray-600";
    label.textContent = tac.split(" ").map(x=>x[0]).join("").slice(0,1); // 첫 글자 정도로
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

// ==============================
// Flow (buttons + graph + timeline + events)
// ==============================
let CURRENT_FLOW = null;
let CURRENT_SVG = null;
let CURRENT_SIM = null;

async function drawFlow(query = "") {
  const container = qs("#flowCanvas");
  if (!container) return;
  container.innerHTML = `<div class="w-full h-full flex items-center justify-center text-gray-400">플로우 불러오는 중...</div>`;

  const q = query || buildQuery();
  const res = await fetch(`/api/flow${q}`);
  const data = await res.json();
  const flows = data.flows || [];

  const controls = qs("#flowControls");
  controls.innerHTML = "";

  if (!flows.length) {
    container.innerHTML = `<div class="w-full h-full flex items-center justify-center text-gray-400">표시할 공격 플로우가 없습니다.</div>`;
    updateEventsTable({ events: [] });
    return;
  }

  flows.forEach((f, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "flow-btn px-3 py-1 border rounded text-xs bg-white hover:bg-gray-100";
    btn.textContent = f.title || `flow ${idx + 1}`;
    btn.addEventListener("click", async () => {
      qsa("#flowControls .flow-btn").forEach(b => b.classList.remove("bg-indigo-600","text-white"));
      btn.classList.add("bg-indigo-600","text-white");
    
      renderFlow(f);
      await updateTimeline(f);  // 🔥 공격플로우 버튼 클릭 시에도 timeline 반영
      updateEventsTable(f);
    });
    controls.appendChild(btn);
    if (idx === 0) setTimeout(()=>btn.click(), 10); // 첫 플로우 자동 선택
  });
}

function renderFlow(flow) {
  CURRENT_FLOW = flow;
  const container = qs("#flowCanvas");
  container.innerHTML = "";

  const rect = container.getBoundingClientRect();
  const width = Math.max(640, Math.floor(rect.width || 800));
  const height = Math.max(360, Math.floor(rect.height || 520));

  const svg = d3.select(container).append("svg")
    .attr("width", "100%").attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("display","block")
    .style("background","#f8fafc")
    .style("border-radius","10px");
  CURRENT_SVG = svg;

  const nodes = (flow.nodes || []).map(n => ({ ...n }));
  const links = (flow.edges || []).map(l => ({ ...l, source: l.from, target: l.to }));

  if (!nodes.length) {
    svg.append("text").attr("x", width/2).attr("y", height/2).attr("text-anchor","middle")
      .attr("fill","#6b7280").text("노드 데이터가 없습니다.");
    updateEventsTable(flow);
    return;
  }

  const maxCount = links.length ? d3.max(links, d => d.count) : 1;
  const linkScale = d3.scaleLinear().domain([0, maxCount]).range([1, Math.min(14, Math.max(3, maxCount/3))]);

  function nodeRadius(d) {
    const total = links.reduce((acc, e) => {
      if (e.from === d.id) acc += e.count || 0;
      if (e.to === d.id) acc += e.count || 0;
      return acc;
    }, 0);
    if (total > 120) return 22; // ddos 수준
    if (total > 30) return 18;
    return 14;
  }

  if (CURRENT_SIM) { try { CURRENT_SIM.stop(); } catch(e){} CURRENT_SIM = null; }
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(140).strength(1))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collide", d3.forceCollide().radius(d => nodeRadius(d) + 6));
  CURRENT_SIM = simulation;

  const linkEl = svg.append("g").selectAll("line")
    .data(links).join("line")
    .attr("stroke","#9CA3AF")
    .attr("stroke-width", d => Math.max(1, Math.round(linkScale(d.count))))
    .attr("stroke-linecap","round").style("opacity",0.95);

  const nodeG = svg.append("g").selectAll("g")
    .data(nodes).join("g")
    .style("cursor","pointer")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on("drag", (event, d) => { d.fx=event.x; d.fy=event.y; })
      .on("end", (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; })
    )
    .on("click", async (ev, d) => {
      await updateTimeline(d.id); // IP 기준
      updateEventsTable(flow, d.id);
    });

  nodeG.append("circle")
    .attr("r", d => nodeRadius(d))
    .attr("fill","#fff").attr("stroke","#374151").attr("stroke-width",1.4);

  nodeG.append("text")
    .attr("dy", d => -nodeRadius(d) - 6)
    .attr("text-anchor","middle").attr("font-size",11).attr("fill","#111827")
    .text(d => {
      const big = links.find(e => (e.from===d.id || e.to===d.id) && (e.count||0) > 120);
      return big ? `${d.label} 외 ${big.count}건` : d.label;
    });

  simulation.on("tick", () => {
    linkEl.attr("x1", d=>d.source.x).attr("y1", d=>d.source.y).attr("x2", d=>d.target.x).attr("y2", d=>d.target.y);
    nodeG.attr("transform", d => `translate(${d.x},${d.y})`);
  });

  updateEventsTable(flow); // 초기에는 전체
}

// Flow → Timeline
async function updateTimeline(nodeOrFlow) {
  const tl = qs("#timeline"); if (!tl) return;
  tl.innerHTML = `<li class="text-gray-500">타임라인 불러오는 중...</li>`;

  try {
    // 노드(IP)
    if (typeof nodeOrFlow === "string") {
      const res = await fetch(`/api/timeline/${encodeURIComponent(nodeOrFlow)}`);
      const data = await res.json();
      tl.innerHTML = "";
      if (!data || !data.length) {
        tl.innerHTML = `<li class="text-gray-500">타임라인 데이터 없음</li>`;
        return;
      }
      data.forEach(ev => {
        const li = document.createElement("li");
        li.className = "mb-3 relative pl-4 text-xs";
        li.innerHTML = `
          <div class="absolute -left-1.5 w-2.5 h-2.5 rounded-full bg-indigo-500"></div>
          <time class="text-gray-500">${ev.time}</time>
          <div class="font-medium">${ev.desc}</div>
        `;
        tl.appendChild(li);
      });
      return;
    }

    // 플로우 전체
    if (typeof nodeOrFlow === "object" && nodeOrFlow.events) {
      const events = nodeOrFlow.events.slice().sort((a,b)=>new Date(a.timestamp)-new Date(b.timestamp));
      tl.innerHTML = "";
      if (!events.length) {
        tl.innerHTML = `<li class="text-gray-500">타임라인 데이터 없음</li>`;
        return;
      }
      events.forEach(ev => {
        const li = document.createElement("li");
        li.className = "mb-3 relative pl-4 text-xs";
        li.innerHTML = `
          <div class="absolute -left-1.5 w-2.5 h-2.5 rounded-full bg-indigo-500"></div>
          <time class="text-gray-500">${ev.timestamp}</time>
          <div class="font-medium">${ev.src_ip} → ${ev.dst_ip} (${ev.technique})</div>
        `;
        tl.appendChild(li);
      });
    }
  } catch (e) {
    console.error("timeline error", e);
    tl.innerHTML = `<li class="text-red-500">타임라인 로드 실패</li>`;
  }
}

// Events table
let EVENTS_CACHE = []; let PAGE = 1; const SIZE = 10;
function renderEventsPage() {
  const tbody = qs("#eventsTable tbody");
  tbody.innerHTML = "";
  const totalPages = Math.max(1, Math.ceil(EVENTS_CACHE.length / SIZE));
  PAGE = Math.min(PAGE, totalPages);
  const slice = EVENTS_CACHE.slice((PAGE - 1) * SIZE, (PAGE) * SIZE);
  slice.forEach(ev => {
    const tr = document.createElement("tr"); tr.className = "border-t hover:bg-gray-50 cursor-pointer"; tr.dataset.eventId = ev.id;
    tr.innerHTML = `
      <td class="p-2">${ev.timestamp}</td>
      <td class="p-2">${ev.src_ip}</td>
      <td class="p-2">${ev.src_port ?? "-"}</td>
      <td class="p-2">${ev.dst_ip}</td>
      <td class="p-2">${ev.dst_port ?? "-"}</td>
      <td class="p-2">${ev.tatic || ev.tactic || "-"}</td>
      <td class="p-2">${ev.technique || "-"} ${ev.tid ? `(${ev.tid})` : ""}</td>
    `;
    tr.addEventListener("dblclick", async () => {
      try {
        const res = await fetch(`/api/event_related/${encodeURIComponent(ev.id)}`);
        if (!res.ok) return;
        openEventModal(await res.json());
      } catch(e){ console.error(e); }
    });
    tbody.appendChild(tr);
  });
  qs("#pageInfo").textContent = `${PAGE} / ${totalPages}`;
}
async function loadEvents(query = "") {
  const q = query || buildQuery();
  const res = await fetch(`/api/events${q}`);
  EVENTS_CACHE = await res.json();
  PAGE = 1; renderEventsPage();
  // 페이지 버튼
  qs("#prevPage").onclick = () => { if (PAGE > 1) { PAGE--; renderEventsPage(); } };
  qs("#nextPage").onclick = () => { if (PAGE < Math.max(1, Math.ceil(EVENTS_CACHE.length / SIZE))) { PAGE++; renderEventsPage(); } };
}

// Event modal
(function initEventModal() {
  const modal = qs("#eventModal"); if (!modal) return;
  const title = qs("#eventModalTitle"); const body = qs("#eventModalBody");
  qs("#eventModalClose").addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", e => { if (e.target === modal) modal.classList.add("hidden"); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") modal.classList.add("hidden"); });

  window.openEventModal = function(data) {
    const base = data.base || {}; const related = data.related || [];
    title.textContent = `Raw 이벤트 상세 (${base.id || "-"})`;
    body.innerHTML = `
      <div class="grid grid-cols-2 gap-4 text-sm mb-4">
        <div><span class="text-gray-500">timestamp</span><div class="font-medium">${base.timestamp || "-"}</div></div>
        <div><span class="text-gray-500">src_ip</span><div class="font-medium">${base.src_ip || "-"}</div></div>
        <div><span class="text-gray-500">dst_ip</span><div class="font-medium">${base.dst_ip || "-"}</div></div>
        <div><span class="text-gray-500">technique</span><div class="font-medium">${base.technique || "-"}</div></div>
      </div>
      <div class="border rounded-xl bg-gray-50 text-xs p-2 mb-3 overflow-auto max-h-[40vh]">
        <pre>${JSON.stringify(base.raw || base, null, 2)}</pre>
      </div>
      <h4 class="font-semibold mb-2">같은 src_ip (${base.src_ip || "-"})의 이벤트 목록</h4>
      <div class="overflow-auto max-h-[40vh] border rounded-lg">
        <table class="min-w-full text-xs">
          <thead class="bg-gray-100 text-left">
            <tr>
              <th class="p-1">timestamp</th>
              <th class="p-1">dst_ip</th>
              <th class="p-1">tactic</th>
              <th class="p-1">technique</th>
            </tr>
          </thead>
          <tbody>
            ${
              related.length ? related.map(r => `
                <tr class="border-t hover:bg-gray-50">
                  <td class="p-1">${r.timestamp}</td>
                  <td class="p-1">${r.dst_ip}</td>
                  <td class="p-1">${r.tatic || "-"}</td>
                  <td class="p-1">${r.technique || "-"}</td>
                </tr>
              `).join("") : `<tr><td colspan="4" class="p-2 text-gray-400 text-center">연관 이벤트 없음</td></tr>`
            }
          </tbody>
        </table>
      </div>
    `;
    modal.classList.remove("hidden"); modal.scrollTop = 0;
  };
})();

// Export PNG
qs("#btnExportPng").onclick = () => {
  if (!CURRENT_SVG) { alert("저장할 그래프가 없습니다."); return; }
  try {
    const svgNode = CURRENT_SVG.node();
    const xml = new XMLSerializer().serializeToString(svgNode);
    const svg64 = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xml)));
    const img = new Image();
    img.onload = () => {
      const vb = svgNode.viewBox.baseVal;
      const w = vb && vb.width ? vb.width : svgNode.getBoundingClientRect().width;
      const h = vb && vb.height ? vb.height : svgNode.getBoundingClientRect().height;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(800, Math.floor(w));
      canvas.height = Math.max(400, Math.floor(h));
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#F9FAFB"; ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = `${CURRENT_FLOW ? (CURRENT_FLOW.title || "flow") : "flow"}.png`;
      a.click();
    };
    img.src = svg64;
  } catch(e) {
    console.error("export png error", e);
    alert("이미지 저장 실패");
  }
};

// ==============================
// Refresh orchestration
// ==============================
let LIVE_TIMER = null;

async function refreshAll(forceOnce = false) {
  // Topbar 메트릭 먼저
  await updateTopbarMetrics();

  const q = buildQuery();
  await loadHeatmap(q);
  await loadMiniHeatmap(q);
  await loadEvents(q);
  await drawFlow(q);

  // 실시간 자동 갱신 설정/해제
  const { mode } = currentRange();
  if (mode === "live") {
    if (!LIVE_TIMER) {
      LIVE_TIMER = setInterval(async () => {
        await updateTopbarMetrics();
        const qq = buildQuery();
        await loadHeatmap(qq);
        await loadMiniHeatmap(qq);
        await loadEvents(qq);
        await drawFlow(qq);
      }, 5000);
    }
  } else {
    if (LIVE_TIMER) { clearInterval(LIVE_TIMER); LIVE_TIMER = null; }
  }
}

// ==============================
// Init
// ==============================
document.addEventListener("DOMContentLoaded", () => {
  // 초기값: 실시간
  const quick = qs("#quickRange"); if (quick) quick.value = "realtime";
  const s = qs("#startDate"); const e = qs("#endDate");
  if (s && e) { s.value = ""; e.value = ""; s.disabled = true; e.disabled = true; }
  refreshAll();
});
