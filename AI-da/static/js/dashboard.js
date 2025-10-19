const RISK_COLORS = { low: "#FFC107", middle: "#FF6F3C", high: "#B22222", none: "#E3E7F1" };
const qs  = (sel, root=document) => root.querySelector(sel);
const qsa = (sel, root=document) => Array.from(root.querySelectorAll(sel));
const fmtDate = d => d.toISOString().slice(0,10);

function currentRange() {
  const s = qs("#startDate")?.value || "";
  const e = qs("#endDate")?.value || "";
  const quick = qs("#quickRange")?.value || "realtime";
  const mode = (quick === "realtime") ? "live" : "range";
  return { start: s, end: e, mode };
}
function buildQuery() {
  const { start, end, mode } = currentRange();
  if (mode === "live") return "?mode=live";
  if (start && end) return `?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&mode=range`;
  return "?mode=range";
}
function buildRangeParams() {
  const { start, end } = currentRange();
  const p = new URLSearchParams();
  if (start) p.set("start", start);
  if (end) p.set("end", end);
  return p.toString();
}

// ==============================
// 상단바
// ==============================
(function initTopbar(){
  const quick = qs("#quickRange");
  const s = qs("#startDate"), e = qs("#endDate");
  const applyBtn = qs("#applyRange");
  const helpIcon = qs("#helpIcon"), helpTooltip = qs("#helpTooltip");

  quick.addEventListener("change", () => {
    const today = new Date();
    const end = fmtDate(today);
    let start = end;
    const DAY = 24*60*60*1000;
    if (quick.value === "realtime") {
      s.value = ""; e.value = ""; s.disabled = true; e.disabled = true;
    } else {
      s.disabled = false; e.disabled = false;
      if (quick.value === "24h") start = fmtDate(new Date(today.getTime() - DAY));
      else if (quick.value === "7d") start = fmtDate(new Date(today.getTime() - 7*DAY));
      else if (quick.value === "30d") start = fmtDate(new Date(today.getTime() - 30*DAY));
      if (quick.value !== "custom") { s.value = start; e.value = end; }
    }
  });

  // tooltip hover
  let tipVisible = false;
  helpIcon?.addEventListener("mouseenter", () => { tipVisible = true; helpTooltip.classList.remove("invisible","opacity-0"); });
  helpIcon?.addEventListener("mouseleave", () => { tipVisible = false; helpTooltip.classList.add("invisible","opacity-0"); });
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

  window.updateTopbarMetrics = async function(){
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
(function initTabs(){
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
      if ("#"+panel.id === targetId) { panel.classList.remove("hidden"); panel.classList.add("block"); }
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
  const maxRows = Math.max(...order.map(tac => (tactics[tac]||[]).length));
  const pick = n => n >= 100 ? "#494CA2" : n >= 10 ? "#8186D5" : n >= 1 ? "#C6CBEF" : "#E3E7F1";

  for (let r=0; r<maxRows; r++){
    const tr = document.createElement("tr");
    order.forEach(tac=>{
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
function openTechniqueModal(tech){
  const modal = qs("#heatmapModal");
  if(!modal) return;
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

async function loadMiniHeatmap(query=""){
  const res = await fetch(`/api/heatmap${query}`);
  if (!res.ok) return;
  const data = await res.json();
  const { tactics, order } = data;
  const wrap = qs("#miniHeatmap .grid");
  if(!wrap) return;
  wrap.innerHTML = "";
  const pick = n => n >= 100 ? "#494CA2" : n >= 10 ? "#8186D5" : n >= 1 ? "#C6CBEF" : "#E3E7F1";
  order.forEach(tac=>{
    const row = document.createElement("div");
    row.className = "flex items-center gap-2";
    const label = document.createElement("div");
    label.className = "w-6 text-xs font-semibold text-gray-600";
    label.textContent = tac.split(" ").map(x=>x[0]).join("").slice(0,1);
    row.appendChild(label);
    const col = document.createElement("div");
    col.className = "flex items-center flex-wrap gap-1";
    (tactics[tac]||[]).forEach(tech=>{
      const box = document.createElement("div");
      box.className = "w-4 h-4 rounded";
      box.style.backgroundColor = pick(tech.count);
      col.appendChild(box);
    });
    row.appendChild(col);
    wrap.appendChild(row);
  });
}

async function refreshHeatmap(){
  const q = buildQuery();
  await loadHeatmap(q);
  await loadMiniHeatmap(q);
  await updateTopbarMetrics();
}

// ==============================
// Flow
// ==============================
let FLOW_STATE = { currentSession: null, page:1, pageSize: 10 };

async function refreshFlow(){
  const rangeQS = buildRangeParams();
  const url = `/api/sessions${rangeQS ? ("?"+rangeQS) : ""}`;
  const res = await fetch(url);
  const flowControls = qs("#flowControls");
  flowControls.innerHTML = "";
  if (!res.ok) {
    flowControls.innerHTML = `<div class="text-xs text-red-600">세션을 불러올 수 없습니다.</div>`;
    return;
  }
  const sessions = await res.json();

  sessions.forEach(s=>{
    const btn = document.createElement("button");
    btn.className = "px-3 py-1 border rounded bg-white hover:bg-gray-50 text-xs";
    btn.textContent = `${s.label} (${s.count}건)`;
    btn.dataset.sid = s.id;
    btn.addEventListener("click", async ()=> {
      await openSession(s.id);
    });
    flowControls.appendChild(btn);
  });

  const singlesRes = await fetch(`/api/single_events${rangeQS ? ("?"+rangeQS) : ""}`);
  if (singlesRes.ok) {
    const singles = await singlesRes.json();
    const btn = document.createElement("button");
    btn.className = "px-3 py-1 border rounded border-indigo-600 hover:bg-indigo-400 text-xs ml-2";
    btn.textContent = `단일공격 조회 (${singles.count}건)`;
    btn.addEventListener("click", () => {
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
  }
}

async function openSession(sid){
  const rangeQS = buildRangeParams();
  const url = `/api/session/${sid}${rangeQS ? ("?"+rangeQS) : ""}`;
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
}

function populateEventsTable(events, page=1, per_page=10){
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
  pageEvents.forEach(ev=>{
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
    tr.addEventListener("dblclick", ()=> openEventModal(JSON.parse(tr.dataset.raw)));
    tbody.appendChild(tr);
  });

  pageInfo.textContent = `${page} / ${totalPages}`;
  prev.onclick = ()=>{ if(page > 1) populateEventsTable(events, page-1, per_page); };
  next.onclick = ()=>{ if(page < totalPages) populateEventsTable(events, page+1, per_page); };
}

async function loadSessionEvents(sid){
  const rangeQS = buildRangeParams();
  const url = `/api/session/${sid}${rangeQS ? ("?"+rangeQS) : ""}`;
  const res = await fetch(url);
  if (!res.ok) return console.warn("이벤트 목록 실패:", res.status);
  const session = await res.json();
  if (!session.events) return;
  populateEventsTable(session.events);
}

function renderSingleEventsTable(events){
  qs("#flowCanvas").innerHTML = `<div class="text-xs text-gray-600 p-3">단일 이벤트 목록 (그래프는 공격플로우만 표시됩니다)</div>`;
  qs("#timeline").innerHTML = `<li class="text-gray-500">단일공격은 타임라인이 표시되지 않습니다.</li>`;
  populateEventsTable(events);
}

function renderTimeline(events){
  const timelineEl = qs("#timeline");
  if (!events || events.length === 0) {
    timelineEl.innerHTML = `<li class="text-gray-500">표시할 타임라인이 없습니다.</li>`;
    return;
  }
  const html = (events || []).map(ev => `
    <li class="pl-2 border-l-2 border-gray-300 mb-2">
      <div class="text-gray-600 text-xs">${ev.timestamp}</div>
      <div class="font-medium">${ev.src_ip || "-"} → ${ev.dst_ip || "-"} ${ev.tid ? '['+ev.tid+']' : ''} ${ev.technique || ''}</div>
    </li>`).join("");
  timelineEl.innerHTML = html;
}

function openEventModal(raw){
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
  qs("#eventModalClose").onclick = ()=> modal.classList.add("hidden");
}

function renderGraph(session) {
  const flowCanvas = qs("#flowCanvas");
  flowCanvas.innerHTML = "";
  const width = flowCanvas.clientWidth || 900;
  const height = flowCanvas.clientHeight || 520;

  const svg = d3.select(flowCanvas).append("svg").attr("width", width).attr("height", height).style("background","#fff");
  const container = svg.append("g");

  let nodes = (session.nodes || []).map(n => ({ id: n.id, color: n.color || "#dc2626", count: n.count || 1 }));
  let links = (session.edges || []).map(e => ({ source: e.source, target: e.target, count: e.count || 1 }));

  const degIn = {}, degOut = {};
  links.forEach(l => { degOut[l.source] = (degOut[l.source]||0)+1; degIn[l.target] = (degIn[l.target]||0)+1; });
  const startNode = nodes.find(n => !degIn[n.id] && degOut[n.id])?.id || null;
  const endNode = nodes.find(n => !degOut[n.id] && degIn[n.id])?.id || null;

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d=>d.id).distance(140).strength(0.8))
    .force("charge", d3.forceManyBody().strength(-700))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collide", d3.forceCollide(40))
    .alphaDecay(0.04);

  let tooltip = d3.select(flowCanvas).append("div")
    .attr("id","graphTooltip")
    .style("position","absolute")
    .style("background","rgba(0,0,0,0.75)")
    .style("color","#fff")
    .style("padding","6px 8px")
    .style("border-radius","6px")
    .style("font-size","12px")
    .style("pointer-events","none")
    .style("opacity",0);

  function showTip(ev, text){
    tooltip.transition().duration(80).style("opacity",1);
    tooltip.html(text)
      .style("left",(ev.pageX+12)+"px")
      .style("top",(ev.pageY-20)+"px");
  }
  function hideTip(){ tooltip.transition().duration(120).style("opacity",0); }

  const maxStroke = 8;
  const link = container.append("g").attr("stroke","#9ca3af").attr("stroke-opacity",0.8)
    .selectAll("line").data(links).join("line")
    .attr("stroke-width", d => Math.min(maxStroke, 1 + Math.log(d.count+1)*2))
    .on("mouseover", (ev,d) => showTip(ev, `Count: ${d.count}건`))
    .on("mouseout", hideTip);

  const nodeG = container.append("g").selectAll("g").data(nodes).join("g")
    .attr("class","node-group")
    .style("cursor","pointer")
    .call(d3.drag()
      .on("start", (event,d)=>{ if(!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (event,d)=>{ d.fx = event.x; d.fy = event.y; })
      .on("end",   (event,d)=>{ if(!event.active) simulation.alphaTarget(0); /* keep fixed pos? allow release */ d.fx = null; d.fy = null; })
    )
    .on("mouseover", (ev,d) => showTip(ev, d.id))
    .on("mouseout", hideTip)
    .on("click", async (ev,d) => {
      if (!session || !session.id) return;
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
    .attr("text-anchor","middle")
    .attr("font-size", 14)
    .attr("font-weight","700")
    .text(d => {
      if (d.id === startNode) return "S";
      if (d.id === endNode) return "E";
      return "";
    });

  nodeG.append("text")
    .attr("dy", 42)
    .attr("text-anchor","middle")
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

  svg.call(d3.zoom().scaleExtent([0.5,3]).on("zoom", ev => container.attr("transform", ev.transform)));
}

async function updateTimeline(sid, nodeId=null){
  const session = FLOW_STATE.currentSession;
  if (!session) return;
  if (!session.events) { renderTimeline([]); return; }
  if (!nodeId) { renderTimeline(session.events.slice(0, 50)); return; }
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

async function refreshAll(){
  await Promise.all([ refreshHeatmap(), updateTopbarMetrics() ]);
  await refreshFlow();
}

(async function init(){
  const today = new Date();
  qs("#endDate").value = fmtDate(today);
  qs("#startDate").value = fmtDate(new Date(today.getTime() - 7*24*60*60*1000));
  await refreshAll();
})();

