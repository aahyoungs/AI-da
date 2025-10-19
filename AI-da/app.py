from flask import Flask, jsonify, render_template, request
import os, json, glob
from datetime import datetime, time
from dateutil import parser as dateparser, tz
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENT_DIR = os.path.join(DATA_DIR, "events")
SESSIONS_FILE = os.path.join(EVENT_DIR, "sessions.json")
TTP_FILE = os.path.join(DATA_DIR, "TTPs.json")
ASSETS_FILE = os.path.join(DATA_DIR, "assets.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

TZ_SEOUL = tz.gettz("Asia/Seoul")

def to_aware(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ_SEOUL)
    return dt

def parse_range(start, end):
    s, e = None, None
    try:
        if start:
            d = dateparser.parse(start)
            s = datetime.combine(d.date(), time(0, 0, 0)).replace(tzinfo=TZ_SEOUL)
        if end:
            d = dateparser.parse(end)
            e = datetime.combine(d.date(), time(23, 59, 59, 999999)).replace(tzinfo=TZ_SEOUL)
    except:
        pass
    return s, e

def load_assets():
    if not os.path.exists(ASSETS_FILE):
        return set()
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return set(data)
            if isinstance(data, dict):
                return set(data.get("assets", []))
    except:
        pass
    return set()

ASSETS = load_assets()

def normalize(events):
    out = []
    for i, ev in enumerate(events):
        e = dict(ev)
        ts = e.get("timestamp")
        try:
            e["_ts"] = to_aware(dateparser.parse(ts)) if ts else None
        except:
            e["_ts"] = None
        if "id" not in e:
            e["id"] = f"ev_{i}"
        out.append(e)
    return sorted(out, key=lambda x: x["_ts"] or datetime.min.replace(tzinfo=TZ_SEOUL))

def filter_period(events, start, end):
    s, e = parse_range(start, end)
    if not s and not e:
        return events
    res = []
    for ev in events:
        t = ev.get("_ts")
        if not t:
            continue
        if s and t < s: continue
        if e and t > e: continue
        res.append(ev)
    return res

def load_events_files():
    evs = []
    for path in glob.glob(os.path.join(EVENT_DIR, "*.json")):
        if os.path.basename(path) == "sessions.json":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
                if isinstance(doc, list):
                    evs.extend(doc)
                elif isinstance(doc, dict):
                    evs.append(doc)
        except:
            pass
    return normalize(evs)

def load_sessions():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except:
        return {}

# ==============================
# Heatmap
# ==============================
@app.route("/api/heatmap")
def api_heatmap():
    start, end = request.args.get("start"), request.args.get("end")

    ttps = []
    if os.path.exists(TTP_FILE):
        with open(TTP_FILE, "r", encoding="utf-8") as f:
            ttps = json.load(f)

    sessions = load_sessions()
    all_events = []
    for sdata in sessions.values():
        evs = normalize(sdata.get("events", []))
        all_events.extend(filter_period(evs, start, end))
    file_events = load_events_files()
    all_events.extend(filter_period(file_events, start, end))

    defined_techs = {tech["technique"] for ttp in ttps for tech in ttp["techniques"]}

    tech_count = defaultdict(int)
    unknown_count = 0

    for ev in all_events:
        t = ev.get("technique")
        if not t:
            unknown_count += 1
            continue
        if t not in defined_techs:
            unknown_count += 1
        else:
            tech_count[t] += 1

    tactics, order = {}, []
    for ttp in ttps:
        tac = ttp["tactic"]
        order.append(tac)
        tactics[tac] = []
        for tech in ttp["techniques"]:
            name = tech["technique"]
            tactics[tac].append({
                "technique": name,
                "tid": tech.get("tid"),
                "description": tech.get("description"),
                "count": tech_count.get(name, 0)
            })

    return jsonify({
        "tactics": tactics,
        "order": order,
        "unknown_count": unknown_count
    })


@app.route("/api/metrics")
def api_metrics():
    start, end = request.args.get("start"), request.args.get("end")
    sessions = load_sessions()
    events = []
    for sdata in sessions.values():
        evs = normalize(sdata.get("events", []))
        events.extend(filter_period(evs, start, end))
    events.extend(filter_period(load_events_files(), start, end))
    count = len(events)
    score = 0
    for e in events:
        t = (e.get("technique") or "").lower()
        if "phish" in t: score += 300
        elif "exfil" in t: score += 200
        else: score += 100
    level = "none"
    if score >= 1000: level = "high"
    elif score >= 500: level = "middle"
    elif score >= 100: level = "low"
    return jsonify({"score": score, "level": level, "count": count})

# ==============================
# Flow
# ==============================
@app.route("/api/sessions")
def api_sessions():
    start, end = request.args.get("start"), request.args.get("end")
    sessions = load_sessions()
    res = []
    for sid, sdata in sessions.items():
        evs = normalize(sdata.get("events", []))
        filtered = filter_period(evs, start, end)
        if not filtered:
            continue
        res.append({
            "id": sid,
            "label": f"공격플로우-{sid.split('-')[-1]}",
            "count": len(filtered)
        })
    return jsonify(res)


@app.route("/api/session/<sid>")
def api_session_detail(sid):
    start, end = request.args.get("start"), request.args.get("end")
    sessions = load_sessions()
    sdata = sessions.get(sid)
    if not sdata:
        return jsonify({"error": "not found"}), 404

    events = filter_period(normalize(sdata.get("events", [])), start, end)
    # DDoS
    ddos_events = [e for e in events if e.get("technique") == "Network Denial of Service" and e.get("tid") == "T1498"]
    if ddos_events:
        target_ip = ddos_events[0].get("dst_ip")
        count = len(ddos_events)
        # 그래프에 출발지 1, 목적지 1만 표시
        nodes = [
            {"id": f"{ddos_events[0].get('src_ip')} 외 {count-1}건", "color": "#dc2626", "count": count},
            {"id": target_ip, "color": "#2563eb" if target_ip in ASSETS else "#dc2626", "count": count}
        ]
        edges = [{"source": nodes[0]["id"], "target": target_ip, "count": count}]
    else:
        nodes, edges = {}, defaultdict(int)
        for ev in events:
            src, dst = ev.get("src_ip"), ev.get("dst_ip")
            if not src or not dst:
                continue
            edges[(src, dst)] += 1
            for ip in [src, dst]:
                if ip not in nodes:
                    nodes[ip] = {
                        "id": ip,
                        "color": "#2563eb" if ip in ASSETS else "#dc2626",
                        "count": 0
                    }
                nodes[ip]["count"] += 1
        edges = [{"source": s, "target": t, "count": c} for (s, t), c in edges.items()]
        nodes = list(nodes.values())

    return jsonify({
        "id": sid,
        "label": f"공격플로우-{sid.split('-')[-1]}",
        "nodes": nodes,
        "edges": edges,
        "events": events
    })

@app.route("/api/single_events")
def api_single_events():
    """sessions.json에 속하지 않은 events.json 내 단일 이벤트"""
    start, end = request.args.get("start"), request.args.get("end")
    sessions = load_sessions()
    used = set()
    def key(ev): return (ev.get("timestamp"), ev.get("src_ip"), ev.get("dst_ip"))
    for sdata in sessions.values():
        for ev in sdata.get("events", []):
            used.add(key(ev))
    all_events = filter_period(load_events_files(), start, end)
    singles = [e for e in all_events if key(e) not in used]
    return jsonify({"count": len(singles), "events": singles})

@app.route("/")
def index():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5500)
