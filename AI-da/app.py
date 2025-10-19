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

def filter_period(events, start, end, mode=None):
    if mode == "all":
        return events
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
    mode = request.args.get("mode")
    start = request.args.get("start")
    end = request.args.get("end")

    # TTP 정보 로드
    ttps = []
    if os.path.exists(TTP_FILE):
        with open(TTP_FILE, "r", encoding="utf-8") as f:
            ttps = json.load(f)

    # 모든 이벤트 로드 (events.json들 기준)
    all_events = filter_period(load_events_files(), start, end, mode)

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

# ==============================
# Metrics
# ==============================
@app.route("/api/metrics")
def api_metrics():
    start, end = request.args.get("start"), request.args.get("end")

    events = filter_period(load_events_files(), start, end)
    count = len(events)

    # 위험도 매핑
    BASE_RISK_SCORES = {
        "T1595": 20, "Active Scanning": 20,
        "T1046": 16, "Network Service Discovery": 16,
        "T1598": 36, "Phishing for Information": 36,
        "T1593": 5,  "Search Open Websites/Domains": 5,
        "T1583": 18, "Obtain Infrastructure": 18,
        "T1588": 16, "Acquire Capabilities": 16,
        "T1190": 36, "Exploit Public-Facing Application": 36,
        "T1189": 27, "Drive-by Compromise": 27,
        "T1566.001": 48, "Spearphishing Attachment": 48,
        "T1566.002": 48, "Spearphishing Link": 48,
        "T1566": 48, "Phishing": 48,
        "T1059": 48, "Command and Scripting Interpreter": 48,
        "T1204": 30, "User Execution": 30,
        "T1136": 36, "Create Account": 36,
        "T1053": 36, "Scheduled Task/Job": 36,
        "T1547": 48, "Boot or Logon Autostart Execution": 48,
        "T1068": 60, "Exploitation for Privilege Escalation": 60,
        "T1078": 48, "Valid Accounts": 48,
        "T1027": 80, "Obfuscated Files or Information": 80,
        "T1099": 24, "Timestomping": 24,
        "T1014": 50, "Rootkit": 50,
        "T1110": 36, "Brute Force": 36,
        "T1555.003": 27, "Credentials from Web Browsers": 27,
        "T1557": 36, "Man-in-the-Middle (Network)": 36,
        "T1049": 16, "System Network Connections Discovery": 16,
        "T1135": 27, "Network Share Discovery": 27,
        "T1595.002": 16, "Network Service Scanning (Web)": 16,
        "T1021": 36, "Remote Services": 36,
        "T1558": 40, "Pass the Ticket / Kerberos Abuse": 40,
        "T1021.002": 36, "SMB/Windows Admin Shares": 36,
        "T1039": 36, "Data from Network Shared Drive": 36,
        "T1056": 36, "Input Capture (Keylogging, Screen Capture)": 36,
        "T1123": 36, "API Monitoring": 36,
        "T1537": 36, "Cloud Storage Access": 36,
        "T1071": 64, "Application Layer Protocol": 64,
        "T1071.004": 36, "DNS Tunneling": 36,
        "T1041": 60, "Exfiltration Over C2 Channel": 60,
        "T1485": 48, "Data Destruction / Wiper": 48,
        "T1486": 75, "Encrypt Files for Impact (Ransomware)": 75,
        "T1489": 48, "Service Stop/Disable": 48,
        "T1498": 36, "Network Denial of Service": 36,
        "Hidden/Custom Protocol": 36,
        "Domain Fronting": 36,
        "Exfiltration (DNS/ICMP)": 48,
        "Upload to Cloud Storage": 60
    }

    # 위험도 계산
    def calc_period_risk(events):
        if not events:
            return None

        for e in events:
            tid = e.get("tid")
            tech = e.get("technique")
            score = None
            if tid and tid in BASE_RISK_SCORES:
                score = BASE_RISK_SCORES[tid]
            elif tech and tech in BASE_RISK_SCORES:
                score = BASE_RISK_SCORES[tech]
            else:
                score = 20  # 기본 Low 점수
            e["risk_score"] = score

        scores = [e["risk_score"] for e in events if e.get("risk_score") is not None]
        if not scores:
            return None

        n = len(scores)
        high_count = sum(1 for s in scores if s >= 71)
        weighted = sum(scores) / n
        freq_factor = 1 + min(0.2, n / 1000)
        if high_count >= 1 and weighted < 70:
            weighted = 70
        return round(weighted * freq_factor, 1)

    score = calc_period_risk(events)

    if score >= 71:
        level = "high"
    elif score >= 31:
        level = "middle"
    elif score > 0:
        level = "low"
    else:
        level = "none"

    return jsonify({
        "score": score,
        "level": level,
        "count": count
    })


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
    ddos_events = [e for e in events if e.get("technique") == "Network Denial of Service" and e.get("tid") == "T1498"]

    if ddos_events:
        target_ip = ddos_events[0].get("dst_ip")
        count = len(ddos_events)
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
