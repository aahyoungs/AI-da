from flask import Flask, render_template, jsonify, request, Response
import random, datetime, io, csv, os, json
from collections import defaultdict

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
TTP_PATH = os.path.join(BASE_DIR, "data", "TTPs.json")

# 실시간 데모용 (라이브 모드에서 매 호출 생성)
def load_ttps():
    with open(TTP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    tactics = {}
    for t in data:
        tactics[t["tactic"]] = [{**tech, "count": 0, "events": []} for tech in t["techniques"]]
    return tactics

def generate_live_events(n=30):
    """실시간 모드: 단일 이벤트들 (랜덤)"""
    tactics = load_ttps()
    alltech = []
    for tac, techs in tactics.items():
        for tech in techs:
            alltech.append((tech["tid"], tech["technique"], tac))
    now = datetime.datetime.now()
    events = []
    for i in range(n):
        tid, tech, tac = random.choice(alltech)
        ts = now - datetime.timedelta(seconds=random.randint(0, 300))
        src = f"203.0.113.{random.randint(1, 254)}"
        dst = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
        events.append({
            "id": f"live-{i}",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "tid": tid, "technique": tech, "tatic": tac,
            "src_ip": src, "dst_ip": dst,
            "src_port": random.choice([1234,5678,8080,53,25]),
            "dst_port": random.choice([22,80,443,445,3389]),
            "protocol": random.choice(["TCP","UDP"]),
            "risk": random.choice(["low","middle","high"]),
            "raw": {"host": "demo", "payload": "sample packet"},
        })
    events.sort(key=lambda e: e["timestamp"])
    return events

def sample_attack_flows():
    """기간 모드: 공격플로우 2개 + 단일공격 2개(DDoS 포함) 예시"""
    base_time = datetime.datetime(2025, 10, 18, 17, 40)
    events, idx = [], 1

    # 공격플로우 1: Recon → Lateral Movement → Exfiltration
    flow1 = [
        ("T1595", "Reconnaissance", "Active Scanning", "203.0.113.50", "192.168.10.5"),
        ("T1021", "Lateral Movement", "Remote Services (RDP, SSH, SMB)", "192.168.10.5", "192.168.20.15"),
        ("T1041", "Exfiltration", "Exfiltration Over C2 Channel", "192.168.20.15", "10.10.10.5")
    ]
    for tid, tac, tech, src, dst in flow1:
        events.append({
            "id": f"evt-{idx}",
            "timestamp": (base_time + datetime.timedelta(minutes=idx*2)).strftime("%Y-%m-%d %H:%M:%S"),
            "tid": tid, "tatic": tac, "technique": tech,
            "src_ip": src, "dst_ip": dst,
            "risk": "high" if tech.startswith("Exfiltration") else "middle",
            "raw": {"desc": f"{tac} / {tech}"},
        }); idx += 1

    # 공격플로우 2: Initial Access → Execution → Priv Esc → C2
    flow2 = [
        ("T1566.002", "Initial Access", "Spearphishing Link", "198.51.100.45", "192.168.30.10"),
        ("T1059", "Execution", "Command and Scripting Interpreter", "192.168.30.10", "192.168.30.10"),
        ("T1068", "Privilege Escalation", "Exploitation for Privilege Escalation", "192.168.30.10", "192.168.30.10"),
        ("T1071", "Command and Control", "Application Layer Protocol (HTTP/S)", "192.168.30.10", "8.8.8.8"),
    ]
    for tid, tac, tech, src, dst in flow2:
        events.append({
            "id": f"evt-{idx}",
            "timestamp": (base_time + datetime.timedelta(minutes=idx*2)).strftime("%Y-%m-%d %H:%M:%S"),
            "tid": tid, "tatic": tac, "technique": tech,
            "src_ip": src, "dst_ip": dst,
            "risk": "middle",
            "raw": {"desc": f"{tac} / {tech}"},
        }); idx += 1

    # 단일공격: DDoS
    events.append({
        "id": f"evt-{idx}",
        "timestamp": (base_time + datetime.timedelta(minutes=idx*2)).strftime("%Y-%m-%d %H:%M:%S"),
        "tid": "T1498", "tatic": "Impact", "technique": "Network Denial of Service",
        "src_ip": "203.0.113.100", "dst_ip": "10.0.0.5",
        "risk": "high", "raw": {"desc": "대량 트래픽 (DDoS 공격)"},
        "count": 150
    }); idx += 1

    # 단일공격: Brute Force
    events.append({
        "id": f"evt-{idx}",
        "timestamp": (base_time + datetime.timedelta(minutes=idx*2)).strftime("%Y-%m-%d %H:%M:%S"),
        "tid": "T1110", "tatic": "Credential Access", "technique": "Brute Force",
        "src_ip": "203.0.113.25", "dst_ip": "192.168.5.5",
        "risk": "middle", "raw": {"desc": "로그인 시도 폭주 (Brute Force)"},
        "count": 5
    })
    return events

def filter_events(mode, start_s, end_s):
    if mode == "live":
        return generate_live_events()
    # 기간 모드: 지금은 샘플 고정
    return sample_attack_flows()

def compute_metrics(events):
    weight = {"low": 1, "middle": 3, "high": 7}
    score = sum(weight.get(e.get("risk","low"), 1) for e in events)
    if score >= 300: level = "high"
    elif score >= 120: level = "middle"
    else: level = "low"
    return {"score": score, "level": level, "count": len(events)}

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/metrics")
def api_metrics():
    mode = request.args.get("mode", "live")
    start = request.args.get("start"); end = request.args.get("end")
    evs = filter_events(mode, start, end)
    return jsonify(compute_metrics(evs))

@app.route("/api/events")
def api_events():
    mode = request.args.get("mode", "live")
    start = request.args.get("start"); end = request.args.get("end")
    return jsonify(filter_events(mode, start, end))

@app.route("/api/event/<eid>")
def api_event(eid):
    all_events = sample_attack_flows() + generate_live_events()
    ev = next((e for e in all_events if e["id"] == eid), None)
    if not ev:
        return jsonify({"error": "not found"}), 404
    return jsonify(ev)

@app.route("/api/event_related/<eid>")
def api_event_related(eid):
    all_events = sample_attack_flows() + generate_live_events()
    ev = next((e for e in all_events if e["id"] == eid), None)
    if not ev:
        return jsonify({"error": "event not found"}), 404
    related = [e for e in all_events if e["src_ip"] == ev["src_ip"]]
    return jsonify({"base": ev, "related": related})

@app.route("/api/heatmap")
def api_heatmap():
    mode = request.args.get("mode", "live")
    start = request.args.get("start"); end = request.args.get("end")
    evs = filter_events(mode, start, end)

    tactics = load_ttps()
    order = list(tactics.keys())
    tid_map = {}
    for tac, techs in tactics.items():
        for i, tech in enumerate(techs):
            tid_map[(tech.get("tid") or "").lower()] = (tac, i)

    for ev in evs:
        tid = (ev.get("tid") or "").lower()
        if tid in tid_map:
            tac, i = tid_map[tid]
            tactics[tac][i]["count"] += 1
            tactics[tac][i]["events"].append(ev)

    return jsonify({"tactics": tactics, "order": order})

@app.route("/api/flow")
def api_flow():
    mode = request.args.get("mode", "live")
    evs = filter_events(mode, None, None)
    flows = []

    if mode == "live":
        # 단일 공격들만 (노드 2개 + 링크 1개 형태)
        for i, ev in enumerate(evs[:10], start=1):
            flows.append({
                "id": f"single-{i}",
                "title": f"단일공격 {i}",
                "type": "single",
                "nodes": [{"id": ev["src_ip"], "label": ev["src_ip"]},
                          {"id": ev["dst_ip"], "label": ev["dst_ip"]}],
                "edges": [{"from": ev["src_ip"], "to": ev["dst_ip"], "count": random.randint(1, 6)}],
                "events": [ev]
            })
    else:
        evs = sample_attack_flows()
        flow1 = {
            "id": "session-1",
            "title": "공격플로우 1",
            "type": "session",
            "nodes": [{"id": "203.0.113.50","label": "203.0.113.50"},
                      {"id": "192.168.10.5","label": "192.168.10.5"},
                      {"id": "192.168.20.15","label": "192.168.20.15"},
                      {"id": "10.10.10.5","label": "10.10.10.5"}],
            "edges": [{"from": "203.0.113.50","to": "192.168.10.5","count": 6},
                      {"from": "192.168.10.5","to": "192.168.20.15","count": 3},
                      {"from": "192.168.20.15","to": "10.10.10.5","count": 2}],
            "events": [e for e in evs if e["src_ip"] in ["203.0.113.50","192.168.10.5","192.168.20.15"]],
        }
        flow2 = {
            "id": "session-2",
            "title": "공격플로우 2",
            "type": "session",
            "nodes": [{"id": "198.51.100.45","label": "198.51.100.45"},
                      {"id": "192.168.30.10","label": "192.168.30.10"},
                      {"id": "8.8.8.8","label": "8.8.8.8"}],
            "edges": [{"from": "198.51.100.45","to": "192.168.30.10","count": 4},
                      {"from": "192.168.30.10","to": "8.8.8.8","count": 1}],
            "events": [e for e in evs if e["src_ip"] in ["198.51.100.45","192.168.30.10"]],
        }
        singles = [e for e in evs if e["technique"] in ["Network Denial of Service","Brute Force"]]
        for i, ev in enumerate(singles, start=1):
            flows.append({
                "id": f"single-{i}",
                "title": f"단일공격 {i}",
                "type": "single",
                "nodes": [{"id": ev["src_ip"], "label": ev["src_ip"]},
                          {"id": ev["dst_ip"], "label": ev["dst_ip"]}],
                "edges": [{"from": ev["src_ip"], "to": ev["dst_ip"], "count": ev.get("count", 1)}],
                "events": [ev],
            })
        flows = [flow1, flow2] + flows

    return jsonify({"flows": flows})

@app.route("/api/timeline/<node_id>")
def api_timeline(node_id):
    evs = sample_attack_flows() + generate_live_events()
    node_events = [e for e in evs if e["src_ip"] == node_id or e["dst_ip"] == node_id]
    node_events.sort(key=lambda e: e["timestamp"])
    timeline = [{"time": e["timestamp"], "desc": f"{e['src_ip']} → {e['dst_ip']} ({e['technique']})"} for e in node_events]
    return jsonify(timeline if timeline else [{"time": "-", "desc": "관련 이벤트 없음"}])

@app.route("/api/export_events")
def api_export_events():
    tech = request.args.get("tech", "")
    tac = request.args.get("tac", "")
    start = request.args.get("start"); end = request.args.get("end")
    mode = request.args.get("mode", "live")
    evs = filter_events(mode, start, end)
    if tech:
        evs = [e for e in evs if tech.split("(")[0].strip().lower() in (e.get("technique","").lower())]
    if tac:
        evs = [e for e in evs if tac.lower() in (e.get("tatic","").lower())]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","timestamp","src_ip","src_port","dst_ip","dst_port","tactic","technique","tid"])
    for e in evs:
        writer.writerow([e["id"], e["timestamp"], e["src_ip"], e.get("src_port",""),
                         e["dst_ip"], e.get("dst_port",""), e.get("tatic",""),
                         e.get("technique",""), e.get("tid","")])
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=events_export.csv"})

if __name__ == "__main__":
    app.run(debug=True, port=4000)
