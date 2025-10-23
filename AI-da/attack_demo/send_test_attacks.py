#!/usr/bin/env python3
"""
send_test_attacks.py

로컬 app.py (Flask) 테스트용 공격 이벤트 생성기 & 전송기.

사용 예:
  # 간단 DDoS burst 100건 생성하여 /predict 로 전송 (로컬)
  python send_test_attacks.py --mode ddos --count 100 --host http://127.0.0.1:5000

  # 여러 플로우(Recon->Discovery->Exec) 10개 생성
  python send_test_attacks.py --mode flow --flows 10 --events-per-flow 6

옵션:
  --host       서버 호스트 (기본 http://127.0.0.1:5000)
  --endpoint   예측/수신 endpoint (기본 /predict)
  --mode       ddos | flow | mixed
  --count      ddos 이벤트 개수 (mode=ddos)
  --flows      flow 세션 개수 (mode=flow)
  --events-per-flow  flow 당 이벤트 수 (mode=flow)
  --batch-size 한번에 보낼 batch size (기본 20)
  --delay      batch 간 지연초 (기본 0.2)
  --ttps-path  TTPs.json 경로(있다면 tid/technique 읽음)
  --dry-run    실제 전송 안하고 생성만
"""

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timedelta
import requests
import os

FALLBACK_TTPS = [
    {"tid": "T1595", "technique": "Active Scanning"},
    {"tid": "T1046", "technique": "Network Service Discovery"},
    {"tid": "T1498", "technique": "Network Denial of Service"},
    {"tid": "T1059", "technique": "Command and Scripting Interpreter"},
    {"tid": "T1021", "technique": "Remote Services"},
    {"tid": "T1071", "technique": "Application Layer Protocol"},
    {"tid": "T1566", "technique": "Phishing"},
]

def load_ttps(path):
    if not path:
        return FALLBACK_TTPS
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        found = []
        def walk(o):
            if isinstance(o, dict):
                if any(k.lower() in ("tid", "technique", "name") for k in o.keys()):
                    tid = o.get("tid") or o.get("TID") or o.get("id") or ""
                    tech = o.get("technique") or o.get("Technique") or o.get("name") or ""
                    if tid or tech:
                        found.append({"tid": str(tid), "technique": str(tech)})
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for e in o:
                    walk(e)
        walk(data)
        if found:
            uniq = []
            seen = set()
            for it in found:
                key = (it.get("tid",""), it.get("technique",""))
                if key not in seen:
                    seen.add(key)
                    uniq.append(it)
            return uniq
    except Exception as e:
        print("load_ttps warning:", e)
    return FALLBACK_TTPS

def now_iso():
    return datetime.now().astimezone().isoformat()

def make_event(timestamp, src_ip, src_port, dst_ip, dst_port, protocol,
               fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, tid, technique):
    return {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "protocol": protocol,
        "duration": round(random.uniform(0.01, 300.0), 2),
        "fwd_pkts": fwd_pkts,
        "bwd_pkts": bwd_pkts,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "tid": tid,
        "technique": technique
    }

def random_kr_ip():
    prefixes = ["14.", "121.", "175.", "112.", "211.", "122."]
    p = random.choice(prefixes)
    return p + ".".join(str(random.randint(0,255)) for _ in range(3))

def random_private_ip():
    return "10." + ".".join(str(random.randint(0,255)) for _ in range(3))

def generate_ddos_events(ttps, attacker_ip=None, victim_ip=None, count=100, burst_span_secs=900):
    if not attacker_ip:
        attacker_ip = random_kr_ip()
    if not victim_ip:
        victim_ip = random_private_ip()

    ddos_choices = [t for t in ttps if "ddos" in (t.get("technique") or "").lower() or "denial" in (t.get("technique") or "").lower()]
    if not ddos_choices:
        ddos_choices = [t for t in ttps if t.get("tid","").lower().startswith("t14")] or ttps
    center = datetime.now()
    evs = []
    for i in range(count):
        jitter = random.randint(-burst_span_secs//2, burst_span_secs//2)
        ts = (center + timedelta(seconds=jitter)).astimezone().isoformat()
        src_port = random.randint(1024, 65535) if random.random() < 0.9 else random.randint(1,1023)
        dst_port = random.choice([80,443,53,123,8080,8443,60000])
        protocol = random.choice(["UDP","TCP","ICMP"])
        fwd_pkts = random.randint(20, 1000)  # many packets
        bwd_pkts = max(0, fwd_pkts - random.randint(0, int(fwd_pkts*0.95)))
        fwd_bytes = fwd_pkts * random.randint(40,1500)
        bwd_bytes = bwd_pkts * random.randint(40,1500)
        t = random.choice(ddos_choices)
        evs.append(make_event(ts, attacker_ip, src_port, victim_ip, dst_port, protocol,
                              fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, t.get("tid","T1498"), t.get("technique","Network Denial of Service")))
    return evs

def generate_flow_events(ttps, flows=5, events_per_flow=5):
    def pick_by_keyword(keyword):
        return [t for t in ttps if keyword in (t.get("technique") or "").lower() or keyword in (t.get("tid","").lower())]

    recon = pick_by_keyword("scan") or pick_by_keyword("active") or pick_by_keyword("recon")
    discovery = pick_by_keyword("discovery") or pick_by_keyword("network service")
    access = pick_by_keyword("remote") or pick_by_keyword("rce") or pick_by_keyword("remote services")
    execs = pick_by_keyword("command") or pick_by_keyword("scripting") or pick_by_keyword("exec")
    app = pick_by_keyword("application") or pick_by_keyword("protocol")
    pool = [recon, discovery, access, execs, app]
    pool = [p if p else ttps for p in pool]

    events = []
    for f in range(flows):
        attacker = random_kr_ip()
        victim = random_private_ip()
        start = datetime.now() - timedelta(minutes=random.randint(0, 60*24))
        seq_len = events_per_flow
        for step in range(seq_len):
            ts = (start + timedelta(seconds=step * random.randint(20, 600))).astimezone().isoformat()
            stage = pool[min(step, len(pool)-1)]
            t = random.choice(stage)
            src_port = random.randint(1024,65535)
            dst_port = random.choice([22,80,443,8080,3306,3389,445])
            protocol = random.choice(["TCP","HTTP","HTTPS"])
            fwd_pkts = random.randint(1, 300)
            bwd_pkts = max(0, fwd_pkts - random.randint(0, int(fwd_pkts*0.8)))
            fwd_bytes = fwd_pkts * random.randint(40,1500)
            bwd_bytes = bwd_pkts * random.randint(40,1500)
            events.append(make_event(ts, attacker, src_port, victim, dst_port, protocol,
                                     fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, t.get("tid","T0000"), t.get("technique","Unknown")))
    return events

def send_batches(host, endpoint, events, batch_size=20, delay=0.2, dry_run=False):
    url = host.rstrip("/") + "/" + endpoint.lstrip("/")
    headers = {"Content-Type": "application/json"}
    sent = 0
    for i in range(0, len(events), batch_size):
        batch = events[i:i+batch_size]
        payload = {"samples": batch}
        if dry_run:
            print(f"[DRY] Would POST {len(batch)} events to {url}")
        else:
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                if r.ok:
                    print(f"[OK] Sent batch {i//batch_size + 1} ({len(batch)} events)")
                else:
                    print(f"[ERR] HTTP {r.status_code}: {r.text[:300]}")
            except Exception as e:
                print("[EXCEPTION] send error:", e)
        sent += len(batch)
        time.sleep(delay)
    return sent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:5500", help="Target host (include scheme)")
    p.add_argument("--endpoint", default="/predict", help="API endpoint to POST to")
    p.add_argument("--mode", choices=["ddos","flow","mixed"], default="mixed")
    p.add_argument("--count", type=int, default=100, help="ddos event count")
    p.add_argument("--flows", type=int, default=5, help="number of flows to generate")
    p.add_argument("--events-per-flow", type=int, default=6, help="events per flow session")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--ttps-path", default=None, help="path to TTPs.json (optional)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ttps = load_ttps(args.ttps_path)
    evts = []

    if args.mode in ("ddos", "mixed"):
        ddos_attacker = random_kr_ip()
        ddos_victim = random_private_ip()
        ddos_ev = generate_ddos_events(ttps, attacker_ip=ddos_attacker, victim_ip=ddos_victim, count=args.count)
        evts.extend(ddos_ev)
        print(f"Generated DDoS burst: {len(ddos_ev)} events from {ddos_attacker} -> {ddos_victim}")

    if args.mode in ("flow", "mixed"):
        flows = generate_flow_events(ttps, flows=args.flows, events_per_flow=args.events_per_flow)
        evts.extend(flows)
        print(f"Generated {len(flows)} flow events ({args.flows} flows)")

    random.shuffle(evts)
    print(f"Total events generated: {len(evts)}; sending to {args.host}{args.endpoint} in batches of {args.batch_size}")

    sent = send_batches(args.host, args.endpoint, evts, batch_size=args.batch_size, delay=args.delay, dry_run=args.dry_run)
    print("Done. Events sent:", sent)

if __name__ == "__main__":
    main()