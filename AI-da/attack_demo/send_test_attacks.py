"""
send_test_attacks.py

공격 플로우 이벤트 생성기
각 Flow는 Recon → Impact 까지의 단계적 공격 과정을 생성

사용 예:
  python send_test_attacks_ai.py --mode flow --flows 3 --events-per-flow 10 --host http://127.0.0.1:5500 --endpoint /predict
"""

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timedelta
import requests
import os

# ----------------------------------------
# 기본 TTP fallback
# ----------------------------------------
FALLBACK_TTPS = [
    {"tid": "T1595", "technique": "Active Scanning", "tactic": "Reconnaissance"},
    {"tid": "T1583", "technique": "Obtain Infrastructure", "tactic": "Resource Development"},
    {"tid": "T1566.001", "technique": "Spearphishing Attachment", "tactic": "Initial Access"},
    {"tid": "T1059", "technique": "Command and Scripting Interpreter", "tactic": "Execution"},
    {"tid": "T1053", "technique": "Scheduled Task/Job", "tactic": "Persistence"},
    {"tid": "T1068", "technique": "Exploitation for Privilege Escalation", "tactic": "Privilege Escalation"},
    {"tid": "T1027", "technique": "Obfuscated Files or Information", "tactic": "Defense Evasion"},
    {"tid": "T1110", "technique": "Brute Force", "tactic": "Credential Access"},
    {"tid": "T1049", "technique": "System Network Connections Discovery", "tactic": "Discovery"},
    {"tid": "T1021", "technique": "Remote Services (RDP, SSH, SMB)", "tactic": "Lateral Movement"},
    {"tid": "T1039", "technique": "Data from Network Shared Drive", "tactic": "Collection"},
    {"tid": "T1071", "technique": "Application Layer Protocol", "tactic": "Command and Control"},
    {"tid": "T1041", "technique": "Exfiltration Over C2 Channel", "tactic": "Exfiltration"},
    {"tid": "T1486", "technique": "Encrypt Files for Impact (Ransomware)", "tactic": "Impact"},
]

# ----------------------------------------
# Utility
# ----------------------------------------
def load_ttps(path):
    """TTPs.json 파일을 읽어서 tactic, tid, technique 추출"""
    if not path or not os.path.exists(path):
        return FALLBACK_TTPS

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for group in data:
            tactic = group.get("tactic")
            techs = group.get("techniques", [])
            if not tactic or not isinstance(techs, list):
                continue
            for t in techs:
                tid = t.get("tid")
                tech = t.get("technique")
                if tid and tech:
                    result.append({
                        "tactic": tactic,
                        "tid": tid,
                        "technique": tech
                    })
        if result:
            print(f"[load_ttps] Parsed {len(result)} techniques from {path}")
            return result
    except Exception as e:
        print("[load_ttps] Warning:", e)

    return FALLBACK_TTPS


def now_iso():
    return datetime.now().astimezone().isoformat()

def random_kr_ip():
    prefixes = ["14.", "121.", "175.", "112.", "211.", "122."]
    p = random.choice(prefixes)
    return p + ".".join(str(random.randint(0,255)) for _ in range(3))

def random_private_ip():
    return "10." + ".".join(str(random.randint(0,255)) for _ in range(3))

def make_event(timestamp, src_ip, src_port, dst_ip, dst_port, protocol, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, tid, technique, tactic):
    return {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "protocol": protocol,
        "duration": round(random.uniform(0.05, 300.0), 2),
        "fwd_pkts": fwd_pkts,
        "bwd_pkts": bwd_pkts,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "tid": tid,
        "technique": technique,
        "tactic": tactic
    }

# ----------------------------------------
# Flow generation (단계적 공격 흐름)
# ----------------------------------------
def generate_attack_flows(ttps, flows=3, events_per_flow=10):
    # MITRE ATT&CK의 대표적인 단계 순서
    tactics_order = [
        "Reconnaissance", "Resource Development", "Initial Access", "Execution",
        "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access",
        "Discovery", "Lateral Movement", "Collection", "Command and Control", "Exfiltration", "Impact"
    ]

    # tactic별 기법 묶기
    ttps_by_tactic = {t["tactic"]: [] for t in ttps if t.get("tactic")}
    for t in ttps:
        if t.get("tactic"):
            ttps_by_tactic.setdefault(t["tactic"], []).append(t)

    events = []
    for f in range(flows):
        attacker = random_kr_ip()
        victim = random_private_ip()
        start = datetime.now() - timedelta(minutes=random.randint(0, 300))
        seq = []

        for i in range(events_per_flow):
            # tactic 순서 순회
            tactic = tactics_order[i % len(tactics_order)]
            techs = ttps_by_tactic.get(tactic, FALLBACK_TTPS)
            t = random.choice(techs)
            ts = (start + timedelta(seconds=i * random.randint(20, 300))).astimezone().isoformat()

            src_port = random.randint(1024, 65535)
            dst_port = random.choice([22, 80, 443, 8080, 3389, 445, 53, 8443])
            protocol = random.choice(["TCP", "UDP", "HTTP", "HTTPS"])
            fwd_pkts = random.randint(10, 1000)
            bwd_pkts = random.randint(1, max(1, fwd_pkts // 2))
            fwd_bytes = fwd_pkts * random.randint(40, 1500)
            bwd_bytes = bwd_pkts * random.randint(40, 1500)

            seq.append(make_event(
                ts, attacker, src_port, victim, dst_port, protocol,
                fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
                t.get("tid", "T0000"), t.get("technique", "Unknown"), tactic
            ))
        events.extend(seq)

        print(f"Generated flow #{f+1}: {attacker} -> {victim} ({len(seq)} events)")

    return events

# ----------------------------------------
# Batch send
# ----------------------------------------
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

# ----------------------------------------
# Main
# ----------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:5500")
    p.add_argument("--endpoint", default="/predict")
    p.add_argument("--flows", type=int, default=3)
    p.add_argument("--events-per-flow", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--ttps-path", default="TTPs.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ttps = load_ttps(args.ttps_path)
    events = generate_attack_flows(ttps, flows=args.flows, events_per_flow=args.events_per_flow)

    print(f"Total {len(events)} events generated; sending to {args.host}{args.endpoint}")
    sent = send_batches(args.host, args.endpoint, events, args.batch_size, args.delay, args.dry_run)
    print("Done. Events sent:", sent)

if __name__ == "__main__":
    main()
