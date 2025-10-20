from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
import json

# -----------------------------------------
# Flask 기본 설정
app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
EVENT_DIR = os.path.join(BASE_DIR, "data", "events")

# -----------------------------------------
# 경로 설정 (모델 파일)
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders_all.joblib")
STAGE1_MODEL_PATH = os.path.join(MODEL_DIR, "stage1_xgb.joblib")
STAGE2_MODEL_PATH = os.path.join(MODEL_DIR, "stage2_xgb.joblib")
ATTACK_MAP_PATH = os.path.join(MODEL_DIR, "attack_map.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# -----------------------------------------
# 시간대

KST = timezone(timedelta(hours=9))
def to_kst_iso(dt_like):
    if dt_like is None:
        return datetime.now(tz=KST).replace(microsecond=0).isoformat()
    if isinstance(dt_like, datetime):
        dt = dt_like
    else:
        try:
            dt = dateparser.parse(str(dt_like))
        except Exception:
            return datetime.now(tz=KST).replace(microsecond=0).isoformat()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    else:
        dt = dt.astimezone(KST)
    return dt.replace(microsecond=0).isoformat()

# -----------------------------------------
# 프로토콜 정규화

def normalize_protocol(p):
    if p is None:
        return "OTHER"
    p = str(p).strip().upper()
    if p.startswith("TCP"): return "TCP"
    if p.startswith("UDP"): return "UDP"
    if p in ("ICMP", "ICM"): return "ICMP"
    if p in ("ARP",): return "ARP"
    return "OTHER"

# -----------------------------------------
# 모델 및 인코더 로드
encoders_all = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else None
stage1_model = joblib.load(STAGE1_MODEL_PATH) if os.path.exists(STAGE1_MODEL_PATH) else None
stage2_model = joblib.load(STAGE2_MODEL_PATH) if os.path.exists(STAGE2_MODEL_PATH) else None
attack_map = joblib.load(ATTACK_MAP_PATH) if os.path.exists(ATTACK_MAP_PATH) else None
label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None

# -----------------------------------------
# LabelEncoder safe transform
def safe_label_encode(encoder, values):
    vals = []
    classes = set(getattr(encoder, "classes_", []))
    has_unk = "__UNK__" in classes
    for v in values:
        vs = v if v in classes else ("__UNK__" if has_unk else None)
        if vs is None:
            vals.append(-1)
        else:
            vals.append(int(encoder.transform([vs])[0]))
    return np.array(vals, dtype=int)

# -----------------------------------------
# 전처리 함수
def preprocess_dataframe(df: pd.DataFrame):
    df = df.copy()

    if "datetime" in df.columns:
        dt_parsed = df["datetime"].apply(lambda x: pd.to_datetime(x, errors="coerce"))
        df["hour"] = dt_parsed.dt.hour.fillna(-1).astype(int)
        df["weekday"] = dt_parsed.dt.weekday.fillna(-1).astype(int)
        df.drop(columns=["datetime"], inplace=True)

    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].apply(normalize_protocol)

    for c in ("technique", "data_src"):
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    if encoders_all is not None:
        for col, encoder in encoders_all.items():
            if col not in df.columns:
                continue
            vals = df[col].fillna("__UNK__").astype(str).tolist()
            df[col] = safe_label_encode(encoder, vals)

    df = df.fillna(-1)
    return df

# -----------------------------------------
# /predict 라우트
@app.route("/predict", methods=["POST"])
def predict():
    os.makedirs(EVENT_DIR, exist_ok=True)

    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error": "invalid json"}), 400

    if "sample" in payload:
        samples = [payload["sample"]]
    elif "samples" in payload:
        samples = payload["samples"]
    else:
        return jsonify({"error": "JSON must contain 'sample' or 'samples'"}), 400

    df = pd.DataFrame(samples)
    X_df = preprocess_dataframe(df)

    results_raw = []

    for i, row in df.iterrows():
        result = dict(row)
        result["timestamp"] = to_kst_iso(result.get("datetime"))
        result["protocol"] = normalize_protocol(result.get("protocol"))
        result["tid"] = "T1566"      
        result["technique"] = "Phishing"
        results_raw.append(result)

    # ---- JSON 파일 저장 ----
    events_for_file = []
    for rec in results_raw:
        out = {
            "timestamp": to_kst_iso(rec.get("timestamp")),
            "src_ip": rec.get("src_ip"),
            "src_port": rec.get("src_port"),
            "dst_ip": rec.get("dst_ip"),
            "dst_port": rec.get("dst_port"),
            "protocol": (rec.get("protocol") or "OTHER").upper(),
            "duration": rec.get("duration"),
            "fwd_pkts": rec.get("fwd_pkts"),
            "bwd_pkts": rec.get("bwd_pkts"),
            "fwd_bytes": rec.get("fwd_bytes"),
            "bwd_bytes": rec.get("bwd_bytes"),
            "tid": rec.get("tid"),
            "technique": rec.get("technique")
        }
        events_for_file.append(out)

    events_path = os.path.join(EVENT_DIR, "events.json")
    sessions_path = os.path.join(EVENT_DIR, "sessions.json")

    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events_for_file, f, ensure_ascii=False, indent=2)

    sessions_payload = {"flow-1": {"events": events_for_file}}
    with open(sessions_path, "w", encoding="utf-8") as f:
        json.dump(sessions_payload, f, ensure_ascii=False, indent=2)

    return jsonify({"results": events_for_file, "saved": True}), 200

# -----------------------------------------
# /health 체크
@app.route("/health", methods=["GET"])
def health():
    loaded = {
        "encoders_all": encoders_all is not None,
        "stage1_model": stage1_model is not None,
        "stage2_model": stage2_model is not None,
        "attack_map": attack_map is not None,
        "label_encoder": label_encoder is not None
    }
    return jsonify({"status": "ok", "loaded": loaded})

# -----------------------------------------
# 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
