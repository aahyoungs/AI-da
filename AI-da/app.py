from flask import Flask, jsonify, render_template, request
import os, json, glob
from datetime import datetime, time, timezone, timedelta
from dateutil import parser as dateparser, tz
from collections import defaultdict
from threading import Lock
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pytz

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENT_DIR = os.path.join(DATA_DIR, "events")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVENT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SESSIONS_FILE = os.path.join(EVENT_DIR, "sessions.json")
TTP_FILE = os.path.join(DATA_DIR, "TTPs.json")
ASSETS_FILE = os.path.join(DATA_DIR, "assets.json")

# AI2 모델
AI2_MODEL_PATH = os.path.join(MODEL_DIR, "ai2_nexttid_DEMO4_srcipT4_balanced.keras")
AI2_VOCAB_PATH = os.path.join(MODEL_DIR, "data", "tid_vocab_flow_fixed11.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

TZ_SEOUL = tz.gettz("Asia/Seoul")
KST = timezone(timedelta(hours=9))
WRITE_LOCK = Lock()

def safe_load(path):
    try:
        if not os.path.exists(path):
            print(f"[safe_load] 파일 없음: {path}")
            return None
        print(f"[safe_load] 로드 시도: {path}")
        obj = joblib.load(path)
        print(f"[safe_load] 로드 성공: {path}")
        return obj
    except Exception as e:
        print(f"[safe_load] 로드 실패: {path} → {e}")
        import traceback; traceback.print_exc()
        return None

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

def normalize_protocol(p):
    if p is None:
        return "OTHER"
    p = str(p).strip().upper()
    if p.startswith("TCP"): return "TCP"
    if p.startswith("UDP"): return "UDP"
    if p in ("ICMP", "ICM"): return "ICMP"
    if p in ("ARP",): return "ARP"
    return "OTHER"

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
                if isinstance(doc, list): evs.extend(doc)
                elif isinstance(doc, dict): evs.append(doc)
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

def load_json_safe(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_safe(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def parse_dt_safe(ts):
    if not ts:
        return None
    try:
        dt = dateparser.parse(ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=KST)
        return dt
    except Exception:
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

def next_session_key(existing_sessions):
    nums = []
    for k in existing_sessions.keys():
        if isinstance(k, str) and k.startswith("session-"):
            try:
                n = int(k.split("-", 1)[1])
                nums.append(n)
            except Exception:
                pass
    next_n = max(nums) + 1 if nums else 1
    return f"session-{next_n}"

def is_ddos_event(ev):
    tech = (ev.get("technique") or "").lower()
    tid = (ev.get("tid") or "").lower()
    try:
        fwd_pkts = int(ev.get("fwd_pkts") or 0)
        fwd_bytes = int(ev.get("fwd_bytes") or 0)
    except Exception:
        fwd_pkts = fwd_bytes = 0

    # 명시적 TID 또는 키워드
    if "t1498" in tid or "ddos" in tech or "denial" in tech:
        return True
    # 대량 트래픽 기준
    if fwd_pkts >= 300 and fwd_bytes >= 300 * 40:
        return True
    return False


def event_similarity_score(ev, session_events):
    """세션 유사도 점수 계산 (IP/TID/time 중심으로 강화)"""
    last_ev = session_events[-1]
    score = 0
    try:
        # 동일 src/dst
        if ev.get("src_ip") == last_ev.get("src_ip"):
            score += 50
        if ev.get("dst_ip") == last_ev.get("dst_ip"):
            score += 40

        # 시간 근접성
        ev_dt = parse_dt_safe(ev.get("timestamp"))
        last_dt = parse_dt_safe(last_ev.get("timestamp"))
        if ev_dt and last_dt:
            diff = abs((ev_dt - last_dt).total_seconds())
            if diff <= 120:
                score += 30
            elif diff <= 600:
                score += 15
            elif diff <= 3600:
                score += 5

        # TID/Technique 동일
        if ev.get("tid") == last_ev.get("tid"):
            score += 40
        if (ev.get("technique") or "").lower() == (last_ev.get("technique") or "").lower():
            score += 30

        # DDoS 간 추가 가중치
        if is_ddos_event(ev) and is_ddos_event(last_ev):
            if ev.get("dst_ip") == last_ev.get("dst_ip"):
                score += 70
    except Exception:
        pass
    return score

def assign_events_to_sessions_by_heuristic(existing_sessions, new_events):
    """세션 묶음 개선: 유사도 100 이상만 결합, DDoS는 같은 dst_ip만"""
    sessions = dict(existing_sessions)
    used_ids = {e.get("id") for s in sessions.values() for e in s.get("events", []) if isinstance(e, dict)}

    sorted_new = sorted(new_events, key=lambda e: parse_dt_safe(e.get("timestamp")) or datetime.min.replace(tzinfo=KST))
    for ev in sorted_new:
        if not isinstance(ev, dict):
            continue
        ev_id = ev.get("id")
        if ev_id in used_ids:
            continue

        if not sessions:
            key = "session-1"
            sessions[key] = {"timestamp": ev.get("timestamp"), "events": [ev]}
            used_ids.add(ev_id)
            continue

        best_key, best_score = None, 0
        for k, sdata in sessions.items():
            s_events = sdata.get("events", [])
            if not s_events:
                continue
            score = event_similarity_score(ev, s_events)
            if score > best_score:
                best_score, best_key = score, k

        # 동일 세션 결합 기준 강화
        if best_score >= 100:
            sessions[best_key]["events"].append(ev)
            used_ids.add(ev_id)
            continue

        # DDoS 특수 처리
        if is_ddos_event(ev):
            ddos_candidate = None
            for k, sdata in sessions.items():
                evs = sdata.get("events", [])
                ddos_count = sum(1 for e in evs if is_ddos_event(e))
                if evs and ddos_count >= max(1, int(len(evs) * 0.6)):
                    if any(e.get("dst_ip") == ev.get("dst_ip") for e in evs):
                        ddos_candidate = k
                        break
            if ddos_candidate:
                sessions[ddos_candidate]["events"].append(ev)
                used_ids.add(ev_id)
                continue

        # 새로운 세션 생성
        new_key = next_session_key(sessions)
        sessions[new_key] = {"timestamp": ev.get("timestamp"), "events": [ev]}
        used_ids.add(ev_id)

    return sessions

def call_ai2_grouping_if_available(new_events):
    try:
        return None
    except Exception:
        return None

def save_events_and_sessions_append(new_events):
    events_path = os.path.join(EVENT_DIR, "events.json")
    sessions_path = SESSIONS_FILE

    with WRITE_LOCK:
        # 1) load previous events (list)
        prev = load_json_safe(events_path, [])
        if not isinstance(prev, list):
            prev = []

        # 2) merge and dedupe by 'id' if present, else by full object equality
        merged_map = {}
        for e in prev + new_events:
            if not isinstance(e, dict):
                continue
            eid = e.get("id")
            if eid:
                merged_map[eid] = e
            else:
                key = f"{e.get('timestamp')}_{e.get('src_ip')}_{e.get('dst_ip')}_{e.get('src_port')}_{e.get('dst_port')}"
                merged_map.setdefault(key, e)

        # 3) produce sorted list
        merged_list = list(merged_map.values())
        try:
            merged_sorted = sorted(
                merged_list,
                key=lambda e: parse_dt_safe(e.get("timestamp")) or datetime.min.replace(tzinfo=KST),
            )
        except Exception:
            merged_sorted = merged_list

        # 4) write events.json
        save_json_safe(events_path, merged_sorted)

        # 5) load existing sessions.json
        existing_sessions = load_json_safe(sessions_path, {})
        if not isinstance(existing_sessions, dict):
            existing_sessions = {}

        # 6) try ai2 grouping first (optional)
        ai2_result = call_ai2_grouping_if_available(new_events)
        if ai2_result:
            for sess in ai2_result:
                key = next_session_key(existing_sessions)
                sess_events = []
                for e in sess.get("events", []):
                    if isinstance(e, dict):
                        sess_events.append(e)
                    else:
                        match = next((m for m in merged_sorted if m.get("id") == e), None)
                        if match:
                            sess_events.append(match)
                existing_sessions[key] = {"timestamp": sess.get("timestamp") or (sess_events[0].get("timestamp") if sess_events else None), "events": sess_events}
        else:
            # 7) fallback: assign events to sessions by heuristic
            prev_ids = set()
            for e in prev:
                if isinstance(e, dict) and e.get("id"):
                    prev_ids.add(e.get("id"))
            newly_added = [e for e in merged_sorted if isinstance(e, dict) and e.get("id") not in prev_ids]
            if not newly_added:
                prev_signatures = set(f"{e.get('timestamp')}_{e.get('src_ip')}_{e.get('dst_ip')}" for e in prev if isinstance(e, dict))
                newly_added = [e for e in merged_sorted if f"{e.get('timestamp')}_{e.get('src_ip')}_{e.get('dst_ip')}" not in prev_signatures]

            updated_sessions = assign_events_to_sessions_by_heuristic(existing_sessions, newly_added)
            existing_sessions = updated_sessions

        # 8) save sessions.json
        save_json_safe(sessions_path, existing_sessions)

    return merged_sorted

# =========================================================
# AI1 (공격 탐지/분류)
# =========================================================
encoders_all = safe_load(os.path.join(MODEL_DIR, "encoders_all.joblib"))
stage1_model = safe_load(os.path.join(MODEL_DIR, "stage1_xgb.joblib"))
stage2_model = safe_load(os.path.join(MODEL_DIR, "stage2_xgb.joblib"))
attack_map = safe_load(os.path.join(MODEL_DIR, "attack_map.joblib"))
label_encoder = safe_load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

def safe_label_encode(encoder, values):
    vals = []
    classes = set(getattr(encoder, "classes_", []))
    has_unk = "__UNK__" in classes
    for v in values:
        vs = v if v in classes else ("__UNK__" if has_unk else None)
        vals.append(int(encoder.transform([vs])[0]) if vs else -1)
    return np.array(vals, dtype=int)

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
    if encoders_all:
        for col, encoder in encoders_all.items():
            if col not in df.columns: continue
            vals = df[col].fillna("__UNK__").astype(str).tolist()
            df[col] = safe_label_encode(encoder, vals)
    return df.fillna(-1)

# =========================================================
# AI2 (공격 시퀀스 예측)
# =========================================================
AI2_MODEL = None
AI2_TID2ID, AI2_ID2TID, AI2_UNK = {}, {}, None
AI2_T, AI2_TOPK = 4, 3
TTP_MAP = {}   # tid -> technique

def load_ttps_map():
    global TTP_MAP
    TTP_MAP.clear()
    if not os.path.exists(TTP_FILE):
        return
    try:
        with open(TTP_FILE, "r", encoding="utf-8") as f:
            ttps = json.load(f)
        for ttp in ttps:
            for tech in ttp.get("techniques", []):
                tid = str(tech.get("tid", "")).strip().upper()
                technique = str(tech.get("technique", "")).strip()
                if tid and technique:
                    TTP_MAP[tid] = technique
    except Exception as e:
        print(f"[TTP] failed to load: {e}")

def load_ai2_model():
    global AI2_MODEL, AI2_TID2ID, AI2_ID2TID, AI2_UNK
    if not (os.path.exists(AI2_MODEL_PATH) and os.path.exists(AI2_VOCAB_PATH)):
        print("[AI2] 모델 또는 vocab 없음")
        return
    try:
        AI2_MODEL = tf.keras.models.load_model(AI2_MODEL_PATH)
        with open(AI2_VOCAB_PATH, "r", encoding="utf-8") as f:
            voc = json.load(f)
        AI2_TID2ID = {str(k).upper(): int(v) for k, v in voc.get("tid2id", {}).items()}
        AI2_ID2TID = {v: k for k, v in AI2_TID2ID.items()}
        AI2_UNK = AI2_TID2ID.get("<UNK>", max(AI2_TID2ID.values()) + 1 if len(AI2_TID2ID) > 0 else 0)
    except Exception as e:
        print(f"[AI2] 로드 실패: {e}")

def ai2_predict_next_tids(tid_seq, T=AI2_T, topk=AI2_TOPK):
    if AI2_MODEL is None:
        ctx = (["<UNK>"] * (T - len(tid_seq or [])) + list(tid_seq or []))[-T:]
        return {"context_T": T, "context_tids": ctx, "next_tid_topk": []}

    tid_seq = [str(t).strip().upper() for t in (tid_seq or []) if t]
    ctx_short = tid_seq[-T:] if len(tid_seq) > 0 else []
    pad_count = max(0, T - len(ctx_short))
    ctx_padded = ["<UNK>"] * pad_count + ctx_short

    def enc_id(t):
        return AI2_TID2ID.get(str(t).upper(), AI2_UNK)

    x = np.array([enc_id(t) for t in ctx_padded], dtype=np.int32).reshape(1, T)

    try:
        p = AI2_MODEL.predict(x, verbose=0)[0]
    except Exception:
        return {"context_T": T, "context_tids": ctx_padded, "next_tid_topk": []}

    top = p.argsort()[-topk:][::-1]
    next_topk = []
    for i in top:
        tid_raw = AI2_ID2TID.get(int(i), f"UNK({int(i)})")
        tid_norm = str(tid_raw).strip().upper()
        tech = TTP_MAP.get(tid_norm)
        if not tech and "." in tid_norm:
            tech = TTP_MAP.get(tid_norm.split(".")[0])
        next_topk.append({
            "tid": tid_norm,
            "technique": tech or "-",
            "prob": round(float(p[i]), 3)
        })

    return {"context_T": T, "context_tids": ctx_padded, "next_tid_topk": next_topk}

load_ttps_map()
load_ai2_model()

# =========================================================
# /predict (AI1 inference)
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error": "invalid json"}), 400

        if "sample" in payload:
            samples = [payload["sample"]]
        elif "samples" in payload:
            samples = payload["samples"]
        else:
            return jsonify({"error": "JSON must contain 'sample' or 'samples'"}), 400

        df = pd.DataFrame(samples).reset_index(drop=True)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df.drop(columns=["timestamp"], inplace=True, errors="ignore")
        else:
            df["datetime"] = pd.NaT

        if "protocol" in df.columns:
            df["protocol"] = df["protocol"].fillna("unknown").astype(str)
            df["protocol_cat"] = df["protocol"].astype("category").cat.codes
            df.drop(columns=["protocol"], inplace=True, errors="ignore")

        X_df = preprocess_dataframe(df.copy())

        def get_model_feature_names(model):
            try:
                booster = getattr(model, "get_booster", lambda: None)()
                if booster is not None and getattr(booster, "feature_names", None):
                    return list(booster.feature_names)
            except Exception:
                pass
            fn = getattr(model, "feature_names_in_", None)
            if fn is not None:
                return list(fn)
            return None

        def align_X_to_model(X_df, model, name="stage"):
            feats = get_model_feature_names(model)
            if not feats:
                app.logger.info(f"{name} 모델 feature 정보 없음, 정렬 생략.")
                return X_df
            missing = [c for c in feats if c not in X_df.columns]
            for m in missing:
                X_df[m] = 0
            extra = [c for c in X_df.columns if c not in feats]
            if extra:
                app.logger.debug(f"{name} 모델에서 사용하지 않는 컬럼 제거: {extra}")
            X_df = X_df[[c for c in feats if c in X_df.columns]]
            app.logger.info(f"{name} 모델 feature 정렬 완료 ({len(feats)}개 피처).")
            return X_df

        X_df = align_X_to_model(X_df, stage1_model, "Stage1")
        X_df = X_df.select_dtypes(include=[np.number]).fillna(0)

        if stage1_model is None or stage2_model is None:
            return jsonify({"error": "model not loaded"}), 500

        # Stage 1: 공격 여부
        if hasattr(stage1_model, "predict_proba"):
            y_stage1_proba = stage1_model.predict_proba(X_df)[:, 1]
            y_stage1 = (y_stage1_proba >= 0.5).astype(int)
        else:
            y_stage1 = stage1_model.predict(X_df)
            y_stage1_proba = np.clip(y_stage1, 0, 1)

        # Stage 2: TTP 분류
        X_stage2 = X_df.copy()
        X_stage2["stage1_pred"] = y_stage1
        X_stage2 = align_X_to_model(X_stage2, stage2_model, "Stage2")

        if hasattr(stage2_model, "predict_proba"):
            y_stage2_proba_all = stage2_model.predict_proba(X_stage2)
            y_stage2 = np.argmax(y_stage2_proba_all, axis=1)
            y_stage2_conf = np.max(y_stage2_proba_all, axis=1)
        else:
            y_stage2 = stage2_model.predict(X_stage2)
            y_stage2_conf = np.ones(len(y_stage2)) * 0.5

        decoded_labels = (
            label_encoder.inverse_transform(y_stage2)
            if label_encoder
            else [str(y) for y in y_stage2]
        )

        amap_all = attack_map or {}

        # 결과 매핑
        new_events = []
        for i, row in df.iterrows():
            label = decoded_labels[i]
            amap = amap_all.get(label, {})
            tid = amap.get("tid")
            tech = amap.get("technique")

            # 입력값 우선
            incoming_tid = row.get("tid") or row.get("TID") or None
            incoming_tech = row.get("technique") or row.get("tech") or None
            if incoming_tid:
                tid = str(incoming_tid).upper()
            if incoming_tech:
                tech = str(incoming_tech)

            # TTP_MAP 기반 보완
            if not tech:
                if isinstance(label, str) and label.upper().startswith("T"):
                    if label.upper() in TTP_MAP:
                        tid = label.upper()
                        tech = TTP_MAP[tid]
                    elif "." in label:
                        base = label.split(".")[0].upper()
                        if base in TTP_MAP:
                            tid = label.upper()
                            tech = TTP_MAP[base]
            if not tech and tid:
                key = str(tid).upper()
                tech = TTP_MAP.get(key) or (TTP_MAP.get(key.split(".")[0]) if "." in key else None)

            # 그래도 없으면 UNKNOWN
            tid = (tid or label or "T0000").upper()
            if "." in tid and tid not in TTP_MAP:
                base = tid.split(".")[0]
                if base in TTP_MAP:
                    tech = tech or TTP_MAP[base]
            tech = tech or label or "UNKNOWN"


            out = {
                "timestamp": to_kst_iso(row.get("datetime")),
                "src_ip": row.get("src_ip"),
                "src_port": row.get("src_port"),
                "dst_ip": row.get("dst_ip"),
                "dst_port": row.get("dst_port"),
                "protocol": "unknown",
                "duration": row.get("duration"),
                "fwd_pkts": row.get("fwd_pkts"),
                "bwd_pkts": row.get("bwd_pkts"),
                "fwd_bytes": row.get("fwd_bytes"),
                "bwd_bytes": row.get("bwd_bytes"),
                "tid": tid,
                "technique": tech,
                "attack_label": label,
                "stage1_conf": float(np.nan_to_num(y_stage1_proba[i], nan=0.0)),
                "stage2_conf": float(np.nan_to_num(y_stage2_conf[i], nan=0.0)),
            }
            new_events.append(out)

        save_events_and_sessions_append(new_events)
        return jsonify({"results": new_events, "saved": True}), 200

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        app.logger.error("Predict exception: %s\n%s", str(e), tb)
        return jsonify({"error": "internal server error", "message": str(e)}), 500

# =========================================================
# /health
# =========================================================
@app.route("/health", methods=["GET"])
def health():
    loaded = {
        "encoders_all": encoders_all is not None,
        "stage1_model": stage1_model is not None,
        "stage2_model": stage2_model is not None,
        "attack_map": attack_map is not None,
        "label_encoder": label_encoder is not None,
        "ai2_model": AI2_MODEL is not None,
        "ai2_vocab": bool(AI2_TID2ID),
    }
    return jsonify({"status": "ok", "loaded": loaded})

# =========================================================
# 라우트: /api/heatmap
# =========================================================
@app.route("/api/heatmap")
def api_heatmap():
    mode = request.args.get("mode")
    start = request.args.get("start")
    end = request.args.get("end")

    ttps = []
    if os.path.exists(TTP_FILE):
        try:
            with open(TTP_FILE, "r", encoding="utf-8") as f:
                ttps = json.load(f)
        except Exception:
            ttps = []

    all_events = filter_period(load_events_files(), start, end, mode)

    defined_techs = {tech["technique"] for ttp in ttps for tech in ttp.get("techniques", [])}
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
        tac = ttp.get("tactic")
        if tac is None:
            continue
        order.append(tac)
        tactics[tac] = []
        for tech in ttp.get("techniques", []):
            name = tech.get("technique")
            if not name:
                continue
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

# =========================================================
# /api/metrics (위험도)
# =========================================================
@app.route("/api/metrics")
def api_metrics():
    start, end = request.args.get("start"), request.args.get("end")
    events = filter_period(load_events_files(), start, end)
    count = len(events)

    RISK_MAP = {}
    if os.path.exists(TTP_FILE):
        try:
            with open(TTP_FILE, "r", encoding="utf-8") as f:
                ttps = json.load(f)
                for ttp in ttps:
                    for tech in ttp.get("techniques", []):
                        tid = tech.get("tid")
                        name = tech.get("technique")
                        risk = tech.get("risk", 20)  # 기본값 20 (Low)
                        if tid:
                            RISK_MAP[tid] = risk
                        if name:
                            RISK_MAP[name] = risk
        except Exception as e:
            print(f"[WARN] Failed to load TTPs.json: {e}")

    def calc_period_risk(events_list):
        if not events_list:
            return None

        for e in events_list:
            tid = e.get("tid")
            tech = e.get("technique")
            if tid and tid in RISK_MAP:
                score = RISK_MAP[tid]
            elif tech and tech in RISK_MAP:
                score = RISK_MAP[tech]
            else:
                score = 20
            e["risk_score"] = score

        scores = [e["risk_score"] for e in events_list if e.get("risk_score") is not None]
        if not scores:
            return None

        n = len(scores)
        high_count = sum(1 for s in scores if s >= 71)
        weighted = sum(scores) / n

        # 빈도 보정 (최대 20%)
        freq_factor = 1 + min(0.2, n / 1000)

        # 고위험 보정
        if high_count >= 1 and weighted < 70:
            weighted = 70

        return round(weighted * freq_factor, 1)

    score = calc_period_risk(events)

    if score is None:
        level = "none"
    elif score >= 71:
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

# =========================================================
# /api/sessions
# =========================================================
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

# =========================================================
# /api/session/<sid> (세션 상세 + AI2 예측)
# =========================================================
@app.route("/api/session/<sid>")
def api_session_detail(sid):
    start, end = request.args.get("start"), request.args.get("end")
    sessions = load_sessions()
    sdata = sessions.get(sid)
    if not sdata:
        return jsonify({"error": "not found"}), 404

    events = filter_period(normalize(sdata.get("events", [])), start, end)

    # 1. 그래프
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
        nodes_map, edges_map = {}, defaultdict(int)
        for ev in events:
            src, dst = ev.get("src_ip"), ev.get("dst_ip")
            if not src or not dst:
                continue
            edges_map[(src, dst)] += 1
            for ip in (src, dst):
                if ip not in nodes_map:
                    nodes_map[ip] = {
                        "id": ip,
                        "color": "#2563eb" if ip in ASSETS else "#dc2626",
                        "count": 0
                    }
                nodes_map[ip]["count"] += 1
        edges = [{"source": s, "target": t, "count": c} for (s, t), c in edges_map.items()]
        nodes = list(nodes_map.values())

    # 2. 각 이벤트별 AI2 예측
    tid_seq = []
    for ev in events:
        if ev.get("tid"):
            tid_seq.append(str(ev["tid"]).strip().upper())
            pred = ai2_predict_next_tids(tid_seq)
            ev["ai2_pred_topk"] = pred["next_tid_topk"] if pred and pred.get("next_tid_topk") else []
        else:
            ev["ai2_pred_topk"] = []

    # 3. 세션 일관성(평균 확률)
    s1_conf = [e.get("stage1_conf") for e in events if isinstance(e.get("stage1_conf"), (float, int))]
    s2_conf = [e.get("stage2_conf") for e in events if isinstance(e.get("stage2_conf"), (float, int))]

    session_confidence = {
        "stage1_mean": round(float(np.mean(s1_conf) * 100), 1) if s1_conf else None,
        "stage2_mean": round(float(np.mean(s2_conf) * 100), 1) if s2_conf else None,
        "combined": round(float((np.mean(s1_conf + s2_conf) / 2) * 100), 1)
        if (s1_conf or s2_conf) else None,
    }

    # 4. AI2 요약
    ai2_summary = ai2_predict_next_tids(tid_seq) or {}

    return jsonify({
        "id": sid,
        "label": f"공격플로우-{sid.split('-')[-1]}",
        "nodes": nodes,
        "edges": edges,
        "events": events,
        "session_confidence": session_confidence,
        "ai2_enabled": True,
        **ai2_summary
    })

# =========================================================
# /api/single_events
# =========================================================
@app.route("/api/single_events")
def api_single_events():
    start, end = request.args.get("start"), request.args.get("end")
    sessions = load_sessions()
    used = set()

    def key(ev):
        return (ev.get("timestamp"), ev.get("src_ip"), ev.get("dst_ip"))

    for sdata in sessions.values():
        for ev in sdata.get("events", []):
            used.add(key(ev))

    all_events = filter_period(load_events_files(), start, end)
    singles = [e for e in all_events if key(e) not in used]
    return jsonify({"count": len(singles), "events": singles, "ai2_enabled": False})

@app.route("/")
def index():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5500)