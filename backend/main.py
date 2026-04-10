import csv, re
import threading, time, cv2, logging, numpy as np, torch
import importlib
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime

from detector import (
    load_yolo, load_vlm,
    smolvlm_infer, run_vlm_threat, run_weapons, draw_weapons,
    state, state_lock, yolo_all_classes, yolo_edge_classes,
    ProximityTracker, push_alert, pad_crop, normalize_weapon_label, is_edge_weapon_name,
    RED_HOLD_SEC, RED_CONFIDENCE, vlm_abort,
    DEFAULT_PROXIMITY_PROMPT, DEFAULT_COUNT_CHANGE_PROMPT,
    DEFAULT_WEAPON_PROMPT, DEFAULT_SCENE_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"; UPLOAD_DIR.mkdir(exist_ok=True)
PROMPT_CSV_DIR = BASE_DIR / "logs"; PROMPT_CSV_DIR.mkdir(exist_ok=True)
PROMPT_CSV_PATH = PROMPT_CSV_DIR / "prompt_outputs.csv"
prompt_csv_lock = threading.Lock()

STREAM_W           = 1280
STREAM_H           = 720
STREAM_FPS         = 30
VLM_THREAD_TIMEOUT = 20.0
WEAPON_MIN_FRAMES  = 2
EDGE_WEAPON_MIN_CONF = 0.18
YOLO_DET_CONF      = 0.10
YOLO_DET_IMGSZ     = 960
YOLO_TRACK_CONF    = 0.20
YOLO_TRACK_IMGSZ   = 736
CARRY_OBJECT_MIN_CONF = 0.16
CARRY_TRIGGER_MIN_FRAMES = 2
CARRY_ASSOC_IOU_MIN = 0.02
CARRY_PERSON_PAD_RATIO = 0.22

CARRY_OBJECT_LABELS = {
    "baseball bat",
    "bat",
    "axe",
    "knife",
    "machete",
    "blade",
    "crowbar",
    "scissors",
}

# Classes to suppress from detection overlays/counts.
# "computer" is mapped to COCO's "laptop" label.
EXCLUDED_CLASS_ALIASES = {
    "cell phone",
    "cellphone",
    "cell-phone",
    "chair",
    "bed",
    "mouse",
    "computer",
    "laptop",
    "tv",
    "keyboard",
    "bottle",
    "couch"
}

# Restrict detection to person + weapon-relevant classes to reduce yolo11l noise.
FOCUSED_CLASS_ALIASES = {
    "person",
    "knife",
    "kitchen knife",
    "dagger",
    "scissors",
    "scissor",
    "baseball bat",
    "bat",
    "axe",
    "axes",
    "ax",
    "hatchet",
    "crowbar",
    "blade",
    "machete",
}


def _normalize_class_alias(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def _build_detection_class_ids() -> list[int]:
    keep_ids = []
    kept_labels = []
    skipped_labels = []

    for cid in sorted(yolo_all_classes):
        label = str(yolo_all_classes[cid])
        normalized = _normalize_class_alias(label)
        if (normalized in FOCUSED_CLASS_ALIASES
                or is_edge_weapon_name(normalized)):
            keep_ids.append(cid)
            kept_labels.append(label)
            continue
        skipped_labels.append(label)

    if keep_ids:
        log.info(f"[YOLO] Focused classes enabled: {sorted(set(kept_labels))}")
        log.info(f"[YOLO] Skipping non-focused classes: {len(set(skipped_labels))}")
        return keep_ids

    # Fallback to exclusion mode if focused labels were not available
    # in the model class map.
    keep_ids = []
    skipped_labels = []
    for cid in sorted(yolo_all_classes):
        label = str(yolo_all_classes[cid])
        if _normalize_class_alias(label) in EXCLUDED_CLASS_ALIASES:
            skipped_labels.append(label)
            continue
        keep_ids.append(cid)

    if not keep_ids:
        # Safety fallback: never run with an empty classes list.
        keep_ids = sorted(yolo_all_classes)
        log.warning("[YOLO] Exclusion list removed all classes; using full class list instead")
    else:
        if skipped_labels:
            log.info(f"[YOLO] Excluding classes: {sorted(set(skipped_labels))}")

    return keep_ids


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _expand_box(box, pad_ratio: float, fw: int, fh: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * pad_ratio, bh * pad_ratio
    return [
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(float(fw - 1), x2 + px),
        min(float(fh - 1), y2 + py),
    ]


def _box_center(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _associate_carry_object(person_boxes: list, object_box, fw: int, fh: int):
    """Return best matching person box for a carry-object candidate."""
    if not person_boxes:
        return None

    ocx, ocy = _box_center(object_box)
    best = None
    best_score = -1.0

    for p in person_boxes:
        exp = _expand_box(p, CARRY_PERSON_PAD_RATIO, fw, fh)
        iou = _box_iou(exp, object_box)
        px1, py1, px2, py2 = exp
        center_inside = (px1 <= ocx <= px2) and (py1 <= ocy <= py2)
        if not center_inside and iou < CARRY_ASSOC_IOU_MIN:
            continue

        score = iou + (0.25 if center_inside else 0.0)
        if score > best_score:
            best_score = score
            best = p

    return best


def _merge_boxes(a, b):
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    return [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]


def _incident_weapon_prompt(label: str) -> str:
    return (
        f"A person may be carrying a {label} in this surveillance frame. "
        "Determine if they are merely holding it or using/wielding it as a threat. "
        "Respond ONLY as JSON with keys: "
        "threat (boolean), type (string), confidence (low|medium|high), description (string)."
    )

app = FastAPI(title="CCTV Surveillance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ───────────────────────────────────────────────────────────────
log.info("=" * 50)
yolo_model               = load_yolo()  # all-class detection model
yolo_tracker_model       = load_yolo()  # dedicated person-tracking model
DETECTION_CLASS_IDS      = _build_detection_class_ids()
vlm_model, vlm_processor = load_vlm()

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
    yolo_model.to("cuda"); yolo_model.model.half()
    yolo_tracker_model.to("cuda"); yolo_tracker_model.model.half()
    log.info(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    log.warning("[GPU] No CUDA")
log.info("=" * 50)

# ── Engine ────────────────────────────────────────────────────────────────────
engine = {
    "running":    False,
    "source":     None,
    "thread":     None,
    "frame":      None,
    "frame_lock": threading.Lock(),
}

DIRECT_VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m3u8", ".mpd", ".ts",
    ".wmv", ".flv", ".3gp", ".m4v", ".mjpeg", ".mpg", ".mpeg",
}


def _is_http_url(value: str) -> bool:
    p = urlparse(value)
    return p.scheme.lower() in {"http", "https"}


def _is_direct_media_url(value: str) -> bool:
    lowered = value.lower().split("?", 1)[0]
    return any(lowered.endswith(ext) for ext in DIRECT_VIDEO_EXTS)


def resolve_source_input(source: str) -> tuple[str, dict]:
    """
    Resolve user-provided source into a stream URL/path OpenCV can open.
    - RTSP/RTMP/UDP/TCP/local paths: returned as-is
    - HTTP direct media links: returned as-is
    - Webpage links: resolved using yt-dlp into direct media URL
    """
    raw = (source or "").strip()
    if not raw:
        raise ValueError("Source path/link is empty")

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()

    if scheme in {"rtsp", "rtmp", "udp", "tcp"}:
        return raw, {"resolver": "none", "source_type": "stream"}

    if not _is_http_url(raw):
        return raw, {"resolver": "none", "source_type": "local"}

    if _is_direct_media_url(raw):
        return raw, {"resolver": "none", "source_type": "direct_media_url"}

    try:
        yt_dlp_module = importlib.import_module("yt_dlp")
        YoutubeDL = getattr(yt_dlp_module, "YoutubeDL")
    except Exception as e:
        raise RuntimeError(
            "yt-dlp is required for webpage links. Install it in backend environment."
        ) from e

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "format": "best[protocol!=m3u8_native]/best",
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(raw, download=False)
    except Exception as e:
        raise RuntimeError(f"Could not resolve webpage link ({e})") from e

    if info is None:
        raise RuntimeError("No media metadata returned by extractor")

    if "entries" in info and info["entries"]:
        info = next((x for x in info["entries"] if x), None)
        if info is None:
            raise RuntimeError("Playlist has no playable entries")

    direct_url = info.get("url")
    if not direct_url:
        formats = [f for f in (info.get("formats") or []) if f.get("url")]
        if formats:
            formats.sort(
                key=lambda f: (
                    f.get("height") or 0,
                    f.get("tbr") or 0,
                ),
                reverse=True,
            )
            direct_url = formats[0]["url"]

    if not direct_url:
        raise RuntimeError("Extractor did not provide a playable stream URL")

    return direct_url, {
        "resolver": "yt-dlp",
        "source_type": "webpage_url",
        "title": info.get("title", ""),
    }


def _empty_prompt_outputs() -> dict:
    return {
        "proximity":    {"description": "", "time": ""},
        "count_change": {"description": "", "time": ""},
        "weapon":       {"description": "", "time": ""},
        "scene":        {"description": "", "time": ""},
    }


def _ensure_prompt_csv_header():
    needs_header = (not PROMPT_CSV_PATH.exists()) or PROMPT_CSV_PATH.stat().st_size == 0
    if not needs_header:
        return

    with PROMPT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timeline", "description"])


def _append_prompt_csv_row(prompt_type: str, timeline: str, description: str):
    try:
        with prompt_csv_lock:
            _ensure_prompt_csv_header()
            with PROMPT_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([prompt_type, timeline, description])
    except Exception as e:
        log.warning(f"[PROMPT CSV] Append failed: {e}")


def _read_prompt_csv_rows_unlocked() -> list[dict]:
    _ensure_prompt_csv_header()
    rows = []
    with PROMPT_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            rows.append({
                "id": idx,
                "type": str(row.get("type", "") or ""),
                "timeline": str(row.get("timeline", "") or ""),
                "description": str(row.get("description", "") or ""),
            })
    return rows


def _write_prompt_csv_rows_unlocked(rows: list[dict]):
    with PROMPT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "timeline", "description"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "type": str(row.get("type", "") or ""),
                "timeline": str(row.get("timeline", "") or ""),
                "description": str(row.get("description", "") or ""),
            })


def _matches_prompt_filters(row: dict, type_filter: str, timeline_filter: str, description_filter: str) -> bool:
    t = str(type_filter or "").strip().lower()
    tl = str(timeline_filter or "").strip().lower()
    d = str(description_filter or "").strip().lower()

    row_type = str(row.get("type", "") or "").lower()
    row_timeline = str(row.get("timeline", "") or "").lower()
    row_description = str(row.get("description", "") or "").lower()

    if t and t not in row_type:
        return False
    if tl and tl not in row_timeline:
        return False
    if d and d not in row_description:
        return False
    return True


def _record_prompt_output_locked(prompt_type: str, description: str):
    desc = str(description or "").strip()
    if not desc:
        return

    now_dt = datetime.now()
    timeline = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    ui_time = now_dt.strftime("%H:%M:%S")

    outputs = state.get("prompt_outputs")
    if not isinstance(outputs, dict):
        outputs = _empty_prompt_outputs()
        state["prompt_outputs"] = outputs

    outputs[prompt_type] = {
        "description": desc,
        "time": ui_time,
    }
    _append_prompt_csv_row(prompt_type, timeline, desc)


def _sanitize_prompt_description(text: str) -> str:
    desc = str(text or "").replace("\r", "\n").strip()
    if not desc:
        return ""

    desc = re.sub(r"\s+", " ", desc).strip("`\"' ")
    if not desc:
        return ""

    # Drop malformed punctuation-only fragments like "}" from model glitches.
    if re.fullmatch(r"[\{\}\[\]\(\)\|\\/,:;.!?+\-_]+", desc):
        return ""

    return desc

# ── VLM Priority Task Manager ─────────────────────────────────────────────────
# "trigger" tasks (weapon/proximity/count_change) preempt "passive" tasks.
# Triggers never interrupt other triggers — first one wins.
_vlm_task = {
    "thread": None,
    "type":   "passive",   # "passive" | "trigger"
    "lock":   threading.Lock(),
}

def launch_vlm(task_type: str, fn, args) -> bool:
    """
    Launch VLM task with priority.
    - trigger preempts passive (aborts it via vlm_abort event)
    - passive skips if anything is running
    - trigger skips if another trigger is running
    Returns True if launched.
    """
    with _vlm_task["lock"]:
        t      = _vlm_task["thread"]
        active = t is not None and t.is_alive()

        if active:
            if task_type == "trigger" and _vlm_task["type"] == "passive":
                log.info("[VLM] Aborting passive task — trigger incoming")
                vlm_abort.set()
                t.join(timeout=2.0)
                vlm_abort.clear()
                # fall through to start trigger
            else:
                return False  # skip passive-on-anything or trigger-on-trigger

        def _run():
            try:
                fn(*args)
            except Exception as e:
                log.warning(f"[VLM task] {e}")

        new_t = threading.Thread(target=_run, daemon=True)
        new_t._start_time    = time.time()
        _vlm_task["thread"]  = new_t
        _vlm_task["type"]    = task_type
        vlm_abort.clear()
        new_t.start()
        return True


def vlm_running() -> bool:
    t = _vlm_task["thread"]
    return t is not None and t.is_alive()


# ── VRAM ──────────────────────────────────────────────────────────────────────
def get_vram_pct() -> float:
    try:
        return (torch.cuda.memory_reserved() /
                torch.cuda.get_device_properties(0).total_memory * 100)
    except Exception:
        return 0.0

def offload_vlm():
    global vlm_model
    if vlm_model is None: return
    try:
        vlm_model = vlm_model.to("cpu")
        torch.cuda.empty_cache()
        log.info("[VLM] Offloaded to CPU")
    except Exception as e:
        log.warning(f"[VLM] Offload: {e}")

def reload_vlm():
    global vlm_model
    if vlm_model is None: return
    try:
        vlm_model = vlm_model.to("cuda")
        torch.cuda.synchronize()
        log.info("[VLM] Reloaded to GPU")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.error("[VLM] OOM reload — staying CPU")
    except Exception as e:
        log.warning(f"[VLM] Reload: {e}")


# ── Engine ────────────────────────────────────────────────────────────────────
def run_engine(source):
    prox               = ProximityTracker()
    described_ids      = set()
    prev_count         = -1
    frame_count        = 0
    weapon_consecutive = 0
    carry_consecutive  = 0

    try:    src = int(source)
    except: src = source

    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
        cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)

    if not cap.isOpened():
        log.error(f"Cannot open: {source}")
        engine["running"] = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120: fps = 30.0
    delay = 1.0 / fps
    with state_lock:
        state["source_fps"] = round(fps, 2)
    log.info(f"Stream: {source} @ {fps:.0f}fps")

    # ── VLM callbacks ─────────────────────────────────────────────────────────
    def do_weapon_vlm(crop, prompt, detected_label="weapon"):
        r = run_vlm_threat(crop, vlm_model, vlm_processor, prompt)
        raw_kind = (r.get("type") or "").strip() if isinstance(r.get("type"), str) else ""
        norm_kind = normalize_weapon_label(raw_kind) if raw_kind else ""
        fallback_kind = normalize_weapon_label(detected_label or "weapon")
        kind = norm_kind if norm_kind and norm_kind != "none" else fallback_kind

        desc = _sanitize_prompt_description(r.get("description", ""))
        payload = dict(r)
        payload["type"] = kind

        if not desc:
            desc = (
                f"Threat review indicates potentially aggressive handling of {kind}."
                if payload.get("threat")
                else f"Carry incident reviewed for {kind}."
            )
        payload["description"] = desc

        with state_lock:
            state["vlm_description"] = desc
            if desc:
                state["scene_description"] = desc
            _record_prompt_output_locked("weapon", desc)
        if payload.get("threat") and payload.get("confidence") in RED_CONFIDENCE:
            push_alert("RED", f"Threat confirmed: {kind}", payload)
        else:
            push_alert("YELLOW", f"Carry incident reviewed: {kind}", payload)

    def do_proximity_vlm(crop, prompt):
        desc = _sanitize_prompt_description(
            smolvlm_infer(crop, prompt or DEFAULT_PROXIMITY_PROMPT,
                          vlm_model, vlm_processor, max_tokens=80)
        )
        if not desc: return
        threat = any(w in desc.lower() for w in
                     ["threatening", "assault", "attack", "fight", "danger"])
        payload = {
            "threat": threat,
            "type": "proximity",
            "confidence": "medium" if threat else "low",
            "description": desc,
        }
        if threat:
            push_alert("RED",    f"Proximity threat: {desc[:80]}", payload)
        else:
            push_alert("YELLOW", f"Sustained contact: {desc[:80]}", payload)
        with state_lock:
            state["scene_description"] = desc
            state["vlm_description"] = desc
            _record_prompt_output_locked("proximity", desc)

    def do_count_change_vlm(crop, prompt):
        desc = _sanitize_prompt_description(
            smolvlm_infer(crop, prompt or DEFAULT_COUNT_CHANGE_PROMPT,
                          vlm_model, vlm_processor, max_tokens=60)
        )
        if desc:
            with state_lock:
                state["scene_description"] = desc
                state["vlm_description"] = desc
                _record_prompt_output_locked("count_change", desc)
            log.info(f"[COUNT VLM] {desc}")

    def do_scene_vlm(crop, prompt):
        desc = _sanitize_prompt_description(
            smolvlm_infer(crop, prompt or DEFAULT_SCENE_PROMPT,
                          vlm_model, vlm_processor, max_tokens=60)
        )
        if desc:
            with state_lock:
                state["scene_description"] = desc
                state["vlm_description"] = desc
                _record_prompt_output_locked("scene", desc)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while engine["running"]:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break

        try:
            frame_count += 1
            now = time.time()

            with state_lock:
                yolo_on  = state["yolo_enabled"]
                vlm_on   = state["vlm_enabled"]
                vlm_ivl  = state["vlm_interval"]
                last_vlm = state["last_vlm_time"]
                switching= state["mode_switching"]
                prompts  = dict(state["trigger_prompts"])

            # ── RAW mode ──────────────────────────────────────────────────────
            if not yolo_on and not vlm_on:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with engine["frame_lock"]:
                    engine["frame"] = buf.tobytes()
                time.sleep(max(0, delay-(time.time()-t0))); continue

            # ── VLM thread watchdog ────────────────────────────────────────────
            t_obj = _vlm_task["thread"]
            if (t_obj and t_obj.is_alive()
                    and hasattr(t_obj, "_start_time")
                    and now - t_obj._start_time > VLM_THREAD_TIMEOUT):
                log.warning("[VLM] Watchdog: thread hung >20s — aborting")
                vlm_abort.set()
                t_obj.join(timeout=1.0)
                vlm_abort.clear()

            # ── YOLO detection + tracking ─────────────────────────────────────
            annotated    = frame
            people_boxes = {}
            class_counts = {}
            weapon_dets  = []
            edge_dets    = []   # axe/scissor/crowbar from yolo26n
            yolo_trigger = None
            person_total_for_summary = 0

            if yolo_on:
                # Detect all classes with a lower threshold so small objects
                # like remote/scissors are less likely to be dropped.
                det = yolo_model.predict(
                    frame,
                    conf=YOLO_DET_CONF,
                    iou=0.45,
                    classes=DETECTION_CLASS_IDS,
                    imgsz=YOLO_DET_IMGSZ,
                    verbose=False,
                )
                det_res   = det[0]
                annotated = det_res.plot()

                person_count_total = 0
                person_det_boxes = []
                carry_candidates = []
                det_boxes = det_res.boxes if det_res.boxes is not None else []
                for box in det_boxes:
                    cid  = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    name = str(yolo_all_classes.get(cid, cid))
                    conf = float(box.conf[0])
                    class_counts[name] = class_counts.get(name, 0) + 1

                    # Person count includes all current person detections.
                    if cid == 0:
                        person_count_total += 1
                        if conf >= YOLO_TRACK_CONF:
                            person_det_boxes.append(xyxy.tolist())

                    normalized_name = normalize_weapon_label(name)
                    if (name.lower() in CARRY_OBJECT_LABELS
                            or normalized_name in CARRY_OBJECT_LABELS):
                        if conf >= CARRY_OBJECT_MIN_CONF:
                            x1,y1,x2,y2 = map(int, xyxy)
                            carry_candidates.append({
                                "label": normalized_name if normalized_name in CARRY_OBJECT_LABELS else name.lower(),
                                "confidence": round(conf, 2),
                                "bbox": [x1, y1, x2, y2],
                            })

                    # Edge weapon from yolo26n class names
                    if is_edge_weapon_name(name):
                        if conf >= EDGE_WEAPON_MIN_CONF:
                            x1,y1,x2,y2 = map(int, xyxy)
                            edge_dets.append({
                                "label":      normalize_weapon_label(name),
                                "confidence": round(conf, 2),
                                "bbox":       [x1,y1,x2,y2],
                            })

                # Track persons only for proximity/persistent person IDs.
                tr_people = yolo_tracker_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=YOLO_TRACK_CONF,
                    imgsz=YOLO_TRACK_IMGSZ,
                    verbose=False,
                )
                tr_boxes = tr_people[0].boxes if tr_people and tr_people[0].boxes is not None else []
                for box in tr_boxes:
                    if box.id is None:
                        continue
                    tid  = int(box.id[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    people_boxes[tid] = xyxy

                # If detection-time person boxes were sparse, reuse tracked boxes
                # so carry-association can still run.
                if not person_det_boxes and people_boxes:
                    person_det_boxes = [b.tolist() for b in people_boxes.values()]

                fh, fw = frame.shape[:2]
                carry_dets = []
                for cand in carry_candidates:
                    person_box = _associate_carry_object(
                        person_det_boxes,
                        cand["bbox"],
                        fw,
                        fh,
                    )
                    if person_box is None:
                        continue

                    context_bbox = _merge_boxes(person_box, cand["bbox"])
                    carry_dets.append({
                        "label": cand["label"],
                        "confidence": cand["confidence"],
                        "bbox": cand["bbox"],
                        "context_bbox": context_bbox,
                    })

                with state_lock:
                    state["person_count"] = person_count_total
                    state["class_counts"] = dict(class_counts)
                person_total_for_summary = person_count_total

                # ── Weapon detection ──────────────────────────────────────────
                w_dets, w_trigger, w_crop = run_weapons(frame, edge_dets)
                combined_weapon_dets = list(w_dets)
                seen_weapon_keys = {
                    (d.get("label"), *(d.get("bbox") or []))
                    for d in combined_weapon_dets
                }
                for d in carry_dets:
                    key = (d["label"], *d["bbox"])
                    if key in seen_weapon_keys:
                        continue
                    combined_weapon_dets.append({
                        "label": d["label"],
                        "confidence": d["confidence"],
                        "bbox": d["bbox"],
                    })

                if w_dets:
                    weapon_consecutive += 1
                else:
                    weapon_consecutive  = 0

                if carry_dets:
                    carry_consecutive += 1
                else:
                    carry_consecutive = 0

                weapon_dets = combined_weapon_dets

                if carry_consecutive >= CARRY_TRIGGER_MIN_FRAMES:
                    best_carry = max(carry_dets, key=lambda d: d["confidence"])
                    carry_reason = (
                        f"Person carrying {best_carry['label']} "
                        f"({int(best_carry['confidence']*100)}%)"
                    )
                    carry_crop = pad_crop(frame, best_carry["context_bbox"], 80)
                    yolo_trigger = (
                        "weapon",
                        carry_reason,
                        carry_crop,
                        _incident_weapon_prompt(best_carry["label"]),
                        best_carry["label"],
                    )
                    annotated = draw_weapons(annotated, weapon_dets)
                elif weapon_consecutive >= WEAPON_MIN_FRAMES and w_trigger:
                    best_weapon = max(w_dets, key=lambda d: d["confidence"]) if w_dets else None
                    yolo_trigger = (
                        "weapon",
                        w_trigger,
                        w_crop,
                        "",
                        (best_weapon or {}).get("label", "weapon"),
                    )
                    annotated    = draw_weapons(annotated, weapon_dets)

                with state_lock:
                    state["weapon_detections"] = weapon_dets

                # ── Proximity trigger ─────────────────────────────────────────
                if yolo_trigger is None:
                    pr = prox.update(people_boxes)
                    if pr:
                        pair_ids, mb = pr
                        crop = pad_crop(frame, mb)
                        yolo_trigger = (
                            "proximity",
                            f"Sustained contact — IDs {pair_ids}",
                            crop,
                            "",
                            "",
                        )

                # ── Count change trigger ───────────────────────────────────────
                cur_count = person_count_total
                if cur_count != prev_count and prev_count != -1:
                    log.info(f"[COUNT] {prev_count} → {cur_count}")
                    if yolo_trigger is None:
                        yolo_trigger = (
                            "count_change",
                            f"Person count changed: {prev_count} → {cur_count}",
                            frame.copy(),
                            "",
                            "",
                        )
                prev_count = cur_count

            # ── VLM trigger dispatch ──────────────────────────────────────────
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
                last_vlm  = state["last_vlm_time"]

            if yolo_trigger:
                kind, reason, crop, auto_prompt, label_hint = yolo_trigger
                if vlm_on and vlm_model and get_vram_pct() < 75:
                    with state_lock:
                        state["last_vlm_time"] = now
                    if kind == "weapon":
                        weapon_prompt = (prompts.get("weapon", "") or "").strip() or auto_prompt
                        launch_vlm("trigger", do_weapon_vlm,
                                   (crop.copy(), weapon_prompt, label_hint))
                    elif kind == "proximity":
                        launch_vlm("trigger", do_proximity_vlm,
                                   (crop.copy(), prompts.get("proximity", "")))
                    elif kind == "count_change":
                        launch_vlm("trigger", do_count_change_vlm,
                                   (crop.copy(), prompts.get("count_change", "")))
                if cur_alert == "CLEAR" and kind != "count_change":
                    push_alert("YELLOW", reason)

            # ── Passive scene VLM (interval-based) ────────────────────────────
            if (vlm_on and vlm_model
                    and now - last_vlm >= vlm_ivl
                    and not yolo_trigger
                    and get_vram_pct() < 75):
                with state_lock:
                    state["last_vlm_time"] = now
                launch_vlm("passive", do_scene_vlm, (frame.copy(), ""))

            # ── Person description (VLM + YOLO both on) ───────────────────────
            if yolo_on and vlm_on and vlm_model:
                for tid, xyxy in people_boxes.items():
                    if tid in described_ids:
                        continue

                    crop = pad_crop(frame, xyxy, 40)
                    if crop.size <= 0:
                        continue

                    def _person_desc(t_id=tid, c=crop.copy()):
                        desc = smolvlm_infer(
                            c,
                            "Describe this person's actions in 1 sentence. "
                            "What is most noticeable about them?",
                            vlm_model, vlm_processor, max_tokens=60
                        )
                        if desc:
                            with state_lock:
                                state["person_log"].append({
                                    "time":        datetime.now().strftime("%H:%M:%S"),
                                    "track_id":    t_id,
                                    "description": desc,
                                })
                                state["person_log"] = state["person_log"][-50:]

                    launched = launch_vlm("passive", _person_desc, ())
                    if launched:
                        described_ids.add(tid)
                        # Only one passive VLM task can run at a time.
                        break

            # ── Clear stale alert ─────────────────────────────────────────────
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
            if (cur_alert != "CLEAR"
                    and not yolo_trigger
                    and now - last_red >= RED_HOLD_SEC
                    and not vlm_running()):
                push_alert("CLEAR", "")

            # ── Detection summary ──────────────────────────────────────────────
            w_names = [d["label"] for d in weapon_dets]
            p_str   = f"{person_total_for_summary} person(s)" if person_total_for_summary else ""
            w_str   = f"⚠️ {', '.join(w_names)}"      if w_names    else ""
            o_str   = ", ".join(f"{v}× {k}" for k,v in class_counts.items()
                                if k != "person" and k not in w_names)
            with state_lock:
                state["detection_summary"] = (
                    " | ".join(p for p in [p_str,w_str,o_str] if p)
                    or ("Streaming…" if not yolo_on else "Nothing detected")
                )

            # ── Overlay ────────────────────────────────────────────────────────
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with engine["frame_lock"]:
                engine["frame"] = buf.tobytes()

        except torch.cuda.OutOfMemoryError:
            log.error(f"OOM frame #{frame_count}")
            torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(0.5)
        except Exception as e:
            log.error(f"Engine #{frame_count}: {e}")

        time.sleep(max(0, delay-(time.time()-t0)))

    cap.release()
    engine["running"] = False
    with state_lock:
        state["weapon_detections"] = []
        state["source_fps"]        = 0.0
        state["class_counts"]      = {}
    push_alert("CLEAR", "")
    log.info("[ENGINE] Stopped")


# ── MJPEG ─────────────────────────────────────────────────────────────────────
def mjpeg_gen():
    while True:
        with engine["frame_lock"]:
            f = engine["frame"]
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.033)


def _stop():
    engine["running"] = False
    # Abort any running VLM
    vlm_abort.set()
    if engine["thread"] and engine["thread"].is_alive():
        engine["thread"].join(timeout=4)
    vlm_abort.clear()
    with state_lock:
        state.update({
            "alert":"CLEAR","reason":"","person_count":0,
            "weapon_detections":[],"source_fps":0.0,
            "class_counts":{},
            "detection_summary":"","scene_description":"",
            "prompt_outputs": _empty_prompt_outputs(),
        })
    with engine["frame_lock"]:
        engine["frame"] = None

def _start(source):
    _stop()
    engine["source"]  = source
    engine["running"] = True
    engine["thread"]  = threading.Thread(target=run_engine,
                                         args=(source,), daemon=True)
    engine["thread"].start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start/camera")
def start_camera(index: int = 0):
    _start(index); return {"status":"started","source":f"camera:{index}"}

@app.post("/start/video")
async def start_video(file: UploadFile = File(...)):
    p = UPLOAD_DIR / file.filename
    p.write_bytes(await file.read())
    _start(str(p)); return {"status":"started","source":file.filename}

@app.post("/start/path")
def start_path(path: str):
    try:
        resolved_source, meta = resolve_source_input(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to start from link/path: {e}")

    _start(resolved_source)
    return {
        "status": "started",
        "source": path,
        "resolved": meta,
    }

@app.post("/stop")
def stop():
    _stop(); return {"status":"stopped"}

@app.get("/status")
def get_status():
    with state_lock:
        prompt_outputs = state.get("prompt_outputs")
        if not isinstance(prompt_outputs, dict):
            prompt_outputs = _empty_prompt_outputs()

        return {
            "running":           engine["running"],
            "source":            str(engine["source"]),
            "alert":             state["alert"],
            "reason":            state["reason"],
            "description":       state["vlm_description"],
            "threat_type":       state["threat_type"],
            "scene_description": state["scene_description"],
            "detection_summary": state["detection_summary"],
            "person_count":      state["person_count"],
            "class_counts":      dict(state["class_counts"]),
            "weapon_detections": state["weapon_detections"],
            "source_fps":        state["source_fps"],
            "yolo_enabled":      state["yolo_enabled"],
            "vlm_enabled":       state["vlm_enabled"],
            "vlm_interval":      state["vlm_interval"],
            "mode_switching":    state["mode_switching"],
            "trigger_prompts":   dict(state["trigger_prompts"]),
            "prompt_outputs": {
                "proximity":    dict(prompt_outputs.get("proximity", {"description": "", "time": ""})),
                "count_change": dict(prompt_outputs.get("count_change", {"description": "", "time": ""})),
                "weapon":       dict(prompt_outputs.get("weapon", {"description": "", "time": ""})),
                "scene":        dict(prompt_outputs.get("scene", {"description": "", "time": ""})),
            },
            "vram_pct":          round(get_vram_pct(), 1),
        }


@app.get("/prompt_outputs/csv")
def get_prompt_outputs_csv():
    try:
        with prompt_csv_lock:
            _ensure_prompt_csv_header()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to prepare prompt output CSV: {e}")

    return FileResponse(
        path=str(PROMPT_CSV_PATH),
        media_type="text/csv",
        filename="prompt_outputs.csv",
    )


@app.get("/prompt_outputs/records")
def get_prompt_output_records(type: str = "", timeline: str = "", description: str = ""):
    try:
        with prompt_csv_lock:
            rows = _read_prompt_csv_rows_unlocked()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to read prompt output records: {e}")

    filtered = [
        row for row in rows
        if _matches_prompt_filters(row, type, timeline, description)
    ]
    filtered.sort(key=lambda r: int(r.get("id", 0)), reverse=True)

    return {
        "records": filtered,
        "count": len(filtered),
        "total": len(rows),
        "filters": {
            "type": type,
            "timeline": timeline,
            "description": description,
        },
    }


@app.delete("/prompt_outputs/records/{record_id}")
def delete_prompt_output_record(record_id: int):
    if record_id <= 0:
        raise HTTPException(status_code=400, detail="record_id must be > 0")

    try:
        with prompt_csv_lock:
            rows = _read_prompt_csv_rows_unlocked()
            exists = any(int(row.get("id", 0)) == record_id for row in rows)
            if not exists:
                raise HTTPException(status_code=404, detail="Record not found")

            remaining = [row for row in rows if int(row.get("id", 0)) != record_id]
            _write_prompt_csv_rows_unlocked(remaining)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to delete record: {e}")

    return {
        "deleted_id": record_id,
        "remaining": len(remaining),
    }


@app.delete("/prompt_outputs/records")
def clear_prompt_output_records(type: str = "", timeline: str = "", description: str = ""):
    try:
        with prompt_csv_lock:
            rows = _read_prompt_csv_rows_unlocked()
            to_delete = [
                row for row in rows
                if _matches_prompt_filters(row, type, timeline, description)
            ]
            delete_ids = {int(row.get("id", 0)) for row in to_delete}
            remaining = [row for row in rows if int(row.get("id", 0)) not in delete_ids]
            _write_prompt_csv_rows_unlocked(remaining)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to clear records: {e}")

    return {
        "deleted": len(to_delete),
        "remaining": len(remaining),
        "filters": {
            "type": type,
            "timeline": timeline,
            "description": description,
        },
    }

@app.get("/alerts")
def get_alerts():
    with state_lock: return state["alert_log"]

@app.get("/persons")
def get_persons():
    with state_lock: return state["person_log"]

@app.get("/vram")
def get_vram():
    try:
        p = torch.cuda.get_device_properties(0)
        t = p.total_memory / 1024**3
        return {
            "gpu_name":     p.name,
            "total_gb":     round(t, 2),
            "allocated_gb": round(torch.cuda.memory_allocated()/1024**3, 2),
            "reserved_gb":  round(torch.cuda.memory_reserved()/1024**3,  2),
            "free_gb":      round(t - torch.cuda.memory_reserved()/1024**3, 2),
            "usage_pct":    round(get_vram_pct(), 1),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/weapon_classes")
def get_weapon_classes():
    return {
        "threat_model": [],
        "yolo_edge":    sorted(set(yolo_edge_classes.values())),
        "yolo_all":     [yolo_all_classes[cid] for cid in sorted(yolo_all_classes)],
    }

# ── YOLO toggle ────────────────────────────────────────────────────────────────
@app.post("/yolo/enable")
def yolo_enable():
    with state_lock: state["yolo_enabled"] = True
    return {"yolo_enabled": True}

@app.post("/yolo/disable")
def yolo_disable():
    with state_lock:
        state["yolo_enabled"]      = False
        state["weapon_detections"] = []
        state["person_count"]      = 0
        state["class_counts"]      = {}
        state["detection_summary"] = ""
    return {"yolo_enabled": False}

# ── VLM toggle ─────────────────────────────────────────────────────────────────
@app.post("/vlm/enable")
def vlm_enable():
    with state_lock:
        if state["mode_switching"]:
            return {"error": "Already switching"}
        if vlm_model is None or vlm_processor is None:
            return {"error": "VLM model unavailable in current environment"}
        state["mode_switching"] = True
    def do():
        reload_vlm()
        with state_lock:
            state["vlm_enabled"]    = True
            state["mode_switching"] = False
    threading.Thread(target=do, daemon=True).start()
    return {"vlm_enabled": True, "mode_switching": True}

@app.post("/vlm/disable")
def vlm_disable():
    vlm_abort.set()
    with state_lock:
        state["vlm_enabled"]       = False
        state["scene_description"] = ""
    time.sleep(0.2)
    vlm_abort.clear()
    offload_vlm()
    return {"vlm_enabled": False}

# ── VLM interval (2–30 s) ─────────────────────────────────────────────────────
@app.post("/vlm/interval")
def set_interval(seconds: float):
    seconds = max(2.0, min(seconds, 30.0))
    with state_lock: state["vlm_interval"] = seconds
    return {"vlm_interval": seconds}

# ── Trigger prompts ────────────────────────────────────────────────────────────
@app.get("/trigger_prompts")
def get_trigger_prompts():
    with state_lock: return dict(state["trigger_prompts"])

@app.post("/trigger_prompts/{trigger_type}")
def set_trigger_prompt(trigger_type: str, prompt: str = ""):
    if trigger_type not in ("proximity", "count_change", "weapon"):
        return {"error": "trigger_type must be proximity | count_change | weapon"}
    with state_lock:
        state["trigger_prompts"][trigger_type] = prompt.strip()
    return {"trigger_type": trigger_type, "prompt": prompt.strip()}

@app.delete("/trigger_prompts/{trigger_type}")
def clear_trigger_prompt(trigger_type: str):
    if trigger_type not in ("proximity", "count_change", "weapon"):
        return {"error": "Invalid trigger type"}
    with state_lock:
        state["trigger_prompts"][trigger_type] = ""
    return {"trigger_type": trigger_type, "prompt": ""}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)