import cv2, time, json, threading, logging, re
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR        = Path(__file__).resolve().parent
YOLO_MODEL_PATH  = MODEL_DIR / "yolo11l.pt"
VLM_MODEL_ID    = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# yolo11l is the single detector model used by this service.
# Edge weapons are a subset of YOLO class names that we elevate to alerts.
YOLO_EDGE_NAMES = {"knife", "axe", "crowbar", "scissors", "blade", "machete"}
EDGE_WEAPON_PATTERNS = (
    re.compile(r"\b(?:knife|knives|dagger|switchblade)\b"),
    re.compile(r"\bscissor(?:s)?\b"),
    re.compile(r"\bcrowbar(?:s)?\b"),
    re.compile(r"\b(?:axe|axes|ax|hatchet)\b"),
    re.compile(r"\bmachete(?:s)?\b"),
    re.compile(r"\bblade(?:s)?\b"),
)

PROXIMITY_DURATION = 2.5
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}

WEAPON_COLORS = {
    "knife":   (0,   0,   255),   # red
    "axe":     (0,  128,  255),   # orange
    "baseball bat": (255, 170, 0),
    "bat":     (255, 170, 0),
    "crowbar": (0,  200,  150),   # teal
    "scissors":(200, 0,   200),   # magenta
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "alert":             "CLEAR",
    "reason":            "",
    "vlm_description":   "",
    "threat_type":       "none",
    "last_vlm_time":     0.0,
    "last_red_time":     0.0,
    "alert_log":         [],
    "scene_description": "",
    "detection_summary": "",
    "weapon_detections": [],
    "source_fps":        0.0,
    "person_log":        [],
    "person_count":      0,
    "class_counts":      {},
    "yolo_enabled":      False,
    "vlm_enabled":       False,
    "vlm_interval":      10.0,       # passive scene interval 2–30s
    "mode_switching":    False,
    "trigger_prompts": {             # per-trigger customizable prompts
        "proximity":    "",
        "count_change": "",
        "weapon":       "",
    },
    "prompt_outputs": {
        "proximity":    {"description": "", "time": ""},
        "count_change": {"description": "", "time": ""},
        "weapon":       {"description": "", "time": ""},
        "scene":        {"description": "", "time": ""},
    },
}
state_lock = threading.Lock()

# Filled after model load
yolo_all_classes: dict = {}  # from yolo26n names  {cls_id: label}
yolo_edge_classes: dict = {} # from yolo26n names  {cls_id: label}


def normalize_weapon_label(name: str) -> str:
    """Normalize label variants so downstream logic/UI use consistent names."""
    n = str(name).strip().lower()
    if re.search(r"\b(?:knife|knives|dagger|switchblade)\b", n):
        return "knife"
    if "scissor" in n:
        return "scissors"
    if "crowbar" in n:
        return "crowbar"
    if re.search(r"\b(?:axe|axes|ax|hatchet)\b", n):
        return "axe"
    if "machete" in n:
        return "machete"
    if "blade" in n:
        return "blade"
    return n


def is_edge_weapon_name(name: str) -> bool:
    n = str(name).strip().lower()
    if normalize_weapon_label(n) in YOLO_EDGE_NAMES:
        return True
    return any(p.search(n) for p in EDGE_WEAPON_PATTERNS)

# ── VLM abort event (used by StoppingCriteria) ────────────────────────────────
vlm_abort = threading.Event()

# StoppingCriteria defined at module level — imported lazily so no crash if
# transformers is not installed (vlm_model will be None anyway)
_abort_sc = None
def _init_abort_criteria():
    global _abort_sc
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
        class _AbortCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return vlm_abort.is_set()
        _abort_sc = StoppingCriteriaList([_AbortCriteria()])
        log.info("[VLM] AbortStopCriteria ready")
    except Exception as e:
        log.warning(f"[VLM] StoppingCriteria unavailable: {e}")
        _abort_sc = None


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_yolo():
    log.info("Loading yolo11l...")
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"yolo11l model not found: {YOLO_MODEL_PATH}")

    model = YOLO(str(YOLO_MODEL_PATH))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    yolo_all_classes.clear()
    yolo_all_classes.update({int(cid): str(name) for cid, name in names.items()})

    # Scan class names for edge-case weapons (axe, crowbar, scissors…)
    yolo_edge_classes.clear()
    yolo_edge_classes.update({
        cid: normalize_weapon_label(name)
        for cid, name in yolo_all_classes.items()
        if is_edge_weapon_name(name)
    })

    if yolo_edge_classes:
        log.info(f"[YOLO] Edge weapon classes found: {yolo_edge_classes}")
    else:
        log.info("[YOLO] No edge-weapon class labels found in yolo11l names")
    log.info(f"[YOLO] Ready — {len(yolo_all_classes)} classes")
    return model


def load_vlm():
    try:
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )
        log.info("Loading SmolVLM2-2.2B-Instruct (4-bit)...")
        bnb  = BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        proc = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        mdl  = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_ID, quantization_config=bnb,
            device_map="cuda", _attn_implementation="eager",
        )
        mdl.eval()
        _init_abort_criteria()
        log.info("SmolVLM2-2.2B ready")
        return mdl, proc
    except Exception as e:
        log.warning(f"[VLM] Load failed ({e}) — YOLO-only mode")
        return None, None


# ── Default prompts ───────────────────────────────────────────────────────────
DEFAULT_PROXIMITY_PROMPT = (
    "Describe what people are doing. "
    "Start with: Safe / Suspicious / Threatening — then explain."
)
DEFAULT_COUNT_CHANGE_PROMPT = (
    "The number of people in this camera view just changed. "
    "Describe what is currently happening. "
    "Focus on actions and movement."
)
DEFAULT_WEAPON_PROMPT = (
    "A potential dangerous object is visible in this surveillance image. "
    "Decide if a person is carrying/holding it and whether behavior appears threatening. "
    "Respond ONLY as JSON."
)
DEFAULT_SCENE_PROMPT = (
    "Describe exactly what is happening in this surveillance scene in one sentence. "
    "Focus on people's actions."
)


# ── VLM inference ─────────────────────────────────────────────────────────────
def smolvlm_infer(crop_bgr: np.ndarray, prompt: str,
                  vlm_model, processor, max_tokens: int = 120) -> str:
    if vlm_abort.is_set():
        return ""
    if vlm_model is None or processor is None:
        return ""
    try:
        h, w  = crop_bgr.shape[:2]
        scale = 512 / max(h, w)
        if scale < 1.0:
            crop_bgr = cv2.resize(crop_bgr, (int(w*scale), int(h*scale)),
                                  interpolation=cv2.INTER_AREA)
        img  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        tp  = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = processor(text=tp, images=[img], return_tensors="pt")\
              .to(vlm_model.device, dtype=torch.bfloat16)

        gen_kwargs = dict(do_sample=True, temperature=0.3,
                          top_p=0.9, max_new_tokens=max_tokens)
        if _abort_sc is not None:
            gen_kwargs["stopping_criteria"] = _abort_sc

        with torch.no_grad():
            out = vlm_model.generate(**inp, **gen_kwargs)

        # Discard result if we were aborted mid-generation
        if vlm_abort.is_set():
            log.info("[VLM] Generation aborted — discarding result")
            return ""

        res = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        if "Assistant:" in res:
            res = res.split("Assistant:")[-1].strip()
        lines = [l.strip() for l in res.split("\n") if l.strip()]
        return lines[-1] if lines else res

    except torch.cuda.OutOfMemoryError:
        log.error("[VLM] OOM — clearing VRAM")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return ""
    except Exception as e:
        log.warning(f"[VLM] Infer error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1", "threat", "danger", "violent"}


def _normalize_confidence(value: str, threat: bool) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    if any(k in text for k in ["high", "certain", "strong"]):
        return "high"
    if any(k in text for k in ["medium", "moderate", "likely"]):
        return "medium"
    return "medium" if threat else "low"


def run_vlm_threat(crop_bgr: np.ndarray, vlm_model, processor,
                   custom_prompt: str = "") -> dict:
    schema_hint = (
        "Return ONLY valid JSON with keys: "
        "threat (boolean), type (string), confidence (low|medium|high), description (string)."
    )
    base_prompt = (custom_prompt or DEFAULT_WEAPON_PROMPT).strip()
    prompt = f"{base_prompt}\n{schema_hint}"

    raw = smolvlm_infer(crop_bgr, prompt, vlm_model, processor, max_tokens=120)
    if not raw:
        return {
            "threat": False,
            "type": "none",
            "confidence": "low",
            "description": "Aborted or no response",
        }

    text = raw.strip()
    parsed = None
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            parsed = json.loads(match.group(0))
    except Exception:
        parsed = None

    if parsed is not None:
        threat = _as_bool(parsed.get("threat", False))
        threat_type = str(parsed.get("type", "")).strip().lower() or ("weapon" if threat else "none")
        confidence = _normalize_confidence(parsed.get("confidence", ""), threat)
        description = str(parsed.get("description", "")).strip() or text[:180]
        return {
            "threat": threat,
            "type": threat_type,
            "confidence": confidence,
            "description": description,
        }

    lower = text.lower()
    danger_terms = ["threat", "attack", "assault", "violent", "weapon", "wield", "strike"]
    safe_terms = ["not threatening", "no threat", "safe", "non-threatening", "harmless"]
    threat = any(t in lower for t in danger_terms) and not any(t in lower for t in safe_terms)

    threat_type = "none"
    for t in ["baseball bat", "bat", "axe", "knife", "crowbar", "scissors", "blade", "machete"]:
        if t in lower:
            threat_type = "bat" if t in {"baseball bat", "bat"} else t
            break
    if threat and threat_type == "none":
        threat_type = "weapon"

    return {
        "threat": threat,
        "type": threat_type,
        "confidence": "medium" if threat else "low",
        "description": text[:180],
    }


# ── Weapon detection ──────────────────────────────────────────────────────────
def run_weapons(frame: np.ndarray, yolo_extra_boxes: list = None) -> tuple:
    """
    Use weapon detections already found in the main YOLO26n pass.
    Returns: (detections, trigger_reason, trigger_crop)
    """
    dets       = []
    fh, fw     = frame.shape[:2]

    # ── Edge weapons from main YOLO26n ────────────────────────────────────────
    if yolo_extra_boxes:
        for d in yolo_extra_boxes:
            bbox = d.get("bbox", [])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            x2 = max(0, min(x2, fw - 1))
            y2 = max(0, min(y2, fh - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            conf = float(d.get("confidence", 0.0))
            dets.append({
                "label":      normalize_weapon_label(d.get("label", "")),
                "confidence": max(0.0, min(conf, 1.0)),
                "bbox":       [x1, y1, x2, y2],
            })
            log.info(f"[WEAPON] {dets[-1]['label']} @ {int(dets[-1]['confidence']*100)}%")

    trigger = None
    crop    = None
    if dets:
        best    = max(dets, key=lambda d: d["confidence"])
        trigger = f"Weapon detected: {best['label']} ({int(best['confidence']*100)}%)"
        b       = best["bbox"]
        pad     = 60
        crop    = frame[max(0,b[1]-pad):min(fh,b[3]+pad),
                        max(0,b[0]-pad):min(fw,b[2]+pad)]
    return dets, trigger, crop


def draw_weapons(frame: np.ndarray, dets: list) -> np.ndarray:
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        lbl   = f"{d['label']} {int(d['confidence']*100)}%"
        color = WEAPON_COLORS.get(d["label"], (0, 0, 255))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+6,y1), color, -1)
        cv2.putText(frame, lbl, (x1+3,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame


# ── Utilities ─────────────────────────────────────────────────────────────────
def people_close(b1, b2) -> bool:
    c1   = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2   = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    dist = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
    avg_h = ((b1[3]-b1[1]) + (b2[3]-b2[1])) / 2
    return dist < avg_h * 0.7


def merged_bbox(b1, b2) -> list:
    return [min(b1[0],b2[0]), min(b1[1],b2[1]),
            max(b1[2],b2[2]), max(b1[3],b2[3])]


def pad_crop(frame: np.ndarray, box, pad: int = 60) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[max(0,int(box[1])-pad):min(h,int(box[3])+pad),
                 max(0,int(box[0])-pad):min(w,int(box[2])+pad)]


def push_alert(alert: str, reason: str, vlm_result: dict = None):
    with state_lock:
        state["alert"]  = alert
        state["reason"] = reason
        if vlm_result:
            state["vlm_description"] = vlm_result.get("description", "")
            state["threat_type"]     = vlm_result.get("type", "none")
        if alert == "RED":
            state["last_red_time"] = time.time()
        if alert != "CLEAR":
            entry = {"time":  datetime.now().strftime("%H:%M:%S"),
                     "alert": alert, "reason": reason}
            if vlm_result:
                entry["vlm"] = vlm_result.get("description", "")
            state["alert_log"].append(entry)
            state["alert_log"] = state["alert_log"][-100:]
    log.info(f"[{alert}] {reason}")


class ProximityTracker:
    def __init__(self):
        self.pair_since: dict = {}

    def update(self, boxes: dict):
        ids, now  = list(boxes.keys()), time.time()
        active    = set()
        result    = None
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                pair = (min(a,b), max(a,b))
                if people_close(boxes[a], boxes[b]):
                    active.add(pair)
                    self.pair_since.setdefault(pair, now)
                    if now - self.pair_since[pair] >= PROXIMITY_DURATION:
                        result = (pair, merged_bbox(boxes[a], boxes[b]))
        for p in list(self.pair_since):
            if p not in active:
                del self.pair_since[p]
        return result