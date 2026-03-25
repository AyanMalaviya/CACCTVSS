import cv2, time, json, threading, logging
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "yolo26n.pt"
VLM_MODEL_ID    = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

THREAT_REPO     = "Subh775/Threat-Detection-YOLOv8n"
THREAT_FILE     = "weights/best.pt"

# Gun (cls 1) removed — too many false positives on furniture/chairs
# Knife (cls 4) only from threat model
# Axe / crowbar / scissors scanned from yolo26n class names
ACTIVE_THREATS  = {"knife"}
YOLO_EDGE_NAMES = {"axe", "crowbar", "scissors", "scissor", "blade", "machete"}

PROXIMITY_DURATION = 2.5
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}

WEAPON_COLORS = {
    "knife":   (0,   0,   255),   # red
    "axe":     (0,  128,  255),   # orange
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
    "yolo_enabled":      False,
    "vlm_enabled":       False,
    "vlm_interval":      10.0,       # passive scene interval 2–30s
    "mode_switching":    False,
    "trigger_prompts": {             # per-trigger customizable prompts
        "proximity":    "",
        "count_change": "",
        "weapon":       "",
    },
}
state_lock = threading.Lock()

# Filled after model load
threat_classes: dict = {}    # from threat model  {cls_id: label}
yolo_edge_classes: dict = {} # from yolo26n names  {cls_id: label}

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
    log.info("Loading YOLO26n...")
    model = YOLO(YOLO_MODEL_PATH)
    # Scan class names for edge-case weapons (axe, crowbar, scissors…)
    global yolo_edge_classes
    yolo_edge_classes = {
        cid: name
        for cid, name in model.names.items()
        if name.lower() in YOLO_EDGE_NAMES
    }
    if yolo_edge_classes:
        log.info(f"[YOLO] Edge weapon classes found: {yolo_edge_classes}")
    else:
        log.info("[YOLO] No axe/crowbar/scissors in yolo26n — only knife from threat model")
    log.info(f"[YOLO] Ready — {len(model.names)} classes")
    return model


def load_threat_model():
    global threat_classes
    try:
        log.info(f"[THREAT] Downloading {THREAT_REPO}...")
        path  = hf_hub_download(repo_id=THREAT_REPO, filename=THREAT_FILE)
        model = YOLO(path)
        # Only keep classes in ACTIVE_THREATS (knife only — gun excluded)
        threat_classes = {
            cid: name
            for cid, name in model.names.items()
            if name.lower() in ACTIVE_THREATS
        }
        if not threat_classes:
            threat_classes = {4: "knife"}  # known ID fallback
        log.info(f"[THREAT] Loaded. Active: {threat_classes} | Skipped: gun/explosive/grenade")
        return model
    except Exception as e:
        log.warning(f"[THREAT] Download failed: {e} — fallback YOLOv8n COCO")
        model          = YOLO("yolov8n.pt")
        threat_classes = {49: "knife"}
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
    "Two people have come very close together in this surveillance image. "
    "Describe what they are doing in one sentence. "
    "Start with: Safe / Suspicious / Threatening — then explain."
)
DEFAULT_COUNT_CHANGE_PROMPT = (
    "The number of people in this camera view just changed. "
    "Describe in one sentence what is currently happening. "
    "Focus on actions and movement."
)
DEFAULT_WEAPON_PROMPT = (
    "A potential bladed weapon has been detected in this surveillance image. "
    "Is someone actively holding or using it? "
    "Describe in one factual sentence what you see."
)
DEFAULT_SCENE_PROMPT = (
    "Describe exactly what is happening in this surveillance scene in one sentence. "
    "Focus on people's actions."
)


# ── VLM inference ─────────────────────────────────────────────────────────────
def smolvlm_infer(crop_bgr: np.ndarray, prompt: str,
                  vlm_model, processor, max_tokens: int = 80) -> str:
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


def run_vlm_threat(crop_bgr: np.ndarray, vlm_model, processor,
                   custom_prompt: str = "") -> dict:
    prompt = custom_prompt or DEFAULT_WEAPON_PROMPT
    raw    = smolvlm_infer(crop_bgr, prompt, vlm_model, processor, max_tokens=100)
    if not raw:
        return {"threat": False, "type": "none",
                "confidence": "low", "description": "Aborted or no response"}
    try:
        js = raw[raw.rfind("{"):raw.rfind("}")+1]
        return json.loads(js)
    except Exception:
        threat = "threat" in raw.lower() and "true" in raw.lower()
        return {
            "threat":      threat,
            "type":        "assault" if threat else "none",
            "confidence":  "medium"  if threat else "low",
            "description": raw[:120],
        }


# ── Weapon detection ──────────────────────────────────────────────────────────
def run_weapons(threat_model: YOLO, frame: np.ndarray,
                yolo_extra_boxes: list = None) -> tuple:
    """
    Run threat model (knife only) + optionally pass axe/scissor detections
    already found in yolo_extra_boxes from the main YOLO pass.
    Returns: (detections, trigger_reason, trigger_crop)
    """
    dets       = []
    fh, fw     = frame.shape[:2]
    frame_area = fh * fw

    # ── Threat model (knife) ──────────────────────────────────────────────────
    if threat_model is not None:
        results = threat_model(frame, conf=0.60, imgsz=640, verbose=False)
        for box in results[0].boxes:
            cid = int(box.cls[0])
            if cid not in threat_classes:
                continue
            conf  = float(box.conf[0])
            label = threat_classes[cid]
            xyxy  = box.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = map(int, xyxy)
            bw, bh = x2-x1, y2-y1
            area   = bw*bh
            aspect = bw / max(bh, 1)
            if area < frame_area*0.015:   continue
            if label == "knife" and not (1.2 < aspect < 8.0): continue
            if bw > fw*0.85 or bh > fh*0.85:  continue
            dets.append({"label": label, "confidence": round(conf,2),
                         "bbox": [x1,y1,x2,y2]})
            log.info(f"[WEAPON] {label} @ {int(conf*100)}%")

    # ── Edge weapons from main YOLO (axe/crowbar/scissors) ───────────────────
    if yolo_extra_boxes:
        dets.extend(yolo_extra_boxes)

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