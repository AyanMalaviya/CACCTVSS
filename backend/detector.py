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

PROXIMITY_DURATION = 2.5
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}

# ── Threat model config ───────────────────────────────────────────────────────
# Subh775/Threat-Detection-YOLOv8n classes:
#   0: Bike  1: Gun  2: Explosive  3: Grenade  4: Knife
# We only activate Gun and Knife — ignore Bike/Explosive/Grenade
THREAT_MODEL_REPO     = "Subh775/Threat-Detection-YOLOv8n"
THREAT_MODEL_FILENAME = "weights/best.pt"
ACTIVE_THREAT_NAMES   = {"gun", "knife"}   # lowercase match
threat_classes: dict  = {}   # filled after load: {cls_id: label}

# Colour per weapon in overlay
WEAPON_COLORS = {
    "gun":   (128, 0, 255),   # purple
    "knife": (0,   0, 255),   # red
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Shared State ──────────────────────────────────────────────────────────────
state = {
    "alert":             "CLEAR",
    "reason":            "",
    "vlm_description":   "",
    "threat_type":       "none",
    "last_vlm_time":     0.0,
    "last_red_time":     0.0,
    "alert_log":         [],
    "scene_description": "Waiting for first analysis...",
    "detection_summary": "",
    "weapon_detections": [],
    "source_fps":        0.0,
    "person_log":        [],
    "person_count":      0,
    "vlm_enabled":       True,
    "detection_mode":    "both",
    "vlm_interval":      15.0,
    "mode_switching":    False,
    "custom_prompt":     "",
}

state_lock = threading.Lock()


# ── Model Loaders ─────────────────────────────────────────────────────────────
def load_yolo():
    log.info("Loading YOLO26n...")
    model = YOLO(YOLO_MODEL_PATH)
    log.info(f"YOLO26n ready — {len(model.names)} classes.")
    return model


def load_threat_model():
    """
    Downloads Subh775/Threat-Detection-YOLOv8n.
    Only activates Gun (cls 1) and Knife (cls 4).
    Falls back to YOLOv8n COCO knife if download fails.
    """
    global threat_classes
    try:
        log.info(f"[THREAT] Downloading {THREAT_MODEL_REPO} ...")
        path  = hf_hub_download(
            repo_id=THREAT_MODEL_REPO,
            filename=THREAT_MODEL_FILENAME,
        )
        model = YOLO(path)
        # Only keep classes whose names are in ACTIVE_THREAT_NAMES
        threat_classes = {
            cls_id: name
            for cls_id, name in model.names.items()
            if name.lower() in ACTIVE_THREAT_NAMES
        }
        log.info(
            f"[THREAT] Loaded. All classes: {model.names} "
            f"| Active: {threat_classes}"
        )
        return model
    except Exception as e:
        log.warning(f"[THREAT] Download failed: {e}. Falling back to YOLOv8n COCO.")
        model          = YOLO("yolov8n.pt")
        threat_classes = {49: "knife"}   # COCO knife class
        return model


def load_vlm():
    try:
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )
        log.info("Loading SmolVLM2-2.2B-Instruct (4-bit)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        model     = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_ID,
            quantization_config=bnb,
            device_map="cuda",
            _attn_implementation="eager",
        )
        model.eval()
        log.info("SmolVLM2-2.2B ready.")
        return model, processor
    except Exception as e:
        log.warning(f"VLM load failed ({e}). Running YOLO-only mode.")
        return None, None


# ── VLM Prompts ───────────────────────────────────────────────────────────────
VLM_THREAT_PROMPT = """Look at this surveillance image and answer: is there any dangerous or threatening behavior happening?

Answer YES only if you can clearly see one of these:
- Someone physically attacking or hitting another person
- Someone breaking a door, window, or barrier by force
- A person on the ground being hurt or restrained

Answer NO for everything else including:
- People standing, walking, talking, or sitting
- People near each other or touching normally
- Any object not being actively used as a weapon right now
- Anything you are not completely sure about

Be conservative. If you are not certain, answer NO.

Reply with JSON only:
{"threat": false, "type": "none", "confidence": "low", "description": "No threat detected"}
{"threat": true, "type": "fight|intrusion|assault", "confidence": "low|medium|high", "description": "One factual sentence"}"""

PERSON_PROMPT = """Look at this person in the surveillance image.

Write 2 short sentences:
1. What are they doing right now? (walking, standing, carrying something, using phone, etc.)
2. What is the single most noticeable thing about them that would help identify them?

Do not mention age. Do not describe full outfit.
Start sentence 1 with an action verb. Start sentence 2 with a noun.
Reply with only the 2 sentences."""

SCENE_PROMPT = """Look at this surveillance image and describe what is happening in one sentence.
Focus on people's actions and any objects they are interacting with.
Be specific about what you actually see, not what you expect to see.
Reply with only one sentence."""

COUNT_CHANGE_PROMPT = """The number of people in this camera view just changed.
Look at the image and describe in one sentence what you now see happening.
Focus on movement and actions.
Reply with only one sentence."""


# ── Core VLM Inference ────────────────────────────────────────────────────────
def _smolvlm_infer(crop_bgr: np.ndarray, prompt: str, vlm_model, processor,
                   max_tokens: int = 80) -> str:
    if vlm_model is None or processor is None:
        return ""
    try:
        h, w  = crop_bgr.shape[:2]
        scale = 512 / max(h, w)
        if scale < 1.0:
            crop_bgr = cv2.resize(
                crop_bgr, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        image    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role":    "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs      = processor(
            text=text_prompt, images=[image], return_tensors="pt"
        ).to(vlm_model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            out = vlm_model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                max_new_tokens=max_tokens,
            )

        result = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return lines[-1] if lines else result

    except torch.cuda.OutOfMemoryError:
        log.error("[VLM] OOM — clearing VRAM")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return ""
    except Exception as e:
        log.warning(f"[VLM] Inference error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()


def run_vlm(crop_bgr: np.ndarray, vlm_model, processor) -> dict:
    if vlm_model is None or processor is None:
        return {"threat": False, "type": "none", "confidence": "low",
                "description": "VLM not loaded"}
    raw = _smolvlm_infer(crop_bgr, VLM_THREAT_PROMPT, vlm_model, processor, max_tokens=100)
    try:
        js = raw[raw.rfind("{") : raw.rfind("}") + 1]
        return json.loads(js)
    except (json.JSONDecodeError, ValueError):
        is_threat = '"threat": true' in raw.lower()
        return {
            "threat":      is_threat,
            "type":        "assault" if is_threat else "none",
            "confidence":  "medium"  if is_threat else "low",
            "description": raw[:120],
        }


# ── Weapon Inference ──────────────────────────────────────────────────────────
def run_weapon_inference(threat_model: YOLO, frame: np.ndarray) -> tuple:
    """
    Run Subh775/Threat-Detection-YOLOv8n.
    Only reports classes in threat_classes (gun + knife).
    Returns: (detections, trigger_reason, trigger_crop)
    """
    if threat_model is None:
        return [], None, None

    results    = threat_model(frame, conf=0.60, imgsz=640, verbose=False)
    detections = []
    fh, fw     = frame.shape[:2]
    frame_area = fh * fw

    for box in results[0].boxes:
        cls_id = int(box.cls[0])

        # Skip classes we don't want (Bike, Explosive, Grenade)
        if cls_id not in threat_classes:
            continue

        conf  = float(box.conf[0])
        label = threat_classes[cls_id]
        xyxy  = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = map(int, xyxy)
        box_w  = x2 - x1
        box_h  = y2 - y1
        area   = box_w * box_h
        aspect = box_w / max(box_h, 1)

        # ── False-positive filters ─────────────────────────────────────────
        if area < frame_area * 0.015:
            log.debug(f"[THREAT] {label} skipped — too small ({area}px²)")
            continue

        if label == "gun" and not (0.5 < aspect < 4.0):
            log.debug(f"[THREAT] gun skipped — aspect {aspect:.2f}")
            continue

        if label == "knife" and not (1.2 < aspect < 8.0):
            log.debug(f"[THREAT] knife skipped — aspect {aspect:.2f}")
            continue

        if box_w > fw * 0.85 or box_h > fh * 0.85:
            log.debug(f"[THREAT] {label} skipped — bbox too large")
            continue

        detections.append({
            "label":      label,
            "confidence": round(conf, 2),
            "bbox":       [x1, y1, x2, y2],
        })
        log.info(
            f"[THREAT] ✅ {label} @ {int(conf*100)}% | "
            f"area={area} aspect={aspect:.2f}"
        )

    trigger_reason = None
    trigger_crop   = None
    if detections:
        best           = max(detections, key=lambda d: d["confidence"])
        trigger_reason = f"Weapon detected: {best['label']} ({int(best['confidence']*100)}%)"
        trigger_crop   = pad_crop(frame, best["bbox"])

    return detections, trigger_reason, trigger_crop


def draw_weapon_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label           = f"{det['label']} {int(det['confidence'] * 100)}%"
        color           = WEAPON_COLORS.get(det["label"], (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ── Utilities ─────────────────────────────────────────────────────────────────
def are_people_close(b1, b2) -> bool:
    c1         = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2         = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    dist       = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    avg_height = ((b1[3] - b1[1]) + (b2[3] - b2[1])) / 2
    return dist < avg_height * 0.7


def pad_crop(frame: np.ndarray, box, pad: int = 60) -> np.ndarray:
    h, w = frame.shape[:2]
    x1   = max(0, int(box[0]) - pad)
    y1   = max(0, int(box[1]) - pad)
    x2   = min(w, int(box[2]) + pad)
    y2   = min(h, int(box[3]) + pad)
    return frame[y1:y2, x1:x2]


def merged_bbox(b1, b2) -> list:
    return [min(b1[0], b2[0]), min(b1[1], b2[1]),
            max(b1[2], b2[2]), max(b1[3], b2[3])]


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
            entry = {
                "time":   datetime.now().strftime("%H:%M:%S"),
                "alert":  alert,
                "reason": reason,
            }
            if vlm_result:
                entry["vlm"] = vlm_result.get("description", "")
            state["alert_log"].append(entry)
            state["alert_log"] = state["alert_log"][-100:]
    log.info(f"[{alert}] {reason}")


# ── Proximity Tracker ─────────────────────────────────────────────────────────
class ProximityTracker:
    def __init__(self):
        self.pair_since: dict = {}

    def update(self, people_boxes: dict):
        ids          = list(people_boxes.keys())
        now          = time.time()
        active_pairs = set()
        result       = None

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair     = (min(id1, id2), max(id1, id2))
                if are_people_close(people_boxes[id1], people_boxes[id2]):
                    active_pairs.add(pair)
                    self.pair_since.setdefault(pair, now)
                    if now - self.pair_since[pair] >= PROXIMITY_DURATION:
                        mb     = merged_bbox(people_boxes[id1], people_boxes[id2])
                        result = (pair, mb)

        for p in list(self.pair_since):
            if p not in active_pairs:
                del self.pair_since[p]

        return result
