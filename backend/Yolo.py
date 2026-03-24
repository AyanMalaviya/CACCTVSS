import threading
import time
import cv2
import logging
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime

from detector import (
    load_yolo, load_threat_model, load_vlm,
    run_vlm, run_weapon_inference, draw_weapon_boxes,
    state, state_lock,
    threat_classes,
    ProximityTracker, push_alert, pad_crop,
    RED_HOLD_SEC, RED_CONFIDENCE,
    _smolvlm_infer,
    PERSON_PROMPT, SCENE_PROMPT, COUNT_CHANGE_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STREAM_WIDTH        = 640
STREAM_HEIGHT       = 480
STREAM_FPS          = 30
NEW_PERSON_COOLDOWN = 30.0
VLM_COOLDOWN        = 8.0
COUNT_CHANGE_DELAY  = 2.0
VLM_THREAD_TIMEOUT  = 20.0
WEAPON_MIN_FRAMES   = 3

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="CCTV Surveillance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State — both YOLO and VLM OFF by default ──────────────────────────────────
state.update({
    "scene_description": "",
    "detection_summary": "",
    "person_log":        [],
    "person_count":      0,
    "weapon_detections": [],
    "source_fps":        0.0,
    "yolo_enabled":      False,   # ← off by default
    "vlm_enabled":       False,   # ← off by default
    "vlm_interval":      15.0,
    "mode_switching":    False,
    "custom_prompt":     "",
})

# ── Load models ───────────────────────────────────────────────────────────────
log.info("=" * 50)
log.info("Loading models...")
log.info("=" * 50)
yolo_model               = load_yolo()
threat_model             = load_threat_model()
vlm_model, vlm_processor = load_vlm()

# ── GPU setup ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
    yolo_model.to("cuda")
    yolo_model.model.half()
    if threat_model is not None:
        threat_model.to("cuda")
        threat_model.model.half()
    log.info(
        f"[GPU] {torch.cuda.get_device_name(0)} | "
        f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
    )
else:
    log.warning("[GPU] CUDA not available — running on CPU")

log.info("=" * 50)
log.info("Ready. All routes registered.")
log.info("=" * 50)

# ── Engine state ──────────────────────────────────────────────────────────────
engine = {
    "running":    False,
    "source":     None,
    "thread":     None,
    "frame":      None,
    "frame_lock": threading.Lock(),
}


# ── VRAM helpers ──────────────────────────────────────────────────────────────
def get_vram_usage_pct() -> float:
    try:
        reserved = torch.cuda.memory_reserved()
        total    = torch.cuda.get_device_properties(0).total_memory
        return (reserved / total) * 100
    except Exception:
        return 0.0


def offload_vlm_to_cpu():
    global vlm_model
    if vlm_model is None:
        return
    try:
        vlm_model = vlm_model.to("cpu")
        torch.cuda.empty_cache()
        log.info("[VLM] Offloaded to CPU")
    except Exception as e:
        log.warning(f"[VLM] Offload failed: {e}")


def reload_vlm_to_gpu():
    global vlm_model
    if vlm_model is None:
        return
    try:
        vlm_model = vlm_model.to("cuda")
        torch.cuda.synchronize()
        log.info("[VLM] Reloaded to GPU")
    except torch.cuda.OutOfMemoryError:
        log.error("[VLM] OOM reloading — staying on CPU")
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"[VLM] Reload failed: {e}")


def safe_vlm_call(fn, args):
    if get_vram_usage_pct() > 80.0:
        log.warning("[VLM] VRAM > 80% — skipping")
        return
    try:
        torch.cuda.empty_cache()
        fn(*args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as e:
        log.warning(f"[VLM] Error: {e}")


def _start_vlm_thread(target, args):
    def wrapped():
        safe_vlm_call(target, args)
    t = threading.Thread(target=wrapped, daemon=True)
    t._start_time = time.time()
    t.start()
    return t


def call_vlm_text(crop: np.ndarray, prompt: str) -> str:
    return _smolvlm_infer(crop, prompt, vlm_model, vlm_processor, max_tokens=80)


# ── Detection engine ──────────────────────────────────────────────────────────
def run_engine(source):
    prox               = ProximityTracker()
    vlm_thread         = None
    described_ids      = {}
    prev_person_count  = 0
    count_changed_at   = None
    frame_count        = 0
    weapon_consecutive = 0

    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)

    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        engine["running"] = False
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 30.0
    frame_delay = 1.0 / actual_fps

    with state_lock:
        state["source_fps"] = round(actual_fps, 2)

    log.info(
        f"Stream opened → {source} | "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.1f}fps"
    )

    # ── VLM targets ───────────────────────────────────────────────────────────
    def threat_vlm(crop, reason_prefix):
        result     = run_vlm(crop, vlm_model, vlm_processor)
        is_threat  = result.get("threat", False)
        confidence = result.get("confidence", "low")
        if is_threat and confidence in RED_CONFIDENCE:
            push_alert("RED",    f"{reason_prefix} — confirmed: {result.get('type')}", result)
        else:
            push_alert("YELLOW", f"{reason_prefix} — unconfirmed by VLM", result)

    def person_vlm(track_id, crop, prompt_override=None):
        desc = call_vlm_text(crop, prompt_override or PERSON_PROMPT)
        if not desc:
            return
        with state_lock:
            state["person_log"].append({
                "time":        datetime.now().strftime("%H:%M:%S"),
                "track_id":    track_id,
                "description": desc,
            })
            state["person_log"] = state["person_log"][-50:]
        log.info(f"[PERSON] ID#{track_id}: {desc}")

    def scene_vlm(crop, prompt=None):
        desc = call_vlm_text(crop, prompt or SCENE_PROMPT)
        if desc:
            with state_lock:
                state["scene_description"] = desc
            log.info(f"[SCENE] {desc}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while engine["running"]:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        try:
            frame_count += 1
            now = time.time()

            with state_lock:
                yolo_on       = state["yolo_enabled"]
                vlm_on        = state["vlm_enabled"]
                vlm_ivl       = state["vlm_interval"]
                last_vlm      = state["last_vlm_time"]
                is_switching  = state["mode_switching"]
                custom_prompt = state["custom_prompt"].strip()

            # ── CASE 1: Both off — raw frame, 30fps, no inference ─────────────
            if not yolo_on and not vlm_on:
                h_f, w_f = frame.shape[:2]
                display  = frame.copy()
                cv2.rectangle(display, (0, 0), (w_f, 40), (30, 30, 30), -1)
                cv2.putText(display, "RAW FEED — YOLO & VLM disabled",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
                cv2.putText(display, f"{actual_fps:.0f}fps",
                            (w_f-55, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)
                _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with engine["frame_lock"]:
                    engine["frame"] = buf.tobytes()
                time.sleep(max(0, frame_delay - (time.time() - loop_start)))
                continue

            if is_switching:
                time.sleep(max(0, frame_delay - (time.time() - loop_start)))
                continue

            # ── VLM thread watchdog ────────────────────────────────────────────
            if (vlm_thread is not None
                    and vlm_thread.is_alive()
                    and hasattr(vlm_thread, "_start_time")
                    and now - vlm_thread._start_time > VLM_THREAD_TIMEOUT):
                log.warning("[VLM] Thread hung >20s — abandoning")
                vlm_thread = None
                torch.cuda.empty_cache()

            # ── CASE 2: YOLO only ─────────────────────────────────────────────
            annotated         = frame
            people_boxes      = {}
            class_counts      = {}
            weapon_detections = []
            yolo_trigger      = None

            if yolo_on:
                track_results = yolo_model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=0.4, imgsz=640, verbose=False,
                )
                annotated = track_results[0].plot()

                # Weapon detection
                w_dets, w_trigger, _ = run_weapon_inference(threat_model, frame)
                if w_dets:
                    weapon_consecutive += 1
                else:
                    weapon_consecutive = 0
                confirmed_weapon = weapon_consecutive >= WEAPON_MIN_FRAMES

                if confirmed_weapon and w_trigger:
                    weapon_detections = w_dets
                    yolo_trigger      = w_trigger
                    annotated         = draw_weapon_boxes(annotated, weapon_detections)

                with state_lock:
                    state["weapon_detections"] = weapon_detections

                # Person boxes from YOLO
                for box in track_results[0].boxes:
                    if box.id is None:
                        continue
                    cls_id   = int(box.cls)
                    track_id = int(box.id)
                    xyxy     = box.xyxy[0].cpu().numpy()
                    name     = yolo_model.names[cls_id]
                    class_counts[name] = class_counts.get(name, 0) + 1

                    if cls_id == 0:
                        people_boxes[track_id] = xyxy

                        # Person VLM description (only if VLM also on)
                        if vlm_on and vlm_model is not None:
                            cx       = int((xyxy[0] + xyxy[2]) / 2 / 80)
                            cy_grid  = int((xyxy[1] + xyxy[3]) / 2 / 80)
                            pos_hash = cx * 1000 + cy_grid
                            last_e   = described_ids.get(track_id, {"time": 0, "pos_hash": -1})
                            if (last_e["time"] == 0
                                    or (now - last_e["time"] > NEW_PERSON_COOLDOWN
                                        and last_e["pos_hash"] != pos_hash)):
                                described_ids[track_id] = {"time": now, "pos_hash": pos_hash}
                                crop = pad_crop(frame, xyxy, pad=40)
                                if crop.size > 0 and (vlm_thread is None or not vlm_thread.is_alive()):
                                    vlm_thread = _start_vlm_thread(
                                        person_vlm,
                                        (track_id, crop.copy(),
                                         custom_prompt if custom_prompt else None),
                                    )

                with state_lock:
                    state["person_count"] = len(people_boxes)

                # Proximity check
                if yolo_trigger is None:
                    prox_result = prox.update(people_boxes)
                    if prox_result:
                        pair_ids, mb = prox_result
                        yolo_trigger = f"Sustained contact — IDs {pair_ids}"

            # ── CASE 3: VLM logic (only if vlm_on) ───────────────────────────
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
                last_vlm  = state["last_vlm_time"]

            if vlm_on:
                # Count change scene description
                current_count = len(people_boxes)
                if current_count != prev_person_count:
                    count_changed_at  = now
                    prev_person_count = current_count

                if (count_changed_at is not None
                        and now - count_changed_at >= COUNT_CHANGE_DELAY
                        and now - last_vlm >= VLM_COOLDOWN
                        and vlm_model is not None
                        and (vlm_thread is None or not vlm_thread.is_alive())):
                    with state_lock:
                        state["last_vlm_time"] = now
                    count_changed_at = None
                    vlm_thread = _start_vlm_thread(
                        scene_vlm,
                        (frame.copy(), custom_prompt if custom_prompt else COUNT_CHANGE_PROMPT),
                    )

                # Threat VLM on weapon/proximity trigger
                if yolo_trigger and yolo_on:
                    reason = yolo_trigger if isinstance(yolo_trigger, str) else str(yolo_trigger)
                    crop   = pad_crop(frame, weapon_detections[0]["bbox"]) \
                             if weapon_detections else frame
                    if (get_vram_usage_pct() < 75.0
                            and now - last_vlm >= VLM_COOLDOWN
                            and vlm_model is not None
                            and (vlm_thread is None or not vlm_thread.is_alive())):
                        with state_lock:
                            state["last_vlm_time"] = now
                        vlm_thread = _start_vlm_thread(threat_vlm, (crop.copy(), reason))
                    if cur_alert == "CLEAR":
                        push_alert("YELLOW", reason)

                # VLM only — passive scene with no YOLO
                elif not yolo_on:
                    if (now - last_vlm >= vlm_ivl
                            and vlm_model is not None
                            and (vlm_thread is None or not vlm_thread.is_alive())):
                        with state_lock:
                            state["last_vlm_time"] = now
                        vlm_thread = _start_vlm_thread(
                            scene_vlm,
                            (frame.copy(), custom_prompt if custom_prompt else None),
                        )

                # Passive scene when YOLO on but no trigger
                elif (not yolo_trigger
                        and now - last_vlm >= vlm_ivl
                        and count_changed_at is None
                        and vlm_model is not None
                        and (vlm_thread is None or not vlm_thread.is_alive())):
                    with state_lock:
                        state["last_vlm_time"] = now
                    vlm_thread = _start_vlm_thread(
                        scene_vlm,
                        (frame.copy(), custom_prompt if custom_prompt else None),
                    )
            else:
                # VLM off — YOLO-only alert from weapon/proximity
                if yolo_on and yolo_trigger:
                    if cur_alert == "CLEAR":
                        reason = yolo_trigger if isinstance(yolo_trigger, str) else str(yolo_trigger)
                        push_alert("YELLOW", reason)

            # Clear alert when nothing happening
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
            if (cur_alert != "CLEAR"
                    and not yolo_trigger
                    and now - last_red >= RED_HOLD_SEC
                    and (vlm_thread is None or not vlm_thread.is_alive())):
                push_alert("CLEAR", "")

            # ── Detection summary ──────────────────────────────────────────────
            weapon_names = [d["label"] for d in weapon_detections]
            person_str   = f"{len(people_boxes)} person(s)" if people_boxes else ""
            weapon_str   = f"⚠️ {', '.join(weapon_names)}"  if weapon_names else ""
            other_str    = ", ".join(
                f"{v}× {k}" for k, v in class_counts.items()
                if k != "person" and k not in weapon_names
            )
            summary = " | ".join(p for p in [person_str, weapon_str, other_str] if p) \
                      or ("Streaming..." if not yolo_on else "Nothing detected")
            with state_lock:
                state["detection_summary"] = summary

            # ── Overlay ────────────────────────────────────────────────────────
            h_f, w_f = annotated.shape[:2]

            with state_lock:
                alert_now  = state["alert"]
                reason_now = state["reason"]
                desc_now   = state["scene_description"]
                count_now  = state["person_count"]

            # Status bar
            bar_color = {
                "CLEAR":  (30, 160, 30),
                "YELLOW": (0,  180, 255),
                "RED":    (0,  0,   210),
            }.get(alert_now, (40, 40, 40))
            cv2.rectangle(annotated, (0, 0), (w_f, 46), bar_color, -1)
            cv2.putText(annotated, f"[{alert_now}]  {reason_now[:72]}",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # Right badge
            badge = []
            if yolo_on: badge.append(f"YOLO  👥{count_now}")
            if vlm_on:  badge.append("VLM")
            badge_str = "  |  ".join(badge) if badge else "RAW"
            cv2.rectangle(annotated, (w_f-200, 0), (w_f, 46), (25, 25, 25), -1)
            cv2.putText(annotated, badge_str,
                        (w_f-190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # FPS
            cv2.putText(annotated, f"{actual_fps:.0f}fps",
                        (w_f-55, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)

            # Scene description
            if desc_now and vlm_on:
                cv2.putText(annotated, f"💬 {desc_now[:95]}",
                            (8, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with engine["frame_lock"]:
                engine["frame"] = buf.tobytes()

        except torch.cuda.OutOfMemoryError:
            log.error(f"[ENGINE] OOM on frame #{frame_count} — recovering")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.5)
        except Exception as e:
            log.error(f"[ENGINE] Frame #{frame_count} error: {e}")

        time.sleep(max(0, frame_delay - (time.time() - loop_start)))

    cap.release()
    engine["running"] = False
    with state_lock:
        state["weapon_detections"] = []
        state["source_fps"]        = 0.0
    push_alert("CLEAR", "")
    log.info("[ENGINE] Stopped.")


# ── MJPEG stream ──────────────────────────────────────────────────────────────
def mjpeg_generator():
    while True:
        with engine["frame_lock"]:
            frame = engine["frame"]
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


# ── Engine helpers ────────────────────────────────────────────────────────────
def _stop_engine():
    engine["running"] = False
    if engine["thread"] and engine["thread"].is_alive():
        engine["thread"].join(timeout=4)
    with state_lock:
        state.update({
            "alert":             "CLEAR",
            "reason":            "",
            "person_count":      0,
            "weapon_detections": [],
            "source_fps":        0.0,
            "detection_summary": "",
        })
    with engine["frame_lock"]:
        engine["frame"] = None


def _start_engine(source):
    _stop_engine()
    engine["source"]  = source
    engine["running"] = True
    engine["thread"]  = threading.Thread(
        target=run_engine, args=(source,), daemon=True
    )
    engine["thread"].start()


# ── API Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "CCTV Surveillance API running"}


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/start/camera")
def start_camera(index: int = 0):
    _start_engine(index)
    return {"status": "started", "source": f"camera:{index}"}


@app.post("/start/video")
async def start_video(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    _start_engine(str(save_path))
    return {"status": "started", "source": file.filename}


@app.post("/start/path")
def start_path(path: str):
    _start_engine(path)
    return {"status": "started", "source": path}


@app.post("/stop")
def stop():
    _stop_engine()
    return {"status": "stopped"}


@app.get("/status")
def get_status():
    with state_lock:
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
            "weapon_detections": state["weapon_detections"],
            "source_fps":        state["source_fps"],
            "yolo_enabled":      state["yolo_enabled"],
            "vlm_enabled":       state["vlm_enabled"],
            "vlm_interval":      state["vlm_interval"],
            "mode_switching":    state["mode_switching"],
            "custom_prompt":     state["custom_prompt"],
            "vram_pct":          round(get_vram_usage_pct(), 1),
        }


@app.get("/alerts")
def get_alerts():
    with state_lock:
        return state["alert_log"]


@app.get("/persons")
def get_persons():
    with state_lock:
        return state["person_log"]


@app.get("/vram")
def get_vram():
    try:
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        return {
            "gpu_name":     props.name,
            "total_gb":     round(total, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb":  round(torch.cuda.memory_reserved()  / 1024**3, 2),
            "free_gb":      round(total - torch.cuda.memory_reserved() / 1024**3, 2),
            "usage_pct":    round(get_vram_usage_pct(), 1),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/weapon_classes")
def get_weapon_classes():
    return {"active_classes": list(threat_classes.values())}


# ── YOLO toggle ────────────────────────────────────────────────────────────────
@app.post("/yolo/enable")
def yolo_enable():
    with state_lock:
        state["yolo_enabled"] = True
    log.info("[YOLO] Enabled")
    return {"yolo_enabled": True}


@app.post("/yolo/disable")
def yolo_disable():
    with state_lock:
        state["yolo_enabled"] = False
        state["weapon_detections"] = []
        state["person_count"]      = 0
        state["detection_summary"] = ""
    log.info("[YOLO] Disabled")
    return {"yolo_enabled": False}


# ── VLM toggle ─────────────────────────────────────────────────────────────────
@app.post("/vlm/enable")
def vlm_enable():
    with state_lock:
        switching = state["mode_switching"]
    if switching:
        return {"error": "Mode switch in progress"}
    with state_lock:
        state["mode_switching"] = True

    def do_enable():
        reload_vlm_to_gpu()
        with state_lock:
            state["vlm_enabled"]  = True
            state["mode_switching"] = False
        log.info("[VLM] Enabled")

    threading.Thread(target=do_enable, daemon=True).start()
    return {"vlm_enabled": True, "mode_switching": True}


@app.post("/vlm/disable")
def vlm_disable():
    with state_lock:
        state["vlm_enabled"]     = False
        state["scene_description"] = ""
    offload_vlm_to_cpu()
    log.info("[VLM] Disabled")
    return {"vlm_enabled": False}


@app.post("/vlm/interval")
def set_vlm_interval(seconds: float):
    seconds = max(5.0, min(seconds, 120.0))
    with state_lock:
        state["vlm_interval"] = seconds
    return {"vlm_interval": seconds}


@app.post("/vlm/prompt")
def set_custom_prompt(prompt: str = ""):
    with state_lock:
        state["custom_prompt"] = prompt.strip()
    return {"custom_prompt": state["custom_prompt"]}


@app.delete("/vlm/prompt")
def clear_custom_prompt():
    with state_lock:
        state["custom_prompt"] = ""
    return {"custom_prompt": ""}
