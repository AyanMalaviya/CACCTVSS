import threading, time, cv2, logging, numpy as np, torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime

from detector import (
    load_yolo, load_threat_model, load_vlm,
    smolvlm_infer, run_vlm_threat, run_weapons, draw_weapons,
    state, state_lock, threat_classes, yolo_edge_classes,
    ProximityTracker, push_alert, pad_crop,
    RED_HOLD_SEC, RED_CONFIDENCE, vlm_abort,
    DEFAULT_PROXIMITY_PROMPT, DEFAULT_COUNT_CHANGE_PROMPT,
    DEFAULT_WEAPON_PROMPT, DEFAULT_SCENE_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

STREAM_W           = 640
STREAM_H           = 480
STREAM_FPS         = 30
NEW_PERSON_COOL    = 30.0
VLM_THREAD_TIMEOUT = 20.0
WEAPON_MIN_FRAMES  = 3

app = FastAPI(title="CCTV Surveillance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ───────────────────────────────────────────────────────────────
log.info("=" * 50)
yolo_model               = load_yolo()
threat_model             = load_threat_model()
vlm_model, vlm_processor = load_vlm()

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
    yolo_model.to("cuda"); yolo_model.model.half()
    if threat_model:
        threat_model.to("cuda"); threat_model.model.half()
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
    described_ids      = {}
    prev_count         = -1
    frame_count        = 0
    weapon_consecutive = 0

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
    def do_weapon_vlm(crop, prompt):
        r = run_vlm_threat(crop, vlm_model, vlm_processor, prompt)
        if r.get("threat") and r.get("confidence") in RED_CONFIDENCE:
            push_alert("RED", f"Weapon confirmed: {r.get('type')}", r)
        else:
            push_alert("YELLOW", f"Weapon detected — unconfirmed by VLM", r)

    def do_proximity_vlm(crop, prompt):
        desc = smolvlm_infer(crop, prompt or DEFAULT_PROXIMITY_PROMPT,
                             vlm_model, vlm_processor, max_tokens=80)
        if not desc: return
        threat = any(w in desc.lower() for w in
                     ["threatening", "assault", "attack", "fight", "danger"])
        if threat:
            push_alert("RED",    f"Proximity threat: {desc[:80]}")
        else:
            push_alert("YELLOW", f"Sustained contact: {desc[:80]}")
        with state_lock:
            state["scene_description"] = desc

    def do_count_change_vlm(crop, prompt):
        desc = smolvlm_infer(crop, prompt or DEFAULT_COUNT_CHANGE_PROMPT,
                             vlm_model, vlm_processor, max_tokens=60)
        if desc:
            with state_lock:
                state["scene_description"] = desc
            log.info(f"[COUNT VLM] {desc}")

    def do_scene_vlm(crop, prompt):
        desc = smolvlm_infer(crop, prompt or DEFAULT_SCENE_PROMPT,
                             vlm_model, vlm_processor, max_tokens=60)
        if desc:
            with state_lock:
                state["scene_description"] = desc

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

            # ── YOLO tracking ─────────────────────────────────────────────────
            annotated    = frame
            people_boxes = {}
            class_counts = {}
            weapon_dets  = []
            edge_dets    = []   # axe/scissor/crowbar from yolo26n
            yolo_trigger = None

            if yolo_on:
                tr = yolo_model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=0.4, imgsz=640, verbose=False,
                )
                annotated = tr[0].plot()

                for box in tr[0].boxes:
                    if box.id is None: continue
                    cid  = int(box.cls)
                    tid  = int(box.id)
                    xyxy = box.xyxy[0].cpu().numpy()
                    name = yolo_model.names[cid]
                    class_counts[name] = class_counts.get(name, 0) + 1

                    # Person
                    if cid == 0:
                        people_boxes[tid] = xyxy

                    # Edge weapon from yolo26n class names
                    if cid in yolo_edge_classes:
                        conf = float(box.conf[0])
                        if conf >= 0.60:
                            x1,y1,x2,y2 = map(int, xyxy)
                            edge_dets.append({
                                "label":      yolo_edge_classes[cid],
                                "confidence": round(conf, 2),
                                "bbox":       [x1,y1,x2,y2],
                            })

                with state_lock:
                    state["person_count"] = len(people_boxes)

                # ── Weapon detection ──────────────────────────────────────────
                w_dets, w_trigger, w_crop = run_weapons(
                    threat_model, frame, edge_dets
                )
                if w_dets:
                    weapon_consecutive += 1
                else:
                    weapon_consecutive  = 0

                if weapon_consecutive >= WEAPON_MIN_FRAMES and w_trigger:
                    weapon_dets  = w_dets
                    yolo_trigger = ("weapon", w_trigger, w_crop)
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
                        )

                # ── Count change trigger ──────────────────────────────────────
                cur_count = len(people_boxes)
                if cur_count != prev_count and prev_count != -1:
                    # Always fire VLM on count change if enabled
                    log.info(f"[COUNT] {prev_count} → {cur_count}")
                    if vlm_on and vlm_model and get_vram_pct() < 75:
                        with state_lock:
                            state["last_vlm_time"] = now
                        launched = launch_vlm(
                            "trigger", do_count_change_vlm,
                            (frame.copy(), prompts.get("count_change", ""))
                        )
                        if launched:
                            push_alert("YELLOW",
                                       f"Person count: {prev_count}→{cur_count}")
                prev_count = cur_count

            # ── VLM trigger dispatch ──────────────────────────────────────────
            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
                last_vlm  = state["last_vlm_time"]

            if yolo_trigger:
                kind, reason, crop = yolo_trigger
                if vlm_on and vlm_model and get_vram_pct() < 75:
                    with state_lock:
                        state["last_vlm_time"] = now
                    if kind == "weapon":
                        launch_vlm("trigger", do_weapon_vlm,
                                   (crop.copy(), prompts.get("weapon", "")))
                    elif kind == "proximity":
                        launch_vlm("trigger", do_proximity_vlm,
                                   (crop.copy(), prompts.get("proximity", "")))
                if cur_alert == "CLEAR":
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
                    cx = int((xyxy[0]+xyxy[2])/2/80)
                    cy = int((xyxy[1]+xyxy[3])/2/80)
                    ph = cx*1000+cy
                    le = described_ids.get(tid, {"time":0,"ph":-1})
                    if (le["time"]==0
                            or (now-le["time"]>NEW_PERSON_COOL and le["ph"]!=ph)):
                        described_ids[tid] = {"time":now,"ph":ph}
                        crop = pad_crop(frame, xyxy, 40)
                        if crop.size > 0:
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
                            launch_vlm("passive", _person_desc, ())

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
            p_str   = f"{len(people_boxes)} person(s)" if people_boxes else ""
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
            "detection_summary":"","scene_description":"",
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
    _start(path); return {"status":"started","source":path}

@app.post("/stop")
def stop():
    _stop(); return {"status":"stopped"}

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
            "trigger_prompts":   dict(state["trigger_prompts"]),
            "vram_pct":          round(get_vram_pct(), 1),
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
    return {"threat_model": list(threat_classes.values()),
            "yolo_edge":    list(yolo_edge_classes.values())}

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
        state["detection_summary"] = ""
    return {"yolo_enabled": False}

# ── VLM toggle ─────────────────────────────────────────────────────────────────
@app.post("/vlm/enable")
def vlm_enable():
    with state_lock:
        if state["mode_switching"]:
            return {"error": "Already switching"}
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