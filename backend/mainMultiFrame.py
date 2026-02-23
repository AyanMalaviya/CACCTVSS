from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
from PIL import Image
import numpy as np
import json
import base64
from transformers import AutoProcessor, AutoModelForImageTextToText
from datetime import datetime, timedelta
import asyncio
import gc
import json
import os
from pathlib import Path



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("✓ Model loaded")

THREAT_KEYWORDS = [
    'gun', 'weapon', 'knife', 'axe', 'armed',
    'breaking', 'smashing', 'attacking', 'fighting',
    'robbery', 'robbing', 'stealing', 'theft',
    'vandalism', 'destroying', 'damage',
    'mask', 'balaclava', 'covered face',
    'threatening', 'aggressive', 'violence',
    'accident', 'crash', 'collision', 'fire',
    'explosion', 'suspicious', 'dangerous',
    'screwdriver', 'hammer', 'bat', 'crowbar', 'wrench',
    'running', 'chasing', 'grabbing', 'pulling',
    'swinging', 'jerking', 'lunging', 'charging',
    'striking', 'hitting', 'stabbing', 'pointing'
]

# ── Log Manager ────────────────────────────────────────
LOG_FILE = Path("surveillance_log.jsonl")  # One JSON per line

def write_log(entry: dict):
    """Append one entry to the log file"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def search_log(query: str, window_minutes: int = 0) -> list:
    """
    Search log for query word/phrase.
    window_minutes=0 means search all time.
    """
    if not LOG_FILE.exists():
        return []

    results = []
    query_lower = query.strip().lower()
    now = datetime.now()

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Time window filter
            if window_minutes > 0:
                try:
                    entry_time = datetime.fromisoformat(entry["frame_timestamp"])
                    if now - entry_time > timedelta(minutes=window_minutes):
                        continue
                except Exception:
                    continue

            # Keyword match
            if query_lower in entry.get("description", "").lower():
                results.append(entry)

    # Return newest first
    return list(reversed(results))


# ─────────────────────────────────────────────────────
# CONFIG — tuned for 6GB VRAM + accuracy
# ─────────────────────────────────────────────────────
BATCH_SIZE       = 2
FRAME_W          = 448    # Up from 320 — enough to see objects clearly
FRAME_H          = 336    # Maintains 4:3 ratio
COLLECT_INTERVAL = 1.0    # Seconds between frames in a batch


# ─────────────────────────────────────────────────────
# VLM INFERENCE
# ─────────────────────────────────────────────────────
def analyze_frames(pil_frames: list, frame_timestamp: str) -> dict:
    n = len(pil_frames)

    content = [{"type": "image"} for _ in pil_frames]
    content.append({
        "type": "text",
        "text": (
            f"Look carefully at these {n} CCTV frames. "
            "Describe ONLY what you can directly see RIGHT NOW in these frames in one sentence. "
            "Do NOT reference anything from outside these frames. "
            "State the person's exact action and any object they are currently holding. "
            "If nothing suspicious is visible, say so clearly."
        )
    })

    messages = [{"role": "user", "content": content}]

    with torch.inference_mode():
        # Fresh prompt every call — no conversation history
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = processor(
            text=prompt,
            images=pil_frames,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            use_cache=False,          # No KV bleedover between calls
        )

        full_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Extract only the assistant's reply
        if "Assistant:" in full_text:
            text = full_text.split("Assistant:")[-1].strip()
        elif "\n" in full_text:
            text = full_text.split("\n")[-1].strip()
        else:
            text = full_text.strip()

        if '.' in text:
            text = text.split('.')[0] + '.'

    is_threat = any(kw in text.lower() for kw in THREAT_KEYWORDS)

    del inputs, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "description": text,
        "is_threat": is_threat,
        "frames_analyzed": n,
        "frame_timestamp": frame_timestamp,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def decode_frame(b64_string: str) -> Image.Image | None:
    try:
        img_bytes = base64.b64decode(b64_string)
        nparr    = np.frombuffer(img_bytes, np.uint8)
        raw      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if raw is None:
            return None
        raw = cv2.resize(raw, (FRAME_W, FRAME_H))
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:
        print(f"⚠️ Decode error: {e}")
        return None


# ─────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "CCTV Backend Running", "model": MODEL_ID}


@app.get("/api/cameras")
async def get_cameras():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append({"id": i, "name": f"Camera {i}"})
            cap.release()
    return {"cameras": available}


# ─────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    await websocket.accept()
    print("✓ Client connected")

    await websocket.send_json({
        "description": "Connected. Ready.",
        "is_threat": False,
        "frames_analyzed": 0,
        "frame_timestamp": datetime.now().isoformat(),
        "analysis_timestamp": datetime.now().isoformat()
    })

    is_processing = False
    batch         = []
    batch_ts      = []

@app.get("/api/logs/search")
async def search_logs(
    q: str = Query(..., description="Search keyword or phrase"),
    window: int = Query(0, description="Time window in minutes (0 = all time)")
):
    """Search surveillance log"""
    if not q.strip():
        return {"query": q, "window_minutes": window, "count": 0, "results": []}

    results = search_log(q.strip(), window)

    return {
        "query": q,
        "window_minutes": window,
        "count": len(results),
        "results": results
    }


@app.get("/api/logs/stats")
async def log_stats():
    """Get log statistics"""
    if not LOG_FILE.exists():
        return {"total": 0, "threats": 0, "normal": 0}

    total = threats = 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total += 1
                if entry.get("is_threat"):
                    threats += 1
            except Exception:
                continue

    return {
        "total": total,
        "threats": threats,
        "normal": total - threats
    }


    try:
        while True:
            try:
                raw_data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                print("⚠️ No frame for 10s")
                continue

            # Skip frames while model is busy
            if is_processing:
                continue

            try:
                frame_data = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            pil_img = decode_frame(frame_data.get('frame', ''))
            if pil_img is None:
                continue

            timestamp = frame_data.get('timestamp', datetime.now().isoformat())
            batch.append(pil_img)
            batch_ts.append(timestamp)

            print(f"📥 Frame {len(batch)}/{BATCH_SIZE} collected at {timestamp}")

            # Wait between frames in same batch
            if len(batch) < BATCH_SIZE:
                await asyncio.sleep(COLLECT_INTERVAL)
                continue

            # Batch full — process it
            is_processing  = True
            frames_to_send = batch.copy()
            ts_to_send     = batch_ts[0]

            batch.clear()
            batch_ts.clear()

            print(f"🔍 Analyzing {len(frames_to_send)} frames...")

            try:
                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    analyze_frames,
                    frames_to_send,
                    ts_to_send
                )

                await websocket.send_json(result)

                if result['is_threat']:
                    print(f"🚨 THREAT [{ts_to_send}]: {result['description']}")
                else:
                    print(f"✅ [{ts_to_send}]: {result['description'][:80]}...")
                # In processor_loop, after result = await loop.run_in_executor(...)
                write_log({
                    "frame_timestamp":    result["frame_timestamp"],
                    "analysis_timestamp": result["analysis_timestamp"],
                    "description":        result["description"],
                    "is_threat":          result["is_threat"],
                    "frames_analyzed":    result.get("frames_analyzed", 1)
                })


            except Exception as e:
                print(f"⚠️ Inference error: {e}")
                await websocket.send_json({
                    "description": "Inference error — retrying next batch",
                    "is_threat": False,
                    "frames_analyzed": len(frames_to_send),
                    "frame_timestamp": ts_to_send,
                    "analysis_timestamp": datetime.now().isoformat()
                })

            finally:
                is_processing = False
                print("✅ Ready for next batch\n")

    except WebSocketDisconnect:
        print("✗ Client disconnected")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
