from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2, torch, base64, json, asyncio, gc
from PIL import Image
import numpy as np
from collections import deque
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static folder ────────────────────────────────
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── GPU Memory Config ──────────────────────────────────
MAX_MEMORY = {
    0: "7GiB",
    1: "15GiB",
}

print("Loading Qwen3-VL-8B...")
MODEL_ID  = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=MAX_MEMORY,
    attn_implementation="sdpa"
)
model.eval()

print("✓ Qwen3-VL-8B loaded")
print(f"  GPU 0: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / 7GB reserved")
print(f"  GPU 1: {torch.cuda.memory_allocated(1)/1e9:.2f}GB / 15GB reserved")

THREAT_KEYWORDS = [
    'gun', 'weapon', 'knife', 'axe', 'armed',
    'breaking', 'smashing', 'attacking', 'fighting',
    'robbery', 'robbing', 'stealing', 'theft',
    'vandalism', 'destroying', 'threatening',
    'aggressive', 'violence', 'fire', 'explosion',
    'screwdriver', 'hammer', 'bat', 'crowbar',
    'swinging', 'lunging', 'striking', 'stabbing',
    'running', 'chasing', 'jerking', 'suspicious',
    'dancing', 'erratic', 'flailing'
]

# ── Config ─────────────────────────────────────────────
BUFFER_SIZE = 4
FRAME_W     = 640
FRAME_H     = 480
LOG_FILE    = Path("surveillance_log.jsonl")

frame_buffer    = deque(maxlen=BUFFER_SIZE)
ts_buffer       = deque(maxlen=BUFFER_SIZE)
buffer_lock     = asyncio.Lock()
new_frame_event = asyncio.Event()
prev_gray       = None


# ── Optical Flow ───────────────────────────────────────
def add_motion_overlay(bgr):
    global prev_gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None or prev_gray.shape != gray.shape:
        prev_gray = gray.copy()
        return bgr
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    overlay = bgr.copy()
    h, w = overlay.shape[:2]
    for y in range(0, h, 20):
        for x in range(0, w, 20):
            fx, fy = flow[y, x]
            if np.sqrt(fx**2 + fy**2) > 2.0:
                cv2.arrowedLine(
                    overlay, (x, y),
                    (int(x + fx*3), int(y + fy*3)),
                    (0, 255, 0), 1, tipLength=0.3
                )
    prev_gray = gray.copy()
    return overlay


def frame_to_base64(bgr) -> str:
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')


# ── Inference ──────────────────────────────────────────
def analyze_multi(pil_frames: list, timestamp: str) -> dict:
    n = len(pil_frames)
    content = [
        {"type": "image", "image": f,
         "min_pixels": 256*28*28,
         "max_pixels": 1280*28*28}
        for f in pil_frames
    ]
    content.append({
        "type": "text",
        "text": (
            f"These are {n} consecutive CCTV frames 1 second apart. "
            "Green arrows show motion. In ONE sentence describe what the "
            "person is doing — focus on actions, movement, and any object "
            "they are holding. Mention any aggressive or suspicious behavior."
        )
    })

    messages = [{"role": "user", "content": content}]
    text     = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=70,
            do_sample=False, temperature=None, top_p=None
        )

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    output  = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    if '.' in output:
        output = output.split('.')[0] + '.'

    is_threat = any(kw in output.lower() for kw in THREAT_KEYWORDS)

    del inputs, gen_ids
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "description": output,
        "is_threat": is_threat,
        "frames_analyzed": n,
        "frame_timestamp": timestamp,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ── Log Routes ─────────────────────────────────────────
@app.get("/api/logs/search")
async def search_logs(
    q: str = Query(...),
    window: int = Query(0)
):
    if not LOG_FILE.exists():
        return {"query": q, "window_minutes": window, "count": 0, "results": []}

    results     = []
    query_lower = q.strip().lower()
    now         = datetime.now()

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if window > 0:
                try:
                    entry_time = datetime.fromisoformat(entry["frame_timestamp"])
                    if now - entry_time > timedelta(minutes=window):
                        continue
                except Exception:
                    continue
            if query_lower in entry.get("description", "").lower():
                results.append(entry)

    return {
        "query": q, "window_minutes": window,
        "count": len(results), "results": list(reversed(results))
    }


@app.get("/api/logs/stats")
async def log_stats():
    if not LOG_FILE.exists():
        return {"total": 0, "threats": 0, "normal": 0}
    total = threats = 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                total += 1
                if entry.get("is_threat"):
                    threats += 1
            except Exception:
                continue
    return {"total": total, "threats": threats, "normal": total - threats}


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


# ── WebSocket ──────────────────────────────────────────
async def receiver(ws: WebSocket):
    while True:
        data = await ws.receive_text()
        try:
            fd  = json.loads(data)
            b64 = fd.get("frame")
            ts  = fd.get("timestamp", datetime.now().isoformat())

            nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
            raw   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if raw is None:
                continue

            raw = add_motion_overlay(raw)
            raw = cv2.resize(raw, (FRAME_W, FRAME_H))
            img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))

            async with buffer_lock:
                frame_buffer.append(img)
                ts_buffer.append(ts)

            new_frame_event.set()
        except Exception as e:
            print(f"⚠️ Receiver: {e}")


async def processor_loop(ws: WebSocket):
    while True:
        await new_frame_event.wait()
        new_frame_event.clear()

        async with buffer_lock:
            if len(frame_buffer) == 0:
                continue
            frames     = list(frame_buffer)
            latest_ts  = list(ts_buffer)[-1]
            latest_bgr = cv2.cvtColor(np.array(frames[-1]), cv2.COLOR_RGB2BGR)

        print(f"🔍 {len(frames)} frames | "
              f"GPU0: {torch.cuda.memory_allocated(0)/1e9:.1f}GB | "
              f"GPU1: {torch.cuda.memory_allocated(1)/1e9:.1f}GB")

        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, analyze_multi, frames, latest_ts
            )

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "frame_timestamp"    : result["frame_timestamp"],
                    "analysis_timestamp" : result["analysis_timestamp"],
                    "description"        : result["description"],
                    "is_threat"          : result["is_threat"],
                    "frames_analyzed"    : result["frames_analyzed"]
                }) + "\n")

            result["processed_frame"] = frame_to_base64(latest_bgr)
            await ws.send_json(result)

            icon = "🚨 THREAT" if result["is_threat"] else "✅"
            print(f"{icon} [{latest_ts}] {result['description']}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print("⚠️ OOM — reduce BUFFER_SIZE in config")

        except Exception as e:
            print(f"⚠️ Inference: {e}")


@app.websocket("/ws/camera")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("✓ Client connected")

    async with buffer_lock:
        frame_buffer.clear()
        ts_buffer.clear()
    new_frame_event.clear()

    await ws.send_json({
        "description": f"Qwen3-VL-8B Ready — {BUFFER_SIZE} frame buffer.",
        "is_threat": False, "frames_analyzed": 0,
        "frame_timestamp": datetime.now().isoformat(),
        "analysis_timestamp": datetime.now().isoformat()
    })

    try:
        await asyncio.gather(receiver(ws), processor_loop(ws))
    except WebSocketDisconnect:
        print("✗ Disconnected")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
