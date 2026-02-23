from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2, torch, base64, json, asyncio, gc
from PIL import Image
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Qwen3-VL-8B across both A4000s ───────────────
print("Loading Qwen3-VL-8B...")
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # bfloat16 best for A4000
    device_map="auto",            # Spreads across both A4000s automatically
    attn_implementation="flash_attention_2"  # Faster attention on A4000
)
model.eval()

# Print GPU allocation
print("✓ Qwen3-VL-8B loaded")
print(f"  GPU 0: {torch.cuda.memory_allocated(0)/1e9:.1f}GB used")
print(f"  GPU 1: {torch.cuda.memory_allocated(1)/1e9:.1f}GB used")

THREAT_KEYWORDS = [
    'gun', 'weapon', 'knife', 'axe', 'armed',
    'breaking', 'smashing', 'attacking', 'fighting',
    'robbery', 'robbing', 'stealing', 'theft',
    'vandalism', 'destroying', 'threatening',
    'aggressive', 'violence', 'fire', 'explosion',
    'screwdriver', 'hammer', 'bat', 'crowbar',
    'swinging', 'lunging', 'striking', 'stabbing',
    'running', 'chasing', 'jerking', 'suspicious'
]

prev_gray = None
latest    = {"frame": None, "timestamp": None}


# ── Optical Flow Overlay ───────────────────────────────
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


# ── Single Frame Inference ─────────────────────────────
def analyze(pil_img: Image.Image, timestamp: str) -> dict:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img,
             "min_pixels": 256*28*28,
             "max_pixels": 1280*28*28},   # Dynamic resolution
            {"type": "text", "text":
                "This CCTV frame has green arrows showing motion direction. "
                "Describe what the person is doing in one sentence — "
                "focus on their action, movement, and any object they are holding."}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=None,  # Required when do_sample=False
            top_p=None         # Required when do_sample=False
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
        "mode": "single_frame",
        "frame_timestamp": timestamp,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ── Routes ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "Qwen3-VL-8B Single Frame Running",
        "model": MODEL_ID,
        "gpu_0_vram": f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB",
        "gpu_1_vram": f"{torch.cuda.memory_allocated(1)/1e9:.1f}GB"
    }


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


# ── WebSocket Tasks ────────────────────────────────────
async def receiver(ws: WebSocket):
    while True:
        data = await ws.receive_text()
        try:
            fd = json.loads(data)
            latest["frame"]     = fd.get("frame")
            latest["timestamp"] = fd.get("timestamp", datetime.now().isoformat())
        except Exception:
            pass


async def processor_loop(ws: WebSocket):
    while True:
        if latest["frame"] is None:
            await asyncio.sleep(0.05)
            continue

        b64, ts     = latest["frame"], latest["timestamp"]
        latest["frame"] = None

        try:
            nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
            raw   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if raw is None:
                continue

            raw_arrows = add_motion_overlay(raw)
            resized    = cv2.resize(raw_arrows, (640, 480))
            img        = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, analyze, img, ts)
            result["processed_frame"] = frame_to_base64(resized)

            await ws.send_json(result)

            icon = "🚨 THREAT" if result["is_threat"] else "✅"
            print(f"{icon} [{ts}] {result['description']}")

        except Exception as e:
            print(f"⚠️ Error: {e}")


@app.websocket("/ws/camera")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    latest["frame"] = None
    await ws.send_json({
        "description": "Qwen3-VL-8B Single Frame Ready.",
        "is_threat": False,
        "frame_timestamp": datetime.now().isoformat(),
        "analysis_timestamp": datetime.now().isoformat()
    })
    try:
        await asyncio.gather(receiver(ws), processor_loop(ws))
    except WebSocketDisconnect:
        print("✗ Disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
