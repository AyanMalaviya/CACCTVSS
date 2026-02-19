from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
from PIL import Image
import numpy as np
import json
import base64
from transformers import AutoProcessor, AutoModelForImageTextToText
from datetime import datetime
import asyncio
from collections import deque


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once
print("Loading model...")
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
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
    'screwdriver', 'hammer', 'bat', 'crowbar',
    'running', 'chasing', 'grabbing', 'pulling',
    'swinging', 'jerking', 'lunging', 'charging'
]

# Rolling frame buffer: stores last N frames with timestamps
BUFFER_SIZE = 4  # Number of frames to analyze together
frame_buffer = deque(maxlen=BUFFER_SIZE)

# Latest frame holder (always overwritten with newest frame)
latest_frame_buffer = {
    "frame": None,
    "timestamp": None,
    "lock": asyncio.Lock()
}


def detect_motion(frames: list) -> bool:
    """
    Quick OpenCV motion check before sending to VLM.
    Returns True if significant motion detected between frames.
    Saves GPU time by skipping static scenes.
    """
    if len(frames) < 2:
        return True  # Always analyze if not enough frames yet

    try:
        prev = np.array(frames[-2])
        curr = np.array(frames[-1])

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Motion score = percentage of changed pixels
        motion_score = np.sum(thresh > 0) / thresh.size

        # Trigger VLM if more than 1.5% pixels changed (catches jerks/sudden moves)
        return motion_score > 0.015

    except Exception as e:
        print(f"Motion detection error: {e}")
        return True  # Fallback: always analyze


def analyze_frames(pil_frames: list, frame_timestamp: str) -> dict:
    """
    Send multiple frames as a video sequence to SmolVLM2.
    SmolVLM2 natively understands temporal sequences across frames.
    """
    n = len(pil_frames)

    # Build multi-frame message - SmolVLM2 treats this as a video sequence
    content = []

    # Add all frames first
    for _ in pil_frames:
        content.append({"type": "image"})

    # Action-focused prompt - ask about movement and behavior, not just appearance
    content.append({
        "type": "text",
        "text": (
            f"You are analyzing {n} consecutive CCTV frames taken 1 second apart. "
            "Describe any actions, movements, or behaviors you observe across these frames in one sentence. "
            "Focus on what the person is DOING, not just what they look like. "
            "If you see sudden movements, aggressive actions, or weapons being used, describe them clearly."
        )
    })

    messages = [{"role": "user", "content": content}]

    with torch.inference_mode():
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Pass all frames as images list
        inputs = processor(
            text=prompt,
            images=pil_frames,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,  # More tokens for action descriptions
            do_sample=False
        )

        full_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "Assistant:" in full_text:
            text = full_text.split("Assistant:")[-1].strip()
        else:
            text = full_text.strip()

        if '.' in text:
            text = text.split('.')[0] + '.'

    is_threat = any(kw in text.lower() for kw in THREAT_KEYWORDS)

    return {
        "description": text,
        "is_threat": is_threat,
        "frames_analyzed": n,
        "frame_timestamp": frame_timestamp,
        "analysis_timestamp": datetime.now().isoformat()
    }


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


async def frame_receiver(websocket: WebSocket):
    """Continuously receive frames and keep only the latest"""
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)

            async with latest_frame_buffer["lock"]:
                latest_frame_buffer["frame"] = frame_data['frame']
                latest_frame_buffer["timestamp"] = frame_data.get(
                    'timestamp', datetime.now().isoformat()
                )

    except Exception as e:
        print(f"Frame receiver stopped: {e}")


async def frame_processor(websocket: WebSocket):
    """
    Process latest frame:
    1. Add to rolling buffer
    2. Check for motion
    3. Send buffer to SmolVLM2 for temporal analysis
    """
    try:
        while True:
            # Wait for new frame
            async with latest_frame_buffer["lock"]:
                if latest_frame_buffer["frame"] is None:
                    await asyncio.sleep(0.1)
                    continue

                # Grab latest frame and clear slot
                frame_b64 = latest_frame_buffer["frame"]
                frame_timestamp = latest_frame_buffer["timestamp"]
                latest_frame_buffer["frame"] = None

            try:
                # Decode frame
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Resize
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Add to rolling buffer
                frame_buffer.append(pil_img)

                # Check motion before sending to VLM (saves GPU)
                buffer_list = list(frame_buffer)
                rgb_list = [np.array(f) for f in buffer_list]
                motion_detected = detect_motion(rgb_list)

                if not motion_detected:
                    print(f"⏭️ Static scene - skipping VLM analysis")
                    continue

                # Run temporal inference in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    analyze_frames,
                    buffer_list,
                    frame_timestamp
                )

                # Send result to frontend
                await websocket.send_json(result)

                if result['is_threat']:
                    print(
                        f"🚨 THREAT at {frame_timestamp} "
                        f"[{result['frames_analyzed']} frames]: "
                        f"{result['description']}"
                    )
                else:
                    print(
                        f"✓ {frame_timestamp} "
                        f"[{result['frames_analyzed']} frames]: "
                        f"{result['description'][:60]}..."
                    )

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Processing error: {e}")

            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Frame processor stopped: {e}")


@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    await websocket.accept()
    print("✓ Client connected")

    # Clear buffer for new connection
    frame_buffer.clear()
    async with latest_frame_buffer["lock"]:
        latest_frame_buffer["frame"] = None
        latest_frame_buffer["timestamp"] = None

    await websocket.send_json({
        "description": "Connected. Warming up temporal buffer...",
        "is_threat": False,
        "frames_analyzed": 0,
        "frame_timestamp": datetime.now().isoformat(),
        "analysis_timestamp": datetime.now().isoformat()
    })

    try:
        await asyncio.gather(
            frame_receiver(websocket),
            frame_processor(websocket)
        )

    except WebSocketDisconnect:
        print("✗ Client disconnected")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
