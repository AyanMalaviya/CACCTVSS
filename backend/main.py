from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
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


app = FastAPI()


# CORS for React
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
    dtype=torch.float16,
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
    'explosion', 'suspicious', 'dangerous'
]


# Latest frame buffer (always keep most recent frame)
latest_frame_buffer = {
    "frame": None,
    "timestamp": None,
    "frame_lock": asyncio.Lock()
}


def analyze_frame(frame_pil, frame_timestamp):
    """Analyze a single frame with its capture timestamp"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this CCTV footage in one sentence."}
        ]
    }]
    
    with torch.inference_mode():
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[frame_pil], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        full_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
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
        "frame_timestamp": frame_timestamp,  # When frame was captured
        "analysis_timestamp": datetime.now().isoformat()  # When analysis completed
    }


@app.get("/")
async def root():
    return {
        "status": "CCTV Backend Running", 
        "model": MODEL_ID
    }


@app.get("/api/cameras")
async def get_cameras():
    """List available cameras"""
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
    """Background task to receive and buffer latest frame"""
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Update buffer with latest frame (overwrites old one)
            async with latest_frame_buffer["frame_lock"]:
                latest_frame_buffer["frame"] = frame_data['frame']
                latest_frame_buffer["timestamp"] = frame_data.get('timestamp', datetime.now().isoformat())
                
    except Exception as e:
        print(f"Frame receiver stopped: {e}")


async def frame_processor(websocket: WebSocket):
    """Background task to process latest available frame"""
    try:
        while True:
            # Check if there's a frame to process
            async with latest_frame_buffer["frame_lock"]:
                if latest_frame_buffer["frame"] is None:
                    await asyncio.sleep(0.1)  # Wait for first frame
                    continue
                
                # Grab latest frame and clear buffer
                frame_b64 = latest_frame_buffer["frame"]
                frame_timestamp = latest_frame_buffer["timestamp"]
                latest_frame_buffer["frame"] = None  # Mark as processed
            
            try:
                # Decode base64 to image
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Resize for model
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Run inference in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, analyze_frame, pil_img, frame_timestamp)
                
                # Send result with timestamps
                await websocket.send_json(result)
                
                if result['is_threat']:
                    print(f"🚨 THREAT at {frame_timestamp}: {result['description']}")
                else:
                    print(f"✓ {frame_timestamp}: {result['description'][:60]}...")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                await websocket.send_json({
                    "description": f"Error processing frame",
                    "is_threat": False,
                    "frame_timestamp": frame_timestamp,
                    "analysis_timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop
            
    except Exception as e:
        print(f"Frame processor stopped: {e}")


@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    await websocket.accept()
    print("✓ Client connected")
    
    # Reset buffer for new connection
    async with latest_frame_buffer["frame_lock"]:
        latest_frame_buffer["frame"] = None
        latest_frame_buffer["timestamp"] = None
    
    # Send initial status
    await websocket.send_json({
        "description": "Connected. Waiting for frames...",
        "is_threat": False,
        "frame_timestamp": datetime.now().isoformat(),
        "analysis_timestamp": datetime.now().isoformat()
    })
    
    try:
        # Run receiver and processor concurrently
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
