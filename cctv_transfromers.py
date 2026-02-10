import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2
from PIL import Image
import gc

print("="*60)
print("CCTV SURVEILLANCE DEBUG MODE")
print("="*60)

# Check CUDA
print(f"\n1. Checking CUDA...")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Check video file
VIDEO_PATH = "accident.mp4"
print(f"\n2. Checking video file: {VIDEO_PATH}")
import os
if not os.path.exists(VIDEO_PATH):
    print(f"   ❌ ERROR: Video file not found!")
    print(f"   Looking in: {os.path.abspath(VIDEO_PATH)}")
    exit(1)
else:
    print(f"   ✓ Video file found")
    file_size = os.path.getsize(VIDEO_PATH) / (1024*1024)
    print(f"   File size: {file_size:.2f} MB")

# Try to open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("   ❌ ERROR: Cannot open video file!")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"   ✓ Video opened successfully")
print(f"   FPS: {fps}")
print(f"   Total Frames: {total_frames}")
print(f"   Duration: {duration:.1f} seconds")

cap.release()

# Load model
print(f"\n3. Loading SmolVLM2-2.2B model...")
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("   ✓ Processor loaded")
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"   ✓ Model loaded")
    print(f"   Device: {model.device}")
except Exception as e:
    print(f"   ❌ ERROR loading model: {e}")
    exit(1)

# Threat keywords
THREAT_KEYWORDS = [
    'gun', 'weapon', 'knife', 'axe', 'armed',
    'breaking', 'smashing', 'attacking', 'fighting',
    'robbery', 'robbing', 'stealing', 'theft',
    'vandalism', 'destroying', 'damage',
    'mask', 'balaclava', 'covered face',
    'threatening', 'aggressive', 'violence', 'accident', 'crash', 'collision', 'fire', 'explosion', 'suspicious', 'dangerous'
]

print(f"\n4. Starting analysis...")
print("="*60 + "\n")

# Process video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
processed_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % fps == 0:
        sec = frame_count // fps
        processed_count += 1
        
        print(f"Processing frame at {sec}s... ", end='', flush=True)
        
        try:
            h, w = frame.shape[:2]
            cropped = frame[60:h-60, :]
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            target_h = 384
            target_w = int(384 * rgb.shape[1] / rgb.shape[0])
            rgb = cv2.resize(rgb, (target_w, target_h))
            img = Image.fromarray(rgb)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe what you see in this CCTV footage in one sentence."}
                ]
            }]
            
            with torch.inference_mode():
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                
                inputs = processor(
                    text=prompt,
                    images=[img],
                    return_tensors="pt"
                ).to(model.device)
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                
                full_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if "Assistant:" in full_text:
                    text = full_text.split("Assistant:")[-1].strip()
                elif "\n" in full_text:
                    text = full_text.split("\n")[-1].strip()
                else:
                    text = full_text.strip()
            
            if '.' in text:
                text = text.split('.')[0] + '.'
            
            # Check for threats
            text_lower = text.lower()
            is_threat = any(keyword in text_lower for keyword in THREAT_KEYWORDS)
            
            if is_threat:
                print(f"🚨 ALERT: {text}")
            else:
                print(f"✅ {text}")
            
            del inputs, generated_ids, img
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ ERROR: {e}")

cap.release()

print("\n" + "="*60)
print(f"✓ Analysis complete")
print(f"✓ Total frames: {frame_count}")
print(f"✓ Processed frames: {processed_count}")
print("="*60)
