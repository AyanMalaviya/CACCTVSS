import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import gc

def list_cameras():
    """Detect available cameras"""
    print("\n" + "="*60)
    print("DETECTING AVAILABLE CAMERAS...")
    print("="*60)
    
    available = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"  Camera {i}: Available")
            cap.release()
    
    if not available:
        print("  No cameras detected!")
    
    return available

def select_input_source():
    """Menu to select video source"""
    print("\n" + "="*60)
    print("SELECT VIDEO SOURCE")
    print("="*60)
    print("1. Video file (input.mp4)")
    print("2. Custom video file path")
    print("3. Laptop webcam")
    print("4. Choose camera (ex Phone Link)")
    print("5. Auto-detect cameras")
    print("="*60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        return "input.mp4", "file"
    
    elif choice == "2":
        path = input("Enter video file path: ").strip()
        return path, "file"
    
    elif choice == "3":
        return 0, "camera"  # 0 is usually laptop webcam
    
    elif choice == "4":
        # Phone Link cameras are usually index 1 or 2
        available = list_cameras()
        if len(available) > 1:
            print(f"\nAvailable camera indices: {available}")
            print("Phone Link camera is usually index 1 or 2")
            idx = int(input("Enter camera index: ").strip())
            return idx, "camera"
        else:
            print("\n⚠️ Only one camera found. Using index 0.")
            return 0, "camera"
    
    elif choice == "5":
        available = list_cameras()
        if available:
            print(f"\nAvailable cameras: {available}")
            idx = int(input("Select camera index: ").strip())
            return idx, "camera"
        else:
            print("No cameras found. Defaulting to video file.")
            return "input.mp4", "file"
    
    else:
        print("Invalid choice. Using default: input.mp4")
        return "input.mp4", "file"

# =============================================================================
# MAIN SCRIPT
# =============================================================================

print("="*60)
print("CCTV SURVEILLANCE SYSTEM")
print("="*60)

# Select video source
VIDEO_PATH, SOURCE_TYPE = select_input_source()

print(f"\n✓ Selected: {VIDEO_PATH} ({SOURCE_TYPE})")
print("\nLoading SmolVLM2-2.2B on GPU...")

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("✓ Model loaded on GPU\n")

# Threat keywords
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

# Open video/camera
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ ERROR: Cannot open {VIDEO_PATH}")
    exit(1)

# Get FPS (for cameras, use 1 FPS analysis rate)
if SOURCE_TYPE == "camera":
    fps = 1
    print(f"✓ Camera opened - analyzing 1 frame per second")
    print("✓ Press Ctrl+C to stop\n")
else:
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"✓ Video opened - {duration:.1f}s, {fps} FPS\n")

frame_count = 0
analysis_count = 0

print("="*60)
print("STARTING ANALYSIS...")
print("="*60 + "\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if SOURCE_TYPE == "camera":
                print("\n⚠️ Camera feed lost. Reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(VIDEO_PATH)
                continue
            else:
                break
        
        frame_count += 1
        
        # Analyze every N frames (or every second for cameras)
        interval = fps if SOURCE_TYPE == "file" else 25  # For cameras, analyze ~1 FPS
        
        if frame_count % interval == 0:
            analysis_count += 1
            
            if SOURCE_TYPE == "camera":
                timestamp = f"[LIVE-{analysis_count:03d}]"
            else:
                sec = frame_count // fps
                timestamp = f"[{sec:03d}s]"
            
            try:
                h, w = frame.shape[:2]
                # Crop top/bottom bars if needed
                if h > 200:
                    cropped = frame[60:h-60, :]
                else:
                    cropped = frame
                
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
                    print(f"🚨 ALERT {timestamp} {text}")
                else:
                    print(f"✅ {timestamp} {text}")
                
                del inputs, generated_ids, img
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"⚠️ {timestamp} Error: {e}")

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by user")

finally:
    cap.release()
    print("\n" + "="*60)
    print("✓ Analysis complete")
    print(f"✓ Total frames analyzed: {analysis_count}")
    print("="*60)
