import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import json, gc, os, textwrap, threading

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ── Config ─────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_W      = 854     # 480p widescreen — decent + fast
FRAME_H      = 480
LOG_FILE     = Path("surveillance_log.jsonl")
MAX_MEMORY   = {0: "4GiB", 1: "15GiB"}

THREAT_KEYWORDS = [
    'gun', 'weapon', 'knife', 'axe', 'armed',
    'breaking', 'smashing', 'attacking', 'fighting',
    'robbery', 'robbing', 'stealing', 'theft',
    'vandalism', 'destroying', 'threatening',
    'aggressive', 'violence', 'fire', 'explosion',
    'screwdriver', 'hammer', 'bat', 'crowbar',
    'swinging', 'lunging', 'striking', 'stabbing',
    'running', 'chasing', 'jerking', 'suspicious', 'cable', 'wire','pen'
]

# ── Load Model ─────────────────────────────────────────
print("Loading Qwen3-VL-8B...")
MODEL_ID  = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=MAX_MEMORY,
    attn_implementation="sdpa"
)
model.eval()
print(f"✓ Model loaded")
print(f"  GPU 0 : {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"  GPU 1 : {torch.cuda.memory_allocated(1)/1e9:.2f} GB")

# ── Shared State ───────────────────────────────────────
result_state = {
    "description" : "Starting...",
    "is_threat"   : False,
    "lock"        : threading.Lock()
}
inferring = threading.Event()


# ── Inference (single frame) ───────────────────────────
def run_inference(pil_img: Image.Image, timestamp: str):
    try:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type"       : "image",
                    "image"      : pil_img,
                    "min_pixels" : 256 * 28 * 28,
                    "max_pixels" : 1280 * 28 * 28
                },
                {
                    "type": "text",
                    "text": (
                        "This is a CCTV frame. In ONE sentence describe "
                        "exactly what is happening — what the person is doing, "
                    )
                }
            ]
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        output  = processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()

        if '.' in output:
            output = output.split('.')[0] + '.'

        is_threat = any(kw in output.lower() for kw in THREAT_KEYWORDS)

        with result_state["lock"]:
            result_state["description"] = output
            result_state["is_threat"]   = is_threat

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "frame_timestamp"    : timestamp,
                "analysis_timestamp" : datetime.now().isoformat(),
                "description"        : output,
                "is_threat"          : is_threat
            }) + "\n")

        icon = "🚨 THREAT" if is_threat else "✅"
        print(f"{icon} [{timestamp}] {output}")

        del inputs, gen_ids
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"⚠️ Inference error: {e}")
    finally:
        inferring.clear()


# ── Draw Overlay ───────────────────────────────────────
def draw_overlay(frame, description, is_threat, fps):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 36), (25, 25, 25), -1)
    dot_color = (0, 100, 255) if inferring.is_set() else (0, 220, 90)
    cv2.circle(frame, (18, 18), 7, dot_color, -1)
    status = "Analyzing..." if inferring.is_set() else "Ready"
    cv2.putText(frame, status, (32, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    fps_txt  = f"FPS: {fps:.1f}"
    fps_size = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, fps_txt, (w - fps_size[0] - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    # Bottom bar
    bar_top = h - 80
    bar_bg  = (50, 15, 15) if is_threat else (15, 35, 15)
    border  = (0, 60, 220)  if is_threat else (0, 180, 80)
    cv2.rectangle(frame, (0, bar_top), (w, h), bar_bg, -1)
    cv2.line(frame, (0, bar_top), (w, bar_top), border, 2)

    label   = "THREAT" if is_threat else "SAFE"
    lbl_bg  = (0, 50, 200)  if is_threat else (0, 130, 50)
    lbl_end = 82            if is_threat else 60
    cv2.rectangle(frame, (8, bar_top + 6), (lbl_end, bar_top + 28), lbl_bg, -1)
    cv2.putText(frame, label, (12, bar_top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    txt_color = (180, 180, 255) if is_threat else (180, 255, 180)
    for i, line in enumerate(textwrap.wrap(description, width=95)[:2]):
        cv2.putText(frame, line, (10, bar_top + 46 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, txt_color, 1)

    ts      = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ts_size = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(frame, ts, (w - ts_size[0] - 8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    return frame


# ── Main ───────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {CAMERA_INDEX}")
        return

    print(f"\n✓ Camera opened at {FRAME_W}x{FRAME_H}")
    print("  Press Q to quit\n")

    cv2.namedWindow("CCTV — Qwen3-VL-8B", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV — Qwen3-VL-8B", FRAME_W, FRAME_H)

    fps_counter  = 0
    fps_timer    = cv2.getTickCount()
    fps          = 0.0
    last_trigger = 0.0   # time.time() of last inference trigger

    import time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed")
            break

        # FPS counter
        fps_counter += 1
        elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps         = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = cv2.getTickCount()

        now = time.time()

        # Trigger inference once per second when free
        if (now - last_trigger >= 1.0) and not inferring.is_set():
            last_trigger = now
            resized      = cv2.resize(frame, (FRAME_W, FRAME_H))
            pil_img      = Image.fromarray(
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            )
            timestamp = datetime.now().isoformat()
            inferring.set()
            threading.Thread(
                target=run_inference,
                args=(pil_img, timestamp),
                daemon=True
            ).start()

        # Read latest result
        with result_state["lock"]:
            desc      = result_state["description"]
            is_threat = result_state["is_threat"]

        # Draw and show
        display = draw_overlay(
            cv2.resize(frame, (FRAME_W, FRAME_H)),
            desc, is_threat, fps
        )
        cv2.imshow("CCTV — Qwen3-VL-8B", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✓ Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
