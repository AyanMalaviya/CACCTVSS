# Context-Aware CCTV Surveillance System

AI-powered jewelry store surveillance using Vision Language Models (VLM) to detect suspicious activities in real-time.

## Features

- 🔍 Real-time video analysis using SmolVLM2-2.2B
- 🚨 Automatic threat detection (weapons, robbery, vandalism)
- ⚡ GPU-accelerated inference
- 📊 Per-second frame analysis
- 🎯 Threat keyword detection system

## Requirements

- Python 3.11+
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- CUDA 12.1+ compatible drivers
- Windows 10/11

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cctv-surveillance.git
cd cctv-surveillance

2. Create virtual environment
bash
python -m venv venv
.\venv\Scripts\activate  # Windows
3. Install PyTorch with CUDA
bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
4. Install dependencies
bash
pip install -r requirements.txt
5. Verify GPU detection
bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
Should output: CUDA available: True