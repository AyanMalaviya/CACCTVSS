# 🎥 AI-Powered CCTV Surveillance System

Real-time threat detection for jewelry stores using Vision Language Models (VLM) and deep learning.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![React](https://img.shields.io/badge/react-18.0+-61dafb.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-76B900.svg)

## 🌟 Features

- 🔍 **Real-time Video Analysis** - Process live camera feeds at 1 FPS using SmolVLM2-2.2B
- 🚨 **Automatic Threat Detection** - Detect weapons, violence, robbery attempts, and suspicious behavior
- ⚡ **GPU-Accelerated** - Optimized for NVIDIA GPUs with 6GB+ VRAM
- 🌐 **Web Interface** - Modern React dashboard with live alerts
- 📊 **WebSocket Streaming** - Real-time bidirectional communication between frontend and backend
- 🎯 **Multi-Source Support** - Works with webcams, IP cameras, Phone Link cameras, and video files
- 💾 **Alert Logging** - Timestamped threat logs with descriptions

## 🏗️ Architecture
┌──────────────┐ WebSocket ┌───────────────┐ GPU Process ┌──────────────┐
│ React │ ←────────────→ │ FastAPI │ ──────────────→ │ SmolVLM2 │
│ Frontend │ (frames + │ Backend │ (inference) │ 2.2B Model │
│ (Port 3000) │ alerts) │ (Port 8000) │ │ (CUDA) │
└──────────────┘ └───────────────┘ └──────────────┘
↓ ↓
Live Dashboard Threat Classification
Alert Display Keyword Matching

## 📁 Project Structure

cctv_llamacpp/
├── backend/
│ ├── venv_backend/ # Python virtual environment (not in git)
│ ├── main.py # FastAPI server with WebSocket
│ ├── cctv_transformers.py # Standalone CLI script (original)
│ ├── requirements.txt # Python dependencies
│ └── README_BACKEND.md # Backend documentation
│
├── frontend/
│ ├── node_modules/ # NPM packages (not in git)
│ ├── public/
│ ├── src/
│ │ ├── App.js # Main React component
│ │ ├── App.css # Dashboard styles
│ │ └── index.js
│ ├── package.json # Node dependencies
│ └── README_FRONTEND.md # Frontend documentation
│
├── .gitignore # Git ignore rules
├── README.md # This file
└── LICENSE # MIT License


## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **NVIDIA GPU** with 6GB+ VRAM (RTX 3060 or better)
- **CUDA 12.1+** compatible drivers
- **Windows 10/11** (Linux/Mac compatible with minor tweaks)

### Installation

#### 1. Clone Repository

git clone https://github.com/AyanMalaviya/CACCTVSS.git
cd CACCTVSS



#### 2. Backend setup

# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv_backend

# Activate virtual environment
.\venv_backend\Scripts\activate   # Windows
# source venv_backend/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Verify GPU detection
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"



#### 3. Frontend setup

# Navigate to frontend (new terminal)
cd frontend

# Install Node dependencies
npm install

# Start development server
npm start


#### 4. Running the Application
# Terminal 1: Start Backend

cd backend
.\venv_backend\Scripts\activate
python main.py


# Terminal 2: Start Frontend

cd frontend
npm start
