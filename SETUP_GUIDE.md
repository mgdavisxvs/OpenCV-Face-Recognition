# Setup Guide - OpenCV Face Recognition & Object Detection

Complete setup instructions for getting the application running on your system.

## Table of Contents
- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [Setup Options](#setup-options)
- [Manual Installation](#manual-installation)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Quick Install

### One-Line Installation (Linux/macOS)

```bash
curl -sSL https://raw.githubusercontent.com/your-repo/OpenCV-Face-Recognition/main/setup.sh | bash
```

Or download and run:

```bash
git clone https://github.com/your-repo/OpenCV-Face-Recognition.git
cd OpenCV-Face-Recognition
chmod +x setup.sh
./setup.sh
```

### Quick Install (Windows)

```powershell
# Using Git Bash or WSL
git clone https://github.com/your-repo/OpenCV-Face-Recognition.git
cd OpenCV-Face-Recognition
bash setup.sh
```

---

## Detailed Installation

### Prerequisites

**Required:**
- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ free disk space
- Internet connection (for downloading models)

**Optional:**
- GPU with CUDA support (for acceleration)
- Webcam (for real-time demos)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/OpenCV-Face-Recognition.git
cd OpenCV-Face-Recognition
```

#### 2. Run Setup Script

**Full Installation (Recommended):**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- ✓ Check system requirements
- ✓ Create directory structure
- ✓ Install Python dependencies (~500MB)
- ✓ Download pre-trained models (~60MB)
- ✓ Create configuration files
- ✓ Validate installation
- ✓ Run test suite

**Estimated time:** 5-10 minutes (depending on internet speed)

---

## Setup Options

The setup script supports several command-line options:

### Minimal Installation (Skip Large Downloads)

```bash
./setup.sh --minimal
```

- Skips YOLO model download (~34MB)
- Only installs MobileNet-SSD
- Faster setup, reduced disk usage
- Good for: Testing, limited bandwidth

### Skip Tests

```bash
./setup.sh --no-test
```

- Skips running the test suite
- Faster installation
- Use when: You want to test manually later

### GPU Support

```bash
./setup.sh --gpu
```

- Installs CUDA-enabled PyTorch
- Requires: NVIDIA GPU with CUDA support
- Significant performance boost for neural networks

### Development Mode

```bash
./setup.sh --dev
```

- Installs additional development tools
- Includes: Jupyter, IPython, Black, Flake8
- For: Contributing to the project

### Combined Options

```bash
# Minimal + Skip Tests + GPU
./setup.sh --minimal --no-test --gpu

# Full install with development tools
./setup.sh --dev --gpu
```

---

## Manual Installation

If the automated setup script doesn't work on your system, follow these manual steps:

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy opencv-python opencv-contrib-python pillow scipy
pip install torch torchvision torchaudio
pip install pytest pytest-benchmark hypothesis
pip install pyjwt cryptography cffi
```

### 2. Create Directory Structure

```bash
mkdir -p models dataset captures logs
mkdir -p dataset/example_person1 dataset/example_person2
```

### 3. Download Models

**MobileNet-SSD:**
```bash
cd models

# Prototxt file
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt

# Model weights (manual download may be needed)
# Visit: https://github.com/chuanqi305/MobileNet-SSD
# Download: mobilenet_iter_73000.caffemodel
# Rename to: mobilenet_ssd.caffemodel
```

**YOLO (optional):**
```bash
# Config file
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg

# Weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# Class names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### 4. Verify Installation

```bash
python3 test_models.py
```

---

## Troubleshooting

### Common Issues

#### Issue: "Python 3.8+ required"

**Solution:**
```bash
# Check your Python version
python3 --version

# If too old, install newer version:
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.9

# macOS:
brew install python@3.9

# Windows: Download from python.org
```

#### Issue: "pip3 not found"

**Solution:**
```bash
# Install pip
python3 -m ensurepip --upgrade

# Or on Ubuntu/Debian:
sudo apt install python3-pip
```

#### Issue: "opencv-python installation fails"

**Solution:**
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt install python3-dev libgl1-mesa-glx libglib2.0-0

# macOS:
brew install opencv

# Then retry:
pip3 install opencv-python
```

#### Issue: "Webcam doesn't work"

**Solutions:**
1. **Check permissions:**
   ```bash
   # Linux: Add user to video group
   sudo usermod -a -G video $USER
   # Then logout and login
   ```

2. **Test webcam:**
   ```bash
   # Linux:
   ls /dev/video*

   # Should show /dev/video0, /dev/video1, etc.
   ```

3. **Try different camera index:**
   ```python
   # Instead of cv2.VideoCapture(0)
   cap = cv2.VideoCapture(1)  # Try 1, 2, 3...
   ```

#### Issue: "Model download failed"

**Solution:**
```bash
# Manual download with browser:
# 1. Visit the model URLs (see README)
# 2. Download files to models/ directory
# 3. Verify filenames match exactly

# Or use alternative download tool:
curl -L -O <model-url>
```

#### Issue: "Tests fail with GPU errors"

**Solution:**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Or reinstall PyTorch for CPU:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
# Uninstall all OpenCV versions
pip uninstall opencv-python opencv-contrib-python opencv-python-headless

# Reinstall clean
pip install opencv-contrib-python
```

#### Issue: "Permission denied: ./setup.sh"

**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

#### Issue: "Disk space error"

**Solution:**
```bash
# Check available space
df -h .

# Minimal installation uses less space
./setup.sh --minimal

# Or clean pip cache
pip cache purge
```

### Platform-Specific Issues

#### macOS

**Issue: "command not found: wget"**
```bash
# Install wget
brew install wget

# Or use curl instead
curl -O <url>
```

**Issue: "xcrun: error: invalid active developer path"**
```bash
xcode-select --install
```

#### Windows

**Issue: "bash: ./setup.sh: Permission denied"**
```bash
# Use Git Bash or WSL
bash setup.sh
```

**Issue: "ImportError: DLL load failed"**
```bash
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### Linux

**Issue: "libGL.so.1: cannot open shared object file"**
```bash
sudo apt install libgl1-mesa-glx
```

**Issue: "Failed to initialize NVML: Driver/library version mismatch"**
```bash
# Reboot after NVIDIA driver update
sudo reboot
```

---

## Verification

After installation, verify everything works:

### 1. Check Installed Packages

```bash
pip list | grep -E "opencv|numpy|torch|scipy"
```

Expected output:
```
numpy                1.21.0
opencv-contrib-python 4.5.0
opencv-python        4.5.0
scipy                1.7.0
torch                2.0.0
torchvision          0.15.0
```

### 2. Test Model Loading

```bash
python3 test_models.py
```

Expected output:
```
Testing model files...

✓ Haar Cascade: OK
✓ MobileNet-SSD: OK
✓ YOLOv3-tiny: OK

✅ Model testing complete
```

### 3. Run Quick Demo

```bash
python3 demo_face_detection.py
```

- Should open webcam window
- Should detect faces with green boxes
- Press 'q' to quit

### 4. Run Test Suite

```bash
pytest tests/ -v
```

Expected output:
```
==================== 73 passed in 10.45s ====================
```

### 5. Check Setup Report

```bash
cat logs/setup_report_*.txt
```

Review the complete installation summary.

---

## Post-Installation Steps

### 1. Configure Settings

Edit `settings.json` to customize:
```json
{
    "model": "ssd",           // or "yolo"
    "confidence": 0.5,         // 0.0-1.0
    "input_size": 320,         // larger = slower but more accurate
    "target_fps": 15
}
```

### 2. Prepare Training Data (Optional)

For face recognition:
```bash
# Create person folders
mkdir -p dataset/alice
mkdir -p dataset/bob

# Add 10-20 photos of each person
# Then train:
python3 -c "from how_to_guide import train_recognizer; train_recognizer()"
```

### 3. Read the Documentation

- **HOW_TO_GUIDE.md** - 5 practical examples
- **COMPUTATIONAL_VISION_PARADIGM.md** - Theoretical foundation
- **PARADIGM_USE_CASES.md** - Real-world applications
- **FUTURE_ROADMAP.md** - Upcoming features

### 4. Try the Examples

From HOW_TO_GUIDE.md:
1. Basic face detection
2. Real-time object detection
3. Face recognition
4. Image enhancement pipelines
5. Multi-object tracking

### 5. Join the Community

- Report issues on GitHub
- Contribute improvements
- Share your use cases
- Request features

---

## Uninstallation

To completely remove the installation:

```bash
# Remove Python packages
pip uninstall -y numpy opencv-python opencv-contrib-python \
  pillow scipy torch torchvision pytest pyjwt cryptography

# Remove downloaded files
rm -rf models/* logs/* captures/*

# Remove configuration
rm settings.json .env
```

---

## System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Linux, macOS 10.13+, Windows 10+
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: Quad-core 2.5+ GHz
- **RAM**: 8GB+
- **Storage**: 5GB free space (for models + data)
- **GPU**: NVIDIA with CUDA support (optional)
- **Webcam**: 720p or higher

### Performance Expectations

**Face Detection (Haar Cascade):**
- CPU: 30+ FPS
- Accuracy: ~85%

**Object Detection (MobileNet-SSD):**
- CPU: 15-20 FPS
- GPU: 50+ FPS
- Accuracy: ~75%

**Object Detection (YOLO-tiny):**
- CPU: 8-12 FPS
- GPU: 30+ FPS
- Accuracy: ~80%

**Face Recognition (LBPH):**
- Training: 1-5 seconds for 50 images
- Inference: 30+ FPS
- Accuracy: 90%+ with good training data

---

## Getting Help

1. **Check Documentation**: Review all .md files in the repository
2. **Search Issues**: Look for similar problems on GitHub Issues
3. **Run Diagnostics**: Use `test_models.py` and check logs
4. **Report Issues**: Open a GitHub issue with:
   - System info (OS, Python version)
   - Error messages
   - Steps to reproduce
   - Setup report (from logs/)

---

## Next Steps

✅ **Installation Complete!**

Now you can:
1. 🚀 **Try demos**: `python3 demo_face_detection.py`
2. 📚 **Read guides**: Open `HOW_TO_GUIDE.md`
3. 🧪 **Run tests**: `pytest tests/ -v`
4. 🎯 **Build projects**: Use examples as starting point
5. 🌟 **Contribute**: Submit improvements on GitHub

**Happy coding!** 🎉
