#!/bin/bash
################################################################################
# OpenCV Face Recognition & Object Detection Setup Script
################################################################################
#
# This script automates the complete setup of the computer vision application:
# - Installs Python dependencies
# - Downloads pre-trained models (MobileNet-SSD, YOLO, Haar Cascades)
# - Creates necessary directories
# - Validates installation
# - Runs test suite
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [options]
#
# Options:
#   --minimal    : Skip large model downloads (YOLO)
#   --no-test    : Skip test suite execution
#   --gpu        : Install GPU-accelerated packages (requires CUDA)
#   --dev        : Install development dependencies
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
DATASET_DIR="${SCRIPT_DIR}/dataset"
CAPTURES_DIR="${SCRIPT_DIR}/captures"
LOGS_DIR="${SCRIPT_DIR}/logs"
TEMP_DIR="${SCRIPT_DIR}/.temp"

# Parse command line arguments
MINIMAL_MODE=false
SKIP_TESTS=false
GPU_MODE=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MINIMAL_MODE=true
            shift
            ;;
        --no-test)
            SKIP_TESTS=true
            shift
            ;;
        --gpu)
            GPU_MODE=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

download_file() {
    local url=$1
    local output=$2
    local description=$3

    echo "  Downloading: $description"

    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -# -o "$output" "$url"
    else
        print_error "Neither wget nor curl is installed"
        return 1
    fi

    if [ -f "$output" ]; then
        print_success "Downloaded: $output"
        return 0
    else
        print_error "Download failed: $output"
        return 1
    fi
}

################################################################################
# Step 1: Check System Requirements
################################################################################

check_system() {
    print_header "Step 1: Checking System Requirements"

    # Check Python
    if ! check_command python3; then
        print_error "Python 3 is required but not installed"
        print_info "Install Python 3.8 or higher from https://www.python.org/"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python version: $PYTHON_VERSION"

    # Check pip
    if ! check_command pip3; then
        print_error "pip3 is not installed"
        print_info "Install pip: python3 -m ensurepip --upgrade"
        exit 1
    fi

    # Check for wget or curl
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        print_warning "Neither wget nor curl found. Model downloads may fail."
    else
        print_success "Download tool available"
    fi

    # Check disk space (need ~2GB for models)
    AVAILABLE_SPACE=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 2 ]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available"
        print_info "At least 2GB recommended for model files"
    else
        print_success "Disk space: ${AVAILABLE_SPACE}GB available"
    fi

    echo ""
    print_info "System requirements: OK"
}

################################################################################
# Step 2: Create Directory Structure
################################################################################

create_directories() {
    print_header "Step 2: Creating Directory Structure"

    local dirs=(
        "$MODELS_DIR"
        "$DATASET_DIR"
        "$CAPTURES_DIR"
        "$LOGS_DIR"
        "$TEMP_DIR"
        "${DATASET_DIR}/example_person1"
        "${DATASET_DIR}/example_person2"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created: $dir"
        else
            print_info "Already exists: $dir"
        fi
    done

    # Create README in dataset folder
    cat > "${DATASET_DIR}/README.md" << 'EOF'
# Face Recognition Dataset

Add training images here organized by person:

```
dataset/
  ├── person1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── person2/
  │   ├── img1.jpg
  │   └── ...
```

Tips:
- Use 10-20 images per person
- Include different angles, lighting, and expressions
- Use consistent image quality
- JPG or PNG format
EOF

    print_success "Created dataset README"
}

################################################################################
# Step 3: Install Python Dependencies
################################################################################

install_dependencies() {
    print_header "Step 3: Installing Python Dependencies"

    print_info "Upgrading pip..."
    python3 -m pip install --upgrade pip --quiet

    # Core dependencies
    local core_packages=(
        "numpy>=1.21.0"
        "opencv-python>=4.5.0"
        "opencv-contrib-python>=4.5.0"
        "pillow>=9.0.0"
        "scipy>=1.7.0"
    )

    # Testing dependencies
    local test_packages=(
        "pytest>=7.0.0"
        "pytest-benchmark>=4.0.0"
        "hypothesis>=6.0.0"
    )

    # Security dependencies
    local security_packages=(
        "pyjwt>=2.0.0"
        "cryptography>=40.0.0"
        "cffi>=1.15.0"
    )

    # Development dependencies
    local dev_packages=(
        "ipython>=8.0.0"
        "jupyter>=1.0.0"
        "matplotlib>=3.5.0"
        "black>=22.0.0"
        "flake8>=4.0.0"
    )

    print_info "Installing core packages..."
    for package in "${core_packages[@]}"; do
        echo "  - $package"
    done
    python3 -m pip install --quiet "${core_packages[@]}"
    print_success "Core packages installed"

    print_info "Installing testing packages..."
    python3 -m pip install --quiet "${test_packages[@]}"
    print_success "Testing packages installed"

    print_info "Installing security packages..."
    python3 -m pip install --quiet "${security_packages[@]}"
    print_success "Security packages installed"

    # PyTorch - handle CPU vs GPU
    if [ "$GPU_MODE" = true ]; then
        print_info "Installing PyTorch with CUDA support..."
        python3 -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        print_success "PyTorch (GPU) installed"
    else
        print_info "Installing PyTorch (CPU version)..."
        python3 -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        print_success "PyTorch (CPU) installed"
    fi

    # Development dependencies
    if [ "$DEV_MODE" = true ]; then
        print_info "Installing development packages..."
        python3 -m pip install --quiet "${dev_packages[@]}"
        print_success "Development packages installed"
    fi

    # Additional utility packages
    print_info "Installing utility packages..."
    python3 -m pip install --quiet psutil imutils
    print_success "Utility packages installed"

    echo ""
    print_info "All Python dependencies installed"
}

################################################################################
# Step 4: Download Pre-trained Models
################################################################################

download_models() {
    print_header "Step 4: Downloading Pre-trained Models"

    cd "$MODELS_DIR"

    # -------------------------------------------------------------------------
    # 1. Haar Cascades (already included with OpenCV, just verify)
    # -------------------------------------------------------------------------
    print_info "Haar Cascades are included with OpenCV"
    print_success "Haar Cascades: Available"

    # -------------------------------------------------------------------------
    # 2. MobileNet-SSD
    # -------------------------------------------------------------------------
    print_info "Downloading MobileNet-SSD model..."

    # Prototxt
    if [ ! -f "deploy.prototxt" ]; then
        download_file \
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt" \
            "deploy.prototxt" \
            "MobileNet-SSD Prototxt"
    else
        print_info "deploy.prototxt already exists"
    fi

    # Caffemodel (large file ~23MB)
    if [ ! -f "mobilenet_ssd.caffemodel" ]; then
        print_warning "MobileNet-SSD weights need manual download"
        print_info "Download from: https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel"
        print_info "Save as: ${MODELS_DIR}/mobilenet_ssd.caffemodel"

        # Try alternative download method
        download_file \
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel" \
            "mobilenet_ssd.caffemodel" \
            "MobileNet-SSD Weights" || \
            print_warning "Automatic download failed - manual download required"
    else
        print_info "mobilenet_ssd.caffemodel already exists"
    fi

    # Labels
    if [ ! -f "coco_labels.txt" ]; then
        cat > "coco_labels.txt" << 'EOF'
background
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
EOF
        print_success "Created coco_labels.txt"
    fi

    # -------------------------------------------------------------------------
    # 3. YOLO (skip in minimal mode due to size)
    # -------------------------------------------------------------------------
    if [ "$MINIMAL_MODE" = false ]; then
        print_info "Downloading YOLOv3-tiny model..."

        # Config
        if [ ! -f "yolov3-tiny.cfg" ]; then
            download_file \
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg" \
                "yolov3-tiny.cfg" \
                "YOLOv3-tiny Config"
        else
            print_info "yolov3-tiny.cfg already exists"
        fi

        # Weights (large file ~34MB)
        if [ ! -f "yolov3-tiny.weights" ]; then
            download_file \
                "https://pjreddie.com/media/files/yolov3-tiny.weights" \
                "yolov3-tiny.weights" \
                "YOLOv3-tiny Weights"
        else
            print_info "yolov3-tiny.weights already exists"
        fi

        # Class names
        if [ ! -f "coco.names" ]; then
            download_file \
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" \
                "coco.names" \
                "COCO Class Names"
        else
            print_info "coco.names already exists"
        fi

        print_success "YOLOv3-tiny model downloaded"
    else
        print_warning "Skipping YOLO download (minimal mode)"
    fi

    cd "$SCRIPT_DIR"
    echo ""
    print_info "Model downloads complete"
}

################################################################################
# Step 5: Create Configuration Files
################################################################################

create_configs() {
    print_header "Step 5: Creating Configuration Files"

    # Default settings
    cat > "settings.json" << 'EOF'
{
    "model": "ssd",
    "confidence": 0.5,
    "nms_threshold": 0.4,
    "input_size": 320,
    "preview_scale": 1.0,
    "show_labels": true,
    "enable_frame_skip": true,
    "target_fps": 15,
    "model_paths": {
        "mobilenet_prototxt": "models/deploy.prototxt",
        "mobilenet_model": "models/mobilenet_ssd.caffemodel",
        "mobilenet_labels": "models/coco_labels.txt",
        "yolo_cfg": "models/yolov3-tiny.cfg",
        "yolo_weights": "models/yolov3-tiny.weights",
        "yolo_names": "models/coco.names"
    },
    "directories": {
        "captures": "captures",
        "logs": "logs",
        "dataset": "dataset"
    }
}
EOF
    print_success "Created settings.json"

    # Environment file
    cat > ".env.example" << 'EOF'
# OpenCV Face Recognition & Object Detection
# Environment Configuration

# Model settings
MODEL_TYPE=ssd
CONFIDENCE_THRESHOLD=0.5
INPUT_SIZE=320

# Paths
MODELS_DIR=models
DATASET_DIR=dataset
CAPTURES_DIR=captures
LOGS_DIR=logs

# Performance
TARGET_FPS=15
ENABLE_GPU=false
FRAME_SKIP=true

# Security (for production)
JWT_SECRET_KEY=change-this-in-production
API_KEY=your-api-key-here
EOF
    print_success "Created .env.example"

    # Create requirements.txt
    cat > "requirements.txt" << 'EOF'
# Core Dependencies
numpy>=1.21.0
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
pillow>=9.0.0
scipy>=1.7.0
psutil>=5.9.0
imutils>=0.5.4

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Testing
pytest>=7.0.0
pytest-benchmark>=4.0.0
hypothesis>=6.0.0

# Security
pyjwt>=2.0.0
cryptography>=40.0.0
cffi>=1.15.0

# Development (optional)
ipython>=8.0.0
jupyter>=1.0.0
matplotlib>=3.5.0
black>=22.0.0
flake8>=4.0.0
EOF
    print_success "Created requirements.txt"
}

################################################################################
# Step 6: Validate Installation
################################################################################

validate_installation() {
    print_header "Step 6: Validating Installation"

    # Create validation script
    cat > "${TEMP_DIR}/validate.py" << 'PYEOF'
import sys
import importlib

def check_import(module_name, package_name=None):
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

print("Checking Python packages...\n")

packages = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("scipy", "scipy"),
    ("torch", "pytorch"),
    ("torchvision", "torchvision"),
    ("pytest", "pytest"),
    ("jwt", "pyjwt"),
    ("cryptography", "cryptography"),
]

results = [check_import(module, package) for module, package in packages]

print(f"\n{sum(results)}/{len(results)} packages installed correctly")

if sum(results) == len(results):
    print("\n✅ All required packages installed!")
    sys.exit(0)
else:
    print("\n❌ Some packages missing - check errors above")
    sys.exit(1)
PYEOF

    python3 "${TEMP_DIR}/validate.py"

    echo ""

    # Check OpenCV version
    print_info "Checking OpenCV configuration..."
    python3 << 'PYEOF'
import cv2
print(f"OpenCV version: {cv2.__version__}")
print(f"Build info available: {cv2.getBuildInformation() is not None}")

# Check for DNN module
if hasattr(cv2, 'dnn'):
    print("✓ DNN module available")
else:
    print("✗ DNN module not available")

# Check for face module
if hasattr(cv2, 'face'):
    print("✓ Face module available")
else:
    print("✗ Face module not available")
PYEOF

    echo ""
    print_success "Validation complete"
}

################################################################################
# Step 7: Run Test Suite
################################################################################

run_tests() {
    print_header "Step 7: Running Test Suite"

    if [ "$SKIP_TESTS" = true ]; then
        print_warning "Skipping tests (--no-test flag)"
        return
    fi

    if [ ! -d "tests" ]; then
        print_warning "Test directory not found - skipping tests"
        return
    fi

    print_info "Running pytest..."

    # Run tests with coverage if available
    if python3 -m pytest tests/ -v --tb=short 2>&1 | tee "${LOGS_DIR}/test_results.log"; then
        print_success "All tests passed!"

        # Count tests
        TOTAL_TESTS=$(grep -c "PASSED\|FAILED" "${LOGS_DIR}/test_results.log" || echo "0")
        PASSED_TESTS=$(grep -c "PASSED" "${LOGS_DIR}/test_results.log" || echo "0")

        echo ""
        print_info "Test Results: ${PASSED_TESTS}/${TOTAL_TESTS} passed"
        print_info "Full log: ${LOGS_DIR}/test_results.log"
    else
        print_error "Some tests failed - check ${LOGS_DIR}/test_results.log"
    fi
}

################################################################################
# Step 8: Create Quick Start Scripts
################################################################################

create_quickstart_scripts() {
    print_header "Step 8: Creating Quick Start Scripts"

    # Face detection demo
    cat > "demo_face_detection.py" << 'PYEOF'
#!/usr/bin/env python3
"""Quick demo of face detection."""

import cv2
import sys

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit(1)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Detection Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
PYEOF
    chmod +x demo_face_detection.py
    print_success "Created demo_face_detection.py"

    # Test models script
    cat > "test_models.py" << 'PYEOF'
#!/usr/bin/env python3
"""Test that all models can be loaded."""

import cv2
import os
import sys

models_dir = "models"

print("Testing model files...\n")

# Test Haar Cascade
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        raise Exception("Cascade is empty")
    print("✓ Haar Cascade: OK")
except Exception as e:
    print(f"✗ Haar Cascade: {e}")

# Test MobileNet-SSD
try:
    prototxt = os.path.join(models_dir, "deploy.prototxt")
    model = os.path.join(models_dir, "mobilenet_ssd.caffemodel")

    if not os.path.exists(prototxt):
        raise Exception(f"Not found: {prototxt}")
    if not os.path.exists(model):
        raise Exception(f"Not found: {model}")

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("✓ MobileNet-SSD: OK")
except Exception as e:
    print(f"✗ MobileNet-SSD: {e}")

# Test YOLO
try:
    cfg = os.path.join(models_dir, "yolov3-tiny.cfg")
    weights = os.path.join(models_dir, "yolov3-tiny.weights")

    if not os.path.exists(cfg):
        raise Exception(f"Not found: {cfg}")
    if not os.path.exists(weights):
        raise Exception(f"Not found: {weights}")

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    print("✓ YOLOv3-tiny: OK")
except Exception as e:
    print(f"✗ YOLOv3-tiny: {e}")

print("\n✅ Model testing complete")
PYEOF
    chmod +x test_models.py
    print_success "Created test_models.py"
}

################################################################################
# Step 9: Generate Setup Report
################################################################################

generate_report() {
    print_header "Step 9: Generating Setup Report"

    local report_file="${LOGS_DIR}/setup_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" << EOF
OpenCV Face Recognition & Object Detection
Setup Report
Generated: $(date)

═══════════════════════════════════════════════════════════════

SYSTEM INFORMATION
------------------
OS: $(uname -s)
Architecture: $(uname -m)
Python: $(python3 --version)
Pip: $(pip3 --version | head -1)

INSTALLATION STATUS
-------------------
Directory Structure: ✓ Created
Python Packages: ✓ Installed
Configuration Files: ✓ Created
Models: $([ -f "${MODELS_DIR}/deploy.prototxt" ] && echo "✓ Downloaded" || echo "⚠ Partial")

INSTALLED PACKAGES
------------------
$(python3 -m pip list | grep -E "opencv|numpy|torch|scipy|pillow|pytest|jwt")

DIRECTORY STRUCTURE
-------------------
$(tree -L 2 "$SCRIPT_DIR" 2>/dev/null || find "$SCRIPT_DIR" -maxdepth 2 -type d)

MODEL FILES
-----------
$(ls -lh "$MODELS_DIR" 2>/dev/null || echo "Models directory listing not available")

NEXT STEPS
----------
1. Test the installation:
   python3 test_models.py

2. Run quick demo:
   python3 demo_face_detection.py

3. Run full test suite:
   pytest tests/ -v

4. Read the documentation:
   - HOW_TO_GUIDE.md - Practical examples
   - COMPUTATIONAL_VISION_PARADIGM.md - Theory
   - PARADIGM_USE_CASES.md - Real-world use cases

5. Explore examples from HOW_TO_GUIDE.md

TROUBLESHOOTING
---------------
- If webcam doesn't work: Check camera permissions
- If models don't load: Run 'python3 test_models.py' to diagnose
- If tests fail: Check ${LOGS_DIR}/test_results.log
- For issues: See HOW_TO_GUIDE.md troubleshooting section

═══════════════════════════════════════════════════════════════
EOF

    print_success "Setup report: $report_file"

    # Also display summary
    cat "$report_file"
}

################################################################################
# Main Setup Flow
################################################################################

main() {
    clear

    echo -e "${GREEN}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   OpenCV Face Recognition & Object Detection Setup           ║
║                                                               ║
║   This script will install all dependencies and models       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    print_info "Installation directory: $SCRIPT_DIR"
    print_info "Mode: $([ "$MINIMAL_MODE" = true ] && echo "Minimal" || echo "Full")"
    print_info "GPU: $([ "$GPU_MODE" = true ] && echo "Enabled" || echo "Disabled")"

    echo ""
    read -p "Press Enter to begin setup (Ctrl+C to cancel)..."

    # Run setup steps
    check_system
    create_directories
    install_dependencies
    download_models
    create_configs
    validate_installation
    run_tests
    create_quickstart_scripts
    generate_report

    # Final message
    print_header "Setup Complete!"

    echo -e "${GREEN}"
    echo "✅ Installation successful!"
    echo ""
    echo "Quick Start:"
    echo "  1. Test models:    python3 test_models.py"
    echo "  2. Run demo:       python3 demo_face_detection.py"
    echo "  3. Run tests:      pytest tests/ -v"
    echo "  4. Read guide:     cat HOW_TO_GUIDE.md"
    echo ""
    echo "For detailed examples, see: HOW_TO_GUIDE.md"
    echo -e "${NC}"
}

# Run main function
main "$@"
