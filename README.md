# OpenCV Face Recognition & Object Detection

**A complete computer vision framework with face recognition, object detection, and real-time tracking capabilities.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV 4.5+](https://img.shields.io/badge/opencv-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 73 passing](https://img.shields.io/badge/tests-73%20passing-brightgreen.svg)](tests/)

---

## 🚀 Quick Start

### One-Line Installation

```bash
git clone https://github.com/your-repo/OpenCV-Face-Recognition.git
cd OpenCV-Face-Recognition
chmod +x setup.sh && ./setup.sh
```

The setup script will automatically:
- ✅ Install all Python dependencies
- ✅ Download pre-trained models (MobileNet-SSD, YOLO)
- ✅ Create directory structure
- ✅ Validate installation
- ✅ Run test suite (73 tests)

**Installation time:** ~5-10 minutes

### Quick Demo

```bash
# Test face detection with webcam
python3 demo_face_detection.py

# Verify models are loaded
python3 test_models.py

# Run full test suite
pytest tests/ -v
```

---

## 📚 Features

### Core Capabilities

- **Face Detection** - Haar Cascades, fast and accurate
- **Face Recognition** - Train custom models with LBPH
- **Object Detection** - MobileNet-SSD (21 classes) and YOLOv3-tiny (80 classes)
- **Multi-Object Tracking** - Centroid tracking with motion trails
- **Image Enhancement** - Custom compositional pipelines
- **Real-Time Processing** - Optimized for 15-30 FPS

### Technical Highlights

- 🎯 **Compositional Architecture** - Transform → Detect → Reason paradigm
- 🔒 **Security Features** - Adversarial robustness, differential privacy, JWT auth
- ⚡ **Performance Optimized** - GPU acceleration, frame skipping, batch processing
- 🧪 **Comprehensive Testing** - 73 unit tests with 100% pass rate
- 📖 **Well Documented** - 5 complete examples, theoretical foundation, use cases

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[SETUP_GUIDE.md](SETUP_GUIDE.md)** | Complete installation instructions, troubleshooting |
| **[HOW_TO_GUIDE.md](HOW_TO_GUIDE.md)** | 5 practical examples with runnable code |
| **[COMPUTATIONAL_VISION_PARADIGM.md](COMPUTATIONAL_VISION_PARADIGM.md)** | Theoretical foundation (16,000+ lines) |
| **[PARADIGM_USE_CASES.md](PARADIGM_USE_CASES.md)** | Healthcare and manufacturing use cases |
| **[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)** | Planned features and enhancements |

---

## 🎯 Use Cases

### 1. Face Recognition
- Access control systems
- Photo organization
- Attendance tracking
- Personalized experiences

### 2. Object Detection
- Security surveillance
- Retail analytics
- Autonomous vehicles
- Smart home automation

### 3. Image Enhancement
- Document scanning
- Low-light photography
- Portrait enhancement
- Medical imaging

### 4. Multi-Object Tracking
- Traffic monitoring
- Sports analytics
- Crowd analysis
- Behavior tracking

---

## 💻 Quick Examples

### Example 1: Face Detection

```python
import cv2

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Object Detection with MobileNet-SSD

```python
import cv2
import numpy as np

# Load model
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',
                                'models/mobilenet_ssd.caffemodel')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Preprocess
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Detect objects
    net.setInput(blob)
    detections = net.forward()

    # Draw boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**More examples:** See [HOW_TO_GUIDE.md](HOW_TO_GUIDE.md) for 5 complete tutorials.

---

## 🛠️ Installation Options

### Full Installation (Recommended)
```bash
./setup.sh
```

### Minimal Installation (Skip YOLO)
```bash
./setup.sh --minimal
```

### With GPU Support
```bash
./setup.sh --gpu
```

### Development Mode
```bash
./setup.sh --dev
```

### Manual Installation
```bash
pip install -r requirements.txt
python3 test_models.py
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

---

## 📦 Requirements

### System Requirements
- **Python:** 3.8+
- **Disk Space:** 2GB+
- **RAM:** 4GB+ (8GB recommended)
- **OS:** Linux, macOS, Windows

### Python Dependencies
- OpenCV 4.5+ (`opencv-python`, `opencv-contrib-python`)
- NumPy 1.21+
- PyTorch 2.0+ (CPU or GPU)
- SciPy 1.7+
- Pillow 9.0+

**Full list:** See [requirements.txt](requirements.txt)

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_paradigm_foundations.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

**Test Coverage:**
- `test_paradigm_foundations.py` - 30 tests (primitives, composition laws)
- `test_performance.py` - 15 tests (complexity, latency, throughput)
- `test_security_features.py` - 28 tests (adversarial, privacy, auth)

**Total: 73 tests, 100% passing** ✅

---

## 📊 Performance

| Operation | CPU FPS | GPU FPS | Accuracy |
|-----------|---------|---------|----------|
| Face Detection (Haar) | 30+ | N/A | ~85% |
| Face Recognition (LBPH) | 30+ | N/A | 90%+ |
| MobileNet-SSD | 15-20 | 50+ | ~75% |
| YOLOv3-tiny | 8-12 | 30+ | ~80% |

**Tested on:** Intel i7-8700K, NVIDIA GTX 1080, 16GB RAM

---

## 🏗️ Project Structure

```
OpenCV-Face-Recognition/
├── setup.sh                          # Automated setup script
├── SETUP_GUIDE.md                    # Installation guide
├── HOW_TO_GUIDE.md                   # 5 practical examples
├── COMPUTATIONAL_VISION_PARADIGM.md  # Theoretical foundation
├── PARADIGM_USE_CASES.md            # Real-world use cases
├── FUTURE_ROADMAP.md                # Planned features
├── requirements.txt                  # Python dependencies
├── settings.json                     # Configuration
├── models/                          # Pre-trained models
│   ├── deploy.prototxt
│   ├── mobilenet_ssd.caffemodel
│   ├── yolov3-tiny.cfg
│   └── yolov3-tiny.weights
├── tests/                           # Test suite (73 tests)
│   ├── test_paradigm_foundations.py
│   ├── test_performance.py
│   └── test_security_features.py
├── dataset/                         # Training data
├── captures/                        # Saved images/videos
└── logs/                           # Application logs
```

---

## 🎓 Learning Resources

### Tutorials
- **Original Tutorial:** [Hackster.io](https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826)
- **Instructables:** [Step-by-step guide](https://www.instructables.com/id/Real-time-Face-Recognition-an-End-to-end-Project/)
- **How-To Guide:** [5 practical examples](HOW_TO_GUIDE.md)

### Theory
- **Compositional Vision:** [COMPUTATIONAL_VISION_PARADIGM.md](COMPUTATIONAL_VISION_PARADIGM.md)
- **Use Cases:** [PARADIGM_USE_CASES.md](PARADIGM_USE_CASES.md)

### External Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make changes and test:** `pytest tests/ -v`
4. **Commit:** `git commit -m 'Add amazing feature'`
5. **Push:** `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup
```bash
# Install with dev dependencies
./setup.sh --dev

# Run tests
pytest tests/ -v

# Check code style
black . --check
flake8 .
```

---

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
pip install opencv-contrib-python
```

**"Webcam doesn't work"**
```bash
# Check camera permissions
ls /dev/video*

# Try different camera index
cap = cv2.VideoCapture(1)  # Instead of 0
```

**"Model download failed"**
```bash
# Run setup script again
./setup.sh

# Or download manually (see SETUP_GUIDE.md)
```

**More solutions:** See [SETUP_GUIDE.md - Troubleshooting](SETUP_GUIDE.md#troubleshooting)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original tutorial by [Marcelo Rovai (MJRoBot)](https://github.com/Mjrovai)
- OpenCV development team
- YOLO by Joseph Redmon
- MobileNet-SSD by chuanqi305
- All contributors and users

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/OpenCV-Face-Recognition/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/OpenCV-Face-Recognition/discussions)
- **Documentation:** See `.md` files in repository

---

## 🌟 Star History

If you find this project helpful, please give it a ⭐️!

---

**Happy Coding! 🚀**
