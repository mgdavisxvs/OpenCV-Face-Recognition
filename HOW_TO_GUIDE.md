# How-To Guide: OpenCV Face Recognition & Object Detection
## 5 Practical Examples for Getting Started

This guide provides 5 complete, ready-to-run examples demonstrating the key features of this computer vision application. Each example includes code, explanation, and expected results.

---

## Table of Contents
1. [Example 1: Basic Face Detection Pipeline](#example-1-basic-face-detection-pipeline)
2. [Example 2: Real-Time Object Detection with MobileNet-SSD](#example-2-real-time-object-detection-with-mobilenet-ssd)
3. [Example 3: Face Recognition with Training](#example-3-face-recognition-with-training)
4. [Example 4: Custom Image Enhancement Pipeline](#example-4-custom-image-enhancement-pipeline)
5. [Example 5: Multi-Object Tracking with YOLO](#example-5-multi-object-tracking-with-yolo)

---

## Example 1: Basic Face Detection Pipeline

**Goal**: Detect faces in an image using the compositional vision paradigm (Transform → Detect → Reason).

**Use Case**: Security camera systems, photo organization, attendance systems.

### Code

```python
#!/usr/bin/env python3
"""
Example 1: Basic Face Detection Pipeline
Demonstrates: Transform ∘ Detect ∘ Reason composition
"""

import cv2
import numpy as np

# Initialize face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces_pipeline(image_path, output_path):
    """
    Complete face detection pipeline using compositional approach.

    Pipeline: Normalize ∘ Detect ∘ Filter ∘ Annotate
    Complexity: O(H×W×log(H×W))
    """

    # Step 1: TRANSFORM - Load and normalize image
    print("Step 1: Loading and normalizing image...")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale (Transform operation)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast (Transform operation)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    print(f"  ✓ Image loaded: {image.shape}")

    # Step 2: DETECT - Find faces
    print("Step 2: Detecting faces...")
    faces = face_cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"  ✓ Found {len(faces)} face(s)")

    # Step 3: REASON - Filter and rank detections
    print("Step 3: Filtering and ranking detections...")

    # Sort by face size (largest first)
    faces_sorted = sorted(
        faces,
        key=lambda f: f[2] * f[3],  # area = width × height
        reverse=True
    )

    # Filter out small faces (< 10% of image area)
    min_area = (image.shape[0] * image.shape[1]) * 0.01
    faces_filtered = [
        f for f in faces_sorted
        if f[2] * f[3] >= min_area
    ]

    print(f"  ✓ After filtering: {len(faces_filtered)} significant face(s)")

    # Step 4: ANNOTATE - Visualize results
    print("Step 4: Annotating results...")
    result = image.copy()

    for i, (x, y, w, h) in enumerate(faces_filtered):
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add label
        label = f"Face {i+1} ({w}x{h})"
        cv2.putText(result, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add confidence indicator (based on size)
        area_ratio = (w * h) / (image.shape[0] * image.shape[1])
        confidence = min(area_ratio * 100, 99.9)
        cv2.putText(result, f"{confidence:.1f}%", (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save result
    cv2.imwrite(output_path, result)
    print(f"  ✓ Saved to: {output_path}")

    return len(faces_filtered)


# Run the pipeline
if __name__ == "__main__":
    input_image = "test_image.jpg"  # Replace with your image path
    output_image = "detected_faces.jpg"

    try:
        num_faces = detect_faces_pipeline(input_image, output_image)
        print(f"\n✅ SUCCESS: Detected {num_faces} face(s)")
        print(f"📸 View results: {output_image}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
```

### Expected Output
```
Step 1: Loading and normalizing image...
  ✓ Image loaded: (480, 640, 3)
Step 2: Detecting faces...
  ✓ Found 3 face(s)
Step 3: Filtering and ranking detections...
  ✓ After filtering: 2 significant face(s)
Step 4: Annotating results...
  ✓ Saved to: detected_faces.jpg

✅ SUCCESS: Detected 2 face(s)
📸 View results: detected_faces.jpg
```

### Key Concepts
- **Transform**: Image preprocessing (grayscale, enhancement)
- **Detect**: Face detection with Haar cascades
- **Reason**: Filtering and ranking by size/confidence
- **Composition**: Operations chained sequentially

---

## Example 2: Real-Time Object Detection with MobileNet-SSD

**Goal**: Detect objects in real-time video using MobileNet-SSD neural network.

**Use Case**: Smart home security, retail analytics, autonomous vehicles.

### Prerequisites
Download the MobileNet-SSD model files:
```bash
# 1. Prototxt file
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt

# 2. Caffemodel weights
wget https://drive.google.com/uc?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc -O mobilenet_ssd.caffemodel

# 3. Create labels file
cat > labels.txt << EOF
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
```

### Code

```python
#!/usr/bin/env python3
"""
Example 2: Real-Time Object Detection with MobileNet-SSD
Demonstrates: Neural network inference with DNN module
"""

import cv2
import numpy as np
import time

class ObjectDetector:
    """Real-time object detector using MobileNet-SSD."""

    def __init__(self, prototxt, model, labels, confidence=0.5):
        """
        Initialize detector.

        Args:
            prototxt: Path to deploy.prototxt
            model: Path to mobilenet_ssd.caffemodel
            labels: Path to labels.txt
            confidence: Minimum confidence threshold (0.0-1.0)
        """
        print("Initializing MobileNet-SSD detector...")

        # Load neural network
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load labels
        with open(labels, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.confidence_threshold = confidence
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        print(f"  ✓ Loaded {len(self.classes)} classes")
        print(f"  ✓ Confidence threshold: {confidence}")

    def detect(self, frame, input_size=300):
        """
        Detect objects in frame.

        Pipeline: Preprocess ∘ Infer ∘ Postprocess

        Args:
            frame: Input image (H, W, 3)
            input_size: Network input size (300 or 512)

        Returns:
            List of (class_id, class_name, confidence, bbox)
        """
        h, w = frame.shape[:2]

        # TRANSFORM: Preprocess for network
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,  # 1/127.5
            size=(input_size, input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True
        )

        # DETECT: Neural network inference
        self.net.setInput(blob)
        detections = self.net.forward()

        # REASON: Filter and interpret results
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                class_name = self.classes[class_id]

                # Convert bbox to image coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                results.append((class_id, class_name, confidence, (x1, y1, x2, y2)))

        return results

    def annotate(self, frame, detections):
        """Draw bounding boxes and labels on frame."""
        for class_id, class_name, confidence, (x1, y1, x2, y2) in detections:
            # Draw box
            color = self.colors[class_id].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y1, label_size[1] + 10)

            cv2.rectangle(frame, (x1, y_label - label_size[1] - 10),
                         (x1 + label_size[0], y_label), color, -1)
            cv2.putText(frame, label, (x1, y_label - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame


def run_realtime_detection(video_source=0):
    """
    Run real-time object detection on webcam or video file.

    Args:
        video_source: 0 for webcam, or path to video file
    """
    # Initialize detector
    detector = ObjectDetector(
        prototxt="deploy.prototxt",
        model="mobilenet_ssd.caffemodel",
        labels="labels.txt",
        confidence=0.5
    )

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_source}")

    print("\n▶️  Starting real-time detection...")
    print("   Press 'q' to quit, 's' to save screenshot")

    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame, input_size=300)

        # Annotate frame
        frame = detector.annotate(frame, detections)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        else:
            fps = 0

        # Display FPS and detection count
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("MobileNet-SSD Object Detection", frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved screenshot: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detection stopped")


if __name__ == "__main__":
    # Run on webcam (0) or video file
    run_realtime_detection(0)  # Change to "video.mp4" for file
```

### Expected Output
```
Initializing MobileNet-SSD detector...
  ✓ Loaded 21 classes
  ✓ Confidence threshold: 0.5

▶️  Starting real-time detection...
   Press 'q' to quit, 's' to save screenshot

[Live video window shows]:
FPS: 18.3
Detections: 3
- person: 0.87
- car: 0.92
- bottle: 0.65
```

### Performance Tips
- **Input Size**: Use 300 for speed, 512 for accuracy
- **Confidence**: Increase to 0.6-0.7 to reduce false positives
- **Frame Skip**: Process every 2nd or 3rd frame if FPS is low

---

## Example 3: Face Recognition with Training

**Goal**: Train a face recognizer and identify people in real-time.

**Use Case**: Access control, personalized experiences, photo tagging.

### Code

```python
#!/usr/bin/env python3
"""
Example 3: Face Recognition with Training
Demonstrates: Training pipeline and real-time recognition
"""

import cv2
import numpy as np
import os
from pathlib import Path

class FaceRecognizer:
    """Face recognition system with training capability."""

    def __init__(self):
        """Initialize face detector and recognizer."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_labels = {}
        self.is_trained = False

    def collect_training_data(self, dataset_path):
        """
        Collect faces from organized dataset.

        Expected structure:
        dataset/
            person1/
                img1.jpg
                img2.jpg
            person2/
                img1.jpg
                img2.jpg

        Returns:
            (faces, labels) arrays for training
        """
        print("Collecting training data...")

        faces = []
        labels = []
        label_id = 0

        dataset_path = Path(dataset_path)

        # Process each person's folder
        for person_folder in sorted(dataset_path.iterdir()):
            if not person_folder.is_dir():
                continue

            person_name = person_folder.name
            self.face_labels[label_id] = person_name
            print(f"  Processing: {person_name}")

            # Process each image
            image_count = 0
            for image_path in person_folder.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5
                )

                for (x, y, w, h) in detected_faces:
                    face = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (200, 200))

                    faces.append(face_resized)
                    labels.append(label_id)
                    image_count += 1

            print(f"    ✓ Collected {image_count} face(s)")
            label_id += 1

        print(f"\nTotal: {len(faces)} faces from {len(self.face_labels)} people")
        return np.array(faces), np.array(labels)

    def train(self, dataset_path, model_path="face_model.yml"):
        """
        Train face recognizer on dataset.

        Args:
            dataset_path: Path to organized face dataset
            model_path: Path to save trained model
        """
        print("\n" + "="*60)
        print("TRAINING FACE RECOGNIZER")
        print("="*60)

        # Collect training data
        faces, labels = self.collect_training_data(dataset_path)

        if len(faces) == 0:
            raise ValueError("No training data found!")

        # Train recognizer
        print("\nTraining model...")
        self.recognizer.train(faces, labels)

        # Save model
        self.recognizer.save(model_path)
        print(f"  ✓ Model saved: {model_path}")

        self.is_trained = True
        print("\n✅ Training complete!")
        print(f"📊 Stats:")
        print(f"   - People: {len(self.face_labels)}")
        print(f"   - Total faces: {len(faces)}")
        print(f"   - Avg per person: {len(faces) / len(self.face_labels):.1f}")

    def load_model(self, model_path="face_model.yml", labels_path="labels.npy"):
        """Load pre-trained model."""
        print(f"Loading model from {model_path}...")
        self.recognizer.read(model_path)
        self.face_labels = np.load(labels_path, allow_pickle=True).item()
        self.is_trained = True
        print(f"  ✓ Loaded {len(self.face_labels)} people")

    def recognize(self, frame):
        """
        Recognize faces in frame.

        Returns:
            List of (name, confidence, bbox)
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        results = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))

            # Recognize face
            label_id, confidence = self.recognizer.predict(face_resized)

            # Lower confidence = better match (distance metric)
            # Threshold: < 50 is good match, 50-100 is uncertain, > 100 is unknown
            if confidence < 50:
                name = self.face_labels[label_id]
                match_quality = "excellent"
            elif confidence < 100:
                name = self.face_labels[label_id] + "?"
                match_quality = "uncertain"
            else:
                name = "Unknown"
                match_quality = "poor"

            results.append((name, confidence, (x, y, w, h), match_quality))

        return results

    def annotate(self, frame, recognitions):
        """Draw recognition results on frame."""
        for name, confidence, (x, y, w, h), quality in recognitions:
            # Color based on quality
            if quality == "excellent":
                color = (0, 255, 0)  # Green
            elif quality == "uncertain":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw label with confidence
            label = f"{name} ({confidence:.0f})"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


def train_recognizer():
    """Train face recognizer on dataset."""
    recognizer = FaceRecognizer()

    # Train on dataset
    dataset_path = "dataset"  # Create this folder with subfolders for each person
    recognizer.train(dataset_path, model_path="face_model.yml")

    # Save labels separately
    np.save("labels.npy", recognizer.face_labels)

    return recognizer


def run_recognition(video_source=0):
    """Run real-time face recognition."""
    # Initialize recognizer
    recognizer = FaceRecognizer()

    # Load pre-trained model
    try:
        recognizer.load_model("face_model.yml", "labels.npy")
    except:
        print("❌ No trained model found!")
        print("   Run train_recognizer() first to create a model.")
        return

    # Open video
    cap = cv2.VideoCapture(video_source)
    print("\n▶️  Starting face recognition...")
    print("   Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize faces
        recognitions = recognizer.recognize(frame)

        # Annotate frame
        frame = recognizer.annotate(frame, recognitions)

        # Display stats
        cv2.putText(frame, f"Faces: {len(recognitions)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: Train model (run once)
    print("STEP 1: Training")
    print("Create 'dataset' folder with subfolders for each person")
    print("Example: dataset/alice/, dataset/bob/, etc.")
    input("Press Enter when ready to train...")

    recognizer = train_recognizer()

    # Step 2: Run recognition
    print("\nSTEP 2: Recognition")
    input("Press Enter to start real-time recognition...")
    run_recognition(0)
```

### Setup Instructions

1. **Create Dataset Structure**:
```bash
mkdir -p dataset/alice
mkdir -p dataset/bob
mkdir -p dataset/carol

# Add 10-20 photos of each person to their folder
# Photos should show different angles, lighting, expressions
```

2. **Run Training**:
```bash
python example3_face_recognition.py
```

### Expected Output
```
============================================================
TRAINING FACE RECOGNIZER
============================================================
Collecting training data...
  Processing: alice
    ✓ Collected 15 face(s)
  Processing: bob
    ✓ Collected 18 face(s)
  Processing: carol
    ✓ Collected 12 face(s)

Total: 45 faces from 3 people

Training model...
  ✓ Model saved: face_model.yml

✅ Training complete!
📊 Stats:
   - People: 3
   - Total faces: 45
   - Avg per person: 15.0

▶️  Starting face recognition...
   Press 'q' to quit

[Video shows]:
alice (32)  [Green box - excellent match]
bob (45)    [Green box - excellent match]
```

---

## Example 4: Custom Image Enhancement Pipeline

**Goal**: Build a custom image processing pipeline using compositional operations.

**Use Case**: Medical imaging, satellite imagery, document scanning, photography.

### Code

```python
#!/usr/bin/env python3
"""
Example 4: Custom Image Enhancement Pipeline
Demonstrates: Building complex pipelines with composition
"""

import cv2
import numpy as np
from typing import Callable, List

class ImagePipeline:
    """Compositional image processing pipeline."""

    def __init__(self, name="Pipeline"):
        """Initialize empty pipeline."""
        self.name = name
        self.operations = []

    def add(self, operation: Callable, description: str):
        """Add operation to pipeline."""
        self.operations.append((operation, description))
        return self  # Enable chaining

    def execute(self, image: np.ndarray, verbose=True) -> np.ndarray:
        """
        Execute pipeline on image.

        Composition: f_n ∘ f_{n-1} ∘ ... ∘ f_2 ∘ f_1
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"EXECUTING: {self.name}")
            print(f"{'='*60}")

        result = image.copy()

        for i, (operation, description) in enumerate(self.operations, 1):
            if verbose:
                print(f"Step {i}/{len(self.operations)}: {description}")

            result = operation(result)

            if verbose:
                print(f"  ✓ Shape: {result.shape}, dtype: {result.dtype}")

        if verbose:
            print(f"\n✅ Pipeline complete!")

        return result

    def visualize(self, image: np.ndarray, output_path="pipeline_steps.jpg"):
        """Execute and save visualization of each step."""
        steps = [("Original", image)]
        result = image.copy()

        for operation, description in self.operations:
            result = operation(result)
            steps.append((description, result))

        # Create grid visualization
        n_steps = len(steps)
        grid_h = int(np.ceil(n_steps / 3))
        grid_w = min(3, n_steps)

        cell_h = 300
        cell_w = 400

        canvas = np.ones((grid_h * cell_h, grid_w * cell_w, 3), dtype=np.uint8) * 255

        for idx, (title, img) in enumerate(steps):
            row = idx // 3
            col = idx % 3

            # Resize image to fit cell
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img_resized = cv2.resize(img, (cell_w - 20, cell_h - 40))

            # Place in canvas
            y = row * cell_h + 30
            x = col * cell_w + 10
            canvas[y:y+img_resized.shape[0], x:x+img_resized.shape[1]] = img_resized

            # Add title
            cv2.putText(canvas, title, (x, row * cell_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(output_path, canvas)
        print(f"📊 Saved visualization: {output_path}")

        return result


# ============================================================================
# PREDEFINED PIPELINES
# ============================================================================

def create_document_scanner_pipeline():
    """
    Pipeline for scanning documents with phone camera.

    Steps: Denoise → Sharpen → Contrast → Binarize
    """
    pipeline = ImagePipeline("Document Scanner")

    # 1. Denoise with bilateral filter
    pipeline.add(
        lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        "Denoise (bilateral filter)"
    )

    # 2. Sharpen
    def sharpen(img):
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    pipeline.add(sharpen, "Sharpen (convolution)")

    # 3. Convert to grayscale
    pipeline.add(
        lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "Convert to grayscale"
    )

    # 4. Enhance contrast with CLAHE
    def apply_clahe(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    pipeline.add(apply_clahe, "Enhance contrast (CLAHE)")

    # 5. Adaptive thresholding for binarization
    pipeline.add(
        lambda img: cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        ),
        "Binarize (adaptive threshold)"
    )

    return pipeline


def create_lowlight_enhancement_pipeline():
    """
    Pipeline for enhancing low-light photos.

    Steps: Denoise → Gamma → CLAHE → Color Correction
    """
    pipeline = ImagePipeline("Low-Light Enhancement")

    # 1. Denoise
    pipeline.add(
        lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
        "Denoise (NLM)"
    )

    # 2. Gamma correction (brighten)
    def gamma_correction(img, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(img, table)

    pipeline.add(
        lambda img: gamma_correction(img, 1.5),
        "Brighten (gamma correction)"
    )

    # 3. CLAHE on L channel
    def clahe_lab(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    pipeline.add(clahe_lab, "Enhance contrast (LAB CLAHE)")

    # 4. Color balance
    def auto_white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    pipeline.add(auto_white_balance, "Color correction (white balance)")

    return pipeline


def create_portrait_enhancement_pipeline():
    """
    Pipeline for enhancing portrait photos.

    Steps: Smooth Skin → Sharpen Eyes → Adjust Tone → Vignette
    """
    pipeline = ImagePipeline("Portrait Enhancement")

    # 1. Skin smoothing (bilateral filter)
    pipeline.add(
        lambda img: cv2.bilateralFilter(img, 9, 90, 90),
        "Smooth skin (bilateral)"
    )

    # 2. Selective sharpening
    def selective_sharpen(img):
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    pipeline.add(selective_sharpen, "Sharpen details")

    # 3. Warm tone adjustment
    def warm_tone(img):
        result = img.copy()
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)  # Reduce blue
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)  # Increase red
        return result.astype(np.uint8)

    pipeline.add(warm_tone, "Warm tone adjustment")

    # 4. Vignette effect
    def add_vignette(img):
        rows, cols = img.shape[:2]

        # Create radial gradient
        X = np.linspace(-1, 1, cols)
        Y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(X, Y)
        radius = np.sqrt(X**2 + Y**2)

        # Vignette mask (darker at edges)
        vignette = 1 - np.clip(radius - 0.5, 0, 1) * 0.6
        vignette = np.stack([vignette] * 3, axis=2)

        return (img * vignette).astype(np.uint8)

    pipeline.add(add_vignette, "Add vignette")

    return pipeline


# ============================================================================
# DEMO FUNCTION
# ============================================================================

def demo_pipelines():
    """Demonstrate all enhancement pipelines."""

    # Load test image
    test_image = "test_photo.jpg"  # Replace with your image
    image = cv2.imread(test_image)

    if image is None:
        # Create synthetic test image
        print("⚠️  No test image found, creating synthetic image...")
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    print(f"Input image: {image.shape}")

    # Test each pipeline
    pipelines = [
        ("document", create_document_scanner_pipeline()),
        ("lowlight", create_lowlight_enhancement_pipeline()),
        ("portrait", create_portrait_enhancement_pipeline())
    ]

    for name, pipeline in pipelines:
        print(f"\n{'='*60}")
        print(f"Testing: {pipeline.name}")
        print(f"{'='*60}")

        # Execute and visualize
        result = pipeline.visualize(
            image,
            output_path=f"pipeline_{name}_steps.jpg"
        )

        # Save final result
        output_file = f"enhanced_{name}.jpg"
        cv2.imwrite(output_file, result)
        print(f"💾 Saved result: {output_file}")


if __name__ == "__main__":
    demo_pipelines()
    print("\n✅ All pipelines tested successfully!")
    print("\n📂 Output files:")
    print("   - pipeline_document_steps.jpg")
    print("   - pipeline_lowlight_steps.jpg")
    print("   - pipeline_portrait_steps.jpg")
    print("   - enhanced_document.jpg")
    print("   - enhanced_lowlight.jpg")
    print("   - enhanced_portrait.jpg")
```

### Expected Output
```
============================================================
Testing: Document Scanner
============================================================

============================================================
EXECUTING: Document Scanner
============================================================
Step 1/5: Denoise (bilateral filter)
  ✓ Shape: (480, 640, 3), dtype: uint8
Step 2/5: Sharpen (convolution)
  ✓ Shape: (480, 640, 3), dtype: uint8
Step 3/5: Convert to grayscale
  ✓ Shape: (480, 640), dtype: uint8
Step 4/5: Enhance contrast (CLAHE)
  ✓ Shape: (480, 640), dtype: uint8
Step 5/5: Binarize (adaptive threshold)
  ✓ Shape: (480, 640), dtype: uint8

✅ Pipeline complete!
📊 Saved visualization: pipeline_document_steps.jpg
💾 Saved result: enhanced_document.jpg
```

---

## Example 5: Multi-Object Tracking with YOLO

**Goal**: Track multiple objects across video frames using YOLO and centroid tracking.

**Use Case**: Surveillance, traffic monitoring, sports analytics.

### Prerequisites
```bash
# Download YOLOv3-tiny files
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### Code

```python
#!/usr/bin/env python3
"""
Example 5: Multi-Object Tracking with YOLO
Demonstrates: Object detection + tracking with state management
"""

import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    """
    Track objects using centroid matching algorithm.

    Algorithm:
    1. Detect objects in frame
    2. Compute centroids
    3. Match with existing tracks (minimum distance)
    4. Update tracks or create new ones
    """

    def __init__(self, max_disappeared=50):
        """
        Initialize tracker.

        Args:
            max_disappeared: Max frames object can disappear before removal
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """Register new object with unique ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracked objects with new detections.

        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            OrderedDict of {object_id: centroid}
        """
        # No detections
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # Compute centroids of detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")

        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        # No existing objects - register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        # Match existing objects to new detections
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between all pairs
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Assign detections to objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class YOLOTracker:
    """YOLO object detector with tracking."""

    def __init__(self, cfg, weights, names, confidence=0.5, nms_threshold=0.3):
        """Initialize YOLO and tracker."""
        print("Initializing YOLO tracker...")

        # Load YOLO
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load class names
        with open(names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]

        self.confidence_threshold = confidence
        self.nms_threshold = nms_threshold

        # Initialize tracker
        self.tracker = CentroidTracker(max_disappeared=40)

        # Track history
        self.track_history = {}

        # Random colors for visualization
        self.colors = np.random.uniform(0, 255, size=(100, 3))

        print(f"  ✓ Loaded {len(self.classes)} classes")
        print(f"  ✓ Tracker initialized")

    def detect(self, frame, input_size=416):
        """
        Detect objects in frame.

        Returns:
            List of (class_id, class_name, confidence, bbox)
        """
        h, w = frame.shape[:2]

        # Preprocess
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (input_size, input_size),
            swapRB=True, crop=False
        )

        # Inference
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Process outputs
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # Convert to image coordinates
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)

                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)

                    boxes.append([x1, y1, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.confidence_threshold,
            self.nms_threshold
        )

        results = []
        detections_for_tracker = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x2, y2 = x + w, y + h

                results.append((
                    class_ids[i],
                    self.classes[class_ids[i]],
                    confidences[i],
                    (x, y, x2, y2)
                ))

                detections_for_tracker.append((x, y, x2, y2))

        # Update tracker
        objects = self.tracker.update(detections_for_tracker)

        # Update track history
        for object_id, centroid in objects.items():
            if object_id not in self.track_history:
                self.track_history[object_id] = []

            self.track_history[object_id].append(centroid)

            # Keep last 30 positions
            if len(self.track_history[object_id]) > 30:
                self.track_history[object_id].pop(0)

        return results, objects

    def annotate(self, frame, detections, tracked_objects):
        """Draw detections and tracks on frame."""
        # Draw detections
        for class_id, class_name, confidence, (x1, y1, x2, y2) in detections:
            color = self.colors[class_id % len(self.colors)].tolist()

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracked objects and trails
        for object_id, centroid in tracked_objects.items():
            # Draw ID
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw centroid
            cv2.circle(frame, tuple(centroid), 4, (0, 255, 0), -1)

            # Draw trail
            if object_id in self.track_history:
                points = self.track_history[object_id]
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
                    cv2.line(frame, tuple(points[i-1]), tuple(points[i]),
                            (0, 255, 0), thickness)

        return frame


def run_tracking(video_source=0):
    """Run multi-object tracking."""
    # Initialize tracker
    tracker = YOLOTracker(
        cfg="yolov3-tiny.cfg",
        weights="yolov3-tiny.weights",
        names="coco.names",
        confidence=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_source)

    print("\n▶️  Starting multi-object tracking...")
    print("   Press 'q' to quit, 'r' to reset tracker")

    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track
        detections, tracked_objects = tracker.detect(frame, input_size=416)

        # Annotate
        frame = tracker.annotate(frame, detections, tracked_objects)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        else:
            fps = 0

        # Display stats
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracked: {len(tracked_objects)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("YOLO Multi-Object Tracking", frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.tracker = CentroidTracker(max_disappeared=40)
            tracker.track_history = {}
            print("  ↻ Tracker reset")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Tracking stopped")


if __name__ == "__main__":
    import time
    from scipy.spatial import distance

    # Run tracking
    run_tracking(0)  # Webcam, or path to video file
```

### Expected Output
```
Initializing YOLO tracker...
  ✓ Loaded 80 classes
  ✓ Tracker initialized

▶️  Starting multi-object tracking...
   Press 'q' to quit, 'r' to reset tracker

[Live video shows]:
FPS: 8.2
Detections: 5
Tracked: 5

Objects with trails:
- ID 0: person (0.87) [Green trail]
- ID 1: car (0.93) [Green trail]
- ID 2: bicycle (0.76) [Green trail]
- ID 3: person (0.82) [Green trail]
- ID 4: dog (0.69) [Green trail]
```

### Tracking Features
- **Centroid Matching**: Associates detections across frames
- **Trail Visualization**: Shows object movement history
- **Disappearance Handling**: Maintains IDs even if objects are briefly occluded
- **Automatic ID Management**: Creates/removes IDs as objects enter/leave

---

## Tips and Best Practices

### Performance Optimization
1. **Input Size**: Smaller = faster, larger = more accurate
   - MobileNet-SSD: 300-512px
   - YOLO: 320-608px

2. **Frame Skipping**: Process every Nth frame for real-time
   ```python
   if frame_count % 2 == 0:  # Process every 2nd frame
       detections = detector.detect(frame)
   ```

3. **GPU Acceleration**: Use CUDA if available
   ```python
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
   ```

### Troubleshooting

**Problem**: Low FPS
- **Solution**: Reduce input size, enable frame skipping, use GPU

**Problem**: Too many false positives
- **Solution**: Increase confidence threshold (0.5 → 0.7)

**Problem**: Missing detections
- **Solution**: Decrease confidence threshold, use larger input size

**Problem**: Track ID switching
- **Solution**: Increase `max_disappeared` parameter in tracker

### Next Steps
1. Combine examples (e.g., face recognition + tracking)
2. Add your own custom operations to pipelines
3. Deploy to edge devices (Raspberry Pi, Jetson Nano)
4. Build a web interface with Flask/FastAPI
5. Integrate with cloud services (AWS Rekognition, Azure CV)

---

## Additional Resources

- **OpenCV Documentation**: https://docs.opencv.org/
- **YOLO**: https://pjreddie.com/darknet/yolo/
- **MobileNet-SSD**: https://github.com/chuanqi305/MobileNet-SSD
- **Face Recognition**: https://github.com/ageitgey/face_recognition

## Support

For issues or questions:
1. Check the examples above
2. Review test files in `tests/` directory
3. Read `COMPUTATIONAL_VISION_PARADIGM.md` for theory
4. Check `PARADIGM_USE_CASES.md` for more use cases

---

**Happy Coding! 🚀**
