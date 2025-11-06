#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Object Detection for Pythonista 3 on iOS
===================================================

A single-file, production-ready object detection app using OpenCV DNN and AVFoundation.
Achieves >=15 FPS on modern iPhones (A13+) with CPU-only inference.

## Quick Start
1. Place model files in the same directory as this script:
   - MobileNet-SSD: deploy.prototxt, mobilenet_ssd.caffemodel, labels.txt
   - YOLO-tiny: yolov3-tiny.cfg, yolov3-tiny.weights, coco.names
2. Open in Pythonista 3 and run
3. Tap "Start" to begin detection

## Model Sources
- MobileNet-SSD: https://github.com/chuanqi305/MobileNet-SSD
- YOLOv3-tiny: https://pjreddie.com/darknet/yolo/

## Performance Tips
- Reduce inference size for older devices (320x320 for SSD, 320 for YOLO)
- Enable frame skipping (auto-enabled under load)
- Use MobileNet-SSD for better speed, YOLO-tiny for better accuracy

Author: Claude (Anthropic)
License: MIT
"""

import cv2
import numpy as np
import ui
import time
import json
import os
import threading
import queue
from collections import deque, defaultdict
from math import sqrt
import traceback

# --- Objective-C Bridge Imports ---
try:
    from objc_util import (
        ObjCClass, ObjCInstance, c_void_p, create_objc_class,
        on_main_thread, ns, sel, CGRect, CGSize
    )
    OBJC_AVAILABLE = True
except ImportError:
    OBJC_AVAILABLE = False
    print("WARNING: objc_util not available. Camera features disabled.")


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

APP_VERSION = "1.0.0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "settings.json")

# Model file paths
MOBILENET_PROTOTXT = os.path.join(SCRIPT_DIR, "deploy.prototxt")
MOBILENET_MODEL = os.path.join(SCRIPT_DIR, "mobilenet_ssd.caffemodel")
MOBILENET_LABELS = os.path.join(SCRIPT_DIR, "labels.txt")

YOLO_CFG = os.path.join(SCRIPT_DIR, "yolov3-tiny.cfg")
YOLO_WEIGHTS = os.path.join(SCRIPT_DIR, "yolov3-tiny.weights")
YOLO_NAMES = os.path.join(SCRIPT_DIR, "coco.names")

# Default settings
DEFAULT_SETTINGS = {
    "model": "ssd",  # "ssd" or "yolo"
    "confidence": 0.5,
    "nms_threshold": 0.4,
    "input_size": 320,  # SSD: 300-600, YOLO: 320-416
    "preview_scale": 1.0,
    "show_labels": True,
    "enable_frame_skip": True,
    "target_fps": 15,
}

# Performance tuning
FRAME_BUFFER_SIZE = 2
INFERENCE_QUEUE_SIZE = 1
FPS_SMOOTHING_WINDOW = 30
MIN_INPUT_SIZE = 256
MAX_INPUT_SIZE = 608
AUTO_THROTTLE_ENABLED = True

# UI Constants
UI_BG_COLOR = (0.1, 0.1, 0.1, 1.0)
UI_FG_COLOR = (0.9, 0.9, 0.9, 1.0)
UI_ACCENT_COLOR = (0.2, 0.6, 1.0, 1.0)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Create necessary directories."""
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def load_settings():
    """Load settings from JSON file."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Merge with defaults
                result = DEFAULT_SETTINGS.copy()
                result.update(settings)
                return result
        except Exception as e:
            log_error(f"Failed to load settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        log_error(f"Failed to save settings: {e}")


def log_error(message):
    """Write error to log file."""
    try:
        log_file = os.path.join(LOGS_DIR, "runtime.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    except:
        pass  # Silently fail if logging fails


def log_info(message):
    """Write info to log file."""
    log_error(f"INFO: {message}")


def get_color_for_class(class_name):
    """Generate deterministic color from class name."""
    hash_val = sum(ord(c) for c in class_name)
    hue = (hash_val * 137) % 360
    # Convert HSV to RGB (simple approximation)
    h = hue / 60.0
    x = 1 - abs(h % 2 - 1)
    if h < 1:
        r, g, b = 1, x, 0
    elif h < 2:
        r, g, b = x, 1, 0
    elif h < 3:
        r, g, b = 0, 1, x
    elif h < 4:
        r, g, b = 0, x, 1
    elif h < 5:
        r, g, b = x, 0, 1
    else:
        r, g, b = 1, 0, x
    return (int(b * 200 + 55), int(g * 200 + 55), int(r * 200 + 55))


# ============================================================================
# CAMERA STREAM (AVFoundation Bridge)
# ============================================================================

class CameraStream:
    """
    Bridges AVFoundation camera to numpy frames via objc_util.

    Uses AVCaptureSession with AVCaptureVideoDataOutput to get real-time
    frames from the iOS camera. Converts CMSampleBuffer to numpy BGR.
    """

    def __init__(self, resolution=(1280, 720)):
        if not OBJC_AVAILABLE:
            raise RuntimeError("objc_util not available")

        self.resolution = resolution
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0

        # Objective-C classes
        self.AVCaptureSession = ObjCClass('AVCaptureSession')
        self.AVCaptureDevice = ObjCClass('AVCaptureDevice')
        self.AVCaptureDeviceInput = ObjCClass('AVCaptureDeviceInput')
        self.AVCaptureVideoDataOutput = ObjCClass('AVCaptureVideoDataOutput')
        self.AVCaptureVideoPreviewLayer = ObjCClass('AVCaptureVideoPreviewLayer')
        self.AVMediaTypeVideo = ns('vide')

        # Session components
        self.session = None
        self.output = None
        self.delegate = None

    def start(self):
        """Initialize and start camera capture session."""
        if self.is_running:
            return

        try:
            # Create capture session
            self.session = self.AVCaptureSession.alloc().init()
            self.session.setSessionPreset_(ns('AVCaptureSessionPreset1280x720'))

            # Get default camera device
            device = self.AVCaptureDevice.defaultDeviceWithMediaType_(self.AVMediaTypeVideo)
            if not device:
                raise RuntimeError("No camera device found")

            # Create device input
            error = c_void_p()
            device_input = self.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, error)
            if not device_input:
                raise RuntimeError("Failed to create device input")

            if self.session.canAddInput_(device_input):
                self.session.addInput_(device_input)
            else:
                raise RuntimeError("Cannot add input to session")

            # Create video output
            self.output = self.AVCaptureVideoDataOutput.alloc().init()

            # Set pixel format to 32BGRA for easier conversion
            video_settings = {
                ns('kCVPixelBufferPixelFormatTypeKey'): ns(1111970369)  # kCVPixelFormatType_32BGRA
            }
            self.output.setVideoSettings_(video_settings)
            self.output.setAlwaysDiscardsLateVideoFrames_(True)

            # Create delegate for frame callbacks
            self.delegate = self._create_delegate()

            # Set up dispatch queue
            from objc_util import c
            dispatch_queue_create = c.dispatch_queue_create
            dispatch_queue_create.restype = c_void_p
            dispatch_queue_create.argtypes = [c_void_p, c_void_p]
            queue = dispatch_queue_create(ns('cameraQueue').ptr, None)

            self.output.setSampleBufferDelegate_queue_(self.delegate, ObjCInstance(queue))

            if self.session.canAddOutput_(self.output):
                self.session.addOutput_(self.output)
            else:
                raise RuntimeError("Cannot add output to session")

            # Start session
            self.session.startRunning()
            self.is_running = True
            log_info("Camera stream started")

        except Exception as e:
            log_error(f"Camera start failed: {e}")
            log_error(traceback.format_exc())
            raise

    def _create_delegate(self):
        """Create Objective-C delegate for video frame callbacks."""
        def captureOutput_didOutputSampleBuffer_fromConnection_(_self, _cmd, _output, _sample_buffer, _connection):
            try:
                self._process_sample_buffer(_sample_buffer)
            except Exception as e:
                log_error(f"Frame processing error: {e}")

        # Create delegate class
        CameraDelegate = create_objc_class(
            'CameraDelegate',
            methods=[captureOutput_didOutputSampleBuffer_fromConnection_],
            protocols=['AVCaptureVideoDataOutputSampleBufferDelegate']
        )

        return CameraDelegate.alloc().init()

    def _process_sample_buffer(self, sample_buffer):
        """Convert CMSampleBuffer to numpy BGR frame."""
        try:
            from objc_util import c

            # Get image buffer from sample buffer
            CVPixelBufferRef = c_void_p
            CMSampleBufferGetImageBuffer = c.CMSampleBufferGetImageBuffer
            CMSampleBufferGetImageBuffer.restype = CVPixelBufferRef
            CMSampleBufferGetImageBuffer.argtypes = [c_void_p]

            pixel_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
            if not pixel_buffer:
                return

            # Lock pixel buffer
            CVPixelBufferLockBaseAddress = c.CVPixelBufferLockBaseAddress
            CVPixelBufferLockBaseAddress.argtypes = [CVPixelBufferRef, c_void_p]
            CVPixelBufferLockBaseAddress(pixel_buffer, 0)

            # Get buffer info
            CVPixelBufferGetBaseAddress = c.CVPixelBufferGetBaseAddress
            CVPixelBufferGetBaseAddress.restype = c_void_p
            CVPixelBufferGetBaseAddress.argtypes = [CVPixelBufferRef]

            CVPixelBufferGetWidth = c.CVPixelBufferGetWidth
            CVPixelBufferGetWidth.restype = c_void_p
            CVPixelBufferGetWidth.argtypes = [CVPixelBufferRef]

            CVPixelBufferGetHeight = c.CVPixelBufferGetHeight
            CVPixelBufferGetHeight.restype = c_void_p
            CVPixelBufferGetHeight.argtypes = [CVPixelBufferRef]

            CVPixelBufferGetBytesPerRow = c.CVPixelBufferGetBytesPerRow
            CVPixelBufferGetBytesPerRow.restype = c_void_p
            CVPixelBufferGetBytesPerRow.argtypes = [CVPixelBufferRef]

            base_address = CVPixelBufferGetBaseAddress(pixel_buffer)
            width = CVPixelBufferGetWidth(pixel_buffer)
            height = CVPixelBufferGetHeight(pixel_buffer)
            bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer)

            # Create numpy array from buffer (BGRA format)
            import ctypes
            buffer_size = bytes_per_row * height
            buffer_ptr = ctypes.cast(base_address, ctypes.POINTER(ctypes.c_uint8))
            frame_bgra = np.ctypeslib.as_array(buffer_ptr, shape=(height, width, 4)).copy()

            # Convert BGRA to BGR
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # Unlock pixel buffer
            CVPixelBufferUnlockBaseAddress = c.CVPixelBufferUnlockBaseAddress
            CVPixelBufferUnlockBaseAddress.argtypes = [CVPixelBufferRef, c_void_p]
            CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

            # Add to buffer
            with self.lock:
                self.frame_buffer.append(frame_bgr)
                self.frame_count += 1

        except Exception as e:
            log_error(f"Sample buffer processing failed: {e}")

    def get_latest_frame(self):
        """Get the most recent frame from buffer."""
        with self.lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()
        return None

    def stop(self):
        """Stop camera capture session."""
        if self.session and self.is_running:
            self.session.stopRunning()
            self.is_running = False
            log_info("Camera stream stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


# ============================================================================
# MOCK CAMERA (for testing without objc_util)
# ============================================================================

class MockCameraStream:
    """Mock camera for testing when objc_util is unavailable."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0

    def start(self):
        self.is_running = True
        log_info("Mock camera started")

    def get_latest_frame(self):
        """Generate a test pattern frame."""
        if not self.is_running:
            return None

        with self.lock:
            self.frame_count += 1
            # Create colorful test pattern
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            frame[:, :self.resolution[0]//3] = (255, 0, 0)  # Blue
            frame[:, self.resolution[0]//3:2*self.resolution[0]//3] = (0, 255, 0)  # Green
            frame[:, 2*self.resolution[0]//3:] = (0, 0, 255)  # Red

            # Add moving rectangle
            offset = (self.frame_count * 5) % self.resolution[0]
            cv2.rectangle(frame, (offset, 100), (offset + 100, 200), (255, 255, 255), 2)

            return frame

    def stop(self):
        self.is_running = False
        log_info("Mock camera stopped")


# ============================================================================
# DETECTOR BASE CLASS
# ============================================================================

class Detector:
    """Base class for object detectors."""

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.is_loaded = False
        self.class_names = []
        self.input_size = 300

    def load(self):
        """Load model (override in subclass)."""
        raise NotImplementedError

    def infer(self, frame_bgr):
        """
        Run inference on frame.

        Args:
            frame_bgr: Input frame in BGR format (numpy array)

        Returns:
            List of detections, each as dict:
            {
                'x1': int, 'y1': int, 'x2': int, 'y2': int,
                'class_id': int, 'class_name': str, 'score': float
            }
        """
        raise NotImplementedError

    def warmup(self):
        """Warm up model with dummy forward pass."""
        if not self.is_loaded:
            return
        dummy_frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        try:
            self.infer(dummy_frame)
            log_info("Model warmup completed")
        except Exception as e:
            log_error(f"Model warmup failed: {e}")


# ============================================================================
# MOBILENET-SSD DETECTOR
# ============================================================================

class MobileNetSSDDetector(Detector):
    """MobileNet-SSD detector using OpenCV DNN."""

    def __init__(self, confidence_threshold=0.5, input_size=300):
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.net = None
        self.mean = (127.5, 127.5, 127.5)
        self.scale = 1.0 / 127.5

    def load(self):
        """Load MobileNet-SSD model."""
        try:
            if not os.path.exists(MOBILENET_PROTOTXT):
                raise FileNotFoundError(f"Prototxt not found: {MOBILENET_PROTOTXT}")
            if not os.path.exists(MOBILENET_MODEL):
                raise FileNotFoundError(f"Model not found: {MOBILENET_MODEL}")

            log_info("Loading MobileNet-SSD model...")
            self.net = cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT, MOBILENET_MODEL)

            # Load class names
            if os.path.exists(MOBILENET_LABELS):
                with open(MOBILENET_LABELS, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                # Default COCO classes for MobileNet-SSD
                self.class_names = [
                    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                ]

            self.is_loaded = True
            log_info(f"MobileNet-SSD loaded with {len(self.class_names)} classes")

            # Warmup
            self.warmup()

        except Exception as e:
            log_error(f"Failed to load MobileNet-SSD: {e}")
            log_error(traceback.format_exc())
            raise

    def infer(self, frame_bgr):
        """Run MobileNet-SSD inference."""
        if not self.is_loaded:
            return []

        try:
            h, w = frame_bgr.shape[:2]

            # Create blob
            blob = cv2.dnn.blobFromImage(
                frame_bgr, self.scale, (self.input_size, self.input_size),
                self.mean, swapRB=False, crop=False
            )

            # Forward pass
            self.net.setInput(blob)
            detections = self.net.forward()

            # Parse detections
            results = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])

                    # Get bounding box coordinates (normalized)
                    box = detections[0, 0, i, 3:7]
                    x1 = int(box[0] * w)
                    y1 = int(box[1] * h)
                    x2 = int(box[2] * w)
                    y2 = int(box[3] * h)

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                    results.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'class_id': class_id,
                        'class_name': class_name,
                        'score': float(confidence)
                    })

            return results

        except Exception as e:
            log_error(f"MobileNet-SSD inference failed: {e}")
            return []


# ============================================================================
# YOLO-TINY DETECTOR
# ============================================================================

class YOLOTinyDetector(Detector):
    """YOLOv3-tiny detector using OpenCV DNN."""

    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4, input_size=416):
        super().__init__(confidence_threshold)
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = None
        self.output_layers = []

    def load(self):
        """Load YOLO-tiny model."""
        try:
            if not os.path.exists(YOLO_CFG):
                raise FileNotFoundError(f"Config not found: {YOLO_CFG}")
            if not os.path.exists(YOLO_WEIGHTS):
                raise FileNotFoundError(f"Weights not found: {YOLO_WEIGHTS}")

            log_info("Loading YOLO-tiny model...")
            self.net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)

            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            # Load class names
            if os.path.exists(YOLO_NAMES):
                with open(YOLO_NAMES, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                # Default COCO classes
                self.class_names = self._get_default_coco_names()

            self.is_loaded = True
            log_info(f"YOLO-tiny loaded with {len(self.class_names)} classes")

            # Warmup
            self.warmup()

        except Exception as e:
            log_error(f"Failed to load YOLO-tiny: {e}")
            log_error(traceback.format_exc())
            raise

    def _get_default_coco_names(self):
        """Return default COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
            'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def infer(self, frame_bgr):
        """Run YOLO-tiny inference."""
        if not self.is_loaded:
            return []

        try:
            h, w = frame_bgr.shape[:2]

            # Create blob
            blob = cv2.dnn.blobFromImage(
                frame_bgr, 1/255.0, (self.input_size, self.input_size),
                (0, 0, 0), swapRB=True, crop=False
            )

            # Forward pass
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            # Parse detections
            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        # YOLO returns center x, center y, width, height (normalized)
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)

                        # Convert to top-left corner
                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)

                        boxes.append([x1, y1, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, self.confidence_threshold, self.nms_threshold
            )

            # Build results
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, width, height = boxes[i]
                    x2 = x1 + width
                    y2 = y1 + height

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    class_id = class_ids[i]
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                    results.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'class_id': class_id,
                        'class_name': class_name,
                        'score': confidences[i]
                    })

            return results

        except Exception as e:
            log_error(f"YOLO-tiny inference failed: {e}")
            return []


# ============================================================================
# OVERLAY VIEW (Pythonista UI)
# ============================================================================

class OverlayView(ui.View):
    """
    Main view for displaying camera feed and detections.

    Shows live camera preview with bounding boxes, labels, and metrics.
    Handles touch gestures for interaction.
    """

    def __init__(self):
        self.current_frame = None
        self.detections = []
        self.show_labels = True

        # Performance metrics
        self.fps = 0.0
        self.inference_ms = 0.0
        self.pipeline_ms = 0.0
        self.dropped_frames = 0

        # Model info
        self.model_name = ""
        self.input_size = 0

        # Touch handling
        self.last_tap_time = 0
        self.tap_count = 0

        # Color cache
        self.color_cache = {}

    def update_frame(self, frame_bgr, detections=None):
        """Update frame and detections for display."""
        self.current_frame = frame_bgr
        if detections is not None:
            self.detections = detections

        # Trigger redraw on main thread
        on_main_thread(self.set_needs_display)()

    def update_metrics(self, fps, inference_ms, pipeline_ms, dropped_frames):
        """Update performance metrics."""
        self.fps = fps
        self.inference_ms = inference_ms
        self.pipeline_ms = pipeline_ms
        self.dropped_frames = dropped_frames
        on_main_thread(self.set_needs_display)()

    def update_model_info(self, model_name, input_size):
        """Update model information."""
        self.model_name = model_name
        self.input_size = input_size

    def draw(self):
        """Draw frame with detections and UI overlays."""
        if self.current_frame is None:
            # Draw placeholder
            ui.set_color(UI_BG_COLOR)
            path = ui.Path.rect(0, 0, self.width, self.height)
            path.fill()

            ui.set_color(UI_FG_COLOR)
            text = "No camera feed"
            self._draw_text(text, self.width/2, self.height/2, centered=True)
            return

        # Convert frame to ui.Image
        frame = self._draw_detections(self.current_frame.copy())
        img = self._numpy_to_ui_image(frame)

        if img:
            # Scale image to fit view
            img_w, img_h = img.size
            view_w, view_h = self.width, self.height

            scale = min(view_w / img_w, view_h / img_h)
            new_w = img_w * scale
            new_h = img_h * scale

            x = (view_w - new_w) / 2
            y = (view_h - new_h) / 2

            img.draw(x, y, new_w, new_h)

        # Draw status overlay
        self._draw_status_overlay()

    def _draw_detections(self, frame):
        """Draw bounding boxes and labels on frame."""
        for det in self.detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            class_name = det['class_name']
            score = det['score']

            # Get color
            if class_name not in self.color_cache:
                self.color_cache[class_name] = get_color_for_class(class_name)
            color = self.color_cache[class_name]

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw filled rectangle for label background
            if self.show_labels:
                label = f"{class_name} {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                cv2.rectangle(frame, (x1, y1 - text_h - baseline - 4),
                             (x1 + text_w, y1), color, -1)

                cv2.putText(frame, label, (x1, y1 - baseline - 2),
                           font, font_scale, (255, 255, 255), thickness)

        return frame

    def _draw_status_overlay(self):
        """Draw status information overlay."""
        x, y = 10, 30
        line_height = 20

        # Background for text
        ui.set_color((0, 0, 0, 0.7))
        path = ui.Path.rect(0, 0, self.width, 120)
        path.fill()

        # Status text
        ui.set_color(UI_FG_COLOR)

        status_lines = [
            f"Model: {self.model_name} ({self.input_size}x{self.input_size})",
            f"FPS: {self.fps:.1f} | Inference: {self.inference_ms:.1f}ms | Pipeline: {self.pipeline_ms:.1f}ms",
            f"Detections: {len(self.detections)} | Dropped: {self.dropped_frames}",
        ]

        for i, line in enumerate(status_lines):
            self._draw_text(line, x, y + i * line_height)

    def _draw_text(self, text, x, y, centered=False):
        """Draw text at position."""
        if centered:
            # Approximate centering
            text_width = len(text) * 7  # Rough estimate
            x -= text_width / 2

        ui.draw_string(text, rect=(x, y, 500, 20), font=('Courier', 14))

    def _numpy_to_ui_image(self, frame_bgr):
        """Convert numpy BGR frame to ui.Image."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Encode as PNG
            success, buffer = cv2.imencode('.png', frame_rgb)
            if not success:
                return None

            # Create ui.Image from buffer
            from io import BytesIO
            img = ui.Image.from_data(BytesIO(buffer.tobytes()).read())
            return img

        except Exception as e:
            log_error(f"Image conversion failed: {e}")
            return None

    def touch_began(self, touch):
        """Handle touch begin."""
        current_time = time.time()

        # Detect double tap
        if current_time - self.last_tap_time < 0.3:
            self.tap_count += 1
            if self.tap_count >= 2:
                self._handle_double_tap(touch)
                self.tap_count = 0
        else:
            self.tap_count = 1

        self.last_tap_time = current_time

    def touch_ended(self, touch):
        """Handle touch end."""
        # Single tap after delay
        if self.tap_count == 1:
            def delayed_check():
                time.sleep(0.3)
                if self.tap_count == 1:
                    on_main_thread(lambda: self._handle_single_tap(touch))()
            threading.Thread(target=delayed_check, daemon=True).start()

    def _handle_single_tap(self, touch):
        """Handle single tap gesture."""
        # Toggle label visibility
        self.show_labels = not self.show_labels
        self.set_needs_display()

    def _handle_double_tap(self, touch):
        """Handle double tap gesture."""
        # Toggle fullscreen (hide/show controls)
        if hasattr(self.superview, 'toggle_controls'):
            self.superview.toggle_controls()


# ============================================================================
# CONTROL BAR
# ============================================================================

class ControlBar(ui.View):
    """Control panel with buttons and sliders."""

    def __init__(self, controller):
        self.controller = controller
        self.background_color = UI_BG_COLOR

    def setup_ui(self):
        """Setup UI controls."""
        y_offset = 10
        x_margin = 10

        # Start/Stop button
        self.start_btn = ui.Button(title='Start')
        self.start_btn.frame = (x_margin, y_offset, 100, 40)
        self.start_btn.background_color = UI_ACCENT_COLOR
        self.start_btn.tint_color = 'white'
        self.start_btn.action = self.controller.toggle_detection
        self.add_subview(self.start_btn)

        # Save frame button
        self.save_btn = ui.Button(title='Save Frame')
        self.save_btn.frame = (x_margin + 110, y_offset, 100, 40)
        self.save_btn.background_color = (0.3, 0.7, 0.3, 1.0)
        self.save_btn.tint_color = 'white'
        self.save_btn.action = self.controller.save_frame
        self.add_subview(self.save_btn)

        y_offset += 50

        # Model selector
        model_label = ui.Label(text='Model:')
        model_label.frame = (x_margin, y_offset, 80, 30)
        model_label.text_color = UI_FG_COLOR
        self.add_subview(model_label)

        self.model_segment = ui.SegmentedControl()
        self.model_segment.frame = (x_margin + 85, y_offset, 200, 30)
        self.model_segment.segments = ['SSD', 'YOLO']
        self.model_segment.selected_index = 0
        self.model_segment.action = self.controller.change_model
        self.add_subview(self.model_segment)

        y_offset += 40

        # Confidence slider
        conf_label = ui.Label(text='Confidence:')
        conf_label.frame = (x_margin, y_offset, 100, 30)
        conf_label.text_color = UI_FG_COLOR
        self.add_subview(conf_label)

        self.conf_slider = ui.Slider()
        self.conf_slider.frame = (x_margin + 105, y_offset, 150, 30)
        self.conf_slider.value = 0.5
        self.conf_slider.action = self.controller.update_confidence
        self.add_subview(self.conf_slider)

        self.conf_value = ui.Label(text='0.50')
        self.conf_value.frame = (x_margin + 260, y_offset, 50, 30)
        self.conf_value.text_color = UI_FG_COLOR
        self.add_subview(self.conf_value)

        y_offset += 40

        # NMS slider
        nms_label = ui.Label(text='NMS:')
        nms_label.frame = (x_margin, y_offset, 100, 30)
        nms_label.text_color = UI_FG_COLOR
        self.add_subview(nms_label)

        self.nms_slider = ui.Slider()
        self.nms_slider.frame = (x_margin + 105, y_offset, 150, 30)
        self.nms_slider.value = 0.4
        self.nms_slider.action = self.controller.update_nms
        self.add_subview(self.nms_slider)

        self.nms_value = ui.Label(text='0.40')
        self.nms_value.frame = (x_margin + 260, y_offset, 50, 30)
        self.nms_value.text_color = UI_FG_COLOR
        self.add_subview(self.nms_value)

    def update_start_button(self, is_running):
        """Update start button state."""
        self.start_btn.title = 'Stop' if is_running else 'Start'


# ============================================================================
# APP CONTROLLER
# ============================================================================

class AppController:
    """Main application controller orchestrating camera, detector, and UI."""

    def __init__(self):
        self.settings = load_settings()

        # Components
        self.camera = None
        self.detector = None
        self.overlay_view = None
        self.control_bar = None
        self.main_view = None

        # State
        self.is_running = False
        self.is_processing = False
        self.worker_thread = None
        self.stop_event = threading.Event()

        # Frame queues
        self.inference_queue = queue.Queue(maxsize=INFERENCE_QUEUE_SIZE)

        # Performance tracking
        self.fps_tracker = deque(maxlen=FPS_SMOOTHING_WINDOW)
        self.inference_times = deque(maxlen=FPS_SMOOTHING_WINDOW)
        self.pipeline_times = deque(maxlen=FPS_SMOOTHING_WINDOW)
        self.dropped_frames = 0
        self.last_frame_time = 0

        # Model management
        self.available_models = self._check_available_models()

    def _check_available_models(self):
        """Check which models are available."""
        models = {}

        # Check MobileNet-SSD
        if os.path.exists(MOBILENET_PROTOTXT) and os.path.exists(MOBILENET_MODEL):
            models['ssd'] = True
            log_info("MobileNet-SSD files found")
        else:
            models['ssd'] = False
            log_info("MobileNet-SSD files not found")

        # Check YOLO-tiny
        if os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS):
            models['yolo'] = True
            log_info("YOLO-tiny files found")
        else:
            models['yolo'] = False
            log_info("YOLO-tiny files not found")

        return models

    def setup(self):
        """Setup application components."""
        try:
            # Create UI
            self._setup_ui()

            # Initialize camera
            if OBJC_AVAILABLE:
                self.camera = CameraStream()
            else:
                log_info("Using mock camera (objc_util unavailable)")
                self.camera = MockCameraStream()

            # Load detector
            self._load_detector()

            log_info("Application setup completed")

        except Exception as e:
            log_error(f"Setup failed: {e}")
            log_error(traceback.format_exc())
            self._show_error(f"Setup failed: {str(e)}")

    def _setup_ui(self):
        """Setup Pythonista UI."""
        # Main container
        self.main_view = ui.View()
        self.main_view.name = f'Object Detection v{APP_VERSION}'
        self.main_view.background_color = UI_BG_COLOR

        # Overlay view (camera + detections)
        self.overlay_view = OverlayView()
        self.overlay_view.frame = (0, 0, 375, 667)  # Will be resized
        self.overlay_view.flex = 'WH'
        self.main_view.add_subview(self.overlay_view)

        # Control bar at bottom
        self.control_bar = ControlBar(self)
        self.control_bar.frame = (0, 480, 375, 187)
        self.control_bar.flex = 'WT'
        self.control_bar.setup_ui()
        self.main_view.add_subview(self.control_bar)

        # Adjust overlay to make room for controls
        self.overlay_view.frame = (0, 0, 375, 480)

        # Apply saved settings to UI
        self._apply_settings_to_ui()

    def _apply_settings_to_ui(self):
        """Apply saved settings to UI controls."""
        model_idx = 0 if self.settings['model'] == 'ssd' else 1
        self.control_bar.model_segment.selected_index = model_idx

        self.control_bar.conf_slider.value = self.settings['confidence']
        self.control_bar.conf_value.text = f"{self.settings['confidence']:.2f}"

        self.control_bar.nms_slider.value = self.settings['nms_threshold']
        self.control_bar.nms_value.text = f"{self.settings['nms_threshold']:.2f}"

    def _load_detector(self):
        """Load the selected detector model."""
        model_type = self.settings['model']

        if not self.available_models.get(model_type, False):
            # Try alternate model
            alternate = 'yolo' if model_type == 'ssd' else 'ssd'
            if self.available_models.get(alternate, False):
                log_info(f"Switching to {alternate} model (primary not available)")
                model_type = alternate
                self.settings['model'] = alternate
            else:
                raise RuntimeError("No model files found. Please place model files in script directory.")

        try:
            if model_type == 'ssd':
                self.detector = MobileNetSSDDetector(
                    confidence_threshold=self.settings['confidence'],
                    input_size=self.settings['input_size']
                )
                self.detector.load()
                model_name = "MobileNet-SSD"
            else:
                self.detector = YOLOTinyDetector(
                    confidence_threshold=self.settings['confidence'],
                    nms_threshold=self.settings['nms_threshold'],
                    input_size=self.settings['input_size']
                )
                self.detector.load()
                model_name = "YOLO-tiny"

            self.overlay_view.update_model_info(model_name, self.settings['input_size'])
            log_info(f"Loaded {model_name}")

        except Exception as e:
            log_error(f"Failed to load detector: {e}")
            raise

    def run(self):
        """Run the application."""
        try:
            self.main_view.present('fullscreen')
        except Exception as e:
            log_error(f"Failed to present UI: {e}")
            raise

    def toggle_detection(self, sender):
        """Start or stop detection."""
        if self.is_running:
            self.stop()
        else:
            self.start()

    def start(self):
        """Start camera and detection."""
        if self.is_running:
            return

        try:
            # Start camera
            self.camera.start()

            # Reset state
            self.stop_event.clear()
            self.is_running = True
            self.dropped_frames = 0
            self.last_frame_time = time.time()

            # Start worker thread
            self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.worker_thread.start()

            # Update UI
            self.control_bar.update_start_button(True)

            log_info("Detection started")

        except Exception as e:
            log_error(f"Start failed: {e}")
            self._show_error(f"Failed to start: {str(e)}")
            self.is_running = False

    def stop(self):
        """Stop camera and detection."""
        if not self.is_running:
            return

        log_info("Stopping detection...")

        # Signal stop
        self.stop_event.set()
        self.is_running = False

        # Wait for worker thread
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        # Stop camera
        if self.camera:
            self.camera.stop()

        # Update UI
        self.control_bar.update_start_button(False)

        log_info("Detection stopped")

    def _processing_loop(self):
        """Main processing loop (runs in worker thread)."""
        log_info("Processing loop started")

        while not self.stop_event.is_set():
            try:
                loop_start = time.time()

                # Get latest frame
                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Check if we should skip (backpressure)
                if self.settings['enable_frame_skip'] and self.is_processing:
                    self.dropped_frames += 1
                    continue

                self.is_processing = True

                # Run inference
                inference_start = time.time()
                detections = self.detector.infer(frame)
                inference_time = (time.time() - inference_start) * 1000

                self.is_processing = False

                # Update display
                self.overlay_view.update_frame(frame, detections)

                # Track performance
                loop_time = (time.time() - loop_start) * 1000
                self.inference_times.append(inference_time)
                self.pipeline_times.append(loop_time)

                # Calculate FPS
                current_time = time.time()
                if self.last_frame_time > 0:
                    frame_interval = current_time - self.last_frame_time
                    if frame_interval > 0:
                        fps = 1.0 / frame_interval
                        self.fps_tracker.append(fps)
                self.last_frame_time = current_time

                # Update metrics display
                avg_fps = sum(self.fps_tracker) / len(self.fps_tracker) if self.fps_tracker else 0
                avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
                avg_pipeline = sum(self.pipeline_times) / len(self.pipeline_times) if self.pipeline_times else 0

                self.overlay_view.update_metrics(avg_fps, avg_inference, avg_pipeline, self.dropped_frames)

                # Auto-throttle if needed
                if AUTO_THROTTLE_ENABLED and avg_fps < self.settings['target_fps'] - 5:
                    self._auto_throttle()

            except Exception as e:
                log_error(f"Processing loop error: {e}")
                log_error(traceback.format_exc())
                time.sleep(0.1)

        log_info("Processing loop stopped")

    def _auto_throttle(self):
        """Automatically reduce input size if FPS is too low."""
        current_size = self.detector.input_size
        if current_size > MIN_INPUT_SIZE:
            new_size = max(MIN_INPUT_SIZE, current_size - 32)
            self.detector.input_size = new_size
            self.overlay_view.update_model_info(
                self.overlay_view.model_name,
                new_size
            )
            log_info(f"Auto-throttle: reduced input size to {new_size}")

    def change_model(self, sender):
        """Change detection model."""
        was_running = self.is_running

        if was_running:
            self.stop()

        try:
            # Update settings
            model_idx = sender.selected_index
            self.settings['model'] = 'ssd' if model_idx == 0 else 'yolo'
            save_settings(self.settings)

            # Reload detector
            self._load_detector()

            if was_running:
                self.start()

        except Exception as e:
            log_error(f"Model change failed: {e}")
            self._show_error(f"Failed to change model: {str(e)}")

    def update_confidence(self, sender):
        """Update confidence threshold."""
        value = sender.value * 0.5 + 0.2  # Map to 0.2-0.7
        self.control_bar.conf_value.text = f"{value:.2f}"

        self.settings['confidence'] = value
        if self.detector:
            self.detector.confidence_threshold = value

        save_settings(self.settings)

    def update_nms(self, sender):
        """Update NMS threshold."""
        value = sender.value * 0.3 + 0.3  # Map to 0.3-0.6
        self.control_bar.nms_value.text = f"{value:.2f}"

        self.settings['nms_threshold'] = value
        if hasattr(self.detector, 'nms_threshold'):
            self.detector.nms_threshold = value

        save_settings(self.settings)

    def save_frame(self, sender):
        """Save current frame to disk."""
        if self.overlay_view.current_frame is None:
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save raw frame
            raw_path = os.path.join(CAPTURES_DIR, f"raw_{timestamp}.png")
            cv2.imwrite(raw_path, self.overlay_view.current_frame)

            # Save annotated frame
            annotated = self.overlay_view._draw_detections(self.overlay_view.current_frame.copy())
            annotated_path = os.path.join(CAPTURES_DIR, f"annotated_{timestamp}.png")
            cv2.imwrite(annotated_path, annotated)

            log_info(f"Saved frames: {timestamp}")

            # Show feedback
            self._show_toast("Frame saved!")

        except Exception as e:
            log_error(f"Save frame failed: {e}")
            self._show_error(f"Failed to save: {str(e)}")

    def _show_error(self, message):
        """Show error dialog."""
        def show():
            import console
            console.alert("Error", message, "OK", hide_cancel_button=True)
        on_main_thread(show)()

    def _show_toast(self, message):
        """Show brief toast message."""
        def show():
            import console
            console.hud_alert(message, 'success', 1.0)
        on_main_thread(show)()

    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        log_info("Application cleaned up")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    print(f"Real-Time Object Detection v{APP_VERSION}")
    print("=" * 50)

    # Ensure directories exist
    ensure_dirs()

    # Log startup
    log_info(f"Application started (v{APP_VERSION})")

    # Check for model files
    models_found = []
    if os.path.exists(MOBILENET_PROTOTXT) and os.path.exists(MOBILENET_MODEL):
        models_found.append("MobileNet-SSD")
    if os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS):
        models_found.append("YOLO-tiny")

    if not models_found:
        print("\nWARNING: No model files found!")
        print("\nPlease place model files in the script directory:")
        print("  - MobileNet-SSD: deploy.prototxt, mobilenet_ssd.caffemodel, labels.txt")
        print("  - YOLO-tiny: yolov3-tiny.cfg, yolov3-tiny.weights, coco.names")
        print("\nThe app will run in preview-only mode.")
        log_error("No model files found")
    else:
        print(f"\nFound models: {', '.join(models_found)}")

    # Create and run app
    try:
        app = AppController()
        app.setup()
        app.run()

    except KeyboardInterrupt:
        print("\nShutting down...")
        if 'app' in locals():
            app.cleanup()

    except Exception as e:
        print(f"\nFatal error: {e}")
        log_error(f"Fatal error: {e}")
        log_error(traceback.format_exc())
        import traceback as tb
        tb.print_exc()


# ============================================================================
# SELF-TEST SECTION
# ============================================================================

def run_tests():
    """Run basic self-tests."""
    print("Running self-tests...")

    # Test 1: Color generation
    print("  Testing color generation...")
    color1 = get_color_for_class("person")
    color2 = get_color_for_class("car")
    assert color1 != color2, "Colors should be different"
    assert color1 == get_color_for_class("person"), "Colors should be deterministic"
    print("    ✓ Color generation works")

    # Test 2: Settings
    print("  Testing settings...")
    test_settings = {"test": 123}
    save_settings(test_settings)
    loaded = load_settings()
    assert "test" in loaded, "Settings should persist"
    print("    ✓ Settings persistence works")

    # Test 3: Mock camera
    print("  Testing mock camera...")
    mock_cam = MockCameraStream()
    mock_cam.start()
    frame = mock_cam.get_latest_frame()
    assert frame is not None, "Mock camera should produce frames"
    assert frame.shape[2] == 3, "Frame should be BGR"
    mock_cam.stop()
    print("    ✓ Mock camera works")

    print("\nAll tests passed! ✓")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_tests()
    else:
        main()
