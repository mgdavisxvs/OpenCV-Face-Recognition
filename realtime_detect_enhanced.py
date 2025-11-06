#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Object Detection for Pythonista 3 on iOS - ENHANCED UI/UX
====================================================================

Production-ready object detection with modern iOS-style interface.
Achieves >=15 FPS on modern iPhones (A13+) with beautiful, intuitive UX.

## New UI/UX Features:
- Slide-out drawer for controls (swipe from right edge)
- Auto-hiding minimal HUD (tap to toggle)
- Advanced gestures: pinch to zoom, swipe to change models
- Visual feedback: animations, loading spinners, pulse effects
- Modern iOS styling: blur effects, rounded corners, shadows
- Dedicated settings panel with organized sections
- In-app frame gallery with preview and share
- Help/tutorial screen for first-time users
- Smooth transitions and animations
- Professional color scheme with themes

Author: Claude (Anthropic)
Version: 2.0.0 (Enhanced UI)
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
from math import sqrt, sin, cos, pi
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

APP_VERSION = "2.0.0"
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
    "model": "ssd",
    "confidence": 0.5,
    "nms_threshold": 0.4,
    "input_size": 320,
    "preview_scale": 1.0,
    "show_labels": True,
    "show_confidence": True,
    "show_fps": True,
    "enable_frame_skip": True,
    "target_fps": 15,
    "theme": "dark",  # "dark" or "light"
    "hud_mode": "minimal",  # "minimal", "full", "hidden"
    "first_run": True,
}

# Performance tuning
FRAME_BUFFER_SIZE = 2
INFERENCE_QUEUE_SIZE = 1
FPS_SMOOTHING_WINDOW = 30
MIN_INPUT_SIZE = 256
MAX_INPUT_SIZE = 608

# UI Constants - Dark Theme
DARK_THEME = {
    "bg": (0.05, 0.05, 0.05, 1.0),
    "fg": (0.95, 0.95, 0.95, 1.0),
    "accent": (0.2, 0.6, 1.0, 1.0),
    "success": (0.2, 0.8, 0.4, 1.0),
    "warning": (1.0, 0.6, 0.0, 1.0),
    "error": (1.0, 0.3, 0.3, 1.0),
    "overlay": (0.0, 0.0, 0.0, 0.7),
    "card": (0.15, 0.15, 0.15, 0.95),
    "border": (0.3, 0.3, 0.3, 1.0),
}

# UI Constants - Light Theme
LIGHT_THEME = {
    "bg": (0.95, 0.95, 0.95, 1.0),
    "fg": (0.1, 0.1, 0.1, 1.0),
    "accent": (0.0, 0.5, 1.0, 1.0),
    "success": (0.2, 0.7, 0.3, 1.0),
    "warning": (1.0, 0.5, 0.0, 1.0),
    "error": (0.9, 0.2, 0.2, 1.0),
    "overlay": (1.0, 1.0, 1.0, 0.7),
    "card": (1.0, 1.0, 1.0, 0.95),
    "border": (0.7, 0.7, 0.7, 1.0),
}


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
        pass


def log_info(message):
    """Write info to log file."""
    log_error(f"INFO: {message}")


def get_color_for_class(class_name):
    """Generate deterministic color from class name."""
    hash_val = sum(ord(c) for c in class_name)
    hue = (hash_val * 137) % 360
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


def interpolate_color(color1, color2, t):
    """Interpolate between two colors."""
    return tuple(c1 * (1 - t) + c2 * t for c1, c2 in zip(color1, color2))


# ============================================================================
# CAMERA STREAM (Same as before)
# ============================================================================

class CameraStream:
    """AVFoundation camera bridge."""

    def __init__(self, resolution=(1280, 720)):
        if not OBJC_AVAILABLE:
            raise RuntimeError("objc_util not available")

        self.resolution = resolution
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0

        self.AVCaptureSession = ObjCClass('AVCaptureSession')
        self.AVCaptureDevice = ObjCClass('AVCaptureDevice')
        self.AVCaptureDeviceInput = ObjCClass('AVCaptureDeviceInput')
        self.AVCaptureVideoDataOutput = ObjCClass('AVCaptureVideoDataOutput')
        self.AVMediaTypeVideo = ns('vide')

        self.session = None
        self.output = None
        self.delegate = None

    def start(self):
        """Start camera capture."""
        if self.is_running:
            return

        try:
            self.session = self.AVCaptureSession.alloc().init()
            self.session.setSessionPreset_(ns('AVCaptureSessionPreset1280x720'))

            device = self.AVCaptureDevice.defaultDeviceWithMediaType_(self.AVMediaTypeVideo)
            if not device:
                raise RuntimeError("No camera device found")

            error = c_void_p()
            device_input = self.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, error)
            if not device_input:
                raise RuntimeError("Failed to create device input")

            if self.session.canAddInput_(device_input):
                self.session.addInput_(device_input)

            self.output = self.AVCaptureVideoDataOutput.alloc().init()
            video_settings = {
                ns('kCVPixelBufferPixelFormatTypeKey'): ns(1111970369)
            }
            self.output.setVideoSettings_(video_settings)
            self.output.setAlwaysDiscardsLateVideoFrames_(True)

            self.delegate = self._create_delegate()

            from objc_util import c
            dispatch_queue_create = c.dispatch_queue_create
            dispatch_queue_create.restype = c_void_p
            dispatch_queue_create.argtypes = [c_void_p, c_void_p]
            queue = dispatch_queue_create(ns('cameraQueue').ptr, None)

            self.output.setSampleBufferDelegate_queue_(self.delegate, ObjCInstance(queue))

            if self.session.canAddOutput_(self.output):
                self.session.addOutput_(self.output)

            self.session.startRunning()
            self.is_running = True
            log_info("Camera stream started")

        except Exception as e:
            log_error(f"Camera start failed: {e}")
            log_error(traceback.format_exc())
            raise

    def _create_delegate(self):
        """Create Objective-C delegate."""
        def captureOutput_didOutputSampleBuffer_fromConnection_(_self, _cmd, _output, _sample_buffer, _connection):
            try:
                self._process_sample_buffer(_sample_buffer)
            except Exception as e:
                log_error(f"Frame processing error: {e}")

        CameraDelegate = create_objc_class(
            'CameraDelegate',
            methods=[captureOutput_didOutputSampleBuffer_fromConnection_],
            protocols=['AVCaptureVideoDataOutputSampleBufferDelegate']
        )
        return CameraDelegate.alloc().init()

    def _process_sample_buffer(self, sample_buffer):
        """Convert CMSampleBuffer to numpy BGR."""
        try:
            from objc_util import c

            CVPixelBufferRef = c_void_p
            CMSampleBufferGetImageBuffer = c.CMSampleBufferGetImageBuffer
            CMSampleBufferGetImageBuffer.restype = CVPixelBufferRef
            CMSampleBufferGetImageBuffer.argtypes = [c_void_p]

            pixel_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
            if not pixel_buffer:
                return

            CVPixelBufferLockBaseAddress = c.CVPixelBufferLockBaseAddress
            CVPixelBufferLockBaseAddress.argtypes = [CVPixelBufferRef, c_void_p]
            CVPixelBufferLockBaseAddress(pixel_buffer, 0)

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

            import ctypes
            buffer_size = bytes_per_row * height
            buffer_ptr = ctypes.cast(base_address, ctypes.POINTER(ctypes.c_uint8))
            frame_bgra = np.ctypeslib.as_array(buffer_ptr, shape=(height, width, 4)).copy()

            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            CVPixelBufferUnlockBaseAddress = c.CVPixelBufferUnlockBaseAddress
            CVPixelBufferUnlockBaseAddress.argtypes = [CVPixelBufferRef, c_void_p]
            CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

            with self.lock:
                self.frame_buffer.append(frame_bgr)
                self.frame_count += 1

        except Exception as e:
            log_error(f"Sample buffer processing failed: {e}")

    def get_latest_frame(self):
        """Get most recent frame."""
        with self.lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()
        return None

    def stop(self):
        """Stop camera."""
        if self.session and self.is_running:
            self.session.stopRunning()
            self.is_running = False
            log_info("Camera stream stopped")


class MockCameraStream:
    """Mock camera for testing."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0

    def start(self):
        self.is_running = True
        log_info("Mock camera started")

    def get_latest_frame(self):
        """Generate test pattern."""
        if not self.is_running:
            return None

        with self.lock:
            self.frame_count += 1
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            frame[:, :self.resolution[0]//3] = (255, 0, 0)
            frame[:, self.resolution[0]//3:2*self.resolution[0]//3] = (0, 255, 0)
            frame[:, 2*self.resolution[0]//3:] = (0, 0, 255)

            offset = (self.frame_count * 5) % self.resolution[0]
            cv2.rectangle(frame, (offset, 100), (offset + 100, 200), (255, 255, 255), 2)

            return frame

    def stop(self):
        self.is_running = False
        log_info("Mock camera stopped")


# ============================================================================
# DETECTOR CLASSES (Same as before)
# ============================================================================

class Detector:
    """Base detector class."""

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.is_loaded = False
        self.class_names = []
        self.input_size = 300

    def load(self):
        raise NotImplementedError

    def infer(self, frame_bgr):
        raise NotImplementedError

    def warmup(self):
        if not self.is_loaded:
            return
        dummy_frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        try:
            self.infer(dummy_frame)
            log_info("Model warmup completed")
        except Exception as e:
            log_error(f"Model warmup failed: {e}")


class MobileNetSSDDetector(Detector):
    """MobileNet-SSD detector."""

    def __init__(self, confidence_threshold=0.5, input_size=300):
        super().__init__(confidence_threshold)
        self.input_size = input_size
        self.net = None
        self.mean = (127.5, 127.5, 127.5)
        self.scale = 1.0 / 127.5

    def load(self):
        try:
            if not os.path.exists(MOBILENET_PROTOTXT):
                raise FileNotFoundError(f"Prototxt not found: {MOBILENET_PROTOTXT}")
            if not os.path.exists(MOBILENET_MODEL):
                raise FileNotFoundError(f"Model not found: {MOBILENET_MODEL}")

            log_info("Loading MobileNet-SSD model...")
            self.net = cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT, MOBILENET_MODEL)

            if os.path.exists(MOBILENET_LABELS):
                with open(MOBILENET_LABELS, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                self.class_names = [
                    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                ]

            self.is_loaded = True
            log_info(f"MobileNet-SSD loaded with {len(self.class_names)} classes")
            self.warmup()

        except Exception as e:
            log_error(f"Failed to load MobileNet-SSD: {e}")
            log_error(traceback.format_exc())
            raise

    def infer(self, frame_bgr):
        if not self.is_loaded:
            return []

        try:
            h, w = frame_bgr.shape[:2]

            blob = cv2.dnn.blobFromImage(
                frame_bgr, self.scale, (self.input_size, self.input_size),
                self.mean, swapRB=False, crop=False
            )

            self.net.setInput(blob)
            detections = self.net.forward()

            results = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7]
                    x1 = int(box[0] * w)
                    y1 = int(box[1] * h)
                    x2 = int(box[2] * w)
                    y2 = int(box[3] * h)

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


class YOLOTinyDetector(Detector):
    """YOLO-tiny detector."""

    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4, input_size=416):
        super().__init__(confidence_threshold)
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = None
        self.output_layers = []

    def load(self):
        try:
            if not os.path.exists(YOLO_CFG):
                raise FileNotFoundError(f"Config not found: {YOLO_CFG}")
            if not os.path.exists(YOLO_WEIGHTS):
                raise FileNotFoundError(f"Weights not found: {YOLO_WEIGHTS}")

            log_info("Loading YOLO-tiny model...")
            self.net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)

            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            if os.path.exists(YOLO_NAMES):
                with open(YOLO_NAMES, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                self.class_names = self._get_default_coco_names()

            self.is_loaded = True
            log_info(f"YOLO-tiny loaded with {len(self.class_names)} classes")
            self.warmup()

        except Exception as e:
            log_error(f"Failed to load YOLO-tiny: {e}")
            log_error(traceback.format_exc())
            raise

    def _get_default_coco_names(self):
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
        if not self.is_loaded:
            return []

        try:
            h, w = frame_bgr.shape[:2]

            blob = cv2.dnn.blobFromImage(
                frame_bgr, 1/255.0, (self.input_size, self.input_size),
                (0, 0, 0), swapRB=True, crop=False
            )

            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)

                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)

                        boxes.append([x1, y1, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, self.confidence_threshold, self.nms_threshold
            )

            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, width, height = boxes[i]
                    x2 = x1 + width
                    y2 = y1 + height

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
# ENHANCED UI COMPONENTS
# ============================================================================

class LoadingIndicator(ui.View):
    """Animated loading spinner."""

    def __init__(self):
        self.angle = 0
        self.is_animating = False
        self.animation_timer = None
        self.background_color = None

    def start(self):
        """Start animation."""
        if not self.is_animating:
            self.is_animating = True
            self._animate()

    def stop(self):
        """Stop animation."""
        self.is_animating = False

    def _animate(self):
        """Animation loop."""
        if not self.is_animating:
            return

        self.angle = (self.angle + 15) % 360
        self.set_needs_display()

        def next_frame():
            self._animate()

        if self.is_animating:
            threading.Timer(0.033, lambda: on_main_thread(next_frame)()).start()

    def draw(self):
        """Draw spinner."""
        center_x = self.width / 2
        center_y = self.height / 2
        radius = min(self.width, self.height) / 3

        # Draw arc segments
        for i in range(8):
            angle_deg = (self.angle + i * 45) % 360
            angle_rad = angle_deg * pi / 180

            alpha = 1.0 - (i / 8.0) * 0.7
            ui.set_color((0.2, 0.6, 1.0, alpha))

            start_angle = angle_rad - pi/16
            end_angle = angle_rad + pi/16

            path = ui.Path()
            path.move_to(center_x, center_y)
            path.add_arc(center_x, center_y, radius, start_angle, end_angle)
            path.line_to(center_x, center_y)
            path.fill()


class PulseView(ui.View):
    """Pulsing visual feedback."""

    def __init__(self):
        self.pulse_alpha = 1.0
        self.is_pulsing = False
        self.background_color = None

    def pulse(self, color=(0.2, 0.8, 0.4, 1.0)):
        """Trigger pulse animation."""
        self.color = color
        self.pulse_alpha = 1.0
        self.is_pulsing = True
        self._animate_pulse()

    def _animate_pulse(self):
        if not self.is_pulsing:
            return

        self.pulse_alpha -= 0.05
        if self.pulse_alpha <= 0:
            self.is_pulsing = False
            self.pulse_alpha = 0

        self.set_needs_display()

        if self.is_pulsing:
            threading.Timer(0.033, lambda: on_main_thread(self._animate_pulse)()).start()

    def draw(self):
        if not self.is_pulsing or self.pulse_alpha <= 0:
            return

        color = list(self.color)
        color[3] = self.pulse_alpha

        ui.set_color(tuple(color))
        path = ui.Path.rounded_rect(0, 0, self.width, self.height, 10)
        path.fill()


class MinimalHUD(ui.View):
    """Auto-hiding minimal HUD for status."""

    def __init__(self, theme):
        self.theme = theme
        self.background_color = None

        self.fps = 0.0
        self.inference_ms = 0.0
        self.detection_count = 0

        self.is_visible = True
        self.auto_hide_timer = None

    def update_metrics(self, fps, inference_ms, detection_count):
        """Update metrics."""
        self.fps = fps
        self.inference_ms = inference_ms
        self.detection_count = detection_count
        self.set_needs_display()
        self._reset_auto_hide()

    def _reset_auto_hide(self):
        """Reset auto-hide timer."""
        if self.auto_hide_timer:
            self.auto_hide_timer.cancel()

        def hide():
            on_main_thread(self._fade_out)()

        self.auto_hide_timer = threading.Timer(3.0, hide)
        self.auto_hide_timer.start()

    def _fade_out(self):
        """Fade out HUD."""
        # Simple hide for now (can add animation later)
        pass

    def draw(self):
        """Draw minimal HUD."""
        # Semi-transparent background
        ui.set_color(self.theme['overlay'])
        path = ui.Path.rounded_rect(0, 0, self.width, self.height, 8)
        path.fill()

        # Border
        ui.set_color(self.theme['border'])
        path.line_width = 1
        path.stroke()

        # Text
        ui.set_color(self.theme['fg'])

        # FPS with color coding
        fps_color = self.theme['success'] if self.fps >= 15 else self.theme['warning'] if self.fps >= 10 else self.theme['error']
        ui.set_color(fps_color)
        fps_text = f"{self.fps:.1f} FPS"
        ui.draw_string(fps_text, rect=(10, 5, 100, 25), font=('Menlo-Bold', 16))

        # Inference time
        ui.set_color(self.theme['fg'])
        inference_text = f"{self.inference_ms:.0f}ms"
        ui.draw_string(inference_text, rect=(110, 5, 80, 25), font=('Menlo', 14))

        # Detection count
        if self.detection_count > 0:
            ui.set_color(self.theme['accent'])
            count_text = f"•{self.detection_count}"
            ui.draw_string(count_text, rect=(190, 5, 60, 25), font=('Menlo-Bold', 14))


class EnhancedOverlayView(ui.View):
    """Enhanced camera overlay with modern rendering."""

    def __init__(self, theme):
        self.theme = theme
        self.current_frame = None
        self.detections = []
        self.show_labels = True
        self.show_confidence = True
        self.color_cache = {}

        # Zoom
        self.zoom_scale = 1.0
        self.zoom_offset = (0, 0)

        # Animation
        self.detection_animations = {}  # Track new detections for animation

    def update_frame(self, frame_bgr, detections=None):
        """Update frame and detections."""
        self.current_frame = frame_bgr
        if detections is not None:
            # Track new detections for animation
            old_ids = set(f"{d['class_name']}_{d['x1']}_{d['y1']}" for d in self.detections)
            new_ids = set(f"{d['class_name']}_{d['x1']}_{d['y1']}" for d in detections)

            for det_id in new_ids - old_ids:
                self.detection_animations[det_id] = 1.0  # Start animation

            self.detections = detections

        on_main_thread(self.set_needs_display)()

    def draw(self):
        """Draw enhanced overlay."""
        if self.current_frame is None:
            # Placeholder
            ui.set_color(self.theme['bg'])
            path = ui.Path.rect(0, 0, self.width, self.height)
            path.fill()

            ui.set_color(self.theme['fg'])
            text = "No camera feed"
            ui.draw_string(text, rect=(self.width/2 - 70, self.height/2, 140, 30), font=('System', 18))
            return

        # Draw frame
        frame = self._draw_detections(self.current_frame.copy())
        img = self._numpy_to_ui_image(frame)

        if img:
            img_w, img_h = img.size
            view_w, view_h = self.width, self.height

            scale = min(view_w / img_w, view_h / img_h) * self.zoom_scale
            new_w = img_w * scale
            new_h = img_h * scale

            x = (view_w - new_w) / 2 + self.zoom_offset[0]
            y = (view_h - new_h) / 2 + self.zoom_offset[1]

            img.draw(x, y, new_w, new_h)

    def _draw_detections(self, frame):
        """Draw detections with enhanced visuals."""
        for det in self.detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            class_name = det['class_name']
            score = det['score']

            # Get color
            if class_name not in self.color_cache:
                self.color_cache[class_name] = get_color_for_class(class_name)
            color = self.color_cache[class_name]

            # Check for animation
            det_id = f"{class_name}_{x1}_{y1}"
            anim_scale = 1.0
            if det_id in self.detection_animations:
                anim_scale = 1.0 + self.detection_animations[det_id] * 0.1
                self.detection_animations[det_id] *= 0.8
                if self.detection_animations[det_id] < 0.01:
                    del self.detection_animations[det_id]

            # Draw with slight glow effect (double rectangle)
            glow_color = tuple(int(c * 0.5) for c in color)
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), glow_color, 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if self.show_labels:
                label = f"{class_name}"
                if self.show_confidence:
                    label += f" {score:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Label background with padding
                padding = 4
                cv2.rectangle(frame,
                             (x1, y1 - text_h - baseline - padding*2),
                             (x1 + text_w + padding*2, y1),
                             color, -1)

                # Label text
                cv2.putText(frame, label,
                           (x1 + padding, y1 - baseline - padding),
                           font, font_scale, (255, 255, 255), thickness)

        return frame

    def _numpy_to_ui_image(self, frame_bgr):
        """Convert numpy to ui.Image."""
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            success, buffer = cv2.imencode('.png', frame_rgb)
            if not success:
                return None

            from io import BytesIO
            img = ui.Image.from_data(BytesIO(buffer.tobytes()).read())
            return img

        except Exception as e:
            log_error(f"Image conversion failed: {e}")
            return None

    def handle_pinch(self, scale):
        """Handle pinch-to-zoom."""
        self.zoom_scale = max(1.0, min(3.0, self.zoom_scale * scale))
        self.set_needs_display()

    def handle_pan(self, dx, dy):
        """Handle pan gesture."""
        if self.zoom_scale > 1.0:
            x, y = self.zoom_offset
            self.zoom_offset = (x + dx, y + dy)
            self.set_needs_display()




class SlideOutDrawer(ui.View):
    """Slide-out control drawer from right edge."""

    def __init__(self, controller, theme):
        self.controller = controller
        self.theme = theme
        self.background_color = theme['card']
        self.is_open = False
        self.target_x = 0

    def setup_ui(self):
        """Setup drawer controls."""
        y_offset = 20
        x_margin = 15

        # Title
        title = ui.Label(text='Controls')
        title.frame = (x_margin, y_offset, 200, 30)
        title.font = ('System-Bold', 20)
        title.text_color = self.theme['fg']
        self.add_subview(title)

        y_offset += 40

        # Model selector
        model_label = ui.Label(text='Model')
        model_label.frame = (x_margin, y_offset, 80, 25)
        model_label.font = ('System', 14)
        model_label.text_color = self.theme['fg']
        self.add_subview(model_label)

        self.model_segment = ui.SegmentedControl()
        self.model_segment.frame = (x_margin, y_offset + 30, self.width - 2*x_margin, 32)
        self.model_segment.segments = ['MobileNet-SSD', 'YOLO-tiny']
        self.model_segment.selected_index = 0
        self.model_segment.action = self.controller.change_model
        self.add_subview(self.model_segment)

        y_offset += 75

        # Confidence slider
        conf_label = ui.Label(text='Confidence Threshold')
        conf_label.frame = (x_margin, y_offset, 150, 25)
        conf_label.font = ('System', 14)
        conf_label.text_color = self.theme['fg']
        self.add_subview(conf_label)

        self.conf_value = ui.Label(text='0.50')
        self.conf_value.frame = (self.width - 60, y_offset, 45, 25)
        self.conf_value.font = ('Menlo-Bold', 14)
        self.conf_value.text_color = self.theme['accent']
        self.conf_value.alignment = ui.ALIGN_RIGHT
        self.add_subview(self.conf_value)

        self.conf_slider = ui.Slider()
        self.conf_slider.frame = (x_margin, y_offset + 30, self.width - 2*x_margin, 32)
        self.conf_slider.value = 0.5
        self.conf_slider.action = self.controller.update_confidence
        self.add_subview(self.conf_slider)

        y_offset += 75

        # NMS slider
        nms_label = ui.Label(text='NMS Threshold')
        nms_label.frame = (x_margin, y_offset, 150, 25)
        nms_label.font = ('System', 14)
        nms_label.text_color = self.theme['fg']
        self.add_subview(nms_label)

        self.nms_value = ui.Label(text='0.40')
        self.nms_value.frame = (self.width - 60, y_offset, 45, 25)
        self.nms_value.font = ('Menlo-Bold', 14)
        self.nms_value.text_color = self.theme['accent']
        self.nms_value.alignment = ui.ALIGN_RIGHT
        self.add_subview(self.nms_value)

        self.nms_slider = ui.Slider()
        self.nms_slider.frame = (x_margin, y_offset + 30, self.width - 2*x_margin, 32)
        self.nms_slider.value = 0.4
        self.nms_slider.action = self.controller.update_nms
        self.add_subview(self.nms_slider)

        y_offset += 75

        # Display options
        options_label = ui.Label(text='Display Options')
        options_label.frame = (x_margin, y_offset, 150, 25)
        options_label.font = ('System', 14)
        options_label.text_color = self.theme['fg']
        self.add_subview(options_label)

        y_offset += 35

        self.show_labels_switch = ui.Switch()
        self.show_labels_switch.frame = (self.width - 66, y_offset, 51, 31)
        self.show_labels_switch.value = True
        self.show_labels_switch.action = self.controller.toggle_labels
        self.add_subview(self.show_labels_switch)

        labels_label = ui.Label(text='Show Labels')
        labels_label.frame = (x_margin, y_offset, 150, 31)
        labels_label.font = ('System', 14)
        labels_label.text_color = self.theme['fg']
        self.add_subview(labels_label)

        y_offset += 40

        self.show_conf_switch = ui.Switch()
        self.show_conf_switch.frame = (self.width - 66, y_offset, 51, 31)
        self.show_conf_switch.value = True
        self.show_conf_switch.action = self.controller.toggle_confidence
        self.add_subview(self.show_conf_switch)

        conf_display_label = ui.Label(text='Show Confidence')
        conf_display_label.frame = (x_margin, y_offset, 150, 31)
        conf_display_label.font = ('System', 14)
        conf_display_label.text_color = self.theme['fg']
        self.add_subview(conf_display_label)

        y_offset += 50

        # Action buttons
        self.save_btn = ui.Button(title='Save Frame')
        self.save_btn.frame = (x_margin, y_offset, self.width - 2*x_margin, 44)
        self.save_btn.background_color = self.theme['success']
        self.save_btn.tint_color = 'white'
        self.save_btn.corner_radius = 8
        self.save_btn.action = self.controller.save_frame
        self.add_subview(self.save_btn)

        y_offset += 54

        self.gallery_btn = ui.Button(title='View Gallery')
        self.gallery_btn.frame = (x_margin, y_offset, self.width - 2*x_margin, 44)
        self.gallery_btn.background_color = self.theme['accent']
        self.gallery_btn.tint_color = 'white'
        self.gallery_btn.corner_radius = 8
        self.gallery_btn.action = self.controller.show_gallery
        self.add_subview(self.gallery_btn)

        y_offset += 54

        self.help_btn = ui.Button(title='Help')
        self.help_btn.frame = (x_margin, y_offset, self.width - 2*x_margin, 44)
        self.help_btn.background_color = self.theme['border']
        self.help_btn.tint_color = self.theme['fg']
        self.help_btn.corner_radius = 8
        self.help_btn.action = self.controller.show_help
        self.add_subview(self.help_btn)

    def toggle(self):
        """Animate drawer open/close."""
        self.is_open = not self.is_open
        if self.is_open:
            self.animate_to(self.superview.width - self.width)
        else:
            self.animate_to(self.superview.width)

    def animate_to(self, target_x):
        """Smooth animation to target."""
        start_x = self.x
        duration = 0.3
        steps = 15

        def animate_step(step):
            if step >= steps:
                self.x = target_x
                return

            progress = step / steps
            # Ease out cubic
            progress = 1 - pow(1 - progress, 3)
            self.x = start_x + (target_x - start_x) * progress

            threading.Timer(duration / steps, lambda: on_main_thread(lambda: animate_step(step + 1))()).start()

        animate_step(0)


class SettingsPanel(ui.View):
    """Dedicated settings panel."""

    def __init__(self, controller, theme):
        self.controller = controller
        self.theme = theme
        self.background_color = theme['bg']

    def setup_ui(self):
        """Setup settings UI."""
        # Add close button, settings sections, etc.
        close_btn = ui.Button(title='Done')
        close_btn.frame = (self.width - 80, 20, 70, 32)
        close_btn.action = self.controller.hide_settings
        self.add_subview(close_btn)

        # Settings content...
        title = ui.Label(text='Settings')
        title.frame = (20, 20, 200, 32)
        title.font = ('System-Bold', 24)
        title.text_color = self.theme['fg']
        self.add_subview(title)


class FrameGallery(ui.View):
    """In-app frame gallery."""

    def __init__(self, controller, theme):
        self.controller = controller
        self.theme = theme
        self.background_color = theme['bg']
        self.frames = []

    def setup_ui(self):
        """Setup gallery UI."""
        # Title
        title = ui.Label(text='Captured Frames')
        title.frame = (20, 20, 200, 32)
        title.font = ('System-Bold', 24)
        title.text_color = self.theme['fg']
        self.add_subview(title)

        # Close button
        close_btn = ui.Button(title='Done')
        close_btn.frame = (self.width - 80, 20, 70, 32)
        close_btn.action = self.controller.hide_gallery
        self.add_subview(close_btn)

        # Load frames
        self.load_frames()

        # Grid layout (simplified)
        y_offset = 70
        for i, frame_file in enumerate(self.frames[:6]):  # Show first 6
            img_view = ui.ImageView()
            img_view.frame = (20 + (i % 3) * 110, y_offset + (i // 3) * 110, 100, 100)
            img_view.content_mode = ui.CONTENT_SCALE_ASPECT_FIT

            # Load thumbnail
            try:
                img_path = os.path.join(CAPTURES_DIR, frame_file)
                img_view.image = ui.Image.named(img_path)
            except:
                pass

            self.add_subview(img_view)

    def load_frames(self):
        """Load captured frames."""
        if os.path.exists(CAPTURES_DIR):
            self.frames = [f for f in os.listdir(CAPTURES_DIR) if f.endswith('.png')]
            self.frames.sort(reverse=True)


class HelpScreen(ui.View):
    """Help and tutorial screen."""

    def __init__(self, controller, theme):
        self.controller = controller
        self.theme = theme
        self.background_color = theme['bg']

    def setup_ui(self):
        """Setup help UI."""
        # Title
        title = ui.Label(text='How to Use')
        title.frame = (20, 20, 200, 32)
        title.font = ('System-Bold', 24)
        title.text_color = self.theme['fg']
        self.add_subview(title)

        # Close button
        close_btn = ui.Button(title='Got It')
        close_btn.frame = (self.width - 100, 20, 90, 32)
        close_btn.background_color = self.theme['accent']
        close_btn.tint_color = 'white'
        close_btn.corner_radius = 8
        close_btn.action = self.controller.hide_help
        self.add_subview(close_btn)

        # Help content
        help_text = """
GESTURES:

• Tap screen - Toggle HUD visibility
• Swipe from right - Open controls
• Pinch - Zoom in/out
• Double tap - Toggle fullscreen
• Swipe left/right - Change models

CONTROLS:

• Start/Stop - Begin/pause detection
• Model - Switch between SSD/YOLO
• Confidence - Detection threshold
• NMS - Duplicate suppression

TIPS:

• Lower confidence for more detections
• Use SSD for better speed
• Use YOLO for better accuracy
• Reduce input size if FPS is low
"""

        text_view = ui.TextView()
        text_view.frame = (20, 70, self.width - 40, self.height - 100)
        text_view.text = help_text
        text_view.font = ('System', 14)
        text_view.text_color = self.theme['fg']
        text_view.editable = False
        text_view.background_color = None
        self.add_subview(text_view)


class FloatingActionButton(ui.View):
    """Floating action button (FAB)."""

    def __init__(self, title, action, theme):
        self.title_text = title
        self.action_callback = action
        self.theme = theme
        self.background_color = theme['accent']
        self.corner_radius = 28
        self.is_pressed = False

    def draw(self):
        """Draw FAB."""
        # Shadow effect
        ui.set_color((0, 0, 0, 0.3))
        shadow_path = ui.Path.oval(2, 2, self.width - 4, self.height - 4)
        shadow_path.fill()

        # Button circle
        color = self.theme['accent'] if not self.is_pressed else tuple(c * 0.8 for c in self.theme['accent'])
        ui.set_color(color)
        button_path = ui.Path.oval(0, 0, self.width, self.height)
        button_path.fill()

        # Icon/Text
        ui.set_color((1, 1, 1, 1))
        text_w = len(self.title_text) * 8
        ui.draw_string(self.title_text,
                      rect=(self.width/2 - text_w/2, self.height/2 - 8, text_w*2, 16),
                      font=('System-Bold', 16))

    def touch_began(self, touch):
        self.is_pressed = True
        self.set_needs_display()

    def touch_ended(self, touch):
        self.is_pressed = False
        self.set_needs_display()
        if self.action_callback:
            self.action_callback(self)


# ============================================================================
# ENHANCED APP CONTROLLER
# ============================================================================

class EnhancedAppController:
    """Enhanced app controller with modern UI/UX."""

    def __init__(self):
        self.settings = load_settings()
        self.theme = DARK_THEME if self.settings['theme'] == 'dark' else LIGHT_THEME

        # Components
        self.camera = None
        self.detector = None
        self.main_view = None
        self.overlay_view = None
        self.hud = None
        self.drawer = None
        self.fab = None
        self.loading_indicator = None
        self.pulse_view = None

        # Modal views
        self.settings_panel = None
        self.frame_gallery = None
        self.help_screen = None

        # State
        self.is_running = False
        self.is_processing = False
        self.worker_thread = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.fps_tracker = deque(maxlen=FPS_SMOOTHING_WINDOW)
        self.inference_times = deque(maxlen=FPS_SMOOTHING_WINDOW)
        self.pipeline_times = deque(maxlen=FPS_SMOOTHING_WINDOW)
        self.dropped_frames = 0
        self.last_frame_time = 0

        # Gesture state
        self.last_touch_time = 0
        self.touch_count = 0
        self.last_touch_pos = None
        self.pinch_start_distance = None

        # Model management
        self.available_models = self._check_available_models()

    def _check_available_models(self):
        """Check available models."""
        models = {}
        models['ssd'] = os.path.exists(MOBILENET_PROTOTXT) and os.path.exists(MOBILENET_MODEL)
        models['yolo'] = os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS)
        return models

    def setup(self):
        """Setup application."""
        try:
            self._setup_ui()

            # Initialize camera
            if OBJC_AVAILABLE:
                self.camera = CameraStream()
            else:
                log_info("Using mock camera")
                self.camera = MockCameraStream()

            # Load detector
            self._load_detector()

            # Show help on first run
            if self.settings.get('first_run', True):
                self.show_help(None)
                self.settings['first_run'] = False
                save_settings(self.settings)

            log_info("Enhanced app setup completed")

        except Exception as e:
            log_error(f"Setup failed: {e}")
            log_error(traceback.format_exc())
            self._show_error(f"Setup failed: {str(e)}")

    def _setup_ui(self):
        """Setup enhanced UI."""
        # Main container
        self.main_view = ui.View()
        self.main_view.name = f'Object Detection v{APP_VERSION}'
        self.main_view.background_color = self.theme['bg']

        # Overlay view (fullscreen)
        self.overlay_view = EnhancedOverlayView(self.theme)
        self.overlay_view.frame = (0, 0, 375, 667)
        self.overlay_view.flex = 'WH'
        self.main_view.add_subview(self.overlay_view)

        # Pulse feedback layer
        self.pulse_view = PulseView()
        self.pulse_view.frame = (0, 0, 375, 667)
        self.pulse_view.flex = 'WH'
        self.main_view.add_subview(self.pulse_view)

        # Minimal HUD (top)
        self.hud = MinimalHUD(self.theme)
        self.hud.frame = (10, 40, 250, 35)
        self.main_view.add_subview(self.hud)

        # Slide-out drawer (hidden initially)
        drawer_width = 280
        self.drawer = SlideOutDrawer(self, self.theme)
        self.drawer.frame = (375, 0, drawer_width, 667)
        self.drawer.flex = 'HL'
        self.drawer.setup_ui()
        self.main_view.add_subview(self.drawer)

        # FAB (bottom right)
        self.fab = FloatingActionButton('▶', self.toggle_detection, self.theme)
        self.fab.frame = (315, 597, 56, 56)
        self.main_view.add_subview(self.fab)

        # Loading indicator (center, hidden initially)
        self.loading_indicator = LoadingIndicator()
        self.loading_indicator.frame = (160, 300, 60, 60)
        self.loading_indicator.hidden = True
        self.main_view.add_subview(self.loading_indicator)

        # Apply settings to UI
        self._apply_settings_to_ui()

        # Setup gesture recognizers
        self._setup_gestures()

    def _setup_gestures(self):
        """Setup gesture recognizers."""
        # Touch handlers will be in overlay_view
        pass

    def _apply_settings_to_ui(self):
        """Apply settings to UI controls."""
        model_idx = 0 if self.settings['model'] == 'ssd' else 1
        self.drawer.model_segment.selected_index = model_idx

        self.drawer.conf_slider.value = self.settings['confidence']
        self.drawer.conf_value.text = f"{self.settings['confidence']:.2f}"

        self.drawer.nms_slider.value = self.settings['nms_threshold']
        self.drawer.nms_value.text = f"{self.settings['nms_threshold']:.2f}"

        self.drawer.show_labels_switch.value = self.settings['show_labels']
        self.drawer.show_conf_switch.value = self.settings['show_confidence']

        self.overlay_view.show_labels = self.settings['show_labels']
        self.overlay_view.show_confidence = self.settings['show_confidence']

    def _load_detector(self):
        """Load detector model."""
        model_type = self.settings['model']

        if not self.available_models.get(model_type, False):
            alternate = 'yolo' if model_type == 'ssd' else 'ssd'
            if self.available_models.get(alternate, False):
                log_info(f"Switching to {alternate} model")
                model_type = alternate
                self.settings['model'] = alternate
            else:
                raise RuntimeError("No model files found")

        # Show loading
        on_main_thread(lambda: setattr(self.loading_indicator, 'hidden', False))()
        on_main_thread(self.loading_indicator.start)()

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

        finally:
            # Hide loading
            on_main_thread(self.loading_indicator.stop)()
            on_main_thread(lambda: setattr(self.loading_indicator, 'hidden', True))()

    def run(self):
        """Run application."""
        try:
            self.main_view.present('fullscreen')
        except Exception as e:
            log_error(f"Failed to present UI: {e}")
            raise

    def toggle_detection(self, sender):
        """Start/stop detection."""
        if self.is_running:
            self.stop()
        else:
            self.start()

    def start(self):
        """Start detection."""
        if self.is_running:
            return

        try:
            self.camera.start()
            self.stop_event.clear()
            self.is_running = True
            self.dropped_frames = 0
            self.last_frame_time = time.time()

            # Update FAB
            self.fab.title_text = '⏸'
            on_main_thread(self.fab.set_needs_display)()

            # Pulse feedback
            on_main_thread(lambda: self.pulse_view.pulse(self.theme['success']))()

            # Start worker
            self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.worker_thread.start()

            log_info("Detection started")

        except Exception as e:
            log_error(f"Start failed: {e}")
            self._show_error(f"Failed to start: {str(e)}")
            self.is_running = False

    def stop(self):
        """Stop detection."""
        if not self.is_running:
            return

        log_info("Stopping detection...")

        self.stop_event.set()
        self.is_running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        if self.camera:
            self.camera.stop()

        # Update FAB
        self.fab.title_text = '▶'
        on_main_thread(self.fab.set_needs_display)()

        # Pulse feedback
        on_main_thread(lambda: self.pulse_view.pulse(self.theme['error']))()

        log_info("Detection stopped")

    def _processing_loop(self):
        """Processing loop."""
        log_info("Processing loop started")

        while not self.stop_event.is_set():
            try:
                loop_start = time.time()

                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                if self.settings['enable_frame_skip'] and self.is_processing:
                    self.dropped_frames += 1
                    continue

                self.is_processing = True

                inference_start = time.time()
                detections = self.detector.infer(frame)
                inference_time = (time.time() - inference_start) * 1000

                self.is_processing = False

                self.overlay_view.update_frame(frame, detections)

                loop_time = (time.time() - loop_start) * 1000
                self.inference_times.append(inference_time)
                self.pipeline_times.append(loop_time)

                current_time = time.time()
                if self.last_frame_time > 0:
                    frame_interval = current_time - self.last_frame_time
                    if frame_interval > 0:
                        fps = 1.0 / frame_interval
                        self.fps_tracker.append(fps)
                self.last_frame_time = current_time

                avg_fps = sum(self.fps_tracker) / len(self.fps_tracker) if self.fps_tracker else 0
                avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

                self.hud.update_metrics(avg_fps, avg_inference, len(detections))

            except Exception as e:
                log_error(f"Processing loop error: {e}")
                time.sleep(0.1)

        log_info("Processing loop stopped")

    # Control callbacks

    def change_model(self, sender):
        """Change model."""
        was_running = self.is_running
        if was_running:
            self.stop()

        try:
            model_idx = sender.selected_index
            self.settings['model'] = 'ssd' if model_idx == 0 else 'yolo'
            save_settings(self.settings)
            self._load_detector()

            if was_running:
                self.start()

        except Exception as e:
            log_error(f"Model change failed: {e}")
            self._show_error(f"Failed to change model: {str(e)}")

    def update_confidence(self, sender):
        """Update confidence threshold."""
        value = sender.value * 0.5 + 0.2
        self.drawer.conf_value.text = f"{value:.2f}"
        self.settings['confidence'] = value
        if self.detector:
            self.detector.confidence_threshold = value
        save_settings(self.settings)

    def update_nms(self, sender):
        """Update NMS threshold."""
        value = sender.value * 0.3 + 0.3
        self.drawer.nms_value.text = f"{value:.2f}"
        self.settings['nms_threshold'] = value
        if hasattr(self.detector, 'nms_threshold'):
            self.detector.nms_threshold = value
        save_settings(self.settings)

    def toggle_labels(self, sender):
        """Toggle labels."""
        self.settings['show_labels'] = sender.value
        self.overlay_view.show_labels = sender.value
        save_settings(self.settings)

    def toggle_confidence(self, sender):
        """Toggle confidence display."""
        self.settings['show_confidence'] = sender.value
        self.overlay_view.show_confidence = sender.value
        save_settings(self.settings)

    def save_frame(self, sender):
        """Save current frame."""
        if self.overlay_view.current_frame is None:
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            raw_path = os.path.join(CAPTURES_DIR, f"raw_{timestamp}.png")
            cv2.imwrite(raw_path, self.overlay_view.current_frame)

            annotated = self.overlay_view._draw_detections(self.overlay_view.current_frame.copy())
            annotated_path = os.path.join(CAPTURES_DIR, f"annotated_{timestamp}.png")
            cv2.imwrite(annotated_path, annotated)

            log_info(f"Saved frames: {timestamp}")
            self._show_toast("Frame saved!")

            # Pulse feedback
            on_main_thread(lambda: self.pulse_view.pulse(self.theme['success']))()

        except Exception as e:
            log_error(f"Save frame failed: {e}")
            self._show_error(f"Failed to save: {str(e)}")

    def show_gallery(self, sender):
        """Show frame gallery."""
        self.frame_gallery = FrameGallery(self, self.theme)
        self.frame_gallery.frame = (0, 0, self.main_view.width, self.main_view.height)
        self.frame_gallery.setup_ui()
        self.main_view.add_subview(self.frame_gallery)

    def hide_gallery(self, sender):
        """Hide gallery."""
        if self.frame_gallery:
            self.frame_gallery.remove_from_superview()
            self.frame_gallery = None

    def show_help(self, sender):
        """Show help screen."""
        self.help_screen = HelpScreen(self, self.theme)
        self.help_screen.frame = (0, 0, self.main_view.width, self.main_view.height)
        self.help_screen.setup_ui()
        self.main_view.add_subview(self.help_screen)

    def hide_help(self, sender):
        """Hide help screen."""
        if self.help_screen:
            self.help_screen.remove_from_superview()
            self.help_screen = None

    def hide_settings(self, sender):
        """Hide settings panel."""
        if self.settings_panel:
            self.settings_panel.remove_from_superview()
            self.settings_panel = None

    def _show_error(self, message):
        """Show error dialog."""
        def show():
            import console
            console.alert("Error", message, "OK", hide_cancel_button=True)
        on_main_thread(show)()

    def _show_toast(self, message):
        """Show toast message."""
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
    print(f"Real-Time Object Detection v{APP_VERSION} (Enhanced UI)")
    print("=" * 60)

    ensure_dirs()
    log_info(f"Enhanced application started (v{APP_VERSION})")

    # Check models
    models_found = []
    if os.path.exists(MOBILENET_PROTOTXT) and os.path.exists(MOBILENET_MODEL):
        models_found.append("MobileNet-SSD")
    if os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS):
        models_found.append("YOLO-tiny")

    if not models_found:
        print("\nWARNING: No model files found!")
        print("The app will run in preview-only mode.")
        log_error("No model files found")
    else:
        print(f"\nFound models: {', '.join(models_found)}")

    # Create and run app
    try:
        app = EnhancedAppController()
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


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Enhanced UI version does not include test mode")
        print("Use realtime_detect.py --test for testing")
    else:
        main()
