"""Inference engine for webcam capture and classification"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional, Dict
import time
from datetime import datetime

from .config import Config


class InferenceEngine:
    """Handles webcam capture, preprocessing, and model inference"""
    
    def __init__(self, model: torch.jit.ScriptModule, config: Config):
        """
        Args:
            model: Loaded TorchScript model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.model.use_gpu else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize webcam
        self.camera = None
        self._init_camera()
        
        # Setup preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.model.input_size, config.model.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_camera(self):
        """Initialize webcam"""
        self.camera = cv2.VideoCapture(self.config.inference.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.config.inference.camera_index}")
        
        # Warm up camera
        for _ in range(5):
            self.camera.read()
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from webcam
        
        Returns:
            BGR frame as numpy array, or None if capture failed
        """
        if self.camera is None or not self.camera.isOpened():
            self._init_camera()
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input
        
        Args:
            frame: BGR frame from OpenCV
        
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(frame_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def inference(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on preprocessed tensor
        
        Args:
            tensor: Preprocessed input tensor
        
        Returns:
            Tuple of (logits, probabilities)
        """
        start_time = time.time()
        
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if inference_time > self.config.inference.max_inference_time_ms:
            print(f"Warning: Inference took {inference_time:.2f}ms, exceeding target of {self.config.inference.max_inference_time_ms}ms")
        
        return logits, probabilities
    
    def predict_frame(self, frame: Optional[np.ndarray] = None) -> Dict:
        """Capture and classify a frame
        
        Args:
            frame: Optional pre-captured frame. If None, captures from webcam.
        
        Returns:
            Dictionary containing:
                - timestamp: datetime of capture
                - frame: captured frame (BGR)
                - logits: model output logits
                - probabilities: class probabilities
                - predicted_class: index of predicted class
                - predicted_class_name: name of predicted class
                - confidence: confidence of prediction
        """
        # Capture frame if not provided
        if frame is None:
            frame = self.capture_frame()
            if frame is None:
                raise RuntimeError("Failed to capture frame from webcam")
        
        timestamp = datetime.now()
        
        # Preprocess
        tensor = self.preprocess_frame(frame)
        
        # Inference
        logits, probabilities = self.inference(tensor)
        
        # Get prediction
        predicted_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()
        predicted_class_name = self.config.classes[predicted_idx]
        
        return {
            'timestamp': timestamp,
            'frame': frame,
            'logits': logits.cpu().numpy()[0],
            'probabilities': probabilities.cpu().numpy()[0],
            'predicted_class': predicted_idx,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence
        }
    
    def release(self):
        """Release webcam resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()

