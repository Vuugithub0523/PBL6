"""
REAL-TIME VSL RECOGNITION - ĐỘ CHÍNH XÁC CAO NHẤT
================================================
Các tính năng:
- Motion detection: Phát hiện động tác bắt đầu/kết thúc
- Temporal smoothing: Voting qua nhiều predictions
- Confidence filtering: Chỉ accept > 85%
- Buffer management: Tự động clear khi cần
- Visual feedback: Hiển thị rõ ràng
"""

import torch
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import sys
import codecs
import json
from pathlib import Path
from collections import deque, Counter
import time

# Fix encoding
if sys.platform == 'win32':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

sys.path.insert(0, 'src')
from model import create_model

# Load label map
with open('label_map.json', 'r', encoding='utf-8') as f:
    LABEL_MAP = json.load(f)
    LABEL_MAP = {int(k): v for k, v in LABEL_MAP.items()}


class RealtimeRecognizer:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"\n⏳ Đang load model...")
        self.model = create_model(
            model_type='skeleton_lstm',
            num_classes=27,
            hidden_dim=128,
            num_layers=2,
            dropout=0.5
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded ({self.device})")
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.joint_indices = [0, 0, 11, 12, 13, 14, 15, 16]
        
        # ==================================================
        # PARAMETERS - ĐIỀU CHỈNH ĐỂ TĂNG ĐỘ CHÍNH XÁC
        # ==================================================
        
        # Buffer settings
        self.min_buffer_size = 30           # Tối thiểu 30 frames (1s @ 30fps)
        self.max_buffer_size = 90           # Tối đa 90 frames (3s)
        self.predict_every = 10             # Predict mỗi 10 frames
        
        # Motion detection
        self.motion_threshold = 0.015       # Ngưỡng phát hiện chuyển động
        self.still_frames_threshold = 15   # Số frames đứng yên để kết thúc
        
        # Confidence & Smoothing
        self.confidence_threshold = 0.85    # Chỉ accept nếu > 85%
        self.smoothing_window = 5           # Voting window size
        
        # State management
        self.skeleton_buffer = deque(maxlen=self.max_buffer_size)
        self.prediction_history = deque(maxlen=self.smoothing_window)
        self.prev_skeleton = None
        self.still_frames = 0
        self.is_performing_gesture = False
        self.last_prediction = None
        self.last_confidence = 0.0
        self.frame_count = 0
        
        # Stats
        self.total_predictions = 0
        self.correct_predictions = 0
        
        print(f"\n📊 SETTINGS:")
        print(f"   Buffer: {self.min_buffer_size}-{self.max_buffer_size} frames")
        print(f"   Motion threshold: {self.motion_threshold}")
        print(f"   Confidence threshold: {self.confidence_threshold*100:.0f}%")
        print(f"   Smoothing window: {self.smoothing_window}")
    
    def extract_skeleton(self, landmarks):
        """Extract 8 joints từ MediaPipe landmarks"""
        skeleton = []
        for idx in self.joint_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                skeleton.append([lm.x, lm.y, lm.z])
        return np.array(skeleton, dtype=np.float32)
    
    def calculate_motion(self, current_skeleton):
        """Tính motion giữa frame hiện tại và frame trước"""
        if self.prev_skeleton is None:
            self.prev_skeleton = current_skeleton
            return 0.0
        
        # Tính khoảng cách Euclidean trung bình
        motion = np.mean(np.linalg.norm(current_skeleton - self.prev_skeleton, axis=1))
        self.prev_skeleton = current_skeleton
        
        return motion
    
    def normalize_skeleton(self, frames):
        """Normalize skeleton sequence"""
        if len(frames) == 0:
            return frames
        
        frames = np.array(frames, dtype=np.float32)
        
        # Center by neck
        neck = frames[:, 1:2, :]
        frames = frames - neck
        
        # Scale by shoulder width
        left_shoulder = frames[:, 2, :]
        right_shoulder = frames[:, 3, :]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
        shoulder_width = np.maximum(shoulder_width, 1e-6)
        
        frames = frames / shoulder_width[:, None, :]
        
        return frames
    
    def interpolate_to_fixed_length(self, frames, fixed_length=60):
        """Interpolate to 60 frames"""
        T = len(frames)
        
        if T < 15:  # Quá ít frames
            return None
        
        if T == fixed_length:
            return frames
        
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, fixed_length)
        
        interpolated = []
        for joint_idx in range(8):
            for coord_idx in range(3):
                values = frames[:, joint_idx, coord_idx]
                interpolator = interp1d(old_indices, values, kind='linear', fill_value='extrapolate')
                interpolated.append(interpolator(new_indices))
        
        result = np.array(interpolated).reshape(3, 8, fixed_length).transpose(2, 1, 0)
        return result.astype(np.float32)
    
    def predict(self, skeleton_sequence):
        """Predict từ skeleton sequence"""
        # Normalize
        skeleton_sequence = self.normalize_skeleton(skeleton_sequence)
        
        # Interpolate
        skeleton_sequence = self.interpolate_to_fixed_length(skeleton_sequence)
        
        if skeleton_sequence is None:
            return None, 0.0
        
        # To tensor
        skeleton_tensor = torch.from_numpy(skeleton_sequence).float()
        skeleton_tensor = skeleton_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(skeleton_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return predicted.item(), confidence.item()
    
    def get_smoothed_prediction(self):
        """Lấy prediction sau khi smooth bằng voting"""
        if len(self.prediction_history) < 3:
            return None, 0.0
        
        # Voting: Chọn prediction xuất hiện nhiều nhất
        predictions = [p for p, c in self.prediction_history if c >= self.confidence_threshold]
        
        if not predictions:
            return None, 0.0
        
        # Most common prediction
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]
        pred_label = most_common[0]
        
        # Average confidence của prediction này
        confidences = [c for p, c in self.prediction_history if p == pred_label]
        avg_confidence = np.mean(confidences)
        
        return pred_label, avg_confidence
    
    def reset_state(self, reason=""):
        """Reset state khi kết thúc gesture"""
        if self.is_performing_gesture and len(self.skeleton_buffer) > 0:
            print(f"🔄 Reset: {reason}")
        
        self.skeleton_buffer.clear()
        self.still_frames = 0
        self.is_performing_gesture = False
        self.prev_skeleton = None
    
    def draw_info(self, frame):
        """Vẽ thông tin lên frame"""
        h, w = frame.shape[:2]
        
        # Background cho text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Title
        cv2.putText(frame, "VSL RECOGNITION - REAL-TIME", 
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Buffer status
        buffer_size = len(self.skeleton_buffer)
        buffer_text = f"Buffer: {buffer_size}/{self.max_buffer_size}"
        
        if buffer_size < self.min_buffer_size:
            buffer_color = (100, 100, 100)  # Gray
            status_text = "CHO DOI..."
        elif buffer_size >= self.max_buffer_size * 0.9:
            buffer_color = (0, 165, 255)  # Orange
            status_text = "SAP DAY"
        else:
            buffer_color = (0, 255, 0)  # Green
            status_text = "SAN SANG"
        
        cv2.putText(frame, buffer_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2)
        
        # Motion status
        motion_text = "DANG THUC HIEN" if self.is_performing_gesture else "CHO DOI DONG TAC"
        motion_color = (0, 255, 0) if self.is_performing_gesture else (100, 100, 100)
        cv2.putText(frame, motion_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
        
        # Last prediction
        if self.last_prediction is not None:
            word = LABEL_MAP.get(self.last_prediction, f"Unknown_{self.last_prediction}")
            conf = self.last_confidence
            
            if conf >= self.confidence_threshold:
                pred_color = (0, 255, 0)  # Green
                pred_text = f"KET QUA: {word} ({conf*100:.1f}%)"
            else:
                pred_color = (0, 165, 255)  # Orange
                pred_text = f"KHONG CHAC: {word} ({conf*100:.1f}%)"
            
            cv2.putText(frame, pred_text, (20, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, pred_color, 2)
        
        # Instructions
        cv2.putText(frame, "Nhan 'Q' de thoat | 'R' de reset", 
                    (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """Chạy real-time recognition"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Không thể mở camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n{'='*80}")
        print(f"🎥 REAL-TIME RECOGNITION STARTED")
        print(f"{'='*80}")
        print(f"📹 Camera: {camera_id}")
        print(f"🎬 FPS: {fps:.0f}")
        print(f"\n💡 CÁCH SỬ DỤNG:")
        print(f"   1. Đứng trước camera (từ HÔNG trở lên)")
        print(f"   2. Thực hiện động tác (3-5 giây)")
        print(f"   3. Đứng yên → Model sẽ dự đoán")
        print(f"   4. Nhan 'Q' để thoát, 'R' để reset")
        print(f"{'='*80}\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                frame = cv2.flip(frame, 1)  # Mirror
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe detection
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Draw skeleton
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
                    )
                    
                    # Extract skeleton
                    skeleton = self.extract_skeleton(results.pose_landmarks.landmark)
                    
                    # Calculate motion
                    motion = self.calculate_motion(skeleton)
                    
                    # Motion detection
                    if motion > self.motion_threshold:
                        # Có chuyển động → đang thực hiện gesture
                        self.is_performing_gesture = True
                        self.still_frames = 0
                        self.skeleton_buffer.append(skeleton)
                    else:
                        # Không có chuyển động → đang đứng yên
                        self.still_frames += 1
                        
                        if self.is_performing_gesture:
                            self.skeleton_buffer.append(skeleton)
                            
                            # Nếu đứng yên đủ lâu → kết thúc gesture
                            if self.still_frames >= self.still_frames_threshold:
                                # Predict nếu buffer đủ lớn
                                if len(self.skeleton_buffer) >= self.min_buffer_size:
                                    # Predict
                                    pred_label, confidence = self.predict(list(self.skeleton_buffer))
                                    
                                    if pred_label is not None:
                                        # Add to history
                                        self.prediction_history.append((pred_label, confidence))
                                        
                                        # Get smoothed prediction
                                        smooth_pred, smooth_conf = self.get_smoothed_prediction()
                                        
                                        if smooth_pred is not None:
                                            self.last_prediction = smooth_pred
                                            self.last_confidence = smooth_conf
                                            
                                            word = LABEL_MAP.get(smooth_pred, f"Unknown_{smooth_pred}")
                                            
                                            if smooth_conf >= self.confidence_threshold:
                                                print(f"✅ {word:15s} - {smooth_conf*100:.1f}%")
                                                self.total_predictions += 1
                                            else:
                                                print(f"⚠️  {word:15s} - {smooth_conf*100:.1f}% (LOW CONFIDENCE)")
                                
                                # Reset state
                                self.reset_state("Gesture ended")
                else:
                    # Không detect được người → reset
                    if len(self.skeleton_buffer) > 0:
                        self.reset_state("Lost tracking")
                
                # Auto-clear nếu buffer quá đầy
                if len(self.skeleton_buffer) >= self.max_buffer_size:
                    if len(self.skeleton_buffer) >= self.min_buffer_size:
                        # Try to predict before clearing
                        pred_label, confidence = self.predict(list(self.skeleton_buffer))
                        if pred_label is not None and confidence >= self.confidence_threshold:
                            self.last_prediction = pred_label
                            self.last_confidence = confidence
                            word = LABEL_MAP.get(pred_label, f"Unknown_{pred_label}")
                            print(f"⚡ {word:15s} - {confidence*100:.1f}% (AUTO)")
                    
                    self.reset_state("Buffer full")
                
                # Draw info
                frame = self.draw_info(frame)
                
                # Show frame
                cv2.imshow('VSL Real-time Recognition', frame)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset_state("Manual reset")
                    self.last_prediction = None
                    self.last_confidence = 0.0
                    print("🔄 Manual reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            
            print(f"\n{'='*80}")
            print(f"📊 SESSION STATS")
            print(f"{'='*80}")
            print(f"Total frames: {self.frame_count}")
            print(f"Total predictions: {self.total_predictions}")
            print(f"{'='*80}\n")


def main():
    model_path = "checkpoints_from_video/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        print(f"   Train model trước: python train_from_videos.py")
        return
    
    recognizer = RealtimeRecognizer(model_path)
    recognizer.run(camera_id=0)


if __name__ == "__main__":
    main()

