#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sign Language Recognition Client
Gửi dữ liệu lên unified_server.py
"""
import sys
import codecs

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import argparse
import requests
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# Sử dụng TensorFlow Lite Runtime (nhẹ hơn cho Raspberry Pi)
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using TensorFlow Lite Runtime")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("⚠️ Using full TensorFlow (consider installing tflite-runtime)")
    except ImportError:
        print("❌ TensorFlow not available!")
        exit(1)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ===== Configuration =====
STABLE_THRESHOLD = 5
ADD_INTERVAL = 3.0  # giây giữa các ký tự
AUTO_SEND_DELAY = 5.0  # Tự động gửi sau 5s không có tay
FRAME_SEND_INTERVAL = 0.1  # gửi frame mỗi 100ms (10 FPS)

# ===== Load model và scaler =====
def load_model_and_encoder(model_path, label_path, scaler_path):
    """Load TFLite model, label encoder và scaler"""
    print(f"🔄 Loading model from: {model_path}")
    
    # Load TFLite model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None, None, None
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Load label encoder
    if not os.path.exists(label_path):
        print(f"❌ Label encoder not found: {label_path}")
        return None, None, None
    
    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load scaler
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler not found: {scaler_path}")
        return None, None, None
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✅ Model loaded successfully")
    return interpreter, label_encoder, scaler

def predict_tflite(interpreter, data):
    """Predict sử dụng TFLite interpreter"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# ===== Mediapipe Hands =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def extract_landmarks(hand_landmarks):
    """Trích xuất landmarks từ hand"""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# ===== HTTP Client =====
class SignLanguageClient:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.response_text = None
        print(f"🌐 Sign Language Client → {self.base_url}")
    
    def send_frame(self, frame):
        """Gửi frame đến server"""
        try:
            url = f"{self.base_url}/upload_frame"
            frame_data = pickle.dumps(frame)
            response = requests.post(url, data=frame_data, timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def send_text(self, text):
        """Gửi text không dấu đến server để xử lý"""
        try:
            url = f"{self.base_url}/upload_sign"
            response = requests.post(url, json={"text": text}, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                processed = result.get('processed', '')
                self.response_text = processed
                
                print(f"📤 Sent: {text}")
                print(f"📥 Processed: {processed}")
                return True
            else:
                print(f"❌ Server error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to send text: {e}")
            return False

# ===== Main Recognition Loop =====
def run_recognition(args):
    # Load model
    interpreter, label_encoder, scaler = load_model_and_encoder(
        args.model,
        args.label_encoder,
        args.scaler
    )
    
    if interpreter is None:
        print("❌ Failed to load model")
        return
    
    # Initialize client
    client = SignLanguageClient(host=args.server_host, port=args.server_port)
    
    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Font for display
    try:
        if os.name == "nt":
            font_path = "C:/Windows/Fonts/segoeui.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 30)
    except:
        font = None
        print("⚠️ Could not load font, using CV2 text")
    
    # Buffer logic
    text_buffer = ""
    current_char = None
    stable_count = 0
    last_added_time = 0
    last_activity_time = 0
    no_hand_start_time = 0
    no_hand_duration = 0
    last_frame_time = 0
    
    # Video capture
    cap = cv2.VideoCapture(args.camera)
    
    if args.resolution:
        width, height = map(int, args.resolution.split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    fps_time = time.time()
    
    print("\n" + "=" * 60)
    print("🤟 Sign Language Recognition Started")
    print("=" * 60)
    print("[S] Send text | [C] Clear buffer | [Q] Quit")
    print("=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Send frame to server
        current_time = time.time()
        if current_time - last_frame_time >= FRAME_SEND_INTERVAL:
            client.send_frame(frame)
            last_frame_time = current_time
        
        # Recognition
        predicted_label = "No hand"
        conf = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )
                
                # Predict
                landmarks = extract_landmarks(hand_landmarks).reshape(1, -1)
                landmarks_scaled = scaler.transform(landmarks)
                preds = predict_tflite(interpreter, landmarks_scaled)
                conf = float(np.max(preds))
                label = label_encoder.inverse_transform([np.argmax(preds)])[0]
                
                # Handle special characters
                if label == "dd":
                    label = "đ"
                
                predicted_label = label
        
        # Stability logic
        if predicted_label == current_char:
            stable_count += 1
        else:
            current_char = predicted_label
            stable_count = 0
        
        # Add character to buffer
        if stable_count >= STABLE_THRESHOLD and predicted_label not in ["No hand"]:
            current_time = time.time()
            if current_time - last_added_time >= ADD_INTERVAL:
                text_buffer += predicted_label
                last_added_time = current_time
                last_activity_time = current_time
                stable_count = 0
                print(f"🆕 Added: {predicted_label} → {text_buffer}")
        
        # Auto-send logic
        current_time = time.time()
        
        if predicted_label == "No hand":
            if no_hand_start_time == 0:
                no_hand_start_time = current_time
            no_hand_duration = current_time - no_hand_start_time
        else:
            no_hand_start_time = 0
            no_hand_duration = 0
            last_activity_time = current_time
        
        # Auto-send
        if text_buffer.strip() and no_hand_duration >= AUTO_SEND_DELAY:
            print(f"⏰ Auto-send after {AUTO_SEND_DELAY}s")
            client.send_text(text_buffer)
            text_buffer = ""
            no_hand_start_time = 0
            no_hand_duration = 0
            last_activity_time = current_time
        
        # FPS
        fps = 1 / (time.time() - fps_time + 0.001)
        fps_time = time.time()
        
        # Display
        if font:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            draw.text((20, 20), f"Ký hiệu: {predicted_label}", font=font, fill=(0, 255, 0))
            draw.text((20, 60), f"Câu: {text_buffer}", font=font, fill=(255, 255, 255))
            draw.text((20, 100), f"FPS: {fps:.1f}", font=font, fill=(255, 255, 0))
            
            # Countdown
            if text_buffer.strip() and predicted_label == "No hand":
                remaining = AUTO_SEND_DELAY - no_hand_duration
                if remaining > 0:
                    draw.text((20, 140), f"⏱️  Gửi sau {remaining:.1f}s", font=font, fill=(255, 165, 0))
            
            # Server response
            if client.response_text:
                draw.text((20, 180), f"Server: {client.response_text}", font=font, fill=(255, 100, 255))
            
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to CV2 text
            cv2.putText(frame, f"Sign: {predicted_label}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Buffer: {text_buffer}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Sign Language Recognition", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('c'):
            text_buffer = ""
            no_hand_start_time = 0
            no_hand_duration = 0
            print("🗑️  Buffer cleared")
        elif key == ord('s'):
            if text_buffer.strip():
                client.send_text(text_buffer)
                text_buffer = ""
                no_hand_start_time = 0
                no_hand_duration = 0
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done")

# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="Sign Language Recognition Client")
    parser.add_argument("--model", default="vsl_landmarks_model.tflite",
                       help="Path to TFLite model")
    parser.add_argument("--label-encoder", default="label_encoder.pkl",
                       help="Path to label encoder")
    parser.add_argument("--scaler", default="scaler.pkl",
                       help="Path to scaler")
    parser.add_argument("--server-host", default="127.0.0.1",
                       help="Server host")
    parser.add_argument("--server-port", type=int, default=5000,
                       help="Server port")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index")
    parser.add_argument("--resolution", default=None,
                       help="Camera resolution (e.g., 640x480)")
    
    args = parser.parse_args()
    
    run_recognition(args)

if __name__ == "__main__":
    main()
