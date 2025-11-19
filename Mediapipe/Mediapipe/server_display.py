import os

import numpy as np

import pickle

import time

import mediapipe as mp

import cv2

from PIL import Image, ImageDraw, ImageFont

from collections import deque

import tflite_runtime.interpreter as tflite   # ✅ chỉ dùng TFLite runtime



# ==================== LOAD MODEL & SCALERS ====================

print("="*60)

print("🔄 Loading TFLite model và preprocessors...")

print("="*60)



interpreter = tflite.Interpreter(model_path="vsl_landmarks_model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()



with open("label_encoder.pkl", "rb") as f:

    label_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:

    scaler = pickle.load(f)



# ==================== MEDIAPIPE SETUP ====================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)

mp_draw = mp.solutions.drawing_utils

mp_style = mp.solutions.drawing_styles



# ==================== FUNCTION ====================

def predict_tflite(data):

    interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))

    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])

    return preds



def extract_landmarks(hand_landmarks):

    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()



# ==================== VIETNAMESE COMBINATION LOGIC ====================

def combine_vietnamese(text):

    """

    Ghép các tổ hợp ký tự cơ bản như a^ -> â, o^ -> ô, e^ -> ê, u^ -> ư, o' -> ơ, a' -> ă.

    Không thêm dấu thanh.

    """

    mapping = {

        "a^": "â",

        "A^": "Â",

        "o^": "ô",

        "O^": "Ô",

        "e^": "ê",

        "E^": "Ê",

        "u^": "ư",

        "U'": "Ư",

        "o'": "ơ",

        "O'": "Ơ",

        "a'": "ă",

        "A'": "Ă",

    }

    for k, v in mapping.items():

        if k in text:

            text = text.replace(k, v)

    return text



# ==================== FONT & DISPLAY ====================

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

font = ImageFont.truetype(font_path, 30)



text_buffer = ""

current_char = None

stable_count = 0

STABLE_THRESHOLD = 5

last_added_time = 0

ADD_INTERVAL = 3.0



# ==================== CAMERA ====================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps_time = time.time()



print("✅ Realtime gesture recognition started... (press Q to quit)")



# ==================== MAIN LOOP ====================

while True:

    ret, frame = cap.read()

    if not ret:

        print("⚠️ Không lấy được khung hình từ camera.")

        break



    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)



    predicted_label, conf = "No hand", 0.0



    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,

                                   mp_style.get_default_hand_landmarks_style(),

                                   mp_style.get_default_hand_connections_style())

            landmarks = extract_landmarks(hand_landmarks).reshape(1, -1)

            landmarks_scaled = scaler.transform(landmarks)

            preds = predict_tflite(landmarks_scaled)

            conf = float(np.max(preds))

            label = label_encoder.inverse_transform([np.argmax(preds)])[0]

            if label == "dd": label = "đ"

            predicted_label = label



    # ===== Logic cộng dồn ký tự =====

    if predicted_label == current_char:

        stable_count += 1

    else:

        current_char = predicted_label

        stable_count = 0



    if stable_count >= STABLE_THRESHOLD and predicted_label not in ["No hand"]:

        current_time = time.time()

        if current_time - last_added_time >= ADD_INTERVAL:

            text_buffer += predicted_label

            text_buffer = combine_vietnamese(text_buffer)  # ✅ Tự ghép ký tự đặc biệt

            last_added_time = current_time

            stable_count = 0

            print("🆕 Added:", predicted_label, "→", text_buffer)



    # FPS

    fps = 1 / (time.time() - fps_time)

    fps_time = time.time()



    # ==== Hiển thị ====

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(frame_pil)

    draw.text((20, 20), f"Ký hiệu: {predicted_label}", font=font, fill=(0,255,0))

    draw.text((20, 60), f"Câu: {text_buffer}", font=font, fill=(255,255,255))

    draw.text((20, 100), f"FPS: {fps:.1f}", font=font, fill=(255,255,0))

    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)



    cv2.imshow("Sign Sentence Builder (TFLite)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('q'), 27]:

        break

    elif key == ord(' '):

        text_buffer += " "

    elif key in [ord('x'), 8]:  # Backspace

        text_buffer = text_buffer[:-1]

    elif key == ord('c'):

        text_buffer = ""



cap.release()

cv2.destroyAllWindows()

print("✅ Done.")

