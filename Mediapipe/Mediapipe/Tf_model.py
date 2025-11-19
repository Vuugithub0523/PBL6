import cv2, mediapipe as mp, numpy as np, pickle, time, os
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import threading
import textwrap
import requests

# Sử dụng TensorFlow Lite Runtime (nhẹ hơn cho Raspberry Pi)
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using TensorFlow Lite Runtime")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("⚠️ Using full TensorFlow (consider installing tflite-runtime for better performance)")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model
interpreter = tflite.Interpreter(model_path="vsl_landmarks_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def predict_tflite(data):
    interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Font path
font_path = "C:/Windows/Fonts/segoeui.ttf" if os.name == "nt" else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 30)

# ===== LCD Configuration (Raspberry Pi) =====
FB1 = "/dev/fb1"
LCD_WIDTH, LCD_HEIGHT = 480, 320
LCD_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LCD_FONT_SIZE = 22
MAX_HISTORY = 3
USE_LCD = os.path.exists(FB1) if os.name != "nt" else False

# Lịch sử hiển thị
display_history = deque(maxlen=MAX_HISTORY)
history_lock = threading.Lock()

# ===== Hàm ghi framebuffer LCD =====
def write_to_lcd(img):
    """Ghi hình ảnh lên LCD framebuffer"""
    if not USE_LCD:
        return
    
    try:
        if img.size != (LCD_WIDTH, LCD_HEIGHT):
            img = img.resize((LCD_WIDTH, LCD_HEIGHT))
        arr = np.array(img)
        r = (arr[:, :, 0] >> 3).astype(np.uint16)
        g = (arr[:, :, 1] >> 2).astype(np.uint16)
        b = (arr[:, :, 2] >> 3).astype(np.uint16)
        rgb565 = (r << 11) | (g << 5) | b
        with open(FB1, "wb") as f:
            rgb565.tofile(f)
    except Exception as e:
        print(f"❌ LCD Error: {e}")

def show_on_lcd(history_list, status=""):
    """Hiển thị lịch sử và trạng thái lên LCD"""
    if not USE_LCD:
        return
    
    try:
        img = Image.new("RGB", (LCD_WIDTH, LCD_HEIGHT), "white")
        draw = ImageDraw.Draw(img)
        lcd_font = ImageFont.truetype(LCD_FONT_PATH, LCD_FONT_SIZE)
        
        # Hiển thị trạng thái
        if status:
            status_font = ImageFont.truetype(LCD_FONT_PATH, 18)
            bbox = draw.textbbox((0, 0), status, font=status_font)
            w = bbox[2] - bbox[0]
            draw.text(((LCD_WIDTH - w) // 2, 10), status, font=status_font, fill="blue")
            y = 50
        else:
            y = 10
        
        # Hiển thị lịch sử
        with history_lock:
            for line in history_list:
                wrapped = textwrap.fill(line, width=25)
                for subline in wrapped.split("\n"):
                    bbox = draw.textbbox((0, 0), subline, font=lcd_font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    x = (LCD_WIDTH - w) // 2
                    draw.text((x, y), subline, font=lcd_font, fill="black")
                    y += h + 4
                y += 8
        
        write_to_lcd(img)
    except Exception as e:
        print(f"❌ LCD Display Error: {e}")

def update_lcd_history(text):
    """Cập nhật lịch sử và hiển thị lên LCD"""
    with history_lock:
        display_history.append(text)
    show_on_lcd(list(display_history))
    print(f"📺 LCD: {text}")

# ===== HTTP Client (Sign Language) ====
class ClientSocket:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self.response_text = None
        self.response_lock = threading.Lock()
        print(f"🌐 Sign Language Client → http://{host}:{port}")
    
    def send_frame(self, frame):
        """Gửi frame đến server qua HTTP POST"""
        try:
            url = f"http://{self.host}:{self.port}/upload_frame"
            frame_data = pickle.dumps(frame)
            response = requests.post(url, data=frame_data, timeout=1)
            return response.status_code == 200
        except:
            return False
            
    def send_text(self, text):
        """Gửi văn bản đến server qua HTTP POST (unified server)"""
        try:
            url = f"http://{self.host}:{self.port}/upload_sign"
            response = requests.post(url, json={"text": text}, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                processed = result.get('processed', '')
                
                # Cập nhật response để hiển thị
                with self.response_lock:
                    self.response_text = processed
                
                print(f"📤 Sent: {text}")
                print(f"📥 Processed: {processed}")
                
                # Hiển thị lên LCD
                update_lcd_history(processed)
                return True
            else:
                print(f"❌ Server error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to send text: {e}")
            return False


# Khởi tạo client - Kết nối đến server Windows (192.168.10.117)
client = ClientSocket(host='192.168.10.117', port=5000)

# ==== Buffer logic ====
text_buffer = ""
current_char = None
stable_count = 0
STABLE_THRESHOLD = 5
frame_counter = 0

last_added_time = 0
ADD_INTERVAL = 3.0  # giây

# Auto-send logic
AUTO_SEND_DELAY = 5.0  # Tự động gửi sau 5s không có tay và không thêm ký tự
last_activity_time = 0  # Thời điểm hoạt động cuối cùng
no_hand_start_time = 0  # Thời điểm bắt đầu không phát hiện tay
no_hand_duration = 0  # Thời gian không phát hiện tay

# Frame streaming
FRAME_SEND_INTERVAL = 0.1  # gửi frame mỗi 100ms (10 FPS)
last_frame_time = 0

# ==== Video ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
fps_time = time.time()

# Hiển thị trạng thái ban đầu trên LCD
if USE_LCD:
    show_on_lcd([], "🎥 Đang khởi động...")
    print("📺 LCD initialized")

def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Gửi frame đến server (giảm tần suất để không quá tải)
    current_time = time.time()
    if current_time - last_frame_time >= FRAME_SEND_INTERVAL:
        client.send_frame(frame)
        last_frame_time = current_time

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

    # ====== Cộng dồn thành câu ======
    if predicted_label == current_char:
        stable_count += 1
    else:
        current_char = predicted_label
        stable_count = 0

    # Thêm ký tự vào buffer
    if stable_count >= STABLE_THRESHOLD and predicted_label not in ["No hand"]:
        current_time = time.time()
        if current_time - last_added_time >= ADD_INTERVAL:
            text_buffer += predicted_label
            last_added_time = current_time
            last_activity_time = current_time  # Cập nhật thời gian hoạt động
            stable_count = 0
            print("🆕 Added:", predicted_label, "→", text_buffer)

    # ====== Tự động gửi sau 5s không hoạt động ======
    current_time = time.time()
    
    # Đếm thời gian không phát hiện tay
    if predicted_label == "No hand":
        if no_hand_start_time == 0:  # Bắt đầu đếm
            no_hand_start_time = current_time
        no_hand_duration = current_time - no_hand_start_time
    else:
        no_hand_start_time = 0  # Reset nếu phát hiện tay
        no_hand_duration = 0
        last_activity_time = current_time
    
    # Tự động gửi nếu có text và không hoạt động trong AUTO_SEND_DELAY giây
    if text_buffer.strip() and no_hand_duration >= AUTO_SEND_DELAY:
        print(f"⏰ Auto-send after {AUTO_SEND_DELAY}s of inactivity")
        client.send_text(text_buffer)
        if USE_LCD:
            show_on_lcd(list(display_history), "📤 Tự động gửi...")
        
        # Reset để không gửi liên tục
        text_buffer = ""
        no_hand_start_time = 0
        no_hand_duration = 0
        last_activity_time = current_time

    # FPS
    fps = 1 / (time.time() - fps_time)
    fps_time = time.time()

    # ==== Hiển thị ====
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((20, 20), f"Ký hiệu: {predicted_label}", font=font, fill=(0,255,0))
    draw.text((20, 60), f"Câu: {text_buffer}", font=font, fill=(255,255,255))
    draw.text((20, 100), f"FPS: {fps:.1f}", font=font, fill=(255,255,0))
    
    # Hiển thị hướng dẫn và countdown
    if text_buffer.strip() and predicted_label == "No hand":
        remaining = AUTO_SEND_DELAY - no_hand_duration
        if remaining > 0:
            draw.text((20, 140), f"⏱️  Gửi sau {remaining:.1f}s", font=font, fill=(255,165,0))
        else:
            draw.text((20, 140), f"[S] Send | [C] Clear | [Q] Quit", font=font, fill=(100,200,255))
    else:
        draw.text((20, 140), f"[S] Send | [C] Clear | [Q] Quit", font=font, fill=(100,200,255))
    
    # Hiển thị phản hồi từ server (nếu có)
    with client.response_lock:
        if client.response_text:
            draw.text((20, 180), f"Server: {client.response_text}", font=font, fill=(255,100,255))
    
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Sign Sentence Builder", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]: 
        break
    elif key == ord(' '): 
        text_buffer += " "
    elif key == ord('\b') or key == ord('x'): 
        text_buffer = text_buffer[:-1]
    elif key == ord('c'):  # clear buffer
        text_buffer = ""
        no_hand_start_time = 0  # Reset countdown
        no_hand_duration = 0  # Reset countdown
        last_activity_time = time.time()
        # Xóa LCD
        with history_lock:
            display_history.clear()
        if USE_LCD:
            show_on_lcd([], "🗑️  Đã xóa")
    elif key == ord('s'):  # Nhấn 's' để gửi chuỗi thủ công
        if text_buffer.strip():
            client.send_text(text_buffer)
            # Hiển thị trạng thái gửi trên LCD
            if USE_LCD:
                show_on_lcd(list(display_history), "📤 Đang gửi...")
            text_buffer = ""  # Xóa buffer sau khi gửi
            no_hand_start_time = 0
            no_hand_duration = 0
            last_activity_time = time.time()

cap.release()
cv2.destroyAllWindows()

# Hiển thị trạng thái kết thúc trên LCD
if USE_LCD:
    show_on_lcd(list(display_history), "⏹️  Đã dừng")

print("✅ Done.")
