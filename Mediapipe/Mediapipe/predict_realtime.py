# predict_realtime_tk.py - SERVER
import cv2
import numpy as np
import pickle
import time
import os
import tkinter as tk
from tkinter import ttk
from PIL import ImageFont, ImageDraw, Image, ImageTk
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import socket
import threading
import struct

# ==================== LOAD SIGN MODEL & PREPROCESSORS ====================
# Server không cần load model nữa, chỉ xử lý text

print("=" * 70)
print("Server: Waiting for client connection...")
print("=" * 70)

# ==================== VIETNAMESE ACCENT MODEL & UTILS ====================

def load_trained_transformer_model():
    model_path = "peterhung/vietnamese-accent-marker-xlm-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return model, tokenizer

accent_model, accent_tokenizer = load_trained_transformer_model()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
accent_model.to(device)
accent_model.eval()

def _load_tags_set(fpath):
    labels = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

label_list = _load_tags_set("./selected_tags_names.txt")
TOKENIZER_WORD_PREFIX = "▁"

VIETNAMESE_DICT = {
    # Chao hoi & Giao tiep co ban
    'chao', 'cac', 'ban', 'toi', 'la', 
    'ten', 'gi', 'cam', 'on', 'khong', 'vang',
    'da', 'duoc', 'roi', 'oi', 'a', 'u', 'xin',
    
    # Dai tu & Xung ho
    'chi', 'ban', 'minh', 'chung', 
    'may', 'no', 'ho', 'nguoi',
    'ong', 'ba', 'chu', 'bac', 'co', 'di',
    'cau', 'mo', 'duong', 'thim',
    
    # Gia dinh
    'bo', 'me', 'gai', 
    'chi', 'ong', 'ba', 'noi', 'ngoai',
    'chau', 'bau', 'vo', 'chong', 'dinh',
    
    # Con nguoi & Dac diem
    'nguoi', 'dan', 'ba', 'tre', 'gia',
    'lon', 'nho', 'be', 'thap', 'gay',
    'map', 'dep', 'xau', 'tot', 'thong',
    'nu', 'gioi', 'tinh',
    
    # Dong tu thong dung
    'la', 'co', 'di', 'den', 've', 'len',
    'xuong', 'vao', 'lam', 'an', 'uong',
    'ngu', 'thuc', 'day', 'nhin', 'doc', 'viet', 'noi', 'hoi', 'tra', 'loi',
    'choi', 'hoc', 'day', 'biet', 'hieu',
    'yeu', 'thuong', 'ghet', 'thich', 'muon',
    'can', 'phai', 'nen', 'duoc', 'dung',
    'ngoi', 'dung', 'nam', 'chay', 'nhay',
    'mo', 'dong', 'bat', 'tat', 'mua', 'ban',
    'gui', 'nhan', 'goi', 'tim', 'kiem',
    
    # Tinh tu & Trang tu
    'dep', 'xau', 'tot', 'te', 'gioi',
    'lon', 'nho', 'thap', 'dai', 'ngan',
    'rong', 'hep', 'nang', 'nhe', 'cham',
    'nong', 'lanh', 'am', 'mat', 'kho', 'de',
    'rat', 'qua', 'lam', 'hoi', 'kha', 'tuong',
    'doi', 'nua', 'cung', 'da', 'dang', 'se',
    'vua', 'moi', 'cu', 'sap', 'chua',
    
    # Con so
    'khong', 'mot', 'bon', 'nam',
    'sau', 'bay', 'tam', 'chin', 'muoi',
    'tram', 'ngan', 'van', 'trieu', 'ty',
    'thu', 'lan', 'so', 'luong',
    
    # Thoi gian
    'gio', 'phut', 'giay', 'ngay', 'thang',
    'nam', 'tuan', 'thoi', 'khoang',
    'sang', 'chieu', 'toi', 'dem',
    'som', 'muon', 'mot',
    'truoc', 'luc', 'dang', 'roi',
    'buoi', 'dau', 'cuoi', 'giua',
    
    # Dia diem
    'nha', 'truong', 'lop', 'cong', 'so',
    'vien', 'benh', 'ngan', 'hang', 'cho',
    'sieu', 'thi', 'quan', 'ca', 'phe',
    'khach', 'san', 'duong', 'pho', 'hem',
    'thanh', 'tinh', 'huyen', 'xa', 'thi',
    'tran', 'lang', 'nong', 'thon', 'vung',
    'viet', 'ha', 'gon', 'da', 'nang',
    'noi', 'cho', 'cua', 'ngoai', 'tren',
    'duoi', 'truoc', 'ben', 'canh', 'gan',
    
    # Giao duc
    'hoc', 'giao', 'vien', 'thay', 'co',
    'bai', 'tap', 'kiem', 'diem',
    'vo', 'but', 'thu', 'vien', 'mon',
    'toan', 'van', 'ly', 'hoa', 'su',
    'dia', 'tieu', 'dai',
    
    # Cong viec
    'lam', 'viec', 'cong', 'van', 'phong',
    'hop', 'du', 'an', 'bao', 'luong',
    'nghi', 'phep', 'giai', 'quyet',
    
    # Do vat & Cong nghe
    'dien', 'thoai', 'may', 'tinh',
    'mang',
    'ung', 'dung', 'thu',
    'dien', 'tu', 'do', 'dung',
    'ao', 'quan', 'giay', 'dep', 'tu', 'ban',
    'ghe', 'giuong', 'cua', 'so', 'den',
    
    # Do an & Thuc uong
    'an', 'uong', 'com', 'pho', 'bun', 'banh',
    'mi', 'thit', 'ca', 'tom', 'trai', 'cay',
    'nuoc', 'tra', 'ca', 'phe', 'sua', 'ruou',
    'dan', 'man', 'nhat', 'ngot',
    
    # Suc khoe & Cam xuc
    'khoe', 'manh', 'om', 'dau', 'benh',
    'bac', 'si', 'cham', 'soc', 'kham',
    'buon', 've', 'gian', 'du', 'hanh',
    'phuc', 'kho', 'khan', 'met', 'moi',
    'lang', 'au', 'cang', 'thang',
    
    # Tu nghi van & Lien tu
    'gi', 'nao', 'dau', 'the',
    'nhu', 'nhieu', 'may', 'gio',
    'va', 'hoac', 'nhung', 'ma', 'neu', 'thi',
    'nen', 'vi', 'vay', 'the', 'de',
    'hon', 'bang', 'nhat', 'ca',
    
    # Dong vat & Thuc vat
    'cho', 'meo', 'ga', 'vit', 'lon', 'bo',
    'cay', 'co', 'la', 'canh', 'qua',
    
    # Mau sac
    'mau', 'do', 'vang', 'den', 'trang',
    'tim', 'hong', 'nau', 'xam',
    
    # The thao & Giai tri
    'choi', 'bong', 'da', 'ro', 'cau',
    'boi', 'chay', 'nhac', 'hat',
    
    # Tu khac thuong dung
    'cai', 'chiec', 'cuon', 'to', 'bo',
    'doi', 'cap', 'tam', 'canh', 'mieng',
    'quan', 'ao', 'chan', 'dau', 'mat',
    'mieng', 'mui', 'mat', 'long', 'bung',
    'y', 'kien', 'nghi', 'nghia', 'cach',
    'phuong', 'phap', 'van', 'de', 'su',
    'cuoc', 'song', 'doi', 'doi', 'dieu',
}
def segment_vietnamese_no_accent(text):
    """
    Tách từ tiếng Việt không dấu sử dụng Dynamic Programming
    """
    text = text.lower().strip()
    n = len(text)
    
    # dp[i] = (độ dài từ tốt nhất kết thúc tại i, vị trí bắt đầu từ đó)
    dp = [(-1, -1)] * (n + 1)
    dp[0] = (0, 0)
    
    # Duyệt qua từng vị trí
    for i in range(1, n + 1):
        # Thử tất cả các từ có thể kết thúc tại vị trí i
        for j in range(max(0, i - 15), i):  # Giới hạn độ dài từ <= 15
            word = text[j:i]
            if word in VIETNAMESE_DICT and dp[j][0] != -1:
                if dp[i][0] == -1 or dp[j][0] + 1 > dp[i][0]:
                    dp[i] = (dp[j][0] + 1, j)
    
    # Truy vết để lấy các từ
    if dp[n][0] == -1:
        # Không tách được, trả về từng ký tự
        return list(text)
    
    words = []
    pos = n
    while pos > 0:
        start_pos = dp[pos][1]
        if start_pos == pos:  # Không tách được đoạn này
            break
        words.append(text[start_pos:pos])
        pos = start_pos
    
    words.reverse()
    
    # Nếu không tách được hết, thêm phần còn lại
    if pos > 0:
        words.insert(0, text[:pos])
    
    return words if words else [text]

def chars_to_text(char_array):
    return "".join(char_array).lower()

def insert_accents(tokens, model, tokenizer):
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    subword_tokens = subword_tokens[1:-1]

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    predictions = outputs["logits"].cpu().numpy()
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions[0][1:-1]

    assert len(subword_tokens) == len(predictions)
    return subword_tokens, predictions

def merge_tokens_and_preds(tokens, predictions):
    merged_tokens_preds = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        label_indexes = {int(predictions[i])}
        if tok.startswith(TOKENIZER_WORD_PREFIX):
            tok_no_prefix = tok[len(TOKENIZER_WORD_PREFIX):]
            cur_word_toks = [tok_no_prefix]
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith(TOKENIZER_WORD_PREFIX):
                cur_word_toks.append(tokens[j])
                label_indexes.add(int(predictions[j]))
                j += 1
            cur_word = "".join(cur_word_toks)
            merged_tokens_preds.append((cur_word, label_indexes))
            i = j
        else:
            merged_tokens_preds.append((tok, label_indexes))
            i += 1
    return merged_tokens_preds

def get_accented_words(merged_tokens_preds, label_list):
    accented_words = []
    for word_raw, label_indexes in merged_tokens_preds:
        word_accented = word_raw
        for label_index in label_indexes:
            tag_name = label_list[int(label_index)]
            raw, vowel = tag_name.split("-")
            if raw and raw in word_raw:
                word_accented = word_raw.replace(raw, vowel)
                break
        accented_words.append(word_accented)
    return accented_words

def process_char_array(char_array):
    """
    char_array: ví dụ ['x','i','n','c','h','a','o','c','a','c','b','a','n']
    """
    text_input = chars_to_text(char_array)
    tokens = segment_vietnamese_no_accent(text_input)
    subword_tokens, predictions = insert_accents(tokens, accent_model, accent_tokenizer)
    merged_tokens_preds = merge_tokens_and_preds(subword_tokens, predictions)
    accented_words = get_accented_words(merged_tokens_preds, label_list)
    result = " ".join(accented_words).strip()
    if result:
        result = result[0].upper() + result[1:]
        if result[-1] not in ".?!":
            result += "."
    return result

# ==================== TKINTER APPLICATION ====================

font_path = (
    "C:/Windows/Fonts/segoeui.ttf"
    if os.name == "nt"
    else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
)
overlay_font = ImageFont.truetype(font_path, 30)

# ==================== SERVER SOCKET ====================

class ServerSocket:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.latest_text = ""
        self.frame_lock = threading.Lock()
        
    def start(self):
        """Khởi động server"""
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"🟢 Server started on {self.host}:{self.port}")
        
    def _run_server(self):
        """Chạy server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"🔵 Waiting for client connection...")
            
            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    self.client_socket, addr = self.server_socket.accept()
                    print(f"✅ Client connected from {addr}")
                    
                    # Nhận dữ liệu từ client
                    while self.running and self.client_socket:
                        try:
                            # Nhận loại dữ liệu (1 byte): 'T' = text, 'F' = frame
                            data_type = self.client_socket.recv(1).decode('utf-8')
                            
                            if data_type == 'T':
                                # Nhận text
                                length = struct.unpack('!I', self.client_socket.recv(4))[0]
                                text_data = self.client_socket.recv(length).decode('utf-8')
                                self._process_text(text_data)
                                
                            elif data_type == 'F':
                                # Nhận frame
                                length = struct.unpack('!I', self.client_socket.recv(4))[0]
                                frame_data = b''
                                while len(frame_data) < length:
                                    chunk = self.client_socket.recv(min(4096, length - len(frame_data)))
                                    if not chunk:
                                        break
                                    frame_data += chunk
                                
                                if len(frame_data) == length:
                                    frame = pickle.loads(frame_data)
                                    with self.frame_lock:
                                        self.latest_frame = frame
                                        
                        except socket.timeout:
                            continue
                        except Exception as e:
                            print(f"❌ Error receiving data: {e}")
                            break
                    
                    if self.client_socket:
                        self.client_socket.close()
                        self.client_socket = None
                        print("🔴 Client disconnected")
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"❌ Server error: {e}")
                        
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            
    def _process_text(self, text):
        """Xử lý văn bản nhận được từ client"""
        print("=" * 70)
        print(f"📥 Received from client: {text}")
        
        # Xử lý thêm dấu cho văn bản
        try:
            char_array = list(text.replace(" ", "").lower())
            if char_array:
                processed_text = process_char_array(char_array)
                print(f"✨ Processed text: {processed_text}")
                
                # Gửi lại cho client
                if self.client_socket:
                    try:
                        self.client_socket.send('R'.encode('utf-8'))  # 'R' = Response
                        response_bytes = processed_text.encode('utf-8')
                        self.client_socket.send(struct.pack('!I', len(response_bytes)))
                        self.client_socket.send(response_bytes)
                        print(f"📤 Sent back to client: {processed_text}")
                    except Exception as e:
                        print(f"❌ Error sending response: {e}")
        except Exception as e:
            print(f"❌ Error processing text: {e}")
            
        print("=" * 70)
    
    def get_latest_frame(self):
        """Lấy frame mới nhất từ client"""
        with self.frame_lock:
            return self.latest_frame
        
    def stop(self):
        """Dừng server"""
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("🔴 Server stopped")

# Khởi tạo server - Localhost
# Khởi tạo server - Lắng nghe trên tất cả interfaces để Raspberry Pi có thể kết nối
server = ServerSocket(host='0.0.0.0', port=5000)

class SignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Server - Sign Language Processing")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.fps_time = time.time()

        self.build_ui()
        self.update_frame()

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Camera from client
        ttk.Label(main_frame, text="Client Camera Stream:").grid(row=0, column=0, sticky="w")
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=1, column=0, padx=10, pady=10)

        # Status
        self.status_var = tk.StringVar(value="Waiting for client connection...")
        ttk.Label(main_frame, textvariable=self.status_var, foreground="blue").grid(
            row=2, column=0, sticky="w", padx=10
        )

    def update_frame(self):
        # Lấy frame từ server
        frame = server.get_latest_frame()
        
        if frame is not None:
            # Hiển thị frame từ client
            fps = 1.0 / (time.time() - self.fps_time)
            self.fps_time = time.time()
            
            # Thêm FPS lên frame
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            draw.text((20, 20), f"Client Stream | FPS: {fps:.1f}", 
                     font=overlay_font, fill=(0, 255, 0))
            frame_rgb = np.array(frame_pil)
            
            # Cập nhật TKinter image
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            self.status_var.set("✅ Client connected - Streaming...")
        else:
            # Hiển thị placeholder khi chưa có kết nối
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder_pil = Image.fromarray(placeholder)
            draw = ImageDraw.Draw(placeholder_pil)
            draw.text((220, 240), "Waiting for client...", 
                     font=overlay_font, fill=(255, 255, 255))
            
            imgtk = ImageTk.PhotoImage(image=placeholder_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            self.status_var.set("⏳ Waiting for client connection...")

        self.root.after(30, self.update_frame)

    def on_close(self):
        server.stop()
        self.root.destroy()

if __name__ == "__main__":
    # Khởi động server
    server.start()
    
    # Khởi động GUI
    root = tk.Tk()
    app = SignApp(root)
    
    # Dừng server khi đóng ứng dụng
    def on_app_close():
        server.stop()
        app.on_close()
    
    root.protocol("WM_DELETE_WINDOW", on_app_close)
    root.mainloop()
