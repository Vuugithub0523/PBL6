#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server tích hợp nhận cả Speech-to-Text và Sign Language
File: unified_server.py
Chạy: python unified_server.py
Port: 5000
"""

from flask import Flask, request, jsonify, render_template_string, Response
import os
from datetime import datetime
import threading
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import cv2
import pickle
import struct

app = Flask(__name__)

# ===== Camera Stream Storage =====
latest_frame = None
frame_lock = threading.Lock()

# ===== Cấu hình =====
UPLOAD_FOLDER = "received_data"
SPEECH_FOLDER = os.path.join(UPLOAD_FOLDER, "speech")
SIGN_FOLDER = os.path.join(UPLOAD_FOLDER, "sign")

os.makedirs(SPEECH_FOLDER, exist_ok=True)
os.makedirs(SIGN_FOLDER, exist_ok=True)

# ===== Load Vietnamese Accent Model =====
print("🔄 Loading Vietnamese accent model...")
try:
    model_path = "peterhung/vietnamese-accent-marker-xlm-roberta"
    accent_tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    accent_model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accent_model.to(device)
    accent_model.eval()
    print("✅ Accent model loaded successfully")
except Exception as e:
    print(f"⚠️  Accent model not available: {e}")
    accent_model = None
    accent_tokenizer = None

def _load_tags_set(fpath):
    labels = []
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except:
        pass
    return labels

label_list = _load_tags_set("./selected_tags_names.txt")
TOKENIZER_WORD_PREFIX = "▁"

VIETNAMESE_DICT = {
    'chao', 'cac', 'ban', 'toi', 'la', 'ten', 'gi', 'cam', 'on', 'khong', 'vang',
    'da', 'duoc', 'roi', 'oi', 'a', 'u', 'xin', 'chi', 'minh', 'chung', 
    'may', 'no', 'ho', 'nguoi', 'ong', 'ba', 'chu', 'bac', 'co', 'di',
    'bo', 'me', 'gai', 'noi', 'ngoai', 'chau', 'bau', 'vo', 'chong', 'dinh',
    'tre', 'gia', 'lon', 'nho', 'be', 'thap', 'gay', 'map', 'dep', 'xau', 'tot',
    'di', 'den', 've', 'len', 'xuong', 'vao', 'lam', 'an', 'uong', 'ngu', 
    'thuc', 'day', 'nhin', 'doc', 'viet', 'noi', 'hoi', 'tra', 'loi',
    'hoc', 'choi', 'biet', 'hieu', 'yeu', 'thuong', 'ghet', 'thich', 'muon',
    'can', 'phai', 'nen', 'dung', 'ngoi', 'nam', 'chay', 'nhay',
    'mo', 'dong', 'bat', 'tat', 'mua', 'ban', 'gui', 'nhan', 'goi', 'tim', 'kiem',
    'rat', 'qua', 'lam', 'hoi', 'kha', 'tuong', 'doi', 'nua', 'cung', 'dang', 'se',
    'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi',
    'gio', 'phut', 'ngay', 'thang', 'nam', 'tuan', 'thoi', 'sang', 'chieu', 'toi',
    'nha', 'truong', 'lop', 'vien', 'benh', 'ngan', 'hang', 'cho', 'sieu', 'thi',
    'viet', 'ha', 'noi', 'da', 'nang', 'mau', 'do', 'vang', 'xanh', 'den', 'trang',
}

def segment_vietnamese_no_accent(text):
    """Tách từ tiếng Việt không dấu"""
    text = text.lower().strip()
    n = len(text)
    
    dp = [(-1, -1)] * (n + 1)
    dp[0] = (0, 0)
    
    for i in range(1, n + 1):
        for j in range(max(0, i - 15), i):
            word = text[j:i]
            if word in VIETNAMESE_DICT and dp[j][0] != -1:
                if dp[i][0] == -1 or dp[j][0] + 1 > dp[i][0]:
                    dp[i] = (dp[j][0] + 1, j)
    
    if dp[n][0] == -1:
        return list(text)
    
    words = []
    pos = n
    while pos > 0:
        start_pos = dp[pos][1]
        if start_pos == pos:
            break
        words.append(text[start_pos:pos])
        pos = start_pos
    
    words.reverse()
    if pos > 0:
        words.insert(0, text[:pos])
    
    return words if words else [text]

def insert_accents(tokens, model, tokenizer):
    """Thêm dấu cho tokens"""
    if not model or not tokenizer:
        return tokens, [0] * len(tokens)
    
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

    return subword_tokens, predictions

def merge_tokens_and_preds(tokens, predictions):
    """Merge tokens và predictions"""
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
    """Lấy từ có dấu"""
    accented_words = []
    for word_raw, label_indexes in merged_tokens_preds:
        word_accented = word_raw
        for label_index in label_indexes:
            if label_index < len(label_list):
                tag_name = label_list[int(label_index)]
                if '-' in tag_name:
                    raw, vowel = tag_name.split("-", 1)
                    if raw and raw in word_raw:
                        word_accented = word_raw.replace(raw, vowel)
                        break
        accented_words.append(word_accented)
    return accented_words

def process_text_with_accent(text):
    """Xử lý thêm dấu cho văn bản"""
    if not accent_model or not accent_tokenizer:
        return text
    
    try:
        # Tách từ
        text_input = text.lower().strip()
        tokens = segment_vietnamese_no_accent(text_input)
        
        # Thêm dấu
        subword_tokens, predictions = insert_accents(tokens, accent_model, accent_tokenizer)
        merged_tokens_preds = merge_tokens_and_preds(subword_tokens, predictions)
        accented_words = get_accented_words(merged_tokens_preds, label_list)
        
        # Format
        result = " ".join(accented_words).strip()
        if result:
            result = result[0].upper() + result[1:]
            if result[-1] not in ".?!":
                result += "."
        return result
    except Exception as e:
        print(f"❌ Accent processing error: {e}")
        return text

# ===== Endpoint: Speech-to-Text =====
@app.route('/upload_speech', methods=['POST'])
def upload_speech():
    """Nhận file transcript từ Whisper"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        content = file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if lines:
            latest = lines[-1]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Lưu file
            filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(SPEECH_FOLDER, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"🎤 [{timestamp}] Speech: {latest}")
            return jsonify({
                "status": "success", 
                "type": "speech",
                "text": latest,
                "processed": latest  # Speech đã có dấu từ Whisper
            }), 200
        
        return jsonify({"status": "empty"}), 200
        
    except Exception as e:
        print(f"❌ Speech error: {e}")
        return jsonify({"error": str(e)}), 500

# ===== Endpoint: Camera Frame từ Sign Language Client =====
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Nhận frame camera từ Sign Language client"""
    global latest_frame
    try:
        frame_data = request.data
        frame = pickle.loads(frame_data)
        
        with frame_lock:
            latest_frame = frame
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"❌ Frame error: {e}")
        return jsonify({"error": str(e)}), 500

# ===== Endpoint: Sign Language =====
@app.route('/upload_sign', methods=['POST'])
def upload_sign():
    """Nhận text không dấu từ Sign Language và xử lý"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text"}), 400
        
        raw_text = data['text']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Xử lý thêm dấu
        processed_text = process_text_with_accent(raw_text)
        
        # Lưu file
        filename = f"sign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(SIGN_FOLDER, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Raw: {raw_text}\n")
            f.write(f"Processed: {processed_text}\n")
        
        print(f"🤟 [{timestamp}] Sign: {raw_text} → {processed_text}")
        return jsonify({
            "status": "success",
            "type": "sign",
            "text": raw_text,
            "processed": processed_text
        }), 200
        
    except Exception as e:
        print(f"❌ Sign error: {e}")
        return jsonify({"error": str(e)}), 500

# ===== Endpoint: Upload chung (backward compatible) =====
@app.route('/upload', methods=['POST'])
def upload_legacy():
    """Endpoint tương thích ngược cho speech"""
    return upload_speech()

# ===== Video Stream =====
def generate_frames():
    """Generator để stream video"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                # Frame mặc định khi chưa có camera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Endpoint để stream video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== Web UI =====
@app.route('/')
def index():
    # Lấy danh sách file
    speech_files = sorted(os.listdir(SPEECH_FOLDER), reverse=True)[:10]
    sign_files = sorted(os.listdir(SIGN_FOLDER), reverse=True)[:10]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Speech & Sign Language Server</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 2em; margin-bottom: 10px; }
            .stats {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 15px;
            }
            .stat-item {
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 10px;
            }
            .content {
                padding: 30px;
            }
            .camera-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 30px;
            }
            .camera-section h2 {
                margin-bottom: 15px;
                color: #667eea;
            }
            .camera-view {
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .camera-view img {
                width: 100%;
                height: auto;
                display: block;
            }
            .data-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            .section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
            }
            .section h2 {
                margin-bottom: 20px;
                color: #667eea;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .item {
                background: white;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                transition: transform 0.2s;
            }
            .item:hover {
                transform: translateX(5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .timestamp {
                color: #666;
                font-size: 0.85em;
                margin-bottom: 8px;
            }
            .text {
                color: #333;
                font-size: 1.1em;
                line-height: 1.6;
            }
            .processed {
                color: #667eea;
                font-weight: bold;
                margin-top: 5px;
            }
            .empty {
                text-align: center;
                padding: 40px;
                color: #999;
            }
            @media (max-width: 768px) {
                .content { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ 🤟 Speech & Sign Language Server</h1>
                <div class="stats">
                    <div class="stat-item">🎤 Speech: """ + str(len(speech_files)) + """</div>
                    <div class="stat-item">🤟 Sign: """ + str(len(sign_files)) + """</div>
                </div>
            </div>
            <div class="content">
                <div class="camera-section">
                    <h2>📹 Sign Language Camera Feed</h2>
                    <div class="camera-view">
                        <img src="/video_feed" alt="Camera Feed">
                    </div>
                </div>
                
                <div class="data-grid">
                    <div class="section">
                        <h2><span>🎤</span> Speech-to-Text</h2>
    """
    
    if speech_files:
        for f in speech_files:
            try:
                with open(os.path.join(SPEECH_FOLDER, f), 'r', encoding='utf-8') as file:
                    lines = [line.strip() for line in file.readlines() if line.strip()]
                    if lines:
                        content = lines[-1]
                        timestamp_str = f.replace('speech_', '').replace('.txt', '')
                        try:
                            dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                            time_display = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            time_display = timestamp_str
                        
                        html += f"""
                        <div class="item">
                            <div class="timestamp">⏰ {time_display}</div>
                            <div class="text">{content}</div>
                        </div>
                        """
            except:
                pass
    else:
        html += '<div class="empty">📭 Chưa có dữ liệu</div>'
    
    html += """
                </div>
                <div class="section">
                    <h2><span>🤟</span> Sign Language</h2>
    """
    
    if sign_files:
        for f in sign_files:
            try:
                with open(os.path.join(SIGN_FOLDER, f), 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    raw = ""
                    processed = ""
                    for line in lines:
                        if line.startswith("Raw:"):
                            raw = line.replace("Raw:", "").strip()
                        elif line.startswith("Processed:"):
                            processed = line.replace("Processed:", "").strip()
                    
                    if raw or processed:
                        timestamp_str = f.replace('sign_', '').replace('.txt', '')
                        try:
                            dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                            time_display = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            time_display = timestamp_str
                        
                        html += f"""
                        <div class="item">
                            <div class="timestamp">⏰ {time_display}</div>
                            <div class="text">{raw}</div>
                            <div class="processed">→ {processed}</div>
                        </div>
                        """
            except:
                pass
    else:
        html += '<div class="empty">📭 Chưa có dữ liệu</div>'
    
    html += """
                </div>
                </div>
            </div>
        </div>
        <script>
            // Chỉ reload phần data, không reload video stream
            setInterval(() => {
                fetch(window.location.href)
                    .then(r => r.text())
                    .then(html => {
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        const newDataGrid = doc.querySelector('.data-grid');
                        const oldDataGrid = document.querySelector('.data-grid');
                        if (newDataGrid && oldDataGrid) {
                            oldDataGrid.innerHTML = newDataGrid.innerHTML;
                        }
                    });
            }, 5000);
        </script>
    </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    import socket
    
    # Lấy IP thực của server
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        server_ip = s.getsockname()[0]
        s.close()
    except:
        server_ip = "localhost"
    
    print("=" * 70)
    print("🚀 Unified Server - Speech & Sign Language")
    print("=" * 70)
    print(f"📡 Speech endpoint:  http://{server_ip}:5000/upload_speech")
    print(f"📡 Sign endpoint:    http://{server_ip}:5000/upload_sign")
    print(f"📹 Camera endpoint:  http://{server_ip}:5000/upload_frame")
    print(f"🎥 Video stream:     http://{server_ip}:5000/video_feed")
    print(f"📡 Legacy endpoint:  http://{server_ip}:5000/upload")
    print(f"🌐 Web UI:           http://{server_ip}:5000")
    print(f"🔧 Listening on:     0.0.0.0:5000 (all interfaces)")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
