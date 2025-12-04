#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server tích hợp Authentication + Speech-to-Text + Sign Language
File: unified_server.py
"""
import time
import pyotp
import qrcode
import io
import base64
from flask import Flask, request, jsonify, render_template_string, Response, session, redirect, url_for
import os
from datetime import datetime, timedelta
import threading
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import cv2
import pickle
import hashlib
from functools import wraps
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production-2024'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# ===== Database Files =====
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ===== Login Required Decorator =====
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ===== Camera & Data Storage =====
latest_frame = None
frame_lock = threading.Lock()

UPLOAD_FOLDER = "received_data"
SPEECH_FOLDER = os.path.join(UPLOAD_FOLDER, "speech")
SIGN_FOLDER = os.path.join(UPLOAD_FOLDER, "sign")

os.makedirs(SPEECH_FOLDER, exist_ok=True)
os.makedirs(SIGN_FOLDER, exist_ok=True)

# ===== Load Accent Model =====
print("📄 Loading Vietnamese accent model...")
try:
    model_path = "peterhung/vietnamese-accent-marker-xlm-roberta"
    accent_tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    accent_model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accent_model.to(device)
    accent_model.eval()
    print("✅ Accent model loaded")
except Exception as e:
    print(f"⚠️ Accent model not available: {e}")
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

def process_text_with_accent(text):
    if not accent_model:
        return text
    try:
        tokens = segment_vietnamese_no_accent(text.lower().strip())
        result = " ".join(tokens).strip()
        if result:
            result = result[0].upper() + result[1:] + "."
        return result
    except:
        return text

# ===== 2FA Functions =====
def generate_2fa_secret():
    """Tạo secret key cho 2FA"""
    return pyotp.random_base32()

def generate_qr_code(username, secret):
    """Tạo QR code cho 2FA"""
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(
        name=username,
        issuer_name='Sign Language System'
    )
    
    # Tạo QR code và chuyển thành base64
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

def verify_2fa_code(secret, code):
    """Xác thực mã 2FA"""
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)  # valid_window=1 cho phép code trước/sau 30s

# ===== Authentication Routes =====

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        otp = data.get('otp', '').strip()
        
        users = load_users()
        
        # Bước 1: Đăng ký thông tin và tạo QR code
        if not otp:
            if not username or not password:
                return jsonify({"error": "Vui lòng điền đầy đủ thông tin"}), 400
            
            if username in users:
                return jsonify({"error": "Tên đăng nhập đã tồn tại"}), 400
            
            # Tạo 2FA secret
            secret_2fa = generate_2fa_secret()
            
            # Lưu thông tin tạm thời vào session
            session['temp_register'] = {
                'username': username,
                'password': hash_password(password),
                'secret_2fa': secret_2fa,
                'timestamp': datetime.now().isoformat()
            }
            
            # Tạo QR code
            qr_code = generate_qr_code(username, secret_2fa)
            
            return jsonify({
                "status": "qr_generated",
                "message": "Quét mã QR bằng Google Authenticator",
                "qr_code": qr_code,
                "secret": secret_2fa
            }), 200
        
        # Bước 2: Xác thực OTP từ Authenticator app
        else:
            if 'temp_register' not in session:
                return jsonify({"error": "Phiên đăng ký không hợp lệ"}), 400
            
            temp_data = session['temp_register']
            reg_time = datetime.fromisoformat(temp_data['timestamp'])
            
            # Kiểm tra timeout (10 phút)
            if datetime.now() - reg_time > timedelta(minutes=10):
                session.pop('temp_register', None)
                return jsonify({"error": "Phiên đăng ký đã hết hạn"}), 400
            
            # Xác thực OTP
            if not verify_2fa_code(temp_data['secret_2fa'], otp):
                return jsonify({"error": "Mã OTP không đúng"}), 400
            
            # Lưu user vào database
            users[temp_data['username']] = {
                "password": temp_data['password'],
                "secret_2fa": temp_data['secret_2fa'],
                "created_at": datetime.now().isoformat()
            }
            save_users(users)
            
            # Xóa dữ liệu tạm
            session.pop('temp_register', None)
            
            return jsonify({"status": "success", "message": "Đăng ký thành công!"}), 200
    
    return render_template_string(REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        otp = data.get('otp', '').strip()
        
        users = load_users()
        
        # Bước 1: Kiểm tra username và password
        if not otp:
            if username not in users:
                return jsonify({"error": "Tên đăng nhập không tồn tại"}), 400
            
            user = users[username]
            if hash_password(password) != user['password']:
                return jsonify({"error": "Mật khẩu không đúng"}), 400
            
            # Lưu thông tin vào session để bước 2 sử dụng
            session['temp_login'] = {
                'username': username,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                "status": "otp_required",
                "message": "Nhập mã OTP từ Authenticator app"
            }), 200
        
        # Bước 2: Xác thực OTP
        else:
            if 'temp_login' not in session:
                return jsonify({"error": "Phiên đăng nhập không hợp lệ"}), 400
            
            temp_data = session['temp_login']
            login_time = datetime.fromisoformat(temp_data['timestamp'])
            
            # Kiểm tra timeout (5 phút)
            if datetime.now() - login_time > timedelta(minutes=5):
                session.pop('temp_login', None)
                return jsonify({"error": "Phiên đăng nhập đã hết hạn"}), 400
            
            # Lấy secret từ database
            user = users[temp_data['username']]
            
            # Xác thực OTP
            if not verify_2fa_code(user['secret_2fa'], otp):
                return jsonify({"error": "Mã OTP không đúng"}), 400
            
            # Đăng nhập thành công
            session['user'] = temp_data['username']
            session.permanent = True
            session.pop('temp_login', None)
            
            return jsonify({
                "status": "success",
                "message": "Đăng nhập thành công!",
                "redirect": "/"
            }), 200
    
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ===== Protected Routes =====

@app.route('/upload_speech', methods=['POST'])
@login_required
def upload_speech():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        content = file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if lines:
            latest = lines[-1]
            filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(os.path.join(SPEECH_FOLDER, filename), "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"🎤 Speech: {latest}")
            return jsonify({"status": "success", "text": latest, "processed": latest}), 200
        
        return jsonify({"status": "empty"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_frame', methods=['POST'])
@login_required
def upload_frame():
    global latest_frame
    try:
        frame = pickle.loads(request.data)
        with frame_lock:
            latest_frame = frame
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_sign', methods=['POST'])
@login_required
def upload_sign():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text"}), 400
        
        raw_text = data['text']
        processed_text = process_text_with_accent(raw_text)
        
        filename = f"sign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join(SIGN_FOLDER, filename), "w", encoding="utf-8") as f:
            f.write(f"Raw: {raw_text}\nProcessed: {processed_text}\n")
        
        print(f"🤟 Sign: {raw_text} → {processed_text}")
        return jsonify({"status": "success", "text": raw_text, "processed": processed_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_legacy():
    return upload_speech()

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting...", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
@login_required
def index():
    from templates import get_index_html
    return get_index_html(session.get('user', 'User'), SPEECH_FOLDER, SIGN_FOLDER)

# ===== HTML Templates =====

LOGIN_HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Đăng nhập</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.login-container{background:#fff;padding:40px;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);max-width:450px;width:100%}
h1{text-align:center;color:#667eea;margin-bottom:10px;font-size:2em}
.subtitle{text-align:center;color:#666;margin-bottom:30px;font-size:.9em}
.form-group{margin-bottom:20px}
label{display:block;margin-bottom:8px;color:#333;font-weight:500}
input{width:100%;padding:12px 15px;border:2px solid #e0e0e0;border-radius:10px;font-size:1em;transition:all .3s}
input:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 3px rgba(102,126,234,.1)}
button{width:100%;padding:14px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
color:#fff;border:none;border-radius:10px;font-size:1.1em;font-weight:600;cursor:pointer;transition:all .3s;margin-top:10px}
button:hover{transform:translateY(-2px);box-shadow:0 5px 20px rgba(102,126,234,.4)}
button:disabled{background:#ccc;cursor:not-allowed;transform:none}
.message{padding:12px;border-radius:8px;margin-bottom:20px;display:none}
.message.error{background:#fee;color:#c33;border:1px solid #fcc}
.message.success{background:#efe;color:#3c3;border:1px solid #cfc}
.message.info{background:#eef;color:#33c;border:1px solid #ccf}
.register-link{text-align:center;margin-top:20px;color:#666}
.register-link a{color:#667eea;text-decoration:none;font-weight:600}
.otp-step{display:none}
.step-indicator{text-align:center;margin-bottom:20px;font-size:.9em;color:#667eea;font-weight:600}
.otp-info{background:#f0f4ff;padding:15px;border-radius:8px;margin-bottom:20px;font-size:.9em;color:#555}
</style></head><body>
<div class="login-container">
<h1>🔐 Đăng nhập</h1>
<p class="subtitle">Sign Language System</p>
<div id="message" class="message"></div>
<div id="step-indicator" class="step-indicator">Bước 1: Nhập tên đăng nhập và mật khẩu</div>
<form id="loginForm">
<div id="credentials-step">
<div class="form-group"><label for="username">Tên đăng nhập</label>
<input type="text" id="username" required></div>
<div class="form-group"><label for="password">Mật khẩu</label>
<input type="password" id="password" required></div>
<button type="submit">Tiếp tục</button></div>
<div id="otp-step" class="otp-step">
<div class="otp-info">📱 Mở Google Authenticator và nhập mã 6 chữ số</div>
<div class="form-group"><label for="otp">Mã OTP (6 số)</label>
<input type="text" id="otp" maxlength="6" pattern="[0-9]{6}"></div>
<button type="submit">Xác thực</button></div>
</form>
<div class="register-link">Chưa có tài khoản? <a href="/register">Đăng ký</a></div>
</div>
<script>
let step=1;const form=document.getElementById('loginForm'),msg=document.getElementById('message'),
ind=document.getElementById('step-indicator'),cred=document.getElementById('credentials-step'),
otp=document.getElementById('otp-step');
function showMsg(t,y){msg.textContent=t;msg.className='message '+y;msg.style.display='block'}
function hideMsg(){msg.style.display='none'}
form.addEventListener('submit',async(e)=>{e.preventDefault();hideMsg();
const u=document.getElementById('username').value,p=document.getElementById('password').value,
o=document.getElementById('otp').value,d={username:u,password:p};
if(step===2)d.otp=o;
try{const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}),
res=await r.json();
if(r.ok){if(res.status==='otp_required'){step=2;ind.textContent='Bước 2: Nhập mã từ Authenticator';
cred.style.display='none';otp.style.display='block';document.getElementById('otp').focus();
showMsg(res.message,'info')}else if(res.status==='success'){showMsg(res.message,'success');
setTimeout(()=>window.location.href=res.redirect,1000)}}else showMsg(res.error,'error')
}catch(err){showMsg('Lỗi kết nối','error')}});
</script></body></html>'''

REGISTER_HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Đăng ký</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.register-container{background:#fff;padding:40px;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);max-width:500px;width:100%}
h1{text-align:center;color:#667eea;margin-bottom:10px;font-size:2em}
.subtitle{text-align:center;color:#666;margin-bottom:30px;font-size:.9em}
.form-group{margin-bottom:20px}
label{display:block;margin-bottom:8px;color:#333;font-weight:500}
input{width:100%;padding:12px 15px;border:2px solid #e0e0e0;border-radius:10px;font-size:1em;transition:all .3s}
input:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 3px rgba(102,126,234,.1)}
button{width:100%;padding:14px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
color:#fff;border:none;border-radius:10px;font-size:1.1em;font-weight:600;cursor:pointer;transition:all .3s;margin-top:10px}
button:hover{transform:translateY(-2px);box-shadow:0 5px 20px rgba(102,126,234,.4)}
.message{padding:12px;border-radius:8px;margin-bottom:20px;display:none}
.message.error{background:#fee;color:#c33}
.message.success{background:#efe;color:#3c3}
.message.info{background:#eef;color:#33c}
.login-link{text-align:center;margin-top:20px;color:#666}
.login-link a{color:#667eea;text-decoration:none;font-weight:600}
.otp-step{display:none}
.step-indicator{text-align:center;margin-bottom:20px;font-size:.9em;color:#667eea;font-weight:600}
.qr-container{text-align:center;margin:20px 0;padding:20px;background:#f8f9ff;border-radius:10px}
.qr-container img{max-width:250px;border:3px solid #667eea;border-radius:10px}
.secret-code{background:#fff;padding:15px;border-radius:8px;margin-top:15px;font-family:monospace;font-size:1.1em;color:#667eea;word-break:break-all}
.instructions{background:#f0f4ff;padding:15px;border-radius:8px;margin-bottom:20px;font-size:.9em;line-height:1.6}
.instructions ol{margin-left:20px}
.instructions li{margin:8px 0}
</style></head><body>
<div class="register-container">
<h1>📝 Đăng ký</h1>
<p class="subtitle">Sign Language System</p>
<div id="message" class="message"></div>
<div id="step-indicator" class="step-indicator">Bước 1: Nhập thông tin</div>
<form id="registerForm">
<div id="credentials-step">
<div class="form-group"><label>Tên đăng nhập</label>
<input type="text" id="username" required minlength="3"></div>
<div class="form-group"><label>Mật khẩu</label>
<input type="password" id="password" required minlength="6"></div>
<div class="form-group"><label>Xác nhận mật khẩu</label>
<input type="password" id="confirmPassword" required></div>
<button type="submit">Tiếp tục</button></div>
<div id="otp-step" class="otp-step">
<div class="instructions">
<strong>📱 Hướng dẫn cài đặt 2FA:</strong>
<ol>
<li>Tải <strong>Google Authenticator</strong> trên điện thoại</li>
<li>Mở app và chọn <strong>"Quét mã QR"</strong></li>
<li>Quét mã QR bên dưới</li>
<li>Nhập mã 6 số hiển thị trong app</li>
</ol>
</div>
<div id="qr-container" class="qr-container"></div>
<div class="form-group"><label>Mã xác thực (6 số)</label>
<input type="text" id="otp" maxlength="6" pattern="[0-9]{6}" placeholder="000000"></div>
<button type="submit">Kích hoạt</button></div>
</form>
<div class="login-link">Đã có tài khoản? <a href="/login">Đăng nhập</a></div>
</div>
<script>
let step=1;const form=document.getElementById('registerForm'),msg=document.getElementById('message'),
ind=document.getElementById('step-indicator'),cred=document.getElementById('credentials-step'),
otp=document.getElementById('otp-step'),qrCont=document.getElementById('qr-container');
function showMsg(t,y){msg.textContent=t;msg.className='message '+y;msg.style.display='block'}
function hideMsg(){msg.style.display='none'}
form.addEventListener('submit',async(e)=>{e.preventDefault();hideMsg();
const u=document.getElementById('username').value,
p=document.getElementById('password').value,
cp=document.getElementById('confirmPassword').value,
o=document.getElementById('otp').value;
if(step===1&&p!==cp){showMsg('Mật khẩu không khớp','error');return}
const d={username:u,password:p};if(step===2)d.otp=o;
try{const r=await fetch('/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}),
res=await r.json();
if(r.ok){if(res.status==='qr_generated'){step=2;ind.textContent='Bước 2: Quét mã QR';
cred.style.display='none';otp.style.display='block';
qrCont.innerHTML=`<img src="data:image/png;base64,${res.qr_code}" alt="QR Code">
<div class="secret-code">Hoặc nhập thủ công:<br>${res.secret}</div>`;
document.getElementById('otp').focus();showMsg(res.message,'info')}
else if(res.status==='success'){showMsg(res.message,'success');
setTimeout(()=>window.location.href='/login',2000)}}else showMsg(res.error,'error')
}catch(err){showMsg('Lỗi kết nối','error')}});
</script></body></html>'''

# ===== Run Server =====
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Server with 2FA Authentication")
    print("=" * 70)
    print("🔐 Login: http://localhost:5000/login")
    print("📝 Register: http://localhost:5000/register")
    print("🌐 Home: http://localhost:5000")
    print("=" * 70)
    print("\n✅ 2FA với Google Authenticator đã được kích hoạt")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)