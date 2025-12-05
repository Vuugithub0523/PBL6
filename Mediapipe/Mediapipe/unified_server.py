
import time
import pyotp
import qrcode
import io
import base64
from flask import Flask, request, jsonify, render_template_string, Response, session, redirect, url_for, render_template
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
from templates import get_index_html
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import secrets
import logging
from cryptography.fernet import Fernet
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY')
app = Flask(__name__)
# ===== Camera Stream Storage =====
latest_frame = None
frame_lock = threading.Lock()

# ===== Client Stats Storage =====
client_stats = {
    'fps': 0.0,
    'buffer': '',
    'predicted': '',
    'last_update': None
}
stats_lock = threading.Lock()
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
csrf = CSRFProtect(app)
# ===== Database Files =====
USERS_FILE = "users.json"
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
failed_login_attempts = {}  # {username: {'count': int, 'locked_until': datetime}}
LOCKOUT_THRESHOLD = 5
LOCKOUT_DURATION = timedelta(minutes=15)

def is_account_locked(username):
    """Kiểm tra account có bị lock không"""
    if username in failed_login_attempts:
        attempt = failed_login_attempts[username]
        if attempt['count'] >= LOCKOUT_THRESHOLD:
            if datetime.now() < attempt['locked_until']:
                return True, attempt['locked_until']
            else:
                # Unlock sau khi hết thời gian
                del failed_login_attempts[username]
                return False, None
    return False, None

def record_failed_login(username):
    """Ghi nhận đăng nhập thất bại"""
    if username not in failed_login_attempts:
        failed_login_attempts[username] = {'count': 0, 'locked_until': None}
    
    failed_login_attempts[username]['count'] += 1
    
    if failed_login_attempts[username]['count'] >= LOCKOUT_THRESHOLD:
        failed_login_attempts[username]['locked_until'] = datetime.now() + LOCKOUT_DURATION

def reset_failed_login(username):
    """Reset khi đăng nhập thành công"""
    if username in failed_login_attempts:
        del failed_login_attempts[username]
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'rb') as f: 
            content = f.read()
            
        if not content:
            return {}
        try:
            cipher = Fernet(ENCRYPTION_KEY)
            decrypted_content = cipher.decrypt(content).decode()
            return json.loads(decrypted_content)
        except Exception:
            return json.loads(content.decode('utf-8'))
    except (json.JSONDecodeError, Exception) as e:
        print(f"Lỗi đọc file users.json: {e}")
        return {} 
def encrypt_file_content(content):
    cipher = Fernet(ENCRYPTION_KEY)
    return cipher.encrypt(content.encode())

def decrypt_file_content(encrypted_content):
    cipher = Fernet(ENCRYPTION_KEY)
    return cipher.decrypt(encrypted_content).decode()
def save_users(users):
    cipher = Fernet(ENCRYPTION_KEY)
    encrypted = cipher.encrypt(json.dumps(users).encode())
    with open(USERS_FILE, 'wb') as f:
        f.write(encrypted)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"
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
try:
    model_path = "peterhung/vietnamese-accent-marker-xlm-roberta"
    accent_tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    accent_model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accent_model.to(device)
    accent_model.eval()
except Exception as e:
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
        print(f"Accent processing error: {e}")
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
@limiter.limit("10 per hour")
@csrf.exempt 
def register():
    if request.method == 'POST':
        try:
            # Kiểm tra Content-Type
            if not request.is_json:
                return jsonify({"error": "Content-Type phải là application/json"}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({"error": "Dữ liệu JSON không hợp lệ"}), 400
            
            username = data.get('username', '').strip()
            password = data.get('password', '')
            otp = data.get('otp', '').strip()
            
            users = load_users()
            
            # Bước 1: Đăng ký thông tin và tạo QR code
            if not otp:
                if not username or not password:
                    return jsonify({"error": "Vui lòng điền đầy đủ thông tin"}), 400
                if len(username) < 3:
                    return jsonify({"error": "Username phải có ít nhất 3 ký tự"}), 400
                if not username.isalnum():
                    return jsonify({"error": "Username chỉ chứa chữ cái và số"}), 400
                if username in users:
                    return jsonify({"error": "Tên đăng nhập đã tồn tại"}), 400
                
                # Validate password strength
                is_valid, message = validate_password(password)
                if not is_valid:
                    return jsonify({"error": message}), 400
                
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

                users[temp_data['username']] = {
                    "password": temp_data['password'],
                    "secret_2fa": temp_data['secret_2fa'],
                    "created_at": datetime.now().isoformat()
                }
                save_users(users)
                
                # Xóa dữ liệu tạm
                session.pop('temp_register', None)
                
                return jsonify({"status": "success", "message": "Đăng ký thành công!"}), 200
        
        except Exception as e:
            logger.error(f"Register error: {e}")
            return jsonify({"error": f"Lỗi server: {str(e)}"}), 500
    
    return render_template_string(REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
@csrf.exempt
@limiter.limit("10 per hour")
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        otp = data.get('otp', '').strip()
        
        users = load_users()
        
        # Bước 1: Kiểm tra username và password
        if not otp:
            is_locked, locked_until = is_account_locked(username)
            if is_locked:
                minutes_left = int((locked_until - datetime.now()).total_seconds() / 60)
                return jsonify({
                    "error": f"Tài khoản đã bị khóa. Vui lòng thử lại sau {minutes_left} phút"
                }), 403
            if username not in users:
                record_failed_login(username)
                return jsonify({"error": "Tên đăng nhập không tồn tại"}), 400
            
            user = users[username]
            if hash_password(password) != user['password']:
                record_failed_login(username)
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
            reset_failed_login(temp_data['username'])
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
@csrf.exempt
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
            
            print(f"[{timestamp}] Speech: {latest}")
            return jsonify({
                "status": "success", 
                "type": "speech",
                "text": latest,
                "processed": latest  # Speech đã có dấu từ Whisper
            }), 200
        
        return jsonify({"status": "empty"}), 200
        
    except Exception as e:
        print(f"Speech error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_frame', methods=['POST'])
@login_required
@csrf.exempt
def upload_frame():
    """Nhận frame camera + metadata từ Sign Language client"""
    global latest_frame, client_stats
    try:
        frame_data = request.data
        
        # Try JPEG decode first (from optimized clients)
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG decode failed")
        except:
            # Fallback to pickle (from old clients)
            frame = pickle.loads(frame_data)
        
        with frame_lock:
            latest_frame = frame
        
        # Update client stats from headers
        with stats_lock:
            client_stats['fps'] = float(request.headers.get('X-Client-FPS', 0))
            client_stats['buffer'] = request.headers.get('X-Buffer', '')
            client_stats['predicted'] = request.headers.get('X-Predicted', '')
            client_stats['last_update'] = datetime.now()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Frame error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_sign', methods=['POST'])
@login_required
@csrf.exempt
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
        
        print(f"[{timestamp}] Sign: {raw_text} → {processed_text}")
        return jsonify({
            "status": "success",
            "type": "sign",
            "text": raw_text,
            "processed": processed_text
        }), 200
        
    except Exception as e:
        print(f"Sign error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/client_stats', methods=['GET'])
@login_required
def get_client_stats():
    """API để lấy thống kê client real-time"""
    with stats_lock:
        stats_copy = client_stats.copy()
        # Convert datetime to string
        if stats_copy['last_update']:
            stats_copy['last_update'] = stats_copy['last_update'].strftime('%H:%M:%S')
        return jsonify(stats_copy), 200

@app.route('/upload', methods=['POST'])
@login_required
def upload_legacy():
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
@login_required
def video_feed():
    """Endpoint để stream video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
@login_required
def index():
    """Render trang chủ với dữ liệu lịch sử"""
    # Lấy username từ session
    username = session.get('user', 'Unknown')
    
    # Sử dụng hàm get_index_html từ templates.py
    html_content = get_index_html(username, SPEECH_FOLDER, SIGN_FOLDER)
    
    return html_content
@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """API trả về lịch sử (dùng cho AJAX reload)"""
    speech_files = sorted(os.listdir(SPEECH_FOLDER), reverse=True)[:10]
    sign_files = sorted(os.listdir(SIGN_FOLDER), reverse=True)[:10]
    
    # Parse speech data
    speech_data = []
    for f in speech_files:
        try:
            with open(os.path.join(SPEECH_FOLDER, f), 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
                if lines:
                    content = lines[-1]
                    timestamp_str = f.replace('speech_', '').replace('.txt', '')
                    try:
                        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        time_display = dt.strftime('%H:%M:%S')
                    except:
                        time_display = timestamp_str
                    
                    speech_data.append({
                        'time': time_display,
                        'text': content
                    })
        except:
            pass
    
    # Parse sign data
    sign_data = []
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
                        time_display = dt.strftime('%H:%M:%S')
                    except:
                        time_display = timestamp_str
                    
                    sign_data.append({
                        'time': time_display,
                        'raw': raw,
                        'processed': processed
                    })
        except:
            pass
    
    return jsonify({
        'speech': speech_data,
        'sign': sign_data
    }), 200
@app.route('/api/status')
@limiter.limit("100 per minute")
def api_status():
    """Health check endpoint with rate limiting"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }), 200
LOGIN_HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Login</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',Arial;background:#f5f5f5;min-height:100vh;display:flex;align-items:center;justify-content:center}.container{background:#fff;padding:40px;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);max-width:450px;width:100%}h1{text-align:center;color:#667eea;margin-bottom:30px}input{width:100%;padding:12px;margin:10px 0;border:2px solid #e0e0e0;border-radius:10px}button{width:100%;padding:14px;background:#667eea;color:#fff;border:none;border-radius:10px;font-size:1.1em;cursor:pointer;margin-top:10px}button:hover{opacity:0.9}.msg{padding:10px;margin:10px 0;border-radius:8px;display:none}.msg.error{background:#fee;color:#c33}.msg.info{background:#eef;color:#33c}.otp-step{display:none}</style>
</head><body><div class="container"><h1>Login</h1><div id="msg" class="msg"></div>
<div id="cred-step"><input type="text" id="username" placeholder="Username">
<input type="password" id="password" placeholder="Password">
<button onclick="step1()">Continue</button></div>
<div id="otp-step" class="otp-step"><p style="text-align:center;margin-bottom:15px">Enter OTP from Authenticator</p>
<input type="text" id="otp" maxlength="6" placeholder="000000">
<button onclick="step2()">Verify</button></div>
<div style="text-align:center;margin-top:20px"><a href="/register" style="color:#667eea">Register</a></div></div>
<script>let step=1;async function step1(){const u=document.getElementById('username').value,p=document.getElementById('password').value,m=document.getElementById('msg');
try{const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})}),d=await r.json();
if(r.ok){if(d.status==='otp_required'){step=2;document.getElementById('cred-step').style.display='none';document.getElementById('otp-step').style.display='block';m.textContent=d.message;m.className='msg info';m.style.display='block'}else if(d.status==='success')window.location.href=d.redirect}else{m.textContent=d.error;m.className='msg error';m.style.display='block'}}catch(e){m.textContent='Error';m.className='msg error';m.style.display='block'}}
async function step2(){const o=document.getElementById('otp').value,m=document.getElementById('msg');
try{const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({otp:o})}),d=await r.json();
if(r.ok&&d.status==='success')window.location.href=d.redirect;else{m.textContent=d.error;m.className='msg error';m.style.display='block'}}catch(e){m.textContent='Error';m.className='msg error';m.style.display='block'}}</script></body></html>'''

REGISTER_HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Register</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial;background:#f5f5f5;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.container{background:#fff;padding:40px;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);max-width:500px;width:100%}
h1{text-align:center;color:#667eea;margin-bottom:30px}
input{width:100%;padding:12px;margin:10px 0;border:2px solid #e0e0e0;border-radius:10px;font-size:.95em}
input:focus{outline:none;border-color:#667eea}
button{width:100%;padding:14px;background:#667eea;color:#fff;border:none;border-radius:10px;font-size:1.1em;cursor:pointer;margin-top:10px;transition:all .3s}
button:hover{opacity:0.9;transform:translateY(-2px);box-shadow:0 4px 12px rgba(102,126,234,.4)}
button:active{transform:translateY(0)}
button:disabled{opacity:0.6;cursor:not-allowed}
.msg{padding:12px;margin:10px 0;border-radius:8px;display:none;font-size:.9em}
.msg.error{background:#fee;color:#c33;border-left:4px solid #c33}
.msg.success{background:#efe;color:#3c3;border-left:4px solid #3c3}
.msg.info{background:#eef;color:#33c;border-left:4px solid #33c}
.otp-step{display:none}
.qr{text-align:center;margin:20px 0;padding:20px;background:#f8f9ff;border-radius:10px}
.qr img{max-width:250px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,.1)}
.qr p{margin-top:10px;font-size:.9em;color:#666;word-break:break-all}
.password-requirements{background:#f8f9fa;padding:15px;border-radius:8px;margin:10px 0;font-size:.85em;border-left:4px solid #667eea}
.password-requirements strong{color:#333;display:block;margin-bottom:8px}
.password-requirements ul{margin:5px 0 0 20px;padding:0}
.password-requirements li{margin:5px 0;color:#666;line-height:1.6}
.form-footer{text-align:center;margin-top:20px;padding-top:20px;border-top:1px solid #e0e0e0}
.form-footer a{color:#667eea;text-decoration:none;font-weight:600;transition:color .3s}
.form-footer a:hover{color:#5568d3}
label{display:block;margin:10px 0 5px;color:#666;font-weight:600;font-size:.9em}
</style>
</head><body>
<div class="container">
<h1>Register</h1>
<div id="msg" class="msg"></div>

<div id="cred-step">
<label>Username</label>
<input type="text" id="username" placeholder="Enter username (min 3 chars)" minlength="3" required>

<label>Password</label>
<input type="password" id="password" placeholder="Enter password (min 8 chars)" minlength="8" required>

<div class="password-requirements">
<strong>Password must contain:</strong>
<ul>
<li>At least 8 characters</li>
<li>One uppercase letter (A-Z)</li>
<li>One lowercase letter (a-z)</li>
<li>One number (0-9)</li>
<li>One special character (!@#$%^&*)</li>
</ul>
</div>

<label>Confirm Password</label>
<input type="password" id="confirm" placeholder="Re-enter password" required>

<button onclick="step1()">Continue</button>
</div>

<div id="otp-step" class="otp-step">
<div class="qr" id="qr"></div>
<p style="text-align:center;margin-bottom:15px;color:#666">Scan QR code with Google Authenticator app</p>
<label>Enter OTP Code</label>
<input type="text" id="otp" maxlength="6" placeholder="000000" pattern="[0-9]{6}" required>
<button onclick="step2()">Activate Account</button>
<button onclick="backToStep1()" style="background:#6c757d;margin-top:5px">Back</button>
</div>

<div class="form-footer">
Already have an account? <a href="/login">Login here</a>
</div>
</div>

<script>
let step=1;

async function step1(){
    const u=document.getElementById('username').value.trim(),
          p=document.getElementById('password').value,
          c=document.getElementById('confirm').value,
          m=document.getElementById('msg');
    
    // Client-side validation
    if(!u || !p || !c){
        m.textContent='Please fill in all fields';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    if(u.length < 3){
        m.textContent='Username must be at least 3 characters';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    if(!/^[a-zA-Z0-9]+$/.test(u)){
        m.textContent='Username can only contain letters and numbers';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    if(p !== c){
        m.textContent='Passwords do not match';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    if(p.length < 8){
        m.textContent='Password must be at least 8 characters';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    // Show loading state
    const btn = event.target;
    const originalText = btn.textContent;
    btn.textContent = 'Processing...';
    btn.disabled = true;
    
    try{
        const res = await fetch('/register',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({username:u, password:p})
        });
        
        const d = await res.json();
        
        if(res.ok && d.status === 'qr_generated'){
            step = 2;
            document.getElementById('cred-step').style.display='none';
            document.getElementById('otp-step').style.display='block';
            document.getElementById('qr').innerHTML=`<img src="data:image/png;base64,${d.qr_code}" alt="QR Code"><p><strong>Secret Key:</strong> ${d.secret}</p><p style="font-size:.8em;color:#999;margin-top:10px">Save this secret key in case you lose access to your authenticator app</p>`;
            m.textContent=d.message;
            m.className='msg info';
            m.style.display='block';
            
            // Auto focus OTP input
            setTimeout(() => document.getElementById('otp').focus(), 300);
        }else{
            m.textContent=d.error || 'Registration failed';
            m.className='msg error';
            m.style.display='block';
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }catch(e){
        m.textContent='Network error: ' + e.message;
        m.className='msg error';
        m.style.display='block';
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

async function step2(){
    const o=document.getElementById('otp').value.trim(),
          m=document.getElementById('msg');
    
    if(!o || o.length !== 6){
        m.textContent='Please enter a valid 6-digit OTP code';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    if(!/^[0-9]{6}$/.test(o)){
        m.textContent='OTP must contain only numbers';
        m.className='msg error';
        m.style.display='block';
        return;
    }
    
    // Show loading state
    const btn = event.target;
    const originalText = btn.textContent;
    btn.textContent = 'Verifying...';
    btn.disabled = true;
    
    try{
        const res = await fetch('/register',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({otp:o})
        });
        
        const d = await res.json();
        
        if(res.ok && d.status === 'success'){
            m.textContent=d.message;
            m.className='msg success';
            m.style.display='block';
            
            // Disable all inputs
            document.getElementById('otp').disabled = true;
            btn.style.display = 'none';
            
            // Redirect after 2 seconds
            setTimeout(() => window.location.href='/login', 2000);
        }else{
            m.textContent=d.error || 'Verification failed';
            m.className='msg error';
            m.style.display='block';
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }catch(e){
        m.textContent='Network error: ' + e.message;
        m.className='msg error';
        m.style.display='block';
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

function backToStep1(){
    if(confirm('Are you sure? You will need to scan the QR code again.')){
        step = 1;
        document.getElementById('otp-step').style.display='none';
        document.getElementById('cred-step').style.display='block';
        document.getElementById('msg').style.display='none';
        
        // Clear OTP input
        document.getElementById('otp').value = '';
    }
}

// Auto-focus username on load
window.onload = () => {
    document.getElementById('username').focus();
};

// Allow Enter key to submit
document.addEventListener('keypress', (e) => {
    if(e.key === 'Enter'){
        if(step === 1){
            step1();
        }else if(step === 2){
            step2();
        }
    }
});
</script>
</body></html>'''
# ===== Run Server =====
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
    print("Unified Server - Speech & Sign Language")
    print("=" * 70)
    print(f" Speech endpoint:  http://{server_ip}:5000/upload_speech")
    print(f" Sign endpoint:    http://{server_ip}:5000/upload_sign")
    print(f" Camera endpoint:  http://{server_ip}:5000/upload_frame")
    print(f" Video stream:     http://{server_ip}:5000/video_feed")
    print(f" Legacy endpoint:  http://{server_ip}:5000/upload")
    print(f" Web UI:           http://{server_ip}:5000")
    print(f" Listening on:     0.0.0.0:5000 (all interfaces)")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)