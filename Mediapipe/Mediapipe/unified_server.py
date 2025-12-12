
import time
import pyotp
import qrcode
import io
import base64
from flask import Flask, request, jsonify, render_template_string, Response, session, redirect, url_for, render_template
from flask_sock import Sock
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
sock = Sock(app)
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

sign_processing_state = {
    'current_char': '',
    'current_word': '',
    'current_sentence': '',
    'word_buffer': [],
    'last_char_time': None
}
sign_state_lock = threading.Lock()

# Special tokens
SPECIAL_TOKENS = {
    'SPACE': ' ',
    'END': '.',
    'CLEAR': 'CLEAR',
    'RESET': 'RESET',
    'DELETE': 'DELETE'
}
communication_history = {
    'sign_language': [],
    'speech': []
}
history_lock = threading.Lock()

MAX_HISTORY_ITEMS = 50

app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
csrf = CSRFProtect(app)
# ===== Database Files =====
USERS_FILE = "users.json"
CLIENT_API_KEYS_FILE = "client_api_keys.json"

# ===== Client API Key Management =====
def load_api_keys():
    """Load client API keys"""
    if not os.path.exists(CLIENT_API_KEYS_FILE):
        # Create default API key for development
        default_keys = {
            "default_client_key_12345": {
                "name": "Default Client",
                "created": datetime.now().isoformat(),
                "active": True
            }
        }
        save_api_keys(default_keys)
        return default_keys
    
    try:
        with open(CLIENT_API_KEYS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_api_keys(keys):
    """Save client API keys"""
    with open(CLIENT_API_KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

def verify_api_key(api_key):
    """Verify client API key"""
    keys = load_api_keys()
    if api_key in keys:
        return keys[api_key].get('active', False)
    return False

# Decorator for API key authentication
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not verify_api_key(api_key):
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://",
    # Add these settings to prevent issues
    swallow_errors=True,  # Don't crash on rate limit storage errors
    headers_enabled=True,  # Show rate limit info in response headers
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

# ===== Sign Language Mapping Functions (NEW) =====
def load_sign_language_mapping():
    """Load sign language mapping from JSON file"""
    mapping_file = "sign_mapping.json"
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading sign mapping: {e}")
            return {}
    else:
        logger.warning(f"Sign mapping file not found: {mapping_file}")
        return {}

# Load mapping once when server starts
SIGN_MAPPING = load_sign_language_mapping()
logger.info(f"Loaded {len(SIGN_MAPPING)} sign language mappings")

def normalize_sign_language_text(raw_text):
    """
    Normalize concatenated sign-language text to proper Vietnamese.
    Uses JSON mapping file for direct lookup, falls back to dynamic segmentation.
    Returns ONLY the corrected sentence with accents, spaces, and punctuation.
    """
    if not raw_text:
        return raw_text
    
    # Convert to lowercase and strip whitespace
    text = raw_text.lower().strip()
    
    # Remove all spaces for lookup (handle both with and without spaces)
    text_no_spaces = text.replace(' ', '')
    
    # Direct lookup in mapping
    if text_no_spaces in SIGN_MAPPING:
        result = SIGN_MAPPING[text_no_spaces]
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
        
        # Add period at the end if not present
        if result and result[-1] not in '.?!':
            result += '.'
        
        return result
    
    # Fallback: Try word-by-word segmentation using the mapping as a dictionary
    n = len(text_no_spaces)
    
    # Dynamic programming for word segmentation
    dp = [None] * (n + 1)
    dp[0] = []
    
    for i in range(1, n + 1):
        # Try all possible word lengths (1 to 20 characters for longer phrases)
        for j in range(max(0, i - 20), i):
            word = text_no_spaces[j:i]
            
            # Check if this substring is in our mapping
            if word in SIGN_MAPPING and dp[j] is not None:
                if dp[i] is None or len(dp[j]) + 1 > len(dp[i]):
                    # Get the mapped value but keep it as a single word for segmentation
                    dp[i] = dp[j] + [SIGN_MAPPING[word]]
    
    # If we successfully segmented the entire text
    if dp[n] is not None and len(dp[n]) > 0:
        # Join all the mapped segments
        result = ' '.join(dp[n])
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
        
        # Add period at the end if not present
        if result and result[-1] not in '.?!':
            result += '.'
        
        return result
    
    # If segmentation failed, return original text unchanged
    return raw_text

def reload_sign_mapping():
    """Reload the sign mapping from file (useful for updates without restart)"""
    global SIGN_MAPPING
    SIGN_MAPPING = load_sign_language_mapping()
    logger.info(f"Reloaded {len(SIGN_MAPPING)} sign language mappings")
    return len(SIGN_MAPPING)

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

def process_character(char):
    """
    Process incoming character from sign language recognition
    Returns: dict with current_char, current_word, current_sentence, processed_text
    """
    global sign_processing_state
    
    with sign_state_lock:
        state = sign_processing_state
        
        # Update current character
        state['current_char'] = char
        state['last_char_time'] = datetime.now()
        
        # Handle special tokens
        if char == 'SPACE':
            # Finalize current word
            if state['word_buffer']:
                word = ''.join(state['word_buffer'])
                state['word_buffer'] = []
                state['current_word'] = ''
                
                # Add word to sentence
                if state['current_sentence']:
                    state['current_sentence'] += ' ' + word
                else:
                    state['current_sentence'] = word
        
        elif char == 'END':
            # Finalize word and sentence
            if state['word_buffer']:
                word = ''.join(state['word_buffer'])
                if state['current_sentence']:
                    state['current_sentence'] += ' ' + word
                else:
                    state['current_sentence'] = word
                state['word_buffer'] = []
                state['current_word'] = ''
            
            # Add period
            if state['current_sentence']:
                state['current_sentence'] += '.'
        
        elif char == 'CLEAR' or char == 'RESET':
            # Reset everything
            state['current_char'] = ''
            state['current_word'] = ''
            state['current_sentence'] = ''
            state['word_buffer'] = []
        
        elif char == 'DELETE':
            # Remove last character from word buffer
            if state['word_buffer']:
                state['word_buffer'].pop()
                state['current_word'] = ''.join(state['word_buffer'])
        
        elif char and len(char) == 1 and char.isalpha():
            # Regular character - add to word buffer
            state['word_buffer'].append(char.lower())
            state['current_word'] = ''.join(state['word_buffer'])
        
        # Prepare response
        response = {
            'current_char': state['current_char'],
            'current_word': state['current_word'],
            'current_sentence': state['current_sentence'],
            'processed_text': state['current_sentence']
        }
        
        return response

def get_processing_state():
    """Get current processing state (thread-safe)"""
    with sign_state_lock:
        return {
            'current_char': sign_processing_state['current_char'],
            'current_word': sign_processing_state['current_word'],
            'current_sentence': sign_processing_state['current_sentence'],
            'last_update': sign_processing_state['last_char_time'].isoformat() if sign_processing_state['last_char_time'] else None
        }

def add_to_history(type, text):
    """
    Add entry to communication history
    type: 'sign_language' or 'speech'
    text: the processed text
    """
    if not text or not text.strip():
        return
    
    with history_lock:
        entry = {
            'timestamp': datetime.now().isoformat(),
            'time_display': datetime.now().strftime('%H:%M:%S'),
            'text': text.strip()
        }
        
        # Add to appropriate history list
        if type in communication_history:
            communication_history[type].append(entry)
            
            # Keep only recent entries (prevent memory overflow)
            if len(communication_history[type]) > MAX_HISTORY_ITEMS:
                communication_history[type] = communication_history[type][-MAX_HISTORY_ITEMS:]
        
        # Also save to file for persistence
        try:
            if type == 'sign_language':
                filename = f"sign_history_{datetime.now().strftime('%Y%m%d')}.txt"
                filepath = os.path.join(SIGN_FOLDER, filename)
            else:
                filename = f"speech_history_{datetime.now().strftime('%Y%m%d')}.txt"
                filepath = os.path.join(SPEECH_FOLDER, filename)
            
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"[{entry['time_display']}] {text}\n")
        except Exception as e:
            print(f"Error saving history to file: {e}")
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
@limiter.limit("10 per hour", methods=['POST'])  # Only limit POST requests
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
    
    # GET request - just render the page (no rate limit)
    return render_template_string(REGISTER_HTML)


@app.route('/login', methods=['GET', 'POST'])
@csrf.exempt
@limiter.limit("10 per hour", methods=['POST'])  # Only limit POST requests
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
    
    # GET request - just render the page (no rate limit)
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ===== Protected Routes =====

@app.route('/upload_speech', methods=['POST'])
@api_key_required
@csrf.exempt
@limiter.exempt
def upload_speech():
    """Receive transcription file from Whisper and save to history"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        content = file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if lines:
            latest = lines[-1]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to file
            filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(SPEECH_FOLDER, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            # ADD TO HISTORY
            add_to_history('speech', latest)
            
            print(f"[{timestamp}] Speech: {latest} | Saved to history")
            return jsonify({
                "status": "success", 
                "type": "speech",
                "text": latest,
                "processed": latest
            }), 200
        
        return jsonify({"status": "empty"}), 200
        
    except Exception as e:
        print(f"Speech error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload_frame', methods=['POST'])
@api_key_required  # Dùng API key thay vì session login
@csrf.exempt
@limiter.exempt  # EXEMPT khỏi rate limiting - cần real-time
def upload_frame():
    """Nhận frame camera + metadata từ Sign Language client - OPTIMIZED"""
    global latest_frame, client_stats
    try:
        # OPTIMIZATION 1: Chỉ decode JPEG (bỏ pickle fallback để nhanh hơn)
        frame_data = request.data
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400
        
        # OPTIMIZATION 2: Non-blocking frame update
        with frame_lock:
            nparr = np.frombuffer(frame_data, np.uint8)
            latest_frame = frame
        
        # OPTIMIZATION 3: Minimal stats update (chỉ metadata từ headers)
        with stats_lock:
            client_stats['fps'] = float(request.headers.get('X-Client-FPS', 0))
            client_stats['buffer'] = request.headers.get('X-Buffer', '')
            client_stats['predicted'] = request.headers.get('X-Predicted', '')
            client_stats['last_update'] = datetime.now()
        
        # OPTIMIZATION 4: Trả về response ngay lập tức
        return '', 200  # Empty response nhanh hơn jsonify
        
    except Exception as e:
        # Silent fail - không log quá nhiều
        return '', 500

@app.route('/upload_sign', methods=['POST'])
@api_key_required  # Dùng API key thay vì session login
@csrf.exempt
@limiter.exempt  # EXEMPT - core functionality
def upload_sign():
    """Nhận text không dấu từ Sign Language và xử lý"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text"}), 400
        
        raw_text = data['text']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Xử lý thêm dấu
        processed_text = normalize_sign_language_text(raw_text)
        
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

@app.route('/process_char', methods=['POST'])
@api_key_required
@csrf.exempt
@limiter.exempt
def process_char():
    """
    Process single character from sign language recognition
    Returns processed state for LCD display
    """
    try:
        data = request.get_json()
        if not data or 'char' not in data:
            return jsonify({"error": "No character provided"}), 400
        
        char = data['char'].strip()
        
        # Process character
        result = process_character(char)
        
        # Log for debugging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Char: {char} -> Word: {result['current_word']} | Sentence: {result['current_sentence']}")
        
        # Save to file (append mode for history)
        filename = f"sign_chars_{datetime.now().strftime('%Y%m%d')}.txt"
        filepath = os.path.join(SIGN_FOLDER, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {char} -> {result['current_word']} | {result['current_sentence']}\n")
        
        return jsonify({
            "status": "success",
            "current_char": result['current_char'],
            "current_word": result['current_word'],
            "current_sentence": result['current_sentence'],
            "processed_text": result['processed_text']
        }), 200
        
    except Exception as e:
        print(f"Char processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sign_state', methods=['GET'])
@login_required
@limiter.exempt
def api_sign_state():
    """Get current sign language processing state for dashboard"""
    state = get_processing_state()
    return jsonify(state), 200



@app.route('/client_stats', methods=['GET'])
@login_required
@limiter.exempt  # EXEMPT - gọi mỗi giây
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
@limiter.exempt  # EXEMPT - video streaming continuous
def video_feed():
    """Endpoint để stream video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
@login_required
@limiter.exempt  # EXEMPT - dashboard chính
def index():
    """Render trang chủ - SỬ DỤNG TEMPLATE (tối ưu hơn generate HTML)"""
    username = session.get('user', 'Unknown')
    
    # Lấy dữ liệu speech
    speech_files = sorted(os.listdir(SPEECH_FOLDER), reverse=True)[:10]
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
                    speech_data.append({'time': time_display, 'text': content})
        except:
            pass
    
    # Lấy dữ liệu sign
    sign_files = sorted(os.listdir(SIGN_FOLDER), reverse=True)[:10]
    sign_data = []
    for f in sign_files:
        try:
            with open(os.path.join(SIGN_FOLDER, f), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                raw = processed = ""
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
                    sign_data.append({'time': time_display, 'raw': raw, 'processed': processed})
        except:
            pass
    current_sign_state = get_processing_state()
    # SỬ DỤNG TEMPLATE thay vì generate HTML string - nhanh hơn rất nhiều!
    return render_template('index.html',
                          username=username,
                          speech_count=len(speech_files),
                          sign_count=len(sign_files),
                          speech_data=speech_data,
                          sign_data=sign_data,
                          current_sign_state=current_sign_state)


@app.route('/api/history', methods=['GET'])
@login_required
@limiter.exempt
def api_history():
    """API returns history from memory (real-time) + recent files (fallback)"""
    
    # Get from memory first (faster)
    with history_lock:
        speech_data = list(reversed(communication_history['speech'][-10:]))
        sign_data = list(reversed(communication_history['sign_language'][-10:]))
    
    # ONLY USE FILE FALLBACK IF MEMORY IS EMPTY AND FILES EXIST
    if not speech_data:
        try:
            speech_files = sorted(os.listdir(SPEECH_FOLDER), reverse=True)[:10]
            for f in speech_files:
                if not f.startswith('speech_'):
                    continue
                    
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
        except FileNotFoundError:
            pass  # Folder might be empty after deletion
    else:
        # Convert memory format to API format
        speech_data = [{'time': item['time_display'], 'text': item['text']} for item in speech_data]
    
    if not sign_data:
        try:
            sign_files = sorted(os.listdir(SIGN_FOLDER), reverse=True)[:10]
            for f in sign_files:
                if not f.startswith('sign_'):
                    continue
                    
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
        except FileNotFoundError:
            pass  # Folder might be empty after deletion
    else:
        # Convert memory format to API format
        sign_data = [{'time': item['time_display'], 'raw': '', 'processed': item['text']} for item in sign_data]
    
    # Create response with no-cache headers
    response = jsonify({
        'speech': speech_data,
        'sign': sign_data
    })
    
    # CRITICAL: Add no-cache headers
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response, 200

@app.route('/api/status')
@limiter.limit("100 per minute")
def api_status():
    """Health check endpoint with rate limiting"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/history_count', methods=['GET'])
@login_required
@limiter.exempt
def api_history_count():
    """Get count of history items"""
    with history_lock:
        return jsonify({
            'sign_language': len(communication_history['sign_language']),
            'speech': len(communication_history['speech'])
        }), 200
@app.route('/api/history/all', methods=['DELETE'])
@api_key_required
@csrf.exempt
@limiter.exempt
def delete_all_history():
    """Clear all history (sign, speech, and char state)"""
    try:
        with history_lock:
            communication_history['sign_language'].clear()
            communication_history['speech'].clear()
        
        with sign_state_lock:
            sign_processing_state['current_char'] = ''
            sign_processing_state['current_word'] = ''
            sign_processing_state['current_sentence'] = ''
            sign_processing_state['word_buffer'] = []
            sign_processing_state['last_char_time'] = None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Cleared ALL history via API request")
        
        return jsonify({
            "status": "success",
            "message": "All history cleared"
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing all history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/web/history/sign', methods=['DELETE'])
@login_required
@csrf.exempt
@limiter.exempt
def web_delete_sign_history():
    """Clear sign language history (memory + files)"""
    try:
        # Clear in-memory history
        with history_lock:
            communication_history['sign_language'].clear()
        
        # Clear sign processing state
        with sign_state_lock:
            sign_processing_state['current_char'] = ''
            sign_processing_state['current_word'] = ''
            sign_processing_state['current_sentence'] = ''
            sign_processing_state['word_buffer'] = []
            sign_processing_state['last_char_time'] = None
        
        # DELETE ALL SIGN FILES FROM DISK
        try:
            for filename in os.listdir(SIGN_FOLDER):
                if filename.startswith('sign_') or filename.startswith('sign_chars_'):
                    filepath = os.path.join(SIGN_FOLDER, filename)
                    os.remove(filepath)
                    logger.info(f"Deleted file: {filepath}")
        except Exception as file_error:
            logger.warning(f"Error deleting sign files: {file_error}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Cleared sign language history (memory + files) via web dashboard")
        
        response = jsonify({
            "status": "success",
            "message": "Lịch sử ký hiệu đã được xóa"
        })
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response, 200
        
    except Exception as e:
        logger.error(f"Error clearing sign history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/web/history/speech', methods=['DELETE'])
@login_required
@csrf.exempt
@limiter.exempt
def web_delete_speech_history():
    """Clear speech history (memory + files)"""
    try:
        # Clear in-memory history
        with history_lock:
            communication_history['speech'].clear()
        
        # DELETE ALL SPEECH FILES FROM DISK
        try:
            for filename in os.listdir(SPEECH_FOLDER):
                if filename.startswith('speech_'):
                    filepath = os.path.join(SPEECH_FOLDER, filename)
                    os.remove(filepath)
                    logger.info(f"Deleted file: {filepath}")
        except Exception as file_error:
            logger.warning(f"Error deleting speech files: {file_error}")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Cleared speech history (memory + files) via web dashboard")
        
        response = jsonify({
            "status": "success",
            "message": "Lịch sử giọng nói đã được xóa"
        })
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response, 200
        
    except Exception as e:
        logger.error(f"Error clearing speech history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/web/history/char', methods=['DELETE'])
@login_required
@csrf.exempt
@limiter.exempt
def web_delete_char_history():
    """Clear character processing state"""
    try:
        # Clear character processing state
        with sign_state_lock:
            sign_processing_state['current_char'] = ''
            sign_processing_state['current_word'] = ''
            sign_processing_state['current_sentence'] = ''
            sign_processing_state['word_buffer'] = []
            sign_processing_state['last_char_time'] = None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Cleared character processing state via web dashboard")
        
        response = jsonify({
            "status": "success",
            "message": "Trạng thái ký tự đã được xóa"
        })
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response, 200
        
    except Exception as e:
        logger.error(f"Error clearing char state: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/client_keys', methods=['GET'])
@login_required
def get_client_keys():
    """Get all client API keys (admin only)"""
    keys = load_api_keys()
    # Hide sensitive parts
    safe_keys = {}
    for key, info in keys.items():
        safe_keys[key[:8] + "..." + key[-4:]] = {
            "name": info.get("name", "Unknown"),
            "created": info.get("created", "Unknown"),
            "active": info.get("active", False)
        }
    return jsonify(safe_keys), 200

@app.route('/api/client_keys/generate', methods=['POST'])
@login_required
def generate_client_key():
    """Generate new client API key (admin only)"""
    try:
        data = request.get_json()
        name = data.get('name', 'New Client')
        
        # Generate random API key
        import secrets
        new_key = f"client_key_{secrets.token_urlsafe(32)}"
        
        keys = load_api_keys()
        keys[new_key] = {
            "name": name,
            "created": datetime.now().isoformat(),
            "active": True
        }
        save_api_keys(keys)
        
        return jsonify({
            "status": "success",
            "api_key": new_key,
            "name": name
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


LOGIN_HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Login</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}

:root {
    --primary: #5882aa;
    --primary-dark: #466786;
}

/* Background */
body{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-image: url("/static/images/Background.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
}

/* GLASS CONTAINER */
.container{
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding:40px;
    border-radius:20px;
    box-shadow:0 10px 40px rgba(0,0,0,.25);
    max-width:450px;
    width:100%;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Title */
h1{
    text-align:center;
    color:var(--primary);
    margin-bottom:30px
}

/* Input fields */
input{
    width:100%;
    padding:12px;
    margin:10px 0;
    border:2px solid var(--primary);
    border-radius:10px;
    background: rgba(255,255,255,0.6);
}
input:focus{
    border-color:var(--primary-dark);
    box-shadow:0 0 6px rgba(88,130,170,0.6);
}

/* Buttons */
button{
    width:100%;
    padding:14px;
    background:var(--primary);
    color:#fff;
    border:none;
    border-radius:10px;
    font-size:1.1em;
    cursor:pointer;
    margin-top:10px;
    transition:0.25s;
}
button:hover{
    background:var(--primary-dark);
}

/* Messages */
.msg{
    padding:10px;
    margin:10px 0;
    border-radius:8px;
    display:none;
}
.msg.error{
    background:rgba(255,80,80,0.2);
    color:#c0392b;
}
.msg.info{
    background:rgba(88,130,170,0.15);
    color:var(--primary-dark);
}

/* OTP step hidden initially */
.otp-step{display:none}

/* Register link */
a{
    color:var(--primary);
    font-weight:600;
}
a:hover{
    color:var(--primary-dark);
}
</style>
</head>

<body>
<div class="container">
    <h1>Login</h1>
    <div id="msg" class="msg"></div>

    <div id="cred-step">
        <input type="text" id="username" placeholder="Username">
        <input type="password" id="password" placeholder="Password">
        <button onclick="step1()">Continue</button>
    </div>

    <div id="otp-step" class="otp-step">
        <p style="text-align:center;margin-bottom:15px">Enter OTP from Authenticator</p>
        <input type="text" id="otp" maxlength="6" placeholder="000000">
        <button onclick="step2()">Verify</button>
    </div>

    <div style="text-align:center;margin-top:20px">
        <a href="/register">Register</a>
    </div>
</div>

<script>
let step=1;

async function step1(){
    const u=document.getElementById('username').value,
          p=document.getElementById('password').value,
          m=document.getElementById('msg');

    try{
        const r=await fetch('/login',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({username:u,password:p})
        }),
        d=await r.json();

        if(r.ok){
            if(d.status==='otp_required'){
                step=2;
                document.getElementById('cred-step').style.display='none';
                document.getElementById('otp-step').style.display='block';
                m.textContent=d.message;
                m.className='msg info';
                m.style.display='block';
            } else if(d.status==='success'){
                window.location.href=d.redirect;
            }
        } else {
            m.textContent=d.error;
            m.className='msg error';
            m.style.display='block';
        }
    } catch(e){
        m.textContent='Error';
        m.className='msg error';
        m.style.display='block';
    }
}

async function step2(){
    const o=document.getElementById('otp').value,
          m=document.getElementById('msg');

    try{
        const r=await fetch('/login',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({otp:o})
        }),
        d=await r.json();

        if(r.ok && d.status==='success'){
            window.location.href=d.redirect;
        } else {
            m.textContent=d.error;
            m.className='msg error';
            m.style.display='block';
        }
    } catch(e){
        m.textContent='Error';
        m.className='msg error';
        m.style.display='block';
    }
}
async function updateSignState() {
    try {
        const response = await fetch('/api/sign_state');
        if (response.ok) {
            const state = await response.json();
            
            // Update DOM elements
            document.getElementById('current-char').textContent = state.current_char || '-';
            document.getElementById('current-word').textContent = state.current_word || '-';
            document.getElementById('current-sentence').textContent = state.current_sentence || '-';
        }
    } catch (error) {
        console.error('Error fetching sign state:', error);
    }
}

// Update sign state every 500ms
setInterval(updateSignState, 500);

</script>
</body></html>'''

REGISTER_HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Register</title>

<style>
*{margin:0;padding:0;box-sizing:border-box}

:root{
    --primary:#5882aa;
    --primary-dark:#466786;
}

/* Background */
body{
    font-family:'Segoe UI', system-ui, sans-serif;
    background-image:url("/static/images/Background.png");
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:20px;
}

/* Glass container */
.container{
    background:rgba(255,255,255,0.35);
    backdrop-filter:blur(12px);
    -webkit-backdrop-filter:blur(12px);
    padding:40px;
    border-radius:20px;
    max-width:500px;
    width:100%;
    border:1px solid rgba(255,255,255,0.3);
    box-shadow:0 10px 40px rgba(0,0,0,.25);
}

/* Title */
h1{
    text-align:center;
    color:var(--primary);
    margin-bottom:30px;
}

/* Inputs */
input{
    width:100%;
    padding:12px;
    margin:10px 0;
    border:2px solid var(--primary);
    border-radius:10px;
    background:rgba(255,255,255,0.6);
    font-size:.95em;
}
input:focus{
    outline:none;
    border-color:var(--primary-dark);
    box-shadow:0 0 6px rgba(88,130,170,0.5);
}

/* Labels */
label{
    display:block;
    margin:10px 0 5px;
    color:#333;
    font-weight:600;
    font-size:.9em;
}

/* Buttons */
button{
    width:100%;
    padding:14px;
    background:var(--primary);
    color:#fff;
    border:none;
    border-radius:10px;
    font-size:1.1em;
    cursor:pointer;
    margin-top:10px;
    transition:0.25s;
}
button:hover{
    background:var(--primary-dark);
    transform:translateY(-2px);
}
button:active{
    transform:translateY(0);
}
button:disabled{
    opacity:0.6;
    cursor:not-allowed;
}

/* Back button */
button.back{
    background:#6c757d;
}
button.back:hover{
    background:#5a6268;
}

/* Messages */
.msg{
    padding:12px;
    margin:10px 0;
    border-radius:8px;
    display:none;
    font-size:.9em;
}
.msg.error{
    background:rgba(255,80,80,0.2);
    color:#c0392b;
    border-left:4px solid #c0392b;
}
.msg.success{
    background:rgba(80,255,80,0.2);
    color:#2e8b57;
    border-left:4px solid #2e8b57;
}
.msg.info{
    background:rgba(88,130,170,0.15);
    color:var(--primary-dark);
    border-left:4px solid var(--primary-dark);
}

/* QR */
.qr{
    text-align:center;
    margin:20px 0;
    padding:20px;
    background:rgba(255,255,255,0.4);
    border-radius:10px;
}
.qr img{
    max-width:250px;
    border-radius:10px;
    box-shadow:0 4px 12px rgba(0,0,0,.2);
}

/* Password requirements */
.password-requirements{
    background:rgba(255,255,255,0.5);
    padding:15px;
    border-radius:8px;
    margin:10px 0;
    font-size:.85em;
    border-left:4px solid var(--primary);
}
.password-requirements strong{
    color:#333;
}

/* Footer */
.form-footer{
    text-align:center;
    margin-top:20px;
    padding-top:20px;
    border-top:1px solid rgba(255,255,255,0.4);
}
.form-footer a{
    color:var(--primary);
    font-weight:600;
    text-decoration:none;
}
.form-footer a:hover{
    color:var(--primary-dark);
}

/* Hide OTP step initially */
.otp-step{display:none}

</style>
</head>

<body>
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
<li>One uppercase letter</li>
<li>One lowercase letter</li>
<li>One number</li>
<li>One special character (!@#$%^&*)</li>
</ul>
</div>

<label>Confirm Password</label>
<input type="password" id="confirm" placeholder="Re-enter password" required>

<button onclick="step1()">Continue</button>
</div>

<div id="otp-step" class="otp-step">
<div class="qr" id="qr"></div>
<p style="text-align:center;margin-bottom:15px;color:#333">Scan QR code using Google Authenticator</p>

<label>Enter OTP</label>
<input type="text" id="otp" maxlength="6" placeholder="000000" pattern="[0-9]{6}" required>

<button onclick="step2()">Activate Account</button>
<button class="back" onclick="backToStep1()">Back</button>
</div>

<div class="form-footer">
Already have an account? <a href="/login">Login here</a>
</div>

</div>

<script>
let step = 1;

async function step1() {
    const u = document.getElementById('username').value.trim();
    const p = document.getElementById('password').value;
    const c = document.getElementById('confirm').value;
    const m = document.getElementById('msg');

    // Client-side validation
    if (!u || !p || !c) {
        m.textContent = 'Please fill all fields';
        m.className = 'msg error';
        m.style.display = 'block';
        return;
    }

    if (p !== c) {
        m.textContent = 'Passwords do not match';
        m.className = 'msg error';
        m.style.display = 'block';
        return;
    }

    if (u.length < 3) {
        m.textContent = 'Username must be at least 3 characters';
        m.className = 'msg error';
        m.style.display = 'block';
        return;
    }

    if (p.length < 8) {
        m.textContent = 'Password must be at least 8 characters';
        m.className = 'msg error';
        m.style.display = 'block';
        return;
    }

    try {
        const r = await fetch('/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: u, password: p})
        });
        
        const d = await r.json();

        if (r.ok) {
            if (d.status === 'qr_generated') {
                // Show QR code step
                step = 2;
                document.getElementById('cred-step').style.display = 'none';
                document.getElementById('otp-step').style.display = 'block';
                
                // Display QR code
                document.getElementById('qr').innerHTML = 
                    '<img src="data:image/png;base64,' + d.qr_code + '" alt="QR Code">';
                
                m.textContent = d.message;
                m.className = 'msg info';
                m.style.display = 'block';
            }
        } else {
            m.textContent = d.error || 'Registration failed';
            m.className = 'msg error';
            m.style.display = 'block';
        }
    } catch (e) {
        m.textContent = 'Network error. Please try again.';
        m.className = 'msg error';
        m.style.display = 'block';
    }
}

async function step2() {
    const o = document.getElementById('otp').value.trim();
    const m = document.getElementById('msg');

    if (!o || o.length !== 6) {
        m.textContent = 'Please enter a valid 6-digit OTP';
        m.className = 'msg error';
        m.style.display = 'block';
        return;
    }

    try {
        const r = await fetch('/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({otp: o})
        });
        
        const d = await r.json();

        if (r.ok && d.status === 'success') {
            m.textContent = d.message + ' Redirecting to login...';
            m.className = 'msg success';
            m.style.display = 'block';
            
            // Redirect to login after 2 seconds
            setTimeout(() => {
                window.location.href = '/login';
            }, 2000);
        } else {
            m.textContent = d.error || 'OTP verification failed';
            m.className = 'msg error';
            m.style.display = 'block';
        }
    } catch (e) {
        m.textContent = 'Network error. Please try again.';
        m.className = 'msg error';
        m.style.display = 'block';
    }
}

function backToStep1() {
    step = 1;
    document.getElementById('otp-step').style.display = 'none';
    document.getElementById('cred-step').style.display = 'block';
    document.getElementById('msg').style.display = 'none';
}
</script>

</body></html>'''

@sock.route('/ws/stream')
def websocket_stream(ws):
    """
    WebSocket endpoint for real-time frame streaming AND character processing
    Handles both binary frames and JSON character messages
    """
    global latest_frame, client_stats, sign_processing_state
    
    print("WebSocket client connected")
    
    try:
        while True:
            data = ws.receive()
            if data is None:
                break
            
            # Check if data is JSON (text message) or binary (frame)
            if isinstance(data, str):
                # Text message - character processing
                try:
                    msg = json.loads(data)
                    
                    if 'char' in msg:
                        char = msg['char']
                        
                        # Process character (builds word/sentence)
                        result = process_character(char)
                        
                        # Apply accent processing to the current sentence
                        final_sentence = normalize_sign_language_text(result['current_sentence'])
                        
                        # Send response back to client
                        ws.send(json.dumps({
                            "status": "success",
                            "current_char": result['current_char'],
                            "current_word": result['current_word'],
                            "current_sentence": result['current_sentence'],
                            "processed_text": final_sentence
                        }))
                        
                        # CONDITIONAL RESET + HISTORY UPDATE: Only on SPACE or END tokens
                        if char in ['SPACE', 'END']:
                            # Add to history if there's a complete sentence
                            if final_sentence and final_sentence.strip():
                                add_to_history('sign_language', final_sentence)
                            
                            # Reset state
                            with sign_state_lock:
                                sign_processing_state['current_char'] = ""
                                sign_processing_state['current_word'] = ""
                                sign_processing_state['current_sentence'] = ""
                                sign_processing_state['word_buffer'] = []
                            
                            # Log
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{timestamp}] Sign Language: '{final_sentence}' | Saved to history | State RESET")
                        else:
                            # Log without reset
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{timestamp}] WS Char: '{char}' -> Word: '{result['current_word']}' | Sentence: '{result['current_sentence']}'")
                
                except json.JSONDecodeError:
                    pass
            
            else:
                # Binary data - frame processing (unchanged)
                try:
                    newline_idx = data.find(b'\n')
                    if newline_idx == -1:
                        frame_bytes = data
                        fps = 0.0
                        buffer_text = ""
                        predicted = ""
                    else:
                        metadata_str = data[:newline_idx].decode('utf-8', errors='ignore')
                        frame_bytes = data[newline_idx + 1:]
                        
                        fps = 0.0
                        buffer_text = ""
                        predicted = ""
                        
                        for part in metadata_str.split('|'):
                            if ':' in part:
                                key, value = part.split(':', 1)
                                if key == 'FPS':
                                    fps = float(value) if value else 0.0
                                elif key == 'BUF':
                                    buffer_text = value
                                elif key == 'PRED':
                                    predicted = value
                    
                    # Decode frame
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        with frame_lock:
                            latest_frame = frame
                        
                        with stats_lock:
                            client_stats['fps'] = fps
                            client_stats['buffer'] = buffer_text
                            client_stats['predicted'] = predicted
                            client_stats['last_update'] = datetime.now()
                    
                except:
                    pass
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket client disconnected")

@app.route('/api/reload_mapping', methods=['POST'])
@login_required
@csrf.exempt
def reload_mapping():
    """Reload sign language mapping from file without restarting server"""
    try:
        count = reload_sign_mapping()
        return jsonify({
            "status": "success",
            "message": f"Reloaded {count} sign language mappings"
        }), 200
    except Exception as e:
        logger.error(f"Error reloading mapping: {e}")
        return jsonify({"error": str(e)}), 500

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
    print(f" Web UI:           http://{server_ip}:5000")
    print(f" Default API Key: default_client_key_12345")
    print(f" (Change in client with --api-key argument)")
    print(f"")
    print(f" Listening on:     0.0.0.0:5000 (all interfaces)")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)