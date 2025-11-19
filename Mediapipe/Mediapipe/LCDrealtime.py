#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel
import threading, os, time, textwrap
from PIL import Image, ImageDraw, ImageFont
import requests
from collections import deque

# ===== Cấu hình âm thanh =====
RECORD_RATE = 44100  # Sample rate của thiết bị
WHISPER_RATE = 16000  # Whisper yêu cầu 16kHz
CHANNELS = 1  # USB PnP Sound Device: 1 in (Mono)
DEVICE = 3  # Device 3: USB PnP Sound Device
CHUNK_DURATION = 3.0  # Ghi 3 giây → transcribe → lặp lại

# ===== LCD =====
FB1 = "/dev/fb1"
W, H = 480, 320
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 22

# ===== Khởi tạo mô hình =====
model = WhisperModel("/home/pi/PhoWhisper-tiny-ct2", device="cpu", compute_type="int8")

# ===== Lịch sử =====
HISTORY_FILE = "history.txt"
SERVER_URL = "http://192.168.10.117:5000/upload_speech"  # Unified server endpoint
MAX_HISTORY = 3
history = []
history_lock = threading.Lock()

# ===== Trạng thái =====
is_running = False
processing = False

# ===== Hàm ghi framebuffer =====
def write_to_fb(img):
    if img.size != (W, H):
        img = img.resize((W, H))
    arr = np.array(img)
    r = (arr[:, :, 0] >> 3).astype(np.uint16)
    g = (arr[:, :, 1] >> 2).astype(np.uint16)
    b = (arr[:, :, 2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    with open(FB1, "wb") as f:
        rgb565.tofile(f)

# ===== Hiển thị lịch sử =====
def show_recent_history(history, status=""):
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    
    # Status
    if status:
        status_font = ImageFont.truetype(FONT_PATH, 18)
        bbox = draw.textbbox((0, 0), status, font=status_font)
        w = bbox[2] - bbox[0]
        draw.text(((W - w) // 2, 10), status, font=status_font, fill="blue")
        y = 50
    else:
        y = 10
    
    # History
    with history_lock:
        for line in history:
            wrapped = textwrap.fill(line, width=25)
            for subline in wrapped.split("\n"):
                bbox = draw.textbbox((0, 0), subline, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                x = (W - w) // 2
                draw.text((x, y), subline, font=font, fill="black")
                y += h + 4
            y += 8
    
    write_to_fb(img)

# ===== Resample audio từ 44100Hz → 16000Hz =====
def resample_audio(audio_data, orig_rate, target_rate):
    """Resample đơn giản bằng linear interpolation"""
    # Convert stereo to mono nếu cần
    if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
        audio_data = np.mean(audio_data, axis=1).astype(np.int16)
    else:
        audio_data = audio_data.flatten()
    
    if orig_rate == target_rate:
        return audio_data
    
    duration = len(audio_data) / orig_rate
    target_samples = int(duration * target_rate)
    
    # Linear interpolation
    indices = np.linspace(0, len(audio_data) - 1, target_samples)
    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    return resampled.astype(np.int16)

# ===== Lưu WAV =====
def save_wav(audio_data, filename):
    # Resample xuống 16kHz cho Whisper (sẽ tự convert stereo→mono)
    resampled = resample_audio(audio_data, RECORD_RATE, WHISPER_RATE)
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Whisper cần mono
        wf.setsampwidth(2)
        wf.setframerate(WHISPER_RATE)
        wf.writeframes(resampled.tobytes())

# ===== Gửi file lên server =====
def send_to_server(file_path):
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            r = requests.post(SERVER_URL, files=files, timeout=5)
        print(f"📤 Server: {r.status_code}")
    except Exception as e:
        print(f"❌ Server error: {e}")

# ===== Xử lý transcription =====
def process_audio(audio_data):
    global history, processing
    
    processing = True
    wav_file = "speech_temp.wav"
    save_wav(audio_data, wav_file)
    
    print("🔄 Transcribing...")
    show_recent_history(history, " Đang xử lý...")
    
    try:
        segments, _ = model.transcribe(wav_file, beam_size=1, language="vi")
        full_text = " ".join([s.text for s in segments]).strip()
        
        if full_text:
            print(f"✅ Kết quả: {full_text}")
            
            with history_lock:
                history.append(full_text)
                if len(history) > MAX_HISTORY:
                    history.pop(0)
            
            show_recent_history(history)
            
            # Ghi file
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(full_text + "\n")
            
            # Gửi server
            threading.Thread(target=send_to_server, args=(HISTORY_FILE,), daemon=True).start()
        else:
            print("⚠️  Không phát hiện giọng nói")
            show_recent_history(history, "🎧 Đang nghe...")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        show_recent_history(history, "⚠️  Lỗi xử lý")
    
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)
        processing = False

# ===== Vòng lặp ghi âm liên tục =====
def continuous_recording():
    global is_running
    
    frames_per_chunk = int(RECORD_RATE * CHUNK_DURATION)
    
    with sd.InputStream(samplerate=RECORD_RATE, channels=CHANNELS, dtype="int16", device=DEVICE) as stream:
        print(f"✅ Bắt đầu ghi âm (mỗi {CHUNK_DURATION}s)...\n")
        show_recent_history(history, "🎧 Đang nghe...")
        
        while is_running:
            # Ghi audio trong CHUNK_DURATION giây
            audio_data, _ = stream.read(frames_per_chunk)
            
            # Chờ xử lý xong (nếu đang xử lý)
            while processing and is_running:
                time.sleep(0.1)
            
            if not is_running:
                break
            
            # Xử lý trong thread riêng
            threading.Thread(
                target=process_audio, 
                args=(audio_data.copy(),),
                daemon=True
            ).start()

# ===== Main =====
if __name__ == "__main__":
    print("=" * 60)
    print("🎙️  REAL-TIME WHISPER - Fixed-length (Nhẹ nhất)")
    print("=" * 60)
    print(f"Ghi mỗi {CHUNK_DURATION}s → transcribe → lặp lại")
    print(f"Ghi âm: {RECORD_RATE} Hz → Resample: {WHISPER_RATE} Hz")
    print(f"Device: Card {DEVICE} (USB PnP Sound Device)")
    print("=" * 60)
    print("\nNhấn Enter để bắt đầu...")
    print("Nhấn Ctrl+C để dừng\n")
    
    input()
    
    is_running = True
    
    try:
        continuous_recording()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Đang dừng...")
        is_running = False
        time.sleep(0.5)
        show_recent_history(history, "⏹️  Đã dừng")
        print("Tạm biệt! 👋")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        is_running = False
