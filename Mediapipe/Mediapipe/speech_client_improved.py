#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech-to-Text Client - Improved Version with WebRTC VAD
Hỗ trợ cả Raspberry Pi và Windows
"""
# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sounddevice as sd
import numpy as np
import wave
import threading
import os
import time
import textwrap
import requests
from collections import deque
import argparse
import sys
import struct

# Kiểm tra Whisper model
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("❌ faster-whisper not installed!")
    print("💡 Install: pip install faster-whisper")
    WHISPER_AVAILABLE = False

# Kiểm tra WebRTC VAD
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    print("⚠️  webrtcvad not installed!")
    print("💡 Install: pip install webrtcvad")
    VAD_AVAILABLE = False

# Kiểm tra PIL cho LCD (optional)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ===== Cấu hình =====
RECORD_RATE = 44100  # Sample rate của thiết bị
WHISPER_RATE = 16000  # Whisper yêu cầu 16kHz
CHANNELS = 1  # Mono
CHUNK_DURATION = 3.0  # Ghi 3 giây → transcribe

# WebRTC VAD Configuration
VAD_RATE = 16000  # VAD chỉ hỗ trợ 8000, 16000, 32000, 48000 Hz
VAD_FRAME_MS = 30  # Frame duration: 10, 20, hoặc 30 ms
VAD_AGGRESSIVENESS = 2  # 0-3: 0=least aggressive, 3=most aggressive
VAD_MIN_SPEECH_FRAMES = 10  # Số frames có speech tối thiểu để bắt đầu ghi
VAD_TRAILING_SILENCE_FRAMES = 15  # Số frames im lặng để kết thúc ghi

# LCD (Raspberry Pi only)
FB1 = "/dev/fb1"
LCD_WIDTH, LCD_HEIGHT = 480, 320
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 22
USE_LCD = os.path.exists(FB1) if os.name != "nt" else False

# Lịch sử
HISTORY_FILE = "speech_history.txt"
MAX_HISTORY = 3
history = deque(maxlen=MAX_HISTORY)
history_lock = threading.Lock()

# Trạng thái
is_running = False
processing = False

# ===== Hàm tiện ích =====
def list_audio_devices():
    """Liệt kê tất cả audio devices"""
    print("\n📋 Available Audio Devices:")
    print("=" * 70)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}")
            print(f"      Channels: {dev['max_input_channels']} | Sample Rate: {dev['default_samplerate']} Hz")
    print("=" * 70)

def find_audio_device(device_id=None):
    """Tìm audio device phù hợp"""
    if device_id is not None:
        try:
            dev = sd.query_devices(device_id)
            if dev['max_input_channels'] > 0:
                print(f"✅ Using device [{device_id}]: {dev['name']}")
                return device_id
        except:
            pass
    
    # Tự động tìm device
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            # Ưu tiên USB hoặc mic ngoài
            if any(keyword in dev['name'].lower() for keyword in ['usb', 'microphone', 'mic']):
                print(f"✅ Auto-detected device [{i}]: {dev['name']}")
                return i
    
    # Fallback: device mặc định
    default_input = sd.default.device[0]
    if default_input is not None:
        dev = sd.query_devices(default_input)
        print(f"✅ Using default input [{default_input}]: {dev['name']}")
        return default_input
    
    return None

def check_model_path(model_path):
    """Kiểm tra model path tồn tại"""
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        return model_path
    
    # Thử các đường dẫn thay thế
    alternatives = [
        "./PhoWhisper-tiny-ct2",
        "../PhoWhisper-tiny-ct2",
        "../../PhoWhisper-tiny-ct2",
        "/home/pi/PhoWhisper-tiny-ct2",
        os.path.expanduser("~/PhoWhisper-tiny-ct2"),
    ]
    
    for alt in alternatives:
        if os.path.exists(alt):
            print(f"✅ Model found at alternative path: {alt}")
            return alt
    
    print(f"❌ Model not found at: {model_path}")
    print("💡 Alternative paths tried:")
    for alt in alternatives:
        print(f"   - {alt}")
    return None

# ===== LCD Functions (Raspberry Pi) =====
def write_to_fb(img):
    """Ghi hình ảnh lên LCD framebuffer"""
    if not USE_LCD or not PIL_AVAILABLE:
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
        print(f"⚠️  LCD Error: {e}")

def show_on_lcd(history_list, status=""):
    """Hiển thị lịch sử lên LCD"""
    if not USE_LCD or not PIL_AVAILABLE:
        return
    
    try:
        img = Image.new("RGB", (LCD_WIDTH, LCD_HEIGHT), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        
        # Status
        if status:
            status_font = ImageFont.truetype(FONT_PATH, 18)
            bbox = draw.textbbox((0, 0), status, font=status_font)
            w = bbox[2] - bbox[0]
            draw.text(((LCD_WIDTH - w) // 2, 10), status, font=status_font, fill="blue")
            y = 50
        else:
            y = 10
        
        # History
        with history_lock:
            for line in history_list:
                wrapped = textwrap.fill(line, width=25)
                for subline in wrapped.split("\n"):
                    bbox = draw.textbbox((0, 0), subline, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    x = (LCD_WIDTH - w) // 2
                    draw.text((x, y), subline, font=font, fill="black")
                    y += h + 4
                y += 8
        
        write_to_fb(img)
    except Exception as e:
        print(f"⚠️  LCD Display Error: {e}")

# ===== Audio Processing =====
def resample_audio(audio_data, orig_rate, target_rate):
    """Resample audio từ orig_rate → target_rate"""
    # Convert stereo to mono nếu cần
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
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

# ===== WebRTC VAD =====
class VoiceActivityDetector:
    """Voice Activity Detector using WebRTC VAD"""
    
    def __init__(self, aggressiveness=2, sample_rate=16000, frame_duration_ms=30):
        """
        Initialize VAD
        
        Args:
            aggressiveness: 0-3 (0=least aggressive, 3=most aggressive)
            sample_rate: 8000, 16000, 32000, or 48000 Hz
            frame_duration_ms: 10, 20, or 30 ms
        """
        if not VAD_AVAILABLE:
            raise ImportError("webrtcvad not installed")
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_size * 2  # 16-bit = 2 bytes
        
        # Buffers
        self.speech_buffer = []
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        
        print(f"✅ VAD initialized (aggressiveness={aggressiveness}, rate={sample_rate}Hz, frame={frame_duration_ms}ms)")
    
    def is_speech(self, frame):
        """
        Kiểm tra frame có chứa speech không
        
        Args:
            frame: numpy array int16
        
        Returns:
            bool: True nếu có speech
        """
        # Convert to bytes
        if isinstance(frame, np.ndarray):
            frame_bytes = frame.tobytes()
        else:
            frame_bytes = frame
        
        # VAD check
        try:
            return self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception as e:
            print(f"⚠️  VAD error: {e}")
            return False
    
    def process_audio(self, audio_data, min_speech_frames=10, trailing_silence_frames=15):
        """
        Xử lý audio stream và detect speech segments
        
        Args:
            audio_data: numpy array int16
            min_speech_frames: Số frames có speech tối thiểu
            trailing_silence_frames: Số frames im lặng để kết thúc
        
        Returns:
            tuple: (has_complete_speech, speech_audio)
        """
        # Đảm bảo audio là int16
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        
        # Chia thành frames
        num_frames = len(audio_data) // self.frame_size
        has_complete_speech = False
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio_data[start:end]
            
            if len(frame) != self.frame_size:
                continue
            
            is_speech = self.is_speech(frame)
            
            if is_speech:
                self.speech_frame_count += 1
                self.silence_frame_count = 0
                
                # Bắt đầu ghi khi có đủ speech frames
                if not self.is_speaking and self.speech_frame_count >= min_speech_frames:
                    self.is_speaking = True
                    print("🎤 Speech detected - Recording...")
                    if USE_LCD:
                        show_on_lcd(list(history), "🎤 Đang ghi...")
                
                # Thêm frame vào buffer
                if self.is_speaking:
                    self.speech_buffer.append(frame)
            else:
                # Im lặng
                if self.is_speaking:
                    self.silence_frame_count += 1
                    self.speech_buffer.append(frame)  # Giữ lại trailing silence
                    
                    # Kết thúc khi có đủ silence frames
                    if self.silence_frame_count >= trailing_silence_frames:
                        print(f"🔇 Silence detected - Processing ({len(self.speech_buffer)} frames)...")
                        has_complete_speech = True
                        break
                else:
                    # Reset nếu chưa bắt đầu ghi
                    self.speech_frame_count = max(0, self.speech_frame_count - 1)
        
        # Trả về audio nếu có complete speech
        if has_complete_speech:
            speech_audio = np.concatenate(self.speech_buffer)
            self.reset()
            return True, speech_audio
        
        return False, None
    
    def reset(self):
        """Reset VAD state"""
        self.speech_buffer = []
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
    
    def get_stats(self):
        """Lấy thống kê VAD"""
        return {
            'is_speaking': self.is_speaking,
            'speech_frames': self.speech_frame_count,
            'silence_frames': self.silence_frame_count,
            'buffer_frames': len(self.speech_buffer)
        }

def save_wav(audio_data, filename):
    """Lưu audio thành WAV file"""
    resampled = resample_audio(audio_data, RECORD_RATE, WHISPER_RATE)
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)
        wf.setframerate(WHISPER_RATE)
        wf.writeframes(resampled.tobytes())

def send_to_server(file_path, server_url):
    """Gửi file transcript lên server"""
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            r = requests.post(server_url, files=files, timeout=5)
        
        if r.status_code == 200:
            result = r.json()
            print(f"📤 Server response: {result.get('status', 'unknown')}")
            if 'text' in result:
                print(f"   Text: {result['text']}")
        else:
            print(f"⚠️  Server returned: {r.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server: {server_url}")
    except Exception as e:
        print(f"❌ Server error: {e}")

# ===== Main Processing =====
def process_audio(audio_data, model, server_url, is_16khz=False):
    """Xử lý audio và gửi lên server"""
    global processing
    
    processing = True
    wav_file = "speech_temp.wav"
    
    try:
        # Lưu WAV file
        if is_16khz:
            # Audio đã ở 16kHz (từ VAD), lưu trực tiếp
            with wave.open(wav_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_RATE)
                wf.writeframes(audio_data.tobytes())
        else:
            # Audio ở sample rate gốc, cần resample
            save_wav(audio_data, wav_file)
        
        print("🔄 Transcribing...")
        if USE_LCD:
            show_on_lcd(list(history), "🔄 Đang xử lý...")
        
        # Transcribe
        segments, _ = model.transcribe(wav_file, beam_size=1, language="vi")
        full_text = " ".join([s.text for s in segments]).strip()
        
        if full_text:
            print(f"✅ Transcript: {full_text}")
            
            # Thêm vào history
            with history_lock:
                history.append(full_text)
            
            if USE_LCD:
                show_on_lcd(list(history))
            
            # Ghi file
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {full_text}\n")
            
            # Gửi server
            if server_url:
                threading.Thread(
                    target=send_to_server, 
                    args=(HISTORY_FILE, server_url),
                    daemon=True
                ).start()
        else:
            print("⚠️  No speech detected")
            if USE_LCD:
                show_on_lcd(list(history), "🎧 Đang nghe...")
    
    except Exception as e:
        print(f"❌ Processing error: {e}")
        if USE_LCD:
            show_on_lcd(list(history), "⚠️  Lỗi")
    
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)
        processing = False

def continuous_recording(model, device_id, server_url, use_vad=True):
    """Vòng lặp ghi âm liên tục với VAD"""
    global is_running
    
    # Initialize VAD
    vad = None
    if use_vad and VAD_AVAILABLE:
        try:
            vad = VoiceActivityDetector(
                aggressiveness=VAD_AGGRESSIVENESS,
                sample_rate=VAD_RATE,  # VAD hoạt động ở 16kHz
                frame_duration_ms=VAD_FRAME_MS
            )
            print(f"✅ VAD enabled (aggressiveness={VAD_AGGRESSIVENESS})")
            print(f"   Recording: {RECORD_RATE}Hz → Resample to {VAD_RATE}Hz for VAD")
        except Exception as e:
            print(f"⚠️  VAD initialization failed: {e}")
            print("   Falling back to non-VAD mode")
            vad = None
    elif use_vad:
        print("⚠️  VAD requested but not available")
        print("   Install: pip install webrtcvad")
    
    # Chunk size configuration
    if vad:
        # VAD mode: Đọc ở sample rate gốc với chunks nhỏ (~300ms)
        # Sau đó resample xuống 16kHz cho VAD
        vad_frame_size_16k = int(VAD_RATE * VAD_FRAME_MS / 1000)
        num_vad_frames = 10  # 10 frames = ~300ms
        frames_per_chunk = int(RECORD_RATE * (num_vad_frames * VAD_FRAME_MS / 1000))
    else:
        # Không VAD: đọc chunks lớn như cũ
        frames_per_chunk = int(RECORD_RATE * CHUNK_DURATION)
    
    try:
        with sd.InputStream(
            samplerate=RECORD_RATE,  # Luôn đọc ở sample rate gốc của device
            channels=CHANNELS, 
            dtype="int16", 
            device=device_id
        ) as stream:
            if vad:
                print(f"\n✅ Recording started with VAD (listening...)")
                print(f"   Min speech: {VAD_MIN_SPEECH_FRAMES} frames | Trailing silence: {VAD_TRAILING_SILENCE_FRAMES} frames")
            else:
                print(f"\n✅ Recording started (every {CHUNK_DURATION}s)")
            
            print("Press Ctrl+C to stop\n")
            
            if USE_LCD:
                show_on_lcd(list(history), "🎧 Đang nghe...")
            
            while is_running:
                # Ghi audio ở sample rate gốc
                audio_data, overflowed = stream.read(frames_per_chunk)
                
                if overflowed:
                    print("⚠️  Audio buffer overflowed")
                
                if not is_running:
                    break
                
                # Xử lý với VAD
                if vad:
                    # RESAMPLE xuống 16kHz cho VAD
                    audio_resampled = resample_audio(audio_data, RECORD_RATE, VAD_RATE)
                    
                    has_speech, speech_audio = vad.process_audio(
                        audio_resampled,  # Đưa audio đã resample vào VAD
                        min_speech_frames=VAD_MIN_SPEECH_FRAMES,
                        trailing_silence_frames=VAD_TRAILING_SILENCE_FRAMES
                    )
                    
                    if has_speech:
                        # Chờ xử lý xong
                        while processing and is_running:
                            time.sleep(0.1)
                        
                        if not is_running:
                            break
                        
                        # Xử lý speech trong thread riêng (audio đã ở 16kHz)
                        threading.Thread(
                            target=process_audio,
                            args=(speech_audio.copy(), model, server_url, True),  # is_16khz=True
                            daemon=True
                        ).start()
                        
                        # Reset LCD về trạng thái listening
                        time.sleep(0.5)
                        if USE_LCD and not processing:
                            show_on_lcd(list(history), "🎧 Đang nghe...")
                
                else:
                    # Không VAD: xử lý mọi chunk
                    # Chờ xử lý xong
                    while processing and is_running:
                        time.sleep(0.1)
                    
                    if not is_running:
                        break
                    
                    # Xử lý trong thread riêng (audio ở sample rate gốc)
                    threading.Thread(
                        target=process_audio,
                        args=(audio_data.copy(), model, server_url, False),  # is_16khz=False
                        daemon=True
                    ).start()
    
    except Exception as e:
        print(f"❌ Recording error: {e}")
        is_running = False

# ===== Main =====
def main():
    global is_running
    
    parser = argparse.ArgumentParser(description="Speech-to-Text Client with VAD")
    parser.add_argument("--model", default="/home/pi/PhoWhisper-tiny-ct2", 
                       help="Path to Whisper model")
    parser.add_argument("--server", default="http://172.20.10.2:5000/upload_speech",
                       help="Server URL")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio device ID")
    parser.add_argument("--list-devices", action="store_true",
                       help="List all audio devices")
    parser.add_argument("--no-server", action="store_true",
                       help="Don't send to server")
    parser.add_argument("--no-vad", action="store_true",
                       help="Disable Voice Activity Detection")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                       help="VAD aggressiveness (0-3, higher = more aggressive)")
    
    args = parser.parse_args()
    
    # List devices
    if args.list_devices:
        list_audio_devices()
        return
    
    # Update VAD aggressiveness from args
    global VAD_AGGRESSIVENESS
    VAD_AGGRESSIVENESS = args.vad_aggressiveness
    
    print("=" * 70)
    print("🎙️  SPEECH-TO-TEXT CLIENT with WebRTC VAD")
    print("=" * 70)
    
    # Check Whisper
    if not WHISPER_AVAILABLE:
        print("❌ faster-whisper not available!")
        return
    
    # Check model
    model_path = check_model_path(args.model)
    if not model_path:
        print("\n💡 Download model:")
        print("   ct2-transformers-converter --model vinai/PhoWhisper-tiny \\")
        print("       --output_dir ./PhoWhisper-tiny-ct2 --copy_files tokenizer.json")
        return
    
    # Load model
    print(f"\n🔄 Loading model...")
    try:
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Find audio device
    print("\n🔍 Detecting audio device...")
    list_audio_devices()
    device_id = find_audio_device(args.device)
    
    if device_id is None:
        print("❌ No audio input device found!")
        return
    
    # Server URL
    server_url = None if args.no_server else args.server
    if server_url:
        print(f"\n📡 Server: {server_url}")
    else:
        print("\n📝 Offline mode (no server)")
    
    # VAD status
    use_vad = not args.no_vad
    if use_vad and VAD_AVAILABLE:
        print(f"🎯 VAD: Enabled (aggressiveness={VAD_AGGRESSIVENESS})")
        print(f"   Frame: {VAD_FRAME_MS}ms | Min speech: {VAD_MIN_SPEECH_FRAMES} | Trailing silence: {VAD_TRAILING_SILENCE_FRAMES}")
    elif use_vad:
        print("⚠️  VAD: Requested but not available")
    else:
        print("🔇 VAD: Disabled")
    
    # LCD status
    if USE_LCD:
        print("📺 LCD: Enabled")
    
    print("\n" + "=" * 70)
    if use_vad and VAD_AVAILABLE:
        print(f"Recording: {VAD_RATE} Hz (VAD mode)")
        print(f"Speech detection: Real-time | History: {HISTORY_FILE}")
    else:
        print(f"Recording: {RECORD_RATE} Hz → Resample: {WHISPER_RATE} Hz")
        print(f"Chunk: {CHUNK_DURATION}s | History: {HISTORY_FILE}")
    print("=" * 70)
    print("\nPress Enter to start...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    is_running = True
    
    try:
        continuous_recording(model, device_id, server_url, use_vad=use_vad)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        is_running = False
        time.sleep(0.5)
        
        if USE_LCD:
            show_on_lcd(list(history), "⏹️  Đã dừng")
        
        print("Goodbye! 👋")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        is_running = False

if __name__ == "__main__":
    main()
