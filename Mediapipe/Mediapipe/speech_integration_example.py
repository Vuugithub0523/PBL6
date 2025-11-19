# ===== CÁCH TÍCH HỢP SPEECH-TO-TEXT VÀO TF_MODEL.PY =====

"""
Thay thế hàm speech_to_text_thread() trong Tf_model.py bằng code này:
"""

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading

# Khởi tạo Whisper model (chỉ cần 1 lần)
whisper_model = WhisperModel("/home/pi/PhoWhisper-tiny-ct2", device="cpu", compute_type="int8")

# Cấu hình audio
RECORD_RATE = 44100
WHISPER_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3.0  # Ghi 3 giây

def resample_audio(audio_data, orig_rate, target_rate):
    """Resample từ 44100Hz → 16000Hz"""
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1).astype(np.int16)
    else:
        audio_data = audio_data.flatten()
    
    if orig_rate == target_rate:
        return audio_data
    
    duration = len(audio_data) / orig_rate
    target_samples = int(duration * target_rate)
    indices = np.linspace(0, len(audio_data) - 1, target_samples)
    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
    return resampled.astype(np.int16)

def speech_to_text_thread():
    """Thread xử lý speech-to-text liên tục"""
    global speech_buffer, speech_status
    
    frames_per_chunk = int(RECORD_RATE * CHUNK_DURATION)
    
    with sd.InputStream(samplerate=RECORD_RATE, channels=CHANNELS, dtype="int16", device=3) as stream:
        print("🎤 Speech-to-Text started")
        
        while True:
            try:
                # Ghi audio
                with speech_lock:
                    speech_status = "🎤 Đang nghe..."
                
                audio_data, _ = stream.read(frames_per_chunk)
                
                # Resample
                resampled = resample_audio(audio_data, RECORD_RATE, WHISPER_RATE)
                audio_float = resampled.astype(np.float32) / 32768.0  # Normalize
                
                # Transcribe
                with speech_lock:
                    speech_status = "🔄 Đang xử lý..."
                
                segments, _ = whisper_model.transcribe(audio_float, beam_size=1, language="vi")
                full_text = " ".join([s.text for s in segments]).strip()
                
                if full_text:
                    with speech_lock:
                        speech_buffer = full_text
                        speech_status = "✅ Sẵn sàng"
                    print(f"🎤 Speech: {full_text}")
                else:
                    with speech_lock:
                        speech_status = "⚠️  Không nghe rõ"
                        
            except Exception as e:
                print(f"❌ Speech error: {e}")
                with speech_lock:
                    speech_status = f"❌ Lỗi: {str(e)[:20]}"
                time.sleep(1)

# CÁCH SỬ DỤNG:
# 1. Uncomment dòng này trong Tf_model.py:
#    threading.Thread(target=speech_to_text_thread, daemon=True).start()
#
# 2. Chạy chương trình → cả Sign Language và Speech sẽ chạy song song
#
# 3. Màn hình sẽ hiển thị:
#    - 🤟 Sign: [ký hiệu hiện tại]
#    - Câu (Sign): [chuỗi từ cử chỉ]
#    - 🎤 [trạng thái speech]
#    - Câu (Speech): [chuỗi từ giọng nói]
#
# 4. Phím tắt:
#    - S: Gửi chuỗi Sign Language
#    - M: Gửi chuỗi Speech-to-Text
#    - C: Xóa cả 2 buffer
#    - Q: Thoát

print("""
╔════════════════════════════════════════════════════════════╗
║  TÍCH HỢP SONG SONG: SIGN LANGUAGE + SPEECH-TO-TEXT       ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  📺 HIỂN THỊ TRÊN MÀN HÌNH:                                ║
║  ┌──────────────────────────────────────────────────────┐ ║
║  │ 🤟 Sign: c                                           │ ║
║  │ Câu (Sign): chaocacban                              │ ║
║  │ 🎤 Đang nghe...                                      │ ║
║  │ Câu (Speech): xin chào các bạn                      │ ║
║  │ FPS: 25.3                                            │ ║
║  │ [S] Send Sign | [M] Send Speech | [C] Clear         │ ║
║  │ Server: Chào các bạn.                                │ ║
║  └──────────────────────────────────────────────────────┘ ║
║                                                            ║
║  📺 HIỂN THỊ TRÊN LCD (RASPBERRY PI):                      ║
║  ┌──────────────────────────────────────────────────────┐ ║
║  │              📤 Đang gửi...                          │ ║
║  │                                                       │ ║
║  │         Chào các bạn.                                │ ║
║  │                                                       │ ║
║  │         Hôm nay trời đẹp quá.                        │ ║
║  │                                                       │ ║
║  │         Bạn có khỏe không?                           │ ║
║  └──────────────────────────────────────────────────────┘ ║
║                                                            ║
║  🎯 LƯU TRÌNH HOẠT ĐỘNG:                                   ║
║  1. Sign Language: Làm cử chỉ → tạo chuỗi → gửi          ║
║  2. Speech: Nói → nhận diện → gửi                        ║
║  3. Server: Nhận → xử lý dấu → gửi lại                   ║
║  4. Client: Nhận → hiển thị LCD + console                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
