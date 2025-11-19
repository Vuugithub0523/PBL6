from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Demo Phiên Dịch Người Khiếm Thính</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
    <!-- Header -->
    <header class="bg-white shadow-md">
        <div class="max-w-6xl mx-auto px-4 py-6">
            <div class="flex items-center justify-center space-x-3">
                <i class="fas fa-hands text-indigo-600 text-4xl"></i>
                <h1 class="text-3xl font-bold text-gray-800">Hệ Thống Phiên Dịch Người Khiếm Thính</h1>
            </div>
        </div>
    </header>

    <div class="max-w-6xl mx-auto p-6">
        <!-- Main Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            
            <!-- Camera - Sign Language Detection -->
            <div class="bg-white rounded-2xl shadow-xl p-6 transform hover:scale-105 transition duration-300">
                <div class="flex items-center space-x-3 mb-4">
                    <div class="bg-blue-500 p-3 rounded-full">
                        <i class="fas fa-video text-white text-xl"></i>
                    </div>
                    <h2 class="text-2xl font-bold text-gray-800">Ngôn Ngữ Ký Hiệu</h2>
                </div>
                
                <div class="relative mb-4">
                    <video id="video" autoplay class="w-full h-72 bg-gray-900 rounded-xl shadow-inner object-cover"></video>
                    <canvas id="canvas" class="hidden"></canvas>
                    <div id="camera-status" class="absolute top-2 right-2 bg-gray-800 bg-opacity-75 text-white px-3 py-1 rounded-full text-sm hidden">
                        <i class="fas fa-circle text-red-500 animate-pulse"></i> Đang phát hiện
                    </div>
                </div>

                <button onclick="toggleCamera()" id="camera-btn" 
                    class="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 transition shadow-lg">
                    <i class="fas fa-camera mr-2"></i>Bật Camera
                </button>
                
                <div id="sign-result" class="mt-4 p-4 bg-blue-50 rounded-xl hidden">
                    <p class="text-sm text-gray-600 mb-1">Phát hiện:</p>
                    <p class="text-2xl font-bold text-blue-600" id="sign-text"></p>
                </div>
            </div>

            <!-- Microphone - Speech to Text -->
            <div class="bg-white rounded-2xl shadow-xl p-6 transform hover:scale-105 transition duration-300">
                <div class="flex items-center space-x-3 mb-4">
                    <div class="bg-green-500 p-3 rounded-full">
                        <i class="fas fa-microphone text-white text-xl"></i>
                    </div>
                    <h2 class="text-2xl font-bold text-gray-800">Giọng Nói → Văn Bản</h2>
                </div>
                
                <div id="mic-display" class="h-72 bg-gradient-to-br from-green-50 to-green-100 rounded-xl flex items-center justify-center mb-4 shadow-inner">
                    <div class="text-center">
                        <i class="fas fa-microphone text-gray-400 text-7xl mb-4"></i>
                        <p class="text-gray-500">Nhấn nút để bắt đầu</p>
                    </div>
                </div>

                <button onclick="toggleSpeech()" id="speech-btn" 
                    class="w-full bg-gradient-to-r from-green-500 to-green-600 text-white py-4 rounded-xl font-semibold hover:from-green-600 hover:to-green-700 transition shadow-lg">
                    <i class="fas fa-microphone mr-2"></i>Bắt Đầu Ghi Âm
                </button>
                
                <div id="speech-result" class="mt-4 p-4 bg-green-50 rounded-xl hidden">
                    <p class="text-sm text-gray-600 mb-1">Nhận diện:</p>
                    <p class="text-2xl font-bold text-green-600" id="speech-text"></p>
                </div>
            </div>
        </div>

        <!-- Kết Quả Tổng Hợp -->
        <div class="bg-white rounded-2xl shadow-xl p-8 mb-6">
            <div class="flex items-center space-x-3 mb-6">
                <div class="bg-indigo-500 p-3 rounded-full">
                    <i class="fas fa-comment-dots text-white text-xl"></i>
                </div>
                <h2 class="text-2xl font-bold text-gray-800">Văn Bản Phiên Dịch</h2>
            </div>
            
            <div id="translation-display" class="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-xl min-h-32 flex items-center justify-center">
                <p class="text-gray-400 text-lg">Chưa có phiên dịch nào...</p>
            </div>
        </div>

        <!-- Lịch Sử -->
        <div class="bg-white rounded-2xl shadow-xl p-8">
            <div class="flex items-center justify-between mb-6">
                <div class="flex items-center space-x-3">
                    <div class="bg-purple-500 p-3 rounded-full">
                        <i class="fas fa-history text-white text-xl"></i>
                    </div>
                    <h2 class="text-2xl font-bold text-gray-800">Lịch Sử Demo</h2>
                </div>
                <button onclick="clearHistory()" class="text-red-500 hover:text-red-700 font-semibold transition">
                    <i class="fas fa-trash mr-2"></i>Xóa Hết
                </button>
            </div>

            <div id="history-list" class="space-y-3 max-h-96 overflow-y-auto">
                <div class="text-center py-12 text-gray-400">
                    <i class="fas fa-inbox text-6xl mb-4"></i>
                    <p>Chưa có lịch sử nào</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="text-center py-8 text-gray-600">
        <p> Demo Hệ Thống Phiên Dịch </p>
    </footer>

    <script>
        let cameraActive = false;
        let speechActive = false;
        let cameraInterval = null;
        let recognition = null;
        let history = [];

        // ========== CAMERA FUNCTIONS ==========
        async function toggleCamera() {
            const video = document.getElementById('video');
            const btn = document.getElementById('camera-btn');
            const status = document.getElementById('camera-status');
            
            if (!cameraActive) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 } 
                    });
                    video.srcObject = stream;
                    cameraActive = true;
                    
                    btn.innerHTML = '<i class="fas fa-stop mr-2"></i>Dừng Camera';
                    btn.classList.remove('from-blue-500', 'to-blue-600');
                    btn.classList.add('from-red-500', 'to-red-600');
                    status.classList.remove('hidden');
                    
                    startSignDetection();
                } catch (error) {
                    alert('Không thể truy cập camera!');
                }
            } else {
                stopCamera();
            }
        }

        function stopCamera() {
            const video = document.getElementById('video');
            const btn = document.getElementById('camera-btn');
            const status = document.getElementById('camera-status');
            
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            cameraActive = false;
            btn.innerHTML = '<i class="fas fa-camera mr-2"></i>Bật Camera';
            btn.classList.remove('from-red-500', 'to-red-600');
            btn.classList.add('from-blue-500', 'to-blue-600');
            status.classList.add('hidden');
            
            if (cameraInterval) {
                clearInterval(cameraInterval);
            }
        }

        function startSignDetection() {
            // Giả lập phát hiện mỗi 3 giây
            cameraInterval = setInterval(async () => {
                if (!cameraActive) return;
                
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg');
                
                try {
                    const response = await fetch('/detect-sign', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image: imageData})
                    });
                    
                    const data = await response.json();
                    if (data.text) {
                        showSignResult(data.text);
                        addToHistory('sign', data.text);
                    }
                } catch (error) {
                    console.error('Detection error:', error);
                }
            }, 3000);
        }

        function showSignResult(text) {
            const result = document.getElementById('sign-result');
            const textElem = document.getElementById('sign-text');
            textElem.textContent = text;
            result.classList.remove('hidden');
            updateTranslationDisplay(text, 'blue');
        }

        // ========== SPEECH FUNCTIONS ==========
        function toggleSpeech() {
            if (!speechActive) {
                startSpeech();
            } else {
                stopSpeech();
            }
        }

        function startSpeech() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                alert(' Trình duyệt không hỗ trợ nhận diện giọng nói!');
                return;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            recognition.lang = 'vi-VN';
            recognition.continuous = true;
            recognition.interimResults = true;

            const btn = document.getElementById('speech-btn');
            const display = document.getElementById('mic-display');
            
            recognition.onstart = () => {
                speechActive = true;
                btn.innerHTML = '<i class="fas fa-stop mr-2"></i>Dừng Ghi Âm';
                btn.classList.remove('from-green-500', 'to-green-600');
                btn.classList.add('from-red-500', 'to-red-600');
                
                display.innerHTML = `
                    <div class="text-center">
                        <i class="fas fa-microphone text-green-500 text-7xl mb-4 animate-pulse"></i>
                        <p class="text-green-600 font-semibold">Đang lắng nghe...</p>
                        <div class="mt-4 flex justify-center space-x-2">
                            <div class="w-2 h-8 bg-green-500 rounded animate-pulse"></div>
                            <div class="w-2 h-12 bg-green-500 rounded animate-pulse" style="animation-delay: 0.1s"></div>
                            <div class="w-2 h-10 bg-green-500 rounded animate-pulse" style="animation-delay: 0.2s"></div>
                            <div class="w-2 h-8 bg-green-500 rounded animate-pulse" style="animation-delay: 0.3s"></div>
                        </div>
                    </div>
                `;
            };

            recognition.onresult = (event) => {
                let finalTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    }
                }

                if (finalTranscript) {
                    showSpeechResult(finalTranscript);
                    addToHistory('speech', finalTranscript);
                }
            };

            recognition.onerror = (event) => {
                console.error('Speech error:', event.error);
                stopSpeech();
            };

            recognition.onend = () => {
                if (speechActive) {
                    recognition.start(); // Restart if still active
                }
            };

            recognition.start();
        }

        function stopSpeech() {
            if (recognition) {
                recognition.stop();
                recognition = null;
            }
            
            speechActive = false;
            
            const btn = document.getElementById('speech-btn');
            const display = document.getElementById('mic-display');
            
            btn.innerHTML = '<i class="fas fa-microphone mr-2"></i>Bắt Đầu Ghi Âm';
            btn.classList.remove('from-red-500', 'to-red-600');
            btn.classList.add('from-green-500', 'to-green-600');
            
            display.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-microphone text-gray-400 text-7xl mb-4"></i>
                    <p class="text-gray-500">Nhấn nút để bắt đầu</p>
                </div>
            `;
        }

        function showSpeechResult(text) {
            const result = document.getElementById('speech-result');
            const textElem = document.getElementById('speech-text');
            textElem.textContent = text;
            result.classList.remove('hidden');
            updateTranslationDisplay(text, 'green');
        }

        // ========== DISPLAY FUNCTIONS ==========
        function updateTranslationDisplay(text, color) {
            const display = document.getElementById('translation-display');
            const colorClass = color === 'blue' ? 'text-blue-600' : 'text-green-600';
            const icon = color === 'blue' ? 'fa-video' : 'fa-microphone';
            
            display.innerHTML = `
                <div class="text-center w-full">
                    <i class="fas ${icon} ${colorClass} text-3xl mb-3"></i>
                    <p class="${colorClass} text-3xl font-bold">${text}</p>
                    <p class="text-gray-500 text-sm mt-2">${new Date().toLocaleTimeString('vi-VN')}</p>
                </div>
            `;
        }

        // ========== HISTORY FUNCTIONS ==========
        function addToHistory(type, text) {
            const item = {
                type: type,
                text: text,
                timestamp: new Date().toISOString()
            };
            
            history.unshift(item);
            if (history.length > 20) history.pop();
            
            renderHistory();
        }

        function renderHistory() {
            const container = document.getElementById('history-list');
            
            if (history.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-12 text-gray-400">
                        <i class="fas fa-inbox text-6xl mb-4"></i>
                        <p>Chưa có lịch sử nào</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = history.map(item => {
                const isSign = item.type === 'sign';
                const bgColor = isSign ? 'bg-blue-50' : 'bg-green-50';
                const textColor = isSign ? 'text-blue-600' : 'text-green-600';
                const icon = isSign ? 'fa-video' : 'fa-microphone';
                
                return `
                    <div class="${bgColor} p-4 rounded-xl border-l-4 ${isSign ? 'border-blue-500' : 'border-green-500'} transform hover:scale-102 transition">
                        <div class="flex items-start space-x-3">
                            <i class="fas ${icon} ${textColor} text-xl mt-1"></i>
                            <div class="flex-1">
                                <p class="text-gray-800 font-medium text-lg">${item.text}</p>
                                <p class="text-xs text-gray-500 mt-1">
                                    <i class="fas fa-clock mr-1"></i>
                                    ${new Date(item.timestamp).toLocaleString('vi-VN')}
                                </p>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function clearHistory() {
            if (history.length === 0) return;
            
            if (confirm('🗑️ Xóa toàn bộ lịch sử?')) {
                history = [];
                renderHistory();
                
                document.getElementById('translation-display').innerHTML = 
                    '<p class="text-gray-400 text-lg">Chưa có phiên dịch nào...</p>';
            }
        }

        // Initialize
        renderHistory();
    </script>
</body>
</html>
"""

# ========== SIGN LANGUAGE DETECTION (Giả lập) ==========
def detect_sign_language(image_data):
    """
    Hàm giả lập phát hiện ngôn ngữ ký hiệu
    THAY THẾ bằng model thực tế của bạn
    """
    # TODO: Tích hợp model ML của bạn ở đây
    # Ví dụ:
    # image = preprocess_image(image_data)
    # prediction = your_model.predict(image)
    # text = decode_prediction(prediction)
    
    # Demo: Trả về text ngẫu nhiên
    import random
    sample_texts = [
        'Xin chào',
        'Cảm ơn bạn', 
        'Hẹn gặp lại',
        'Tôi cần giúp đỡ',
        'Tạm biệt',
        'Rất vui được gặp bạn'
    ]
    return random.choice(sample_texts)


# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/detect-sign', methods=['POST'])
def detect_sign():
    """API endpoint phát hiện ngôn ngữ ký hiệu"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Phát hiện (hiện tại là giả lập)
        detected_text = detect_sign_language(frame)
        
        return jsonify({'success': True, 'text': detected_text})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Hệ Thống Phiên Dịch Người Khiếm Thính (DEMO)")
    print("=" * 60)
    print("📱 Mở trình duyệt tại: http://localhost:5000")
    print("💡 Tích hợp model ML của bạn vào hàm detect_sign_language()")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)