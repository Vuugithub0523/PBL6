# 📁 CẤU TRÚC DỰ ÁN - CLEAN VERSION

## 🗂️ CẤU TRÚC THƯ MỤC MỚI:

```
Mediapipe/
├── unified_server_clean.py    ← SERVER MỚI (logic sạch)
├── unified_server.py           ← Server cũ (backup)
│
├── templates/                  ← FRONTEND HTML
│   └── index.html             ← Giao diện chính
│
├── static/                     ← ASSETS
│   ├── style.css              ← CSS riêng
│   └── app.js                 ← JavaScript riêng
│
├── received_data/              ← DỮ LIỆU
│   ├── speech/                ← Giọng nói
│   └── sign/                  ← Ký hiệu
│
├── Client_pi_with_lcd.py       ← CLIENT (Raspberry Pi)
├── vsl_landmarks_model.tflite  ← Model nhận diện
├── label_encoder.pkl
├── scaler.pkl
└── selected_tags_names.txt
```

---

## 🚀 CÁCH CHẠY:

### **1. Chạy Server (trên Windows):**

```bash
cd D:\Hoctap\ki1nam4\PBL6\Dataset\D_VSL_Share\D_VSL_Share\Mediapipe\Mediapipe

# Version MỚI (Clean, tách frontend/backend)
python unified_server_clean.py

# Hoặc version CŨ (All-in-one)
python unified_server.py
```

### **2. Chạy Client (trên Raspberry Pi):**

```bash
cd ~/Mediapipe
python Client_pi_with_lcd.py \
  --server-host 192.168.10.212 \
  --resolution 320x240 \
  --frame-skip 3 \
  --lcd-update-interval 1.0 \
  --frame-send-interval 0.2
```

### **3. Mở Web UI:**

```
http://192.168.10.212:5000/
```

---

## 📊 SO SÁNH 2 PHIÊN BẢN:

| Feature | unified_server.py | unified_server_clean.py |
|---------|-------------------|-------------------------|
| **HTML** | Inline string (~400 dòng) | Template riêng (✅) |
| **CSS** | Trong <style> tag | File riêng (✅) |
| **JavaScript** | Trong <script> tag | File riêng (✅) |
| **Code size** | ~770 dòng | **~350 dòng** (✅) |
| **Maintainability** | Khó sửa | Dễ dàng (✅) |
| **Performance** | Giống nhau | Giống nhau |
| **Frontend/Backend** | Lẫn lộn | **Tách bạch** (✅) |

---

## 🎨 CẤU TRÚC MỚI:

### **Backend (unified_server_clean.py):**
```python
- Flask app setup
- API endpoints:
  ✓ POST /upload_speech    ← Nhận giọng nói
  ✓ POST /upload_sign      ← Nhận ký hiệu
  ✓ POST /upload_frame     ← Nhận camera
  ✓ GET  /client_stats     ← Stats real-time
  ✓ GET  /api/history      ← Lịch sử
  ✓ GET  /video_feed       ← Video stream
  ✓ GET  /                 ← Web UI

- Logic xử lý:
  ✓ Thêm dấu tiếng Việt
  ✓ Tách từ
  ✓ Lưu file
  ✓ Stream video
```

### **Frontend (templates/index.html):**
```html
- Layout HTML
- Jinja2 templates ({{ variable }})
- Semantic structure
```

### **Styles (static/style.css):**
```css
- CSS variables
- Modern design
- Responsive layout
- Animations
```

### **Logic (static/app.js):**
```javascript
- AJAX calls
- Real-time updates
- Stats refresh (1s)
- History refresh (5s)
```

---

## 💡 ƯU ĐIỂM CỦA CLEAN VERSION:

### **1. Dễ bảo trì:**
- ✅ Sửa giao diện → Chỉ sửa `index.html` & `style.css`
- ✅ Sửa logic → Chỉ sửa `unified_server_clean.py`
- ✅ Không lẫn lộn HTML/Python

### **2. Dễ debug:**
- ✅ Syntax highlighting đúng (HTML, CSS, JS riêng biệt)
- ✅ Linter hoạt động tốt hơn
- ✅ Dễ test từng phần

### **3. Performance:**
- ✅ Flask cache templates
- ✅ Browser cache CSS/JS
- ✅ Không cần rebuild HTML mỗi request

### **4. Team collaboration:**
- ✅ Frontend dev sửa HTML/CSS
- ✅ Backend dev sửa Python
- ✅ Không conflict

### **5. Scalability:**
- ✅ Dễ thêm pages mới
- ✅ Dễ thêm static assets (images, fonts)
- ✅ Dễ integrate frameworks (Vue, React nếu cần)

---

## 🔄 MIGRATION:

### **Từ old → new:**

1. **Backup old:**
   ```bash
   cp unified_server.py unified_server_old_backup.py
   ```

2. **Dùng clean version:**
   ```bash
   python unified_server_clean.py
   ```

3. **Test đầy đủ:**
   - Web UI: `http://IP:5000/`
   - Video stream: `http://IP:5000/video_feed`
   - Stats API: `http://IP:5000/client_stats`

4. **Nếu OK, rename:**
   ```bash
   mv unified_server.py unified_server_old.py
   mv unified_server_clean.py unified_server.py
   ```

---

## 📝 NOTES:

- ✅ **Cả 2 version đều hoạt động** (chọn 1 trong 2)
- ✅ **Clean version khuyến nghị** cho dự án dài hạn
- ✅ **Old version** vẫn OK nếu không muốn thay đổi
- ✅ **Templates tự động reload** khi sửa (debug=True)

---

## 🎯 RECOMMENDED:

**Dùng `unified_server_clean.py`** vì:
- Code gọn hơn 50%
- Dễ maintain
- Professional structure
- Scalable

**Run:**
```bash
python unified_server_clean.py
```

**Open:**
```
http://192.168.10.212:5000/
```

🎉 **DONE!**

