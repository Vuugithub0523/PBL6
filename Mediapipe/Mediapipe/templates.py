# templates.py
import os
from datetime import datetime

def get_index_html(username, speech_folder, sign_folder):
    """Generate main page HTML"""
    
    speech_files = sorted(os.listdir(speech_folder), reverse=True)[:10]
    sign_files = sorted(os.listdir(sign_folder), reverse=True)[:10]
    
    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sign Language Server</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',Arial;background:#f5f5f5;
min-height:100vh;padding:20px}}
.container{{max-width:1200px;margin:0 auto;background:#fff;border-radius:20px;
box-shadow:0 20px 60px rgba(0,0,0,.3);overflow:hidden}}
.header{{background:#667eea;color:#fff;
padding:30px;text-align:center;position:relative}}
.header h1{{font-size:2em;margin-bottom:10px}}
.user-info{{position:absolute;top:20px;right:30px;display:flex;align-items:center;gap:15px}}
.user-name{{background:rgba(255,255,255,.2);padding:8px 15px;border-radius:20px;font-size:.9em}}
.logout-btn{{background:rgba(255,255,255,.3);color:#fff;border:none;padding:8px 20px;
border-radius:20px;cursor:pointer;font-size:.9em;transition:all .3s}}
.logout-btn:hover{{background:rgba(255,255,255,.4)}}
.stats{{display:flex;justify-content:center;gap:30px;margin-top:15px}}
.stat-item{{background:rgba(255,255,255,.2);padding:10px 20px;border-radius:10px}}
.content{{padding:30px}}
.camera-section{{background:#f8f9fa;padding:20px;border-radius:15px;margin-bottom:30px}}
.camera-section h2{{margin-bottom:15px;color:#667eea}}
.camera-view{{width:100%;max-width:800px;margin:0 auto;border-radius:10px;overflow:hidden;
box-shadow:0 5px 15px rgba(0,0,0,.2)}}
.camera-view img{{width:100%;height:auto;display:block}}
.data-grid{{display:grid;grid-template-columns:1fr 1fr;gap:30px}}
.section{{background:#f8f9fa;padding:20px;border-radius:15px}}
.section h2{{margin-bottom:20px;color:#667eea;display:flex;align-items:center;gap:10px}}
.item{{background:#fff;padding:15px;margin-bottom:10px;border-radius:10px;border-left:4px solid #667eea;
transition:transform .2s}}
.item:hover{{transform:translateX(5px);box-shadow:0 5px 15px rgba(0,0,0,.1)}}
.timestamp{{color:#666;font-size:.85em;margin-bottom:8px}}
.text{{color:#333;font-size:1.1em;line-height:1.6}}
.processed{{color:#667eea;font-weight:bold;margin-top:5px}}
.empty{{text-align:center;padding:40px;color:#999}}
@media(max-width:768px){{.data-grid{{grid-template-columns:1fr}}
.user-info{{position:static;justify-content:center;margin-top:15px}}}}
</style></head><body>
<div class="container">
<div class="header">
<div class="user-info">
<div class="user-name">{username}</div>
<button class="logout-btn" onclick="location.href='/logout'">Đăng xuất</button>
</div>
<h1>Speech & Sign Language Server</h1>
<div class="stats">
<div class="stat-item">Speech: {len(speech_files)}</div>
<div class="stat-item">Sign: {len(sign_files)}</div>
</div></div>
<div class="content">
<div class="camera-section">
<h2>Sign Language Camera Feed</h2>
<div class="camera-view">
<img src="/video_feed" alt="Camera Feed">
</div></div>
<div class="data-grid">
<div class="section">
<h2><span>Speech-to-Text</span></h2>'''
    
    if speech_files:
        for f in speech_files:
            try:
                with open(os.path.join(speech_folder, f), 'r', encoding='utf-8') as file:
                    lines = [l.strip() for l in file.readlines() if l.strip()]
                    if lines:
                        content = lines[-1]
                        ts = f.replace('speech_', '').replace('.txt', '')
                        try:
                            dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
                            time_display = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            time_display = ts
                        
                        html += f'''<div class="item">
<div class="timestamp">{time_display}</div>
<div class="text">{content}</div>
</div>'''
            except:
                pass
    else:
        html += '<div class="empty">Chưa có dữ liệu</div>'
    
    html += '</div><div class="section"><h2><span>Sign Language</span></h2>'
    
    if sign_files:
        for f in sign_files:
            try:
                with open(os.path.join(sign_folder, f), 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    raw = processed = ""
                    for line in lines:
                        if line.startswith("Raw:"):
                            raw = line.replace("Raw:", "").strip()
                        elif line.startswith("Processed:"):
                            processed = line.replace("Processed:", "").strip()
                    
                    if raw or processed:
                        ts = f.replace('sign_', '').replace('.txt', '')
                        try:
                            dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
                            time_display = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            time_display = ts
                        
                        html += f'''<div class="item">
<div class="timestamp">{time_display}</div>
<div class="text">{raw}</div>
<div class="processed">→ {processed}</div>
</div>'''
            except:
                pass
    else:
        html += '<div class="empty">Chưa có dữ liệu</div>'
    
    html += '''</div></div></div></div>
<script>
setInterval(()=>{
fetch(window.location.href).then(r=>r.text()).then(html=>{
const parser=new DOMParser(),doc=parser.parseFromString(html,'text/html'),
newGrid=doc.querySelector('.data-grid'),oldGrid=document.querySelector('.data-grid');
if(newGrid&&oldGrid)oldGrid.innerHTML=newGrid.innerHTML})},5000);
</script></body></html>'''
    
    return html