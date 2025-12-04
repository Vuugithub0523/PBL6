// App.js - Client-side logic
console.log('🚀 Hệ thống phiên dịch đã khởi động');

// Update client stats every second
function updateClientStats() {
    fetch('/client_stats')
        .then(response => response.json())
        .then(data => {
            // Update FPS
            const fpsEl = document.getElementById('client-fps');
            fpsEl.textContent = data.fps ? data.fps.toFixed(1) : '--';
            
            // Color based on FPS
            if (data.fps >= 8) {
                fpsEl.style.color = 'var(--secondary)';  // Green
            } else if (data.fps >= 5) {
                fpsEl.style.color = 'var(--accent)';     // Orange
            } else if (data.fps > 0) {
                fpsEl.style.color = 'var(--danger)';     // Red
            } else {
                fpsEl.style.color = 'var(--text-light)'; // Gray
            }
            
            // Update predicted
            const predictedEl = document.getElementById('client-predicted');
            predictedEl.textContent = data.predicted || '--';
            
            // Update buffer
            const bufferEl = document.getElementById('client-buffer');
            const bufferText = data.buffer || '(trống)';
            bufferEl.textContent = bufferText.length > 20 ? bufferText.substring(0, 20) + '...' : bufferText;
            
            // Update timestamp
            document.getElementById('client-update').textContent = data.last_update || '--';
        })
        .catch(error => {
            // Silent fail - show offline
            document.getElementById('client-fps').textContent = '--';
            document.getElementById('client-fps').style.color = 'var(--text-light)';
            document.getElementById('client-predicted').textContent = '--';
            document.getElementById('client-buffer').textContent = '(offline)';
        });
}

// Reload history every 5 seconds
function reloadHistory() {
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            // Update speech history
            const speechContainer = document.getElementById('speech-history');
            if (data.speech && data.speech.length > 0) {
                speechContainer.innerHTML = data.speech.map(item => `
                    <div class="history-item">
                        <div class="item-time">
                            <i class="fas fa-clock"></i>
                            ${item.time}
                        </div>
                        <div class="item-text">${item.text}</div>
                    </div>
                `).join('');
            } else {
                speechContainer.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <p>Chưa có dữ liệu giọng nói</p>
                    </div>
                `;
            }
            
            // Update sign history
            const signContainer = document.getElementById('sign-history');
            if (data.sign && data.sign.length > 0) {
                signContainer.innerHTML = data.sign.map(item => `
                    <div class="history-item sign">
                        <div class="item-time">
                            <i class="fas fa-clock"></i>
                            ${item.time}
                        </div>
                        <div class="item-text" style="color: #666; font-size: 0.95em;">
                            <i class="fas fa-keyboard"></i> ${item.raw}
                        </div>
                        <div class="item-text" style="margin-top: 5px; font-weight: 600; color: var(--secondary);">
                            <i class="fas fa-arrow-right"></i> ${item.processed}
                        </div>
                    </div>
                `).join('');
            } else {
                signContainer.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <p>Chưa có dữ liệu ký hiệu</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.log('History reload error:', error);
        });
}

// Check camera stream status
function checkCameraStatus() {
    const img = document.getElementById('camera-stream');
    img.onerror = function() {
        img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><rect width="800" height="600" fill="%23000"/><text x="50%" y="50%" text-anchor="middle" fill="%23fff" font-size="24" font-family="Arial">Đang chờ camera...</text></svg>';
    };
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Page loaded');
    
    // Initial update
    updateClientStats();
    checkCameraStatus();
    
    // Auto-refresh
    setInterval(updateClientStats, 1000);  // Stats every 1s
    setInterval(reloadHistory, 5000);      // History every 5s
    
    console.log('🔄 Auto-refresh enabled (Stats: 1s, History: 5s)');
});

