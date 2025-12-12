// =====================================================
// APP.JS - SIGN LANGUAGE TRANSLATION SYSTEM
// =====================================================
console.log('🚀 Hệ thống phiên dịch đã khởi động');

// =====================================================
// GLOBAL STATE
// =====================================================
let updateHistoryInterval = null;
let updateStatsInterval = null;

// =====================================================
// HELPER FUNCTIONS
// =====================================================

// Helper function to escape HTML (prevent XSS)
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =====================================================
// CLIENT STATS UPDATE
// =====================================================

function updateClientStats() {
    fetch('/client_stats', {
        cache: 'no-store'
    })
        .then(response => response.json())
        .then(data => {
            // Update FPS with color coding
            const fpsEl = document.getElementById('client-fps');
            if (fpsEl) {
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
            }

            // Update predicted character
            const predictedEl = document.getElementById('client-predicted');
            if (predictedEl) {
                predictedEl.textContent = data.predicted || '--';
            }

            // Update buffer (truncate if too long)
            const bufferEl = document.getElementById('client-buffer');
            if (bufferEl) {
                const bufferText = data.buffer || '(trống)';
                bufferEl.textContent = bufferText.length > 20
                    ? bufferText.substring(0, 20) + '...'
                    : bufferText;
            }

            // Update timestamp
            const updateEl = document.getElementById('client-update');
            if (updateEl) {
                updateEl.textContent = data.last_update || '--';
            }
        })
        .catch(error => {
            // Silent fail - show offline state
            const fpsEl = document.getElementById('client-fps');
            if (fpsEl) {
                fpsEl.textContent = '--';
                fpsEl.style.color = 'var(--text-light)';
            }

            const predictedEl = document.getElementById('client-predicted');
            if (predictedEl) predictedEl.textContent = '--';

            const bufferEl = document.getElementById('client-buffer');
            if (bufferEl) bufferEl.textContent = '(offline)';
        });
}

// =====================================================
// HISTORY RELOAD FUNCTION
// =====================================================

function reloadHistory() {
    fetch('/api/history', {
        method: 'GET',
        headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        },
        cache: 'no-store' // CRITICAL: Force no caching
    })
        .then(response => response.json())
        .then(data => {
            // Update Speech History - ALWAYS CLEAR FIRST
            const speechContainer = document.getElementById('speech-history');
            if (speechContainer) {
                speechContainer.innerHTML = ''; // Clear first to prevent duplicates

                if (data.speech && data.speech.length > 0) {
                    speechContainer.innerHTML = data.speech.map(item => `
                        <div class="history-item">
                            <div class="item-time">
                                <i class="fas fa-clock"></i>
                                ${escapeHtml(item.time)}
                            </div>
                            <div class="item-text">${escapeHtml(item.text)}</div>
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
            }

            // Update Sign History - ALWAYS CLEAR FIRST
            const signContainer = document.getElementById('sign-history');
            if (signContainer) {
                signContainer.innerHTML = ''; // Clear first to prevent duplicates

                if (data.sign && data.sign.length > 0) {
                    signContainer.innerHTML = data.sign.map(item => `
                        <div class="history-item sign">
                            <div class="item-time">
                                <i class="fas fa-clock"></i>
                                ${escapeHtml(item.time)}
                            </div>
                            ${item.raw ? `
                                <div class="item-text" style="color: #666; font-size: 0.95em;">
                                    <i class="fas fa-keyboard"></i> ${escapeHtml(item.raw)}
                                </div>
                            ` : ''}
                            <div class="item-text" style="margin-top: 5px; font-weight: 600; color: var(--secondary);">
                                <i class="fas fa-arrow-right"></i> ${escapeHtml(item.processed)}
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
            }
        })
        .catch(error => {
            console.log('History reload error:', error);
        });
}

// =====================================================
// CAMERA STATUS CHECK
// =====================================================

function checkCameraStatus() {
    const img = document.getElementById('camera-stream');
    if (img) {
        img.onerror = function () {
            img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><rect width="800" height="600" fill="%23000"/><text x="50%" y="50%" text-anchor="middle" fill="%23fff" font-size="24" font-family="Arial">Đang chờ camera...</text></svg>';
        };
    }
}

// =====================================================
// DELETE HISTORY FUNCTIONALITY
// =====================================================

let pendingDeleteType = null;

function deleteHistory(type) {
    pendingDeleteType = type;

    const messages = {
        'sign': 'Bạn có chắc chắn muốn xóa toàn bộ lịch sử ký hiệu?',
        'speech': 'Bạn có chắc chắn muốn xóa toàn bộ lịch sử giọng nói?',
        'char': 'Bạn có chắc chắn muốn xóa trạng thái ký tự hiện tại?'
    };

    const modalMessage = document.getElementById('modalMessage');
    if (modalMessage) {
        modalMessage.textContent = messages[type] || 'Bạn có chắc chắn muốn xóa?';
    }

    const modal = document.getElementById('deleteModal');
    if (modal) {
        modal.classList.add('active');
    }

    // Bind confirm button
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    if (confirmBtn) {
        confirmBtn.onclick = confirmDelete;
    }
}

function closeDeleteModal() {
    const modal = document.getElementById('deleteModal');
    if (modal) {
        modal.classList.remove('active');
    }
    pendingDeleteType = null;
}

async function confirmDelete() {
    if (!pendingDeleteType) return;

    const type = pendingDeleteType;
    const confirmBtn = document.getElementById('confirmDeleteBtn');

    // Show loading state
    if (confirmBtn) {
        confirmBtn.classList.add('loading');
        confirmBtn.disabled = true;
    }

    // STEP 1: IMMEDIATELY CLEAR UI (don't wait for server)
    clearHistoryUIImmediate(type);

    try {
        // STEP 2: Send DELETE request to server
        const response = await fetch(`/web/history/${type}`, {
            method: 'DELETE',
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
            cache: 'no-store'
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Show success notification
            showNotification('success', data.message);

            // Close modal
            closeDeleteModal();

            // STEP 3: Force refresh history from server (with cache-busting)
            setTimeout(() => {
                reloadHistory();
            }, 500);
        } else {
            showNotification('error', data.error || 'Xóa thất bại');
            // Reload history to restore UI if delete failed
            reloadHistory();
        }
    } catch (error) {
        console.error('Delete error:', error);
        showNotification('error', 'Lỗi kết nối đến server');
        // Reload history to restore UI if delete failed
        reloadHistory();
    } finally {
        if (confirmBtn) {
            confirmBtn.classList.remove('loading');
            confirmBtn.disabled = false;
        }
    }
}

function clearHistoryUIImmediate(type) {
    let containerId;
    let emptyMessage;

    if (type === 'sign') {
        containerId = 'sign-history';
        emptyMessage = 'Chưa có dữ liệu ký hiệu';
    } else if (type === 'speech') {
        containerId = 'speech-history';
        emptyMessage = 'Chưa có dữ liệu giọng nói';
    }

    if (containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            // CRITICAL: Force clear DOM immediately
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>${emptyMessage}</p>
                </div>
            `;
        }
    }
}

function showNotification(type, message) {
    // Remove any existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(n => n.remove());

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        <span>${escapeHtml(message)}</span>
    `;

    document.body.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease';
        setTimeout(() => {
            notification.remove();
        }, 400);
    }, 3000);
}

// =====================================================
// INITIALIZATION
// =====================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('✅ Page loaded');

    // Initial updates
    updateClientStats();
    reloadHistory();
    checkCameraStatus();

    // Set up periodic updates
    updateStatsInterval = setInterval(updateClientStats, 1000);  // Stats every 1s
    updateHistoryInterval = setInterval(reloadHistory, 5000);    // History every 5s

    console.log('🔄 Auto-refresh enabled (Stats: 1s, History: 5s)');

    // Modal event handlers
    const modal = document.getElementById('deleteModal');
    if (modal) {
        // Close modal when clicking outside
        modal.addEventListener('click', function (e) {
            if (e.target === this) {
                closeDeleteModal();
            }
        });
    }

    // Close modal with Escape key
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            closeDeleteModal();
        }
    });
});

// Clean up on page unload
window.addEventListener('beforeunload', function () {
    if (updateHistoryInterval) {
        clearInterval(updateHistoryInterval);
    }
    if (updateStatsInterval) {
        clearInterval(updateStatsInterval);
    }
});

// Add CSS animation for notification slideout
if (!document.getElementById('notification-styles')) {
    const style = document.createElement('style');
    style.id = 'notification-styles';
    style.textContent = `
        @keyframes slideOutRight {
            to {
                opacity: 0;
                transform: translateX(100px);
            }
        }
    `;
    document.head.appendChild(style);
}

// Make functions globally accessible for inline onclick handlers
window.deleteHistory = deleteHistory;
window.closeDeleteModal = closeDeleteModal;