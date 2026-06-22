/* =============================================================
   companies.js — Agricultural Platform Company JS
   ============================================================= */

/* ── CSRF Token ─────────────────────────────────────────────── */
function getCsrfToken() {
    const el = document.querySelector('[name=csrfmiddlewaretoken]');
    if (el) return el.value;
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? cookie.trim().split('=')[1] : '';
}

/* ── Tabs ────────────────────────────────────────────────────── */
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const tab = document.getElementById('tab-' + tabName);
    if (tab) tab.classList.add('active');
    if (event && event.target) event.target.classList.add('active');
}

/* ── Star Rating ─────────────────────────────────────────────── */
function setRating(value) {
    const input = document.getElementById('ratingInput');
    if (input) input.value = value;
    document.querySelectorAll('.star-input').forEach((star, idx) => {
        star.classList.toggle('active', idx < value);
    });
}

/* ── Grid/List View Toggle ───────────────────────────────────── */
function setView(mode) {
    const grid    = document.getElementById('companiesGrid');
    const gridBtn = document.getElementById('gridViewBtn');
    const listBtn = document.getElementById('listViewBtn');
    if (!grid) return;
    if (mode === 'list') {
        grid.classList.add('list-view');
        if (gridBtn) gridBtn.classList.remove('active');
        if (listBtn) listBtn.classList.add('active');
    } else {
        grid.classList.remove('list-view');
        if (gridBtn) gridBtn.classList.add('active');
        if (listBtn) listBtn.classList.remove('active');
    }
}

/* ── Manage Form Toggle ──────────────────────────────────────── */
function toggleManageForm(header) {
    const card = header.closest('.manage-form');
    const body = card ? card.querySelector('.manage-form-body') : null;
    const icon = card ? card.querySelector('.form-toggle-btn i') : null;
    if (!body) return;
    const open = body.style.display === 'none' || body.style.display === '';
    body.style.display = open ? '' : 'none';
    if (icon) icon.className = open ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
}

/* ── Search/Filter Items ─────────────────────────────────────── */
function filterItems(query, gridId) {
    const grid = document.getElementById(gridId);
    if (!grid) return;
    const cards = grid.querySelectorAll('.manage-card');
    const q = query.toLowerCase().trim();
    cards.forEach(card => {
        const name = (card.getAttribute('data-name') || '').toLowerCase();
        card.style.display = (!q || name.includes(q)) ? '' : 'none';
    });
}

/* ── Animated Counters ───────────────────────────────────────── */
function animateCounters() {
    document.querySelectorAll('.stat-number[data-count]').forEach(el => {
        const target = parseInt(el.dataset.count) || 0;
        if (target === 0) return;
        let current = 0;
        const step = Math.max(1, Math.floor(target / 30));
        const interval = setInterval(() => {
            current = Math.min(current + step, target);
            el.textContent = current;
            if (current >= target) clearInterval(interval);
        }, 30);
    });
}

/* ── Card Animations ─────────────────────────────────────────── */
function initCardAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.stat-card, .action-card, .company-card, .manage-card, .inbox-item').forEach(el => {
        observer.observe(el);
    });
}

/* ── Image Modal ─────────────────────────────────────────────── */
function openImageModal(src) {
    const modal = document.getElementById('imageModal');
    const img   = document.getElementById('modalImage');
    if (modal && img) { img.src = src; modal.classList.add('active'); }
}
function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) modal.classList.remove('active');
}

/* ── HTML Escape ─────────────────────────────────────────────── */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

/* ── Toast Notification ──────────────────────────────────────── */
function showToast(message, type = 'info') {
    let toast = document.getElementById('_companyToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = '_companyToast';
        Object.assign(toast.style, {
            position: 'fixed', bottom: '24px', right: '24px', zIndex: '9999',
            padding: '12px 20px', borderRadius: '12px', fontSize: '14px',
            fontWeight: '700', color: '#fff', boxShadow: '0 8px 24px rgba(0,0,0,.2)',
            transition: 'all .3s ease', pointerEvents: 'none', opacity: '0',
            transform: 'translateY(8px)', maxWidth: '320px',
        });
        document.body.appendChild(toast);
    }
    toast.textContent         = message;
    toast.style.background    = type === 'error' ? '#ef4444' : '#2e7d32';
    toast.style.opacity       = '1';
    toast.style.transform     = 'translateY(0)';
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
        toast.style.opacity   = '0';
        toast.style.transform = 'translateY(8px)';
    }, 3500);
}

/* =============================================================
   CHAT — Send Message
   ============================================================= */
function sendCompanyMessage(event) {
    event.preventDefault();

    const form          = document.getElementById('chatForm');
    const input         = document.getElementById('chatInput');
    const imageInput    = document.getElementById('chatImage');
    const fileInput     = document.getElementById('chatFile');
    const sendBtn       = document.getElementById('sendBtn');
    const conversationId = form ? form.querySelector('[name=conversation_id]').value : null;

    if (!conversationId) return;

    const text = input ? input.value.trim() : '';
    const hasImage = imageInput && imageInput.files.length > 0;
    const hasFile  = fileInput  && fileInput.files.length  > 0;

    if (!text && !hasImage && !hasFile) return;

    // Build FormData — always multipart so files work
    const fd = new FormData();
    fd.append('conversation_id', conversationId);
    fd.append('text', text);
    if (hasImage) fd.append('image', imageInput.files[0]);
    if (hasFile)  fd.append('file',  fileInput.files[0]);

    // Disable button during request
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    }

    fetch('/companies/api/send-message/', {
        method:  'POST',
        headers: { 'X-CSRFToken': getCsrfToken() },
        body:    fd,
    })
    .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
    })
    .then(data => {
        if (data.status === 'ok') {
            appendMessage(data.message);
            // Reset input
            if (input) { input.value = ''; }
            // Reset file inputs
            if (imageInput) imageInput.value = '';
            if (fileInput)  fileInput.value  = '';
            // Hide file preview
            const fp = document.getElementById('filePreview');
            if (fp) fp.style.display = 'none';
            const fn = document.getElementById('fileName');
            if (fn) fn.textContent = '';
            scrollToBottom();
        } else {
            showToast(data.error || 'Failed to send message.', 'error');
        }
    })
    .catch(err => {
        console.error('Send error:', err);
        showToast('Connection error — message not sent. Try again.', 'error');
    })
    .finally(() => {
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
        if (input) input.focus();
    });
}

/* =============================================================
   CHAT — Append Message to DOM
   ============================================================= */
function appendMessage(msg) {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    // Remove empty-state placeholder
    const empty = container.querySelector('.chat-empty-ultra');
    if (empty) empty.remove();

    // Avoid duplicates
    if (container.querySelector(`[data-msg-id="${msg.id}"]`)) return;

    const wrap = document.createElement('div');
    wrap.className = 'msg-ultra-wrap ' + (msg.is_me ? 'msg-mine' : 'msg-other');
    wrap.setAttribute('data-msg-id', String(msg.id));

    const initMe    = window.MY_USERNAME  ? window.MY_USERNAME[0].toUpperCase()  : 'U';
    const initOther = window.OTHER_NAME   ? window.OTHER_NAME[0].toUpperCase()   : 'O';
    const avatar    = msg.is_me ? initMe : initOther;

    let inner = `<div class="msg-avatar-pro">${avatar}</div><div class="msg-bubble-ultra">`;

    if (msg.text) {
        inner += `<div class="msg-text-pro">${escapeHtml(msg.text)}</div>`;
    }
    if (msg.image) {
        inner += `<div class="msg-image-pro" onclick="openImageModal('${msg.image}')">
                    <img src="${msg.image}" alt="Image attachment" style="max-width:200px;border-radius:8px;cursor:pointer;">
                  </div>`;
    }
    if (msg.file) {
        const fname = msg.file_name || 'Download file';
        inner += `<div class="msg-file-pro">
                    <i class="fas fa-file-alt"></i>
                    <a href="${msg.file}" target="_blank" rel="noopener">${escapeHtml(fname)}</a>
                  </div>`;
    }

    const time = msg.created_at || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const tick  = msg.is_me ? '<i class="fas fa-check-double" style="margin-left:4px;"></i>' : '';
    const delBtn = msg.is_me
        ? `<button class="msg-delete-btn" onclick="deleteMessage(${msg.id})" title="Delete message" style="margin-left:6px;background:none;border:none;cursor:pointer;color:inherit;opacity:.5;font-size:11px;">
               <i class="fas fa-trash-alt"></i>
           </button>`
        : '';

    inner += `<div class="msg-meta-ultra">${time}${tick}${delBtn}</div></div>`;
    wrap.innerHTML = inner;

    // Animate in
    wrap.style.opacity   = '0';
    wrap.style.transform = 'translateY(10px)';
    container.appendChild(wrap);
    requestAnimationFrame(() => {
        wrap.style.transition = 'all 0.2s ease';
        wrap.style.opacity    = '1';
        wrap.style.transform  = 'translateY(0)';
    });
}

/* =============================================================
   CHAT — Scroll to Bottom
   ============================================================= */
function scrollToBottom() {
    const c = document.getElementById('chatMessages');
    if (c) c.scrollTop = c.scrollHeight;
}

/* =============================================================
   CHAT — Delete Message
   ============================================================= */
function deleteMessage(msgId) {
    if (!confirm('Delete this message?')) return;

    fetch(`/companies/api/delete-message/${msgId}/`, {
        method:  'POST',
        headers: { 'X-CSRFToken': getCsrfToken() },
    })
    .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
    })
    .then(data => {
        if (data.status === 'ok') {
            const el = document.querySelector(`[data-msg-id="${msgId}"]`);
            if (el) {
                el.style.transition = 'all 0.25s ease';
                el.style.opacity    = '0';
                el.style.transform  = 'scale(0.85)';
                setTimeout(() => el.remove(), 260);
            }
            showToast('Message deleted.', 'info');
        } else {
            showToast(data.error || 'Could not delete message.', 'error');
        }
    })
    .catch(err => {
        console.error('Delete error:', err);
        showToast('Network error — could not delete.', 'error');
    });
}

/* =============================================================
   CHAT — Message Polling (every 4 sec)
   ============================================================= */
let _pollInterval = null;

function startMessagePolling(conversationId) {
    if (_pollInterval) clearInterval(_pollInterval);

    _pollInterval = setInterval(() => {
        const container = document.getElementById('chatMessages');
        if (!container) return;

        // Get last message ID currently in DOM
        const msgs   = container.querySelectorAll('.msg-ultra-wrap[data-msg-id]');
        const lastId = msgs.length ? msgs[msgs.length - 1].getAttribute('data-msg-id') : '0';

        fetch(`/companies/api/messages/${conversationId}/?after=${lastId}`)
        .then(r => r.json())
        .then(data => {
            if (data.status === 'ok' && data.messages && data.messages.length) {
                let appended = false;
                data.messages.forEach(msg => {
                    // Only append OTHER person's messages (own are already appended instantly)
                    if (!msg.is_me) {
                        appendMessage(msg);
                        appended = true;
                    }
                });
                if (appended) {
                    scrollToBottom();
                    // Mark as read
                    fetch(`/companies/api/mark-read/${conversationId}/`, {
                        method:  'POST',
                        headers: { 'X-CSRFToken': getCsrfToken() },
                    }).catch(() => {});
                }
            }
        })
        .catch(() => {}); // silent — polling failures are non-critical
    }, 4000);
}

/* =============================================================
   DOM READY
   ============================================================= */
document.addEventListener('DOMContentLoaded', () => {
    animateCounters();
    initCardAnimations();
});
