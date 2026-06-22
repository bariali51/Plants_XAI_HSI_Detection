/* community.js — Premium Agricultural Social Platform Community JS */

function getCsrfToken() {
    const el = document.querySelector('[name=csrfmiddlewaretoken]');
    if (el) return el.value;
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? cookie.split('=')[1] : '';
}

/* ── Like ────────────────────────────────────────────────── */
function toggleLike(postId) {
    fetch(`/community/api/like/${postId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            const btn = document.getElementById('likeBtn');
            const count = document.getElementById('likeCount');
            btn.classList.toggle('liked', data.liked);
            count.textContent = data.count;
        }
    });
}

/* ── Bookmark ────────────────────────────────────────────── */
function toggleBookmark(postId) {
    fetch(`/community/api/bookmark/${postId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            const btn = document.getElementById('bookmarkBtn');
            btn.classList.toggle('bookmarked', data.bookmarked);
        }
    });
}

/* ── Comment Like ────────────────────────────────────────── */
function toggleCommentLike(commentId) {
    fetch(`/community/api/comment-like/${commentId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            const el = document.getElementById(`cl-${commentId}`);
            if (el) el.textContent = data.count;
        }
    });
}

/* ── Accept Answer ───────────────────────────────────────── */
function acceptAnswer(commentId) {
    fetch(`/community/api/accept-answer/${commentId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            location.reload();
        }
    });
}

/* ── Submit Comment ──────────────────────────────────────── */
function submitComment(event, postId) {
    event.preventDefault();
    const body = document.getElementById('commentBody').value.trim();
    if (!body) return;

    fetch('/community/api/comment/', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ post_id: postId, body: body }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            location.reload();
        } else {
            alert(data.error || 'Error posting comment.');
        }
    });
}

/* ── Reply ───────────────────────────────────────────────── */
function showReplyForm(commentId) {
    const el = document.getElementById(`replyForm-${commentId}`);
    el.style.display = el.style.display === 'none' ? 'flex' : 'none';
    if (el.style.display === 'flex') {
        el.querySelector('textarea').focus();
    }
}

function submitReply(postId, parentId) {
    const body = document.getElementById(`replyBody-${parentId}`).value.trim();
    if (!body) return;

    fetch('/community/api/comment/', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ post_id: postId, body: body, parent_id: parentId }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            location.reload();
        }
    });
}

/* ── Image Preview ───────────────────────────────────────── */
function previewImages(input) {
    const container = document.getElementById('imagePreviews');
    if (!container) return;
    container.innerHTML = '';
    const files = Array.from(input.files).slice(0, 4);
    files.forEach(file => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'preview-thumb-sm';
            container.appendChild(img);
        };
        reader.readAsDataURL(file);
    });
}

/* ── Tag Input ───────────────────────────────────────────── */
function addTag(name) {
    const input = document.querySelector('.tag-input');
    if (!input) return;
    const current = input.value.trim();
    const tags = current ? current.split(',').map(t => t.trim()) : [];
    if (!tags.includes(name)) {
        tags.push(name);
        input.value = tags.join(', ');
    }
}

/* ── Image Modal ─────────────────────────────────────────── */
function openImageModal(src) {
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('modalImage');
    if (modal && img) {
        img.src = src;
        modal.classList.add('active');
    }
}
function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) modal.classList.remove('active');
}

/* ── Recommendations ─────────────────────────────────────── */
function loadRecommendations(postId) {
    const container = document.getElementById('recContent');
    if (!container) return;

    fetch(`/community/api/recommendations/${postId}/`)
    .then(r => r.json())
    .then(data => {
        if (data.status !== 'ok') {
            container.innerHTML = '<p style="font-size:13px;color:#9ca3af;">No recommendations available.</p>';
            return;
        }
        let html = '';
        if (data.companies && data.companies.length) {
            html += '<h4 style="font-size:13px;font-weight:700;margin:8px 0 6px;">🏢 Companies</h4>';
            data.companies.forEach(c => {
                html += `<a href="/companies/${c.id}/" style="display:block;padding:6px 0;font-size:13px;color:var(--text);text-decoration:none;">
                    ${c.name} ${c.verified ? '<i class="fas fa-check-circle" style="color:#3b82f6;font-size:11px;"></i>' : ''}
                </a>`;
            });
        }
        if (data.experts && data.experts.length) {
            html += '<h4 style="font-size:13px;font-weight:700;margin:12px 0 6px;">👨‍🔬 Experts</h4>';
            data.experts.forEach(e => {
                html += `<a href="/community/profile/${e.id}/" style="display:block;padding:6px 0;font-size:13px;color:var(--text);text-decoration:none;">
                    ${e.name} <span style="font-size:11px;color:#9ca3af;">${e.specialization}</span>
                </a>`;
            });
        }
        if (data.similar_posts && data.similar_posts.length) {
            html += '<h4 style="font-size:13px;font-weight:700;margin:12px 0 6px;">📄 Similar Posts</h4>';
            data.similar_posts.forEach(p => {
                html += `<a href="/community/post/${p.id}/" style="display:block;padding:6px 0;font-size:13px;color:var(--text);text-decoration:none;">
                    ${p.title} ${p.status === 'resolved' ? '✅' : ''}
                </a>`;
            });
        }
        if (!html) html = '<p style="font-size:13px;color:#9ca3af;">No recommendations yet. Add tags to get suggestions.</p>';
        container.innerHTML = html;
    })
    .catch(() => {
        container.innerHTML = '<p style="font-size:13px;color:#9ca3af;">Could not load recommendations.</p>';
    });
}

/* ── Edit Comment ────────────────────────────────────────── */
function editComment(commentId) {
    const bodyEl = document.querySelector(`#comment-${commentId} .comment-body`);
    if (!bodyEl) return;
    const currentText = bodyEl.textContent.trim();
    const newText = prompt('Edit your comment:', currentText);
    if (newText === null || newText.trim() === '' || newText.trim() === currentText) return;

    fetch(`/community/api/comment/${commentId}/edit/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ body: newText.trim() }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            bodyEl.textContent = data.comment.body;
        } else {
            alert(data.error || 'Failed to edit comment.');
        }
    });
}

/* ── Delete Comment ──────────────────────────────────────── */
function deleteComment(commentId) {
    if (!confirm('Are you sure you want to delete this comment?')) return;

    fetch(`/community/api/comment/${commentId}/delete/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            const el = document.getElementById(`comment-${commentId}`);
            if (el) {
                el.style.transition = 'opacity 0.3s, transform 0.3s';
                el.style.opacity = '0';
                el.style.transform = 'translateX(-20px)';
                setTimeout(() => el.remove(), 300);
            }
        } else {
            alert(data.error || 'Failed to delete comment.');
        }
    });
}

/* ── Report Post ─────────────────────────────────────────── */
function reportPost(postId) {
    const reason = prompt('Reason for reporting?\n1. Spam\n2. Offensive\n3. Misinformation\n4. Other\n\nEnter number or type reason:');
    if (!reason) return;

    const reasonMap = { '1': 'spam', '2': 'offensive', '3': 'misinformation', '4': 'other' };
    const mappedReason = reasonMap[reason.trim()] || 'other';
    const detail = reason.trim().length > 1 && !reasonMap[reason.trim()] ? reason.trim() : '';

    fetch(`/community/api/report/${postId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: mappedReason, detail: detail }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            alert('Report submitted. Thank you.');
        } else {
            alert(data.error || 'Failed to submit report.');
        }
    });
}

/* ── Share Post ──────────────────────────────────────────── */
function sharePost(postId, title) {
    const url = window.location.origin + '/community/post/' + postId + '/';
    const modal = document.getElementById('shareModal');
    const input = document.getElementById('shareLink');
    if (modal && input) {
        input.value = url;
        modal.classList.add('active');
    } else {
        navigator.clipboard.writeText(url).then(() => {
            alert('Link copied!');
        });
    }
}

function copyShareLink() {
    const input = document.getElementById('shareLink');
    const btn = document.getElementById('copyBtn');
    if (input) {
        navigator.clipboard.writeText(input.value).then(() => {
            if (btn) {
                btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => { btn.innerHTML = '<i class="fas fa-copy"></i> Copy'; }, 2000);
            }
        });
    }
}

function closeShareModal() {
    const modal = document.getElementById('shareModal');
    if (modal) modal.classList.remove('active');
}

/* ── Staggered Card Entrance Animations ──────────────────── */
function initFeedAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.05 });

    document.querySelectorAll('.post-card, .sidebar-card, .trending-item').forEach(el => {
        observer.observe(el);
    });
}

/* ── Init on DOM Ready ───────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function() {
    initFeedAnimations();

    // Keyboard support for image modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeImageModal();
            closeShareModal();
        }
    });
});
