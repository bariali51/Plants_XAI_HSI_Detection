// ============================================================================
// PlantGuard AI — Shared JavaScript Utilities
// ============================================================================

(function () {
    'use strict';

    // ===== THEME MANAGEMENT =====
    window.toggleTheme = function () {
        const html = document.documentElement;
        const current = html.getAttribute('data-theme') || 'light';
        const next = current === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', next);
        localStorage.setItem('plantguard-theme', next);

        // Update theme icon
        const icon = document.getElementById('theme-icon');
        if (icon) {
            icon.className = next === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            icon.style.transform = 'rotate(180deg)';
            setTimeout(() => icon.style.transform = '', 300);
        }

        showToast(`Switched to ${next} mode`, 'success');
    };

    // Initialize theme on load
    function initTheme() {
        const saved = localStorage.getItem('plantguard-theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);

        const icon = document.getElementById('theme-icon');
        if (icon) {
            icon.className = saved === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    // ===== SIDEBAR TOGGLE =====
    window.toggleSidebar = function () {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        sidebar.classList.toggle('collapsed');

        const isCollapsed = sidebar.classList.contains('collapsed');
        const btn = sidebar.querySelector('.toggle-btn');
        if (btn) {
            const icon = btn.querySelector('i');
            btn.setAttribute('aria-expanded', !isCollapsed);
            if (icon) {
                icon.className = isCollapsed ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
            }
        }

        localStorage.setItem('plantguard-sidebar', isCollapsed ? 'collapsed' : 'expanded');
    };

    // Initialize sidebar state
    function initSidebar() {
        const saved = localStorage.getItem('plantguard-sidebar');
        const sidebar = document.querySelector('.sidebar');
        if (saved === 'collapsed' && sidebar) {
            sidebar.classList.add('collapsed');
            const btn = sidebar.querySelector('.toggle-btn i');
            if (btn) btn.className = 'fas fa-chevron-right';
        }
    }

    // ===== TOAST NOTIFICATIONS =====
    window.showToast = function (message, type = 'success') {
        let container = document.getElementById('toastContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container';
            container.setAttribute('aria-live', 'polite');
            document.body.appendChild(container);
        }

        const toast = document.createElement('div');
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle'
        };

        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.success} toast-icon"></i>
            <span>${message}</span>
        `;

        container.appendChild(toast);

        const duration = type === 'error' ? 5000 : 3000;
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    };

    // ===== MODAL MANAGEMENT =====
    window.showModal = function (id) {
        const modal = document.getElementById(id);
        if (!modal) return;
        modal.classList.add('show');

        // Focus first button
        const firstBtn = modal.querySelector('button');
        if (firstBtn) firstBtn.focus();
    };

    window.closeModal = function (id) {
        const modal = document.getElementById(id);
        if (!modal) return;
        modal.classList.remove('show');
    };

    // Close modal on backdrop click
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal') && e.target.classList.contains('show')) {
            closeModal(e.target.id);
        }
    });

    // ===== CSRF TOKEN =====
    window.getCsrfToken = function () {
        const meta = document.querySelector('meta[name="csrf-token"]');
        if (meta) return meta.getAttribute('content');

        let cookieValue = null;
        if (document.cookie) {
            const cookies = document.cookie.split(';');
            for (const cookie of cookies) {
                const trimmed = cookie.trim();
                if (trimmed.startsWith('csrftoken=')) {
                    cookieValue = decodeURIComponent(trimmed.substring(10));
                    break;
                }
            }
        }
        return cookieValue;
    };

    // ===== LOGOUT =====
    window.confirmLogout = function () {
        const logoutUrl = document.querySelector('[data-logout-url]');
        const url = logoutUrl ? logoutUrl.dataset.logoutUrl : '/logout/';

        if (confirm('Are you sure you want to logout?')) {
            window.location.href = url;
        }
    };

    // ===== DRAG & DROP HELPERS =====
    window.setupDragDrop = function (zoneId) {
        const zone = document.getElementById(zoneId);
        if (!zone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            zone.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        ['dragenter', 'dragover'].forEach(event => {
            zone.addEventListener(event, () => zone.classList.add('drag-over'), false);
        });

        ['dragleave', 'drop'].forEach(event => {
            zone.addEventListener(event, () => zone.classList.remove('drag-over'), false);
        });

        zone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) {
                const input = zone.querySelector('input[type="file"]');
                if (input) {
                    input.files = files;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }, false);
    };

    // ===== ANIMATE COUNTER =====
    window.animateCounter = function (element, start, end, duration = 1000) {
        if (!element) return;
        const range = end - start;
        const startTime = performance.now();

        function step(currentTime) {
            const progress = Math.min((currentTime - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + range * eased);
            element.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        }

        requestAnimationFrame(step);
    };

    // ===== KEYBOARD SHORTCUTS =====
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal.show').forEach(modal => {
                closeModal(modal.id);
            });
        }
    });

    // ===== STAGGERED CARD ANIMATION =====
    function animateCards() {
        const cards = document.querySelectorAll('[data-animate]');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            setTimeout(() => {
                card.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 80 * index);
        });
    }

    // ===== INIT =====
    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        initSidebar();
        animateCards();
    });
})();