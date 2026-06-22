// ============================================================================
// PlantGuard AI — Shared JavaScript Utilities (Production)
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

        // Update all theme icons
        document.querySelectorAll('#theme-icon, #theme-icon-mobile').forEach(icon => {
            if (icon) {
                icon.className = next === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
                icon.style.transform = 'rotate(180deg)';
                setTimeout(() => icon.style.transform = '', 300);
            }
        });

        showToast(`Switched to ${next} mode`, 'success');
    };

    function initTheme() {
        const saved = localStorage.getItem('plantguard-theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);

        document.querySelectorAll('#theme-icon, #theme-icon-mobile').forEach(icon => {
            if (icon) {
                icon.className = saved === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
        });
    }

    // ===== SIDEBAR TOGGLE (DESKTOP) =====
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

    function initSidebar() {
        const saved = localStorage.getItem('plantguard-sidebar');
        const sidebar = document.querySelector('.sidebar');
        if (saved === 'collapsed' && sidebar) {
            sidebar.classList.add('collapsed');
            const btn = sidebar.querySelector('.toggle-btn i');
            if (btn) btn.className = 'fas fa-chevron-right';
        }
    }

    // ===== MOBILE MENU =====
    window.toggleMobileMenu = function () {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobileOverlay');
        if (!sidebar || !overlay) return;

        const isOpen = sidebar.classList.contains('mobile-open');

        if (isOpen) {
            closeMobileMenu();
        } else {
            sidebar.classList.add('mobile-open');
            overlay.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
    };

    window.closeMobileMenu = function () {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobileOverlay');
        if (sidebar) sidebar.classList.remove('mobile-open');
        if (overlay) overlay.classList.remove('show');
        document.body.style.overflow = '';
    };

    // Close mobile menu on resize to desktop
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            if (window.innerWidth > 768) {
                closeMobileMenu();
            }
        }, 100);
    });

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

        const firstBtn = modal.querySelector('button');
        if (firstBtn) firstBtn.focus();
    };

    window.closeModal = function (id) {
        const modal = document.getElementById(id);
        if (!modal) return;
        modal.classList.remove('show');
    };

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

    // ===== API CLIENT =====
    window.apiRequest = async function (url, options = {}) {
        const defaults = {
            headers: {
                'X-CSRFToken': getCsrfToken(),
            },
        };

        if (options.json) {
            defaults.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(options.json);
            delete options.json;
        }

        const config = {
            ...defaults,
            ...options,
            headers: { ...defaults.headers, ...options.headers },
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.message || `Request failed (${response.status})`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    };

    // ===== LOADING OVERLAY =====
    window.showLoading = function (text = 'Analyzing...') {
        const overlay = document.getElementById('loadingOverlay');
        if (!overlay) return;
        const textEl = overlay.querySelector('.loading-text');
        if (textEl) textEl.textContent = text;
        overlay.classList.add('show');
    };

    window.hideLoading = function () {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.classList.remove('show');
    };

    // ===== LOGOUT =====
    window.confirmLogout = function () {
        const logoutModal = document.getElementById('logoutModal');
        if (logoutModal) {
            showModal('logoutModal');
        } else {
            // Fallback
            const logoutUrl = document.querySelector('[data-logout-url]');
            const url = logoutUrl ? logoutUrl.dataset.logoutUrl : '/logout/';
            if (confirm('Are you sure you want to logout?')) {
                window.location.href = url;
            }
        }
    };

    function initLogoutModal() {
        const btn = document.getElementById('logoutConfirmBtn');
        const logoutUrl = document.querySelector('[data-logout-url]');
        if (btn && logoutUrl) {
            btn.addEventListener('click', () => {
                window.location.href = logoutUrl.dataset.logoutUrl;
            });
        }
    }

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

    // ===== LAZY LOADING =====
    function initLazyLoading() {
        if ('IntersectionObserver' in window) {
            const lazyImages = document.querySelectorAll('img[loading="lazy"]');
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.removeAttribute('data-src');
                        }
                        observer.unobserve(img);
                    }
                });
            }, { rootMargin: '100px' });

            lazyImages.forEach(img => observer.observe(img));
        }
    }

    // ===== KEYBOARD SHORTCUTS =====
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal.show').forEach(modal => {
                closeModal(modal.id);
            });
            closeMobileMenu();
        }
    });

    // ===== STAGGERED CARD ANIMATION =====
    function animateCards() {
        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const card = entry.target;
                        const delay = parseInt(card.dataset.animateDelay) || 0;
                        setTimeout(() => {
                            card.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        }, delay);
                        observer.unobserve(card);
                    }
                });
            }, { threshold: 0.1 });

            document.querySelectorAll('[data-animate]').forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.dataset.animateDelay = (80 * index).toString();
                observer.observe(card);
            });
        } else {
            // Fallback: show all immediately
            document.querySelectorAll('[data-animate]').forEach(card => {
                card.style.opacity = '1';
                card.style.transform = 'none';
            });
        }
    }

    // ===== OFFLINE DETECTION =====
    function initOfflineDetection() {
        function updateOnlineStatus() {
            let bar = document.querySelector('.offline-bar');
            if (!bar) {
                bar = document.createElement('div');
                bar.className = 'offline-bar';
                bar.textContent = '⚠ You are offline — Some features may be unavailable';
                document.body.prepend(bar);
            }

            if (navigator.onLine) {
                bar.classList.remove('show');
            } else {
                bar.classList.add('show');
            }
        }

        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
        updateOnlineStatus();
    }

    // ===== SWIPE GESTURE (Mobile menu) =====
    function initSwipeGestures() {
        let touchStartX = 0;
        let touchStartY = 0;
        const threshold = 50;

        document.addEventListener('touchstart', (e) => {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        }, { passive: true });

        document.addEventListener('touchend', (e) => {
            if (!e.changedTouches.length) return;
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            const diffX = touchEndX - touchStartX;
            const diffY = Math.abs(touchEndY - touchStartY);

            // Only horizontal swipes (not vertical scrolling)
            if (Math.abs(diffX) > threshold && diffY < threshold) {
                if (diffX > 0 && touchStartX < 30) {
                    // Right swipe from left edge — open menu
                    toggleMobileMenu();
                } else if (diffX < 0) {
                    // Left swipe — close menu
                    const sidebar = document.getElementById('sidebar');
                    if (sidebar && sidebar.classList.contains('mobile-open')) {
                        closeMobileMenu();
                    }
                }
            }
        }, { passive: true });
    }

    // ===== INIT =====
    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        initSidebar();
        initLogoutModal();
        initLazyLoading();
        initOfflineDetection();
        initSwipeGestures();
        animateCards();
    });
})();