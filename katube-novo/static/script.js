// Global utilities for the YouTube Audio Pipeline frontend

/**
 * Format seconds into human readable duration
 * @param {number} seconds 
 * @returns {string}
 */
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${remainingSeconds}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${remainingSeconds}s`;
    } else {
        return `${remainingSeconds}s`;
    }
}

/**
 * Format bytes into human readable size
 * @param {number} bytes 
 * @returns {string}
 */
function formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Validate YouTube URL
 * @param {string} url 
 * @returns {boolean}
 */
function isValidYouTubeURL(url) {
    const patterns = [
        /^https?:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)/,
        /^https?:\/\/(www\.)?youtube\.com\/embed\//,
        /^https?:\/\/(www\.)?youtube\.com\/v\//
    ];
    
    return patterns.some(pattern => pattern.test(url));
}

/**
 * Show toast notification
 * @param {string} message 
 * @param {string} type - 'success', 'error', 'info', 'warning'
 */
function showToast(message, type = 'info') {
    // Remove existing toast
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add toast to body
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast && toast.parentElement) {
            toast.remove();
        }
    }, 5000);
    
    // Add CSS if not exists
    if (!document.querySelector('#toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.textContent = `
            .toast {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 1rem;
                max-width: 400px;
                animation: slideInRight 0.3s ease;
                border-left: 4px solid;
            }
            .toast-success { border-left-color: #28a745; }
            .toast-error { border-left-color: #dc3545; }
            .toast-warning { border-left-color: #ffc107; }
            .toast-info { border-left-color: #17a2b8; }
            .toast-content { flex: 1; display: flex; align-items: center; gap: 0.5rem; }
            .toast-close { background: none; border: none; cursor: pointer; opacity: 0.7; }
            .toast-close:hover { opacity: 1; }
            @keyframes slideInRight { from { transform: translateX(100%); } to { transform: translateX(0); } }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Get icon for toast type
 * @param {string} type 
 * @returns {string}
 */
function getToastIcon(type) {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    return icons[type] || icons.info;
}

/**
 * Copy text to clipboard
 * @param {string} text 
 * @param {string} successMessage 
 */
function copyToClipboard(text, successMessage = 'Copiado para a área de transferência!') {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showToast(successMessage, 'success');
        }).catch(() => {
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

/**
 * Fallback copy method for older browsers
 * @param {string} text 
 */
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showToast('Copiado para a área de transferência!', 'success');
    } catch (err) {
        showToast('Erro ao copiar. Selecione e copie manualmente.', 'error');
    }
    
    document.body.removeChild(textArea);
}

/**
 * Debounce function execution
 * @param {Function} func 
 * @param {number} wait 
 * @returns {Function}
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function execution
 * @param {Function} func 
 * @param {number} limit 
 * @returns {Function}
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

/**
 * Show loading spinner on element
 * @param {HTMLElement} element 
 * @param {string} message 
 */
function showLoading(element, message = 'Carregando...') {
    const spinner = `
        <div class="loading-spinner">
            <i class="fas fa-spinner fa-spin"></i>
            <span>${message}</span>
        </div>
    `;
    element.innerHTML = spinner;
    
    // Add loading styles if not exists
    if (!document.querySelector('#loading-styles')) {
        const style = document.createElement('style');
        style.id = 'loading-styles';
        style.textContent = `
            .loading-spinner {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1rem;
                padding: 2rem;
                color: #666;
            }
            .loading-spinner i {
                font-size: 2rem;
                color: #667eea;
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Hide loading spinner
 * @param {HTMLElement} element 
 */
function hideLoading(element) {
    const spinner = element.querySelector('.loading-spinner');
    if (spinner) {
        spinner.remove();
    }
}

/**
 * Format time ago
 * @param {Date|string} date 
 * @returns {string}
 */
function timeAgo(date) {
    const now = new Date();
    const past = new Date(date);
    const diffInSeconds = Math.floor((now - past) / 1000);
    
    const intervals = {
        year: 31536000,
        month: 2592000,
        week: 604800,
        day: 86400,
        hour: 3600,
        minute: 60
    };
    
    for (const [unit, seconds] of Object.entries(intervals)) {
        const interval = Math.floor(diffInSeconds / seconds);
        if (interval >= 1) {
            return `${interval} ${unit}${interval > 1 ? 's' : ''} atrás`;
        }
    }
    
    return 'Agora mesmo';
}

/**
 * Smooth scroll to element
 * @param {string} selector 
 * @param {number} offset 
 */
function scrollTo(selector, offset = 0) {
    const element = document.querySelector(selector);
    if (element) {
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition = elementPosition - offset;
        
        window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
        });
    }
}

/**
 * Check if element is in viewport
 * @param {HTMLElement} element 
 * @returns {boolean}
 */
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * Add fade in animation to elements when they come into view
 */
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate
    document.querySelectorAll('.feature-card, .summary-card, .speaker-card, .download-card').forEach(el => {
        observer.observe(el);
    });
    
    // Add CSS for animations
    if (!document.querySelector('#scroll-animation-styles')) {
        const style = document.createElement('style');
        style.id = 'scroll-animation-styles';
        style.textContent = `
            .feature-card, .summary-card, .speaker-card, .download-card {
                opacity: 0;
                transform: translateY(20px);
                transition: opacity 0.6s ease, transform 0.6s ease;
            }
            .animate-fade-in {
                opacity: 1 !important;
                transform: translateY(0) !important;
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Initialize common functionality when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize scroll animations
    if (typeof IntersectionObserver !== 'undefined') {
        initScrollAnimations();
    }
    
    // Add click-to-copy functionality for URLs
    document.querySelectorAll('[data-copy]').forEach(element => {
        element.style.cursor = 'pointer';
        element.title = 'Clique para copiar';
        element.addEventListener('click', () => {
            copyToClipboard(element.dataset.copy);
        });
    });
    
    // Add form validation
    document.querySelectorAll('input[type="url"]').forEach(input => {
        input.addEventListener('blur', function() {
            if (this.value && !isValidYouTubeURL(this.value)) {
                this.setCustomValidity('Por favor, insira uma URL válida do YouTube');
            } else {
                this.setCustomValidity('');
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.querySelector('form');
            if (form && !document.getElementById('processBtn').disabled) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to close modals/errors
        if (e.key === 'Escape') {
            const errorSection = document.getElementById('errorSection');
            if (errorSection && !errorSection.classList.contains('hidden')) {
                resetForm();
            }
        }
    });
    
    // Add auto-save for form data
    const form = document.getElementById('processForm');
    if (form) {
        // Load saved data
        const savedData = localStorage.getItem('youtube-pipeline-form');
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.entries(data).forEach(([key, value]) => {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input) {
                        if (input.type === 'checkbox') {
                            input.checked = value;
                        } else {
                            input.value = value;
                        }
                    }
                });
            } catch (e) {
                console.warn('Error loading saved form data:', e);
            }
        }
        
        // Save data on change
        const saveFormData = debounce(() => {
            const formData = new FormData(form);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = value;
            }
            // Add unchecked checkboxes
            form.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                data[checkbox.name] = checkbox.checked;
            });
            localStorage.setItem('youtube-pipeline-form', JSON.stringify(data));
        }, 1000);
        
        form.addEventListener('input', saveFormData);
        form.addEventListener('change', saveFormData);
    }
});

// Export functions for global use
window.PipelineUtils = {
    formatDuration,
    formatFileSize,
    isValidYouTubeURL,
    showToast,
    copyToClipboard,
    debounce,
    throttle,
    showLoading,
    hideLoading,
    timeAgo,
    scrollTo,
    isInViewport
};
