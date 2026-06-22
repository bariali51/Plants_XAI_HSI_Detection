"""
Django settings for Nabtati project — Production-Ready.
"""

import os
from pathlib import Path

from decouple import config

# Environment mode
ENVIRONMENT = config("ENVIRONMENT", default="development")
IS_PRODUCTION = ENVIRONMENT == "production"

# ===============================
# Base Directory
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent

# ===============================
# Security & API Keys
# ===============================
SECRET_KEY = config("SECRET_KEY")
DEBUG = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = ["*"]

CSRF_TRUSTED_ORIGINS = [
    "https://*.pinggy-free.link"
]

OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
GEMINI_API_KEY = config("GEMINI_API_KEY", default="")

# ===============================
# Email Configuration
# ===============================
EMAIL_HOST = config("EMAIL_HOST", default="smtp.gmail.com")
EMAIL_PORT = config("EMAIL_PORT", default=587, cast=int)
EMAIL_USE_TLS = config("EMAIL_USE_TLS", default=True, cast=bool)
EMAIL_HOST_USER = config("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD", default="")
DEFAULT_FROM_EMAIL = config("DEFAULT_FROM_EMAIL", default="noreply@nabtati.ai")

# Auto-detect backend: use SMTP if credentials exist, otherwise console
_default_backend = (
    "django.core.mail.backends.smtp.EmailBackend"
    if EMAIL_HOST_USER and EMAIL_HOST_PASSWORD
    else "django.core.mail.backends.console.EmailBackend"
)
EMAIL_BACKEND = config("EMAIL_BACKEND", default=_default_backend)

# ===============================
# Application Definition
# ===============================
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "analysis",
    "manager",
    "experts",
    "community",
    "companies",
]

ASGI_APPLICATION = "analysis.asgi.application"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "analysis.middleware.SubscriptionMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "analysis.middleware.RequestLoggingMiddleware",
    "analysis.middleware.SecurityHeadersMiddleware",
    "analysis.middleware.UploadRateLimitMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.i18n",
                "analysis.context_processors.subscription_context",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ===============================
# Database
# ===============================
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ===============================
# Password Validation
# ===============================
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ===============================
# Internationalization
# ===============================
from django.utils.translation import gettext_lazy as _

LANGUAGES = [
    ("en", _("English")),
    ("fr", _("Français")),
    ("ar", _("العربية")),
]

LANGUAGE_CODE = "en"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = True

LOCALE_PATHS = [
    BASE_DIR / "locale",
]

# ===============================
# Static & Media Files
# ===============================
STATIC_URL = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")
os.makedirs(MEDIA_ROOT, exist_ok=True)

# ===============================
# File Upload Limits
# ===============================
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB

# ===============================
# Default Primary Key
# ===============================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ===============================
# Custom User Model
# ===============================
AUTH_USER_MODEL = "analysis.CustomUser"
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"

# ===============================
# Async & Task Queue
# ===============================
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

CELERY_BROKER_URL = config("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_BACKEND = config("CELERY_RESULT_BACKEND", default="redis://localhost:6379/0")

# ===============================
# ML Model Configuration
# ===============================
ML_MODEL_PATH = config(
    "ML_MODEL_PATH",
    default=os.path.join(BASE_DIR, "analysis", "model_files", "plant_disease_prediction_model.h5"),
)
ML_CONFIDENCE_THRESHOLD = config("ML_CONFIDENCE_THRESHOLD", default=40.0, cast=float)
ML_IMG_SIZE = (128, 128)

# ===============================
# Security Settings (Production)
# ===============================
if IS_PRODUCTION:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    X_FRAME_OPTIONS = "DENY"
else:
    X_FRAME_OPTIONS = "SAMEORIGIN"

# ===============================
# Logging
# ===============================
_LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(_LOG_DIR, "django.log"),
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 3,
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "analysis": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
        "tensorflow": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}