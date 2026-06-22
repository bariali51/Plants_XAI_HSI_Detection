# config/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('i18n/', include('django.conf.urls.i18n')),  # Language switching
    path('manager/', include('manager.urls')),
    path('experts/', include('experts.urls')),
    path('community/', include('community.urls')),
    path('companies/', include('companies.urls')),
    path('', include('analysis.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)