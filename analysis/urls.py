'''# analysis/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # الصفحة الرئيسية (رفع الملفات والتحليل)
    path('', views.analyze_view, name='analyze_view'),
]'''

# classifier/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
]