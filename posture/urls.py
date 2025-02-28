# urls.py
from django.urls import path
from . import views
from .views import save_recording  # Ensure the correct import
from django.conf import settings
from django.conf.urls.static import static
from .views import fetch_llm_response

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('temp/stop_detection/', views.stop_detection, name='stop_detection'),
    path('temp/get_results/', views.get_results, name='get_results'),
    path('temp/save-upload/', views.save_upload, name='save_upload'),
    path('temp/save-recording/', save_recording, name='save_recording'),
    path("temp/fetch-llm-response/", fetch_llm_response, name="fetch_llm_response"),
    path("temp/", views.record_page, name="record_page"),
    
]
# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


