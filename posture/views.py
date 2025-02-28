
# views.py
import os
import time
import json
import threading


from django.conf import settings
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import gzip
from django.views.decorators.csrf import csrf_exempt

#from .posture_classifier import PostureClassifier
from .videoAnalzer import VideoAnalyzer
from .postureDetector import PostureDetector
from .utils import VideoStorage
from .audioAnalyzer import AudioAnalysisView
from .feedBackResponse import fetch_llm_response

audio_analyzer=AudioAnalysisView()
video_storage = VideoStorage()
video_analyzer = VideoAnalyzer()

def index(request):
    return render(request, 'posture_detection/index.html')

def record_page(request):
    """Render the recording page."""
    return render(request, "posture_detection/video_upload.html")


#____________________________________________________________________________________________________________________________________________________________________

#             realtime recording and Analyzing LOGIC (postureDetector.py)
#____________________________________________________________________________________________________________________________________________________________________


@csrf_exempt
def save_recording(request):
    """Handle saving of recorded video."""
    if request.method == 'POST':
        try:
            # Get the blob data from the request
            video_file = request.FILES.get('video')
            if not video_file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No video data received'
                })

            # Validate the video
            video_storage.validate_video(video_file)

            # Save the recording
            filename = video_storage.save_recording_file(video_file)
            v_path= f"media/recordings//{filename}"
            
            postureList = video_analyzer.analyze_video(v_path) 
            transcript=audio_analyzer.post(v_path)

            return JsonResponse({
                'status': 'success',
                'filename': filename,
                'path': os.path.join(settings.MEDIA_URL, 'recordings', filename),
                'posture':postureList,
                 'transcript' : transcript,          
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })
    

#____________________________________________________________________________________________________________________________________________________________________

#             uploading from user files LOGIC (postureDetector.py)
#____________________________________________________________________________________________________________________________________________________________________


@csrf_exempt
def save_upload(request):
    """Handle saving of uploaded video."""
    if request.method == 'POST':
        try:
            video_file = request.FILES.get('video')
            if not video_file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No video file received'
                })

            # Validate the video
            video_storage.validate_video(video_file)
            
            

            # Save the upload
            filename = video_storage.save_upload_file(video_file)
            v_path= f"media/uploads//{filename}"
            # Construct the correct file path for analysis
            #video_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
            # print(path)
            postureList = video_analyzer.analyze_video(v_path) 
            
            transcript=audio_analyzer.post(v_path)
            
        
            return JsonResponse({
                'status': 'success',
                'filename': filename,
                'path': os.path.join(settings.MEDIA_URL, 'uploads', filename),
                'posture':postureList,
                'transcript':transcript
            })
        
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

# Add this to your cleanup tasks (e.g., management command or celery task)
def cleanup_old_videos():
    """Clean up old video files."""
    video_storage.cleanup_old_files()


#


#____________________________________________________________________________________________________________________________________________________________________

#             realtime Analyzer LOGIC (postureDetector.py)
#____________________________________________________________________________________________________________________________________________________________________

detector = PostureDetector()


@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(detector.generate_frames(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

def stop_detection(request):
    detector.stop()
    return JsonResponse({'status': 'success'})

def get_results(request):
    postures = detector.classified_postures
    reshaped_list = []
    
    # Reshape the list into 8x15 format
    for i in range(8):
        row = []
        for j in range(15):
            if i * 15 + j < len(postures):
                row.append(postures[i * 15 + j])
        if row:
            reshaped_list.append(row)
            
    return JsonResponse({'postures': reshaped_list})

