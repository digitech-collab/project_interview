import os
import uuid
from datetime import datetime
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import ffmpeg

class VideoStorage:
    def __init__(self):
        self.recording_storage = FileSystemStorage(location=settings.RECORDING_DIR)
        self.upload_storage = FileSystemStorage(location=settings.UPLOAD_DIR)
        self.temp_storage = FileSystemStorage(location=settings.TEMP_DIR)

    def generate_filename(self, original_filename):
        """Generate a unique filename with timestamp and UUID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(original_filename)[1]
        return f"{timestamp}_{unique_id}{ext}"

    def save_recording_file(self, file_data):
        """Save a recorded video file."""
        filename = self.generate_filename('recording.webm')
        # Convert WebM to MP4 for better compatibility
        temp_path = self.temp_storage.save(filename, file_data)
        mp4_filename = os.path.splitext(filename)[0] + '.mp4'
        output_path = os.path.join(settings.RECORDING_DIR, mp4_filename)
        
        try:
            # Convert to MP4 using ffmpeg
            stream = ffmpeg.input(os.path.join(settings.TEMP_DIR, temp_path))
            stream = ffmpeg.output(stream, output_path)
            ffmpeg.run(stream, overwrite_output=True)
            
            # Clean up temp file
            self.temp_storage.delete(temp_path)
            
            return mp4_filename
        # except ffmpeg.Error as e:
        #     # Clean up temp file
        #     self.temp_storage.delete(temp_path)
        #     raise Exception(f"Error converting video: {str(e)}")
        except Exception as e:
            # Handle any other exceptions
            self.temp_storage.delete(temp_path)
            raise Exception(f"An unexpected error occurred: {str(e)}")
        

    def save_upload_file(self, uploaded_file):
        """Save an uploaded video file."""
        filename = self.generate_filename(uploaded_file.name)
        return self.upload_storage.save(filename, uploaded_file)

    def validate_video(self, file):
        """Validate video file size and format."""
        if file.size > settings.MAX_VIDEO_SIZE:
            raise ValueError(f"File size exceeds {settings.MAX_VIDEO_SIZE // (1024*1024)}MB limit")
        
        if file.content_type not in settings.ALLOWED_VIDEO_FORMATS:
            raise ValueError("Invalid video format. Allowed formats: WebM, MP4, AVI")

    def cleanup_old_files(self, days=7):
        """Clean up files older than specified days."""
        cleanup_dirs = [settings.RECORDING_DIR, settings.UPLOAD_DIR, settings.TEMP_DIR]
        current_time = datetime.now().timestamp()
        
        for directory in cleanup_dirs:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                file_time = os.path.getctime(filepath)
                
                if (current_time - file_time) > (days * 24 * 60 * 60):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"Error deleting {filepath}: {str(e)}")