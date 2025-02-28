import os
import re
from django.shortcuts import render
import nltk
import whisper
import pyaudio

from pydub import AudioSegment, silence
from pydub.playback import play
from django.http import JsonResponse
from django.views import View
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

nltk.download('punkt')

FILLER_WORDS = {"um", "uh", "like", "you know", "so", "actually", "basically"}
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "real_time_audio.wav"

class AudioAnalysisView:
    def post(self, file_path):
        transcription = self.transcribe_audio(file_path)
        fluency_results = self.analyze_fluency(transcription, file_path)
        
        return {"transcription": transcription,
                "fluency_results": fluency_results}

    def transcribe_audio(self, audio_path):
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
        transcript = []
        words = result['segments']

        for segment in words:
            start_time = segment['start']
            text = segment['text']
            timestamp = f"[{int(start_time // 15) * 15}s]"
            transcript.append({"time":timestamp,"text":text})

        return transcript

    def analyze_fluency(self, transcription, audio_path):
        # words = nltk.word_tokenize(transcription)
        words = nltk.word_tokenize(" ".join([segment["text"] for segment in transcription]))

        total_words = len(words)
        filler_count = sum(1 for word in words if word.lower() in FILLER_WORDS)
        
        # audio = AudioSegment.from_wav(audio_path)
        audio = AudioSegment.from_file(audio_path, format="mp4")

        silent_chunks = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=-40)
        long_pauses = len(silent_chunks)
        
        total_duration = len(audio) / 1000 / 60
        wpm = total_words / total_duration if total_duration > 0 else 0
        
        fluency_score = 10
        if filler_count > 5:
            fluency_score -= 2
        if long_pauses > 3:
            fluency_score -= 2
        if wpm < 90 or wpm > 180:
            fluency_score -= 2
        
        return {
            "total_words": total_words,
            "filler_words": filler_count,
            "long_pauses": long_pauses,
            "speaking_speed_wpm": round(wpm, 2),
            "fluency_rating": max(fluency_score, 1)
        }
