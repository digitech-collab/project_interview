import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
# import cv2
import numpy as np
import time
import os
from .posture_classifier import PostureClassifier
from django.conf import settings
from pathlib import Path
from .audioAnalyzer import AudioAnalysisView

audio_analyzer=AudioAnalysisView()

class VideoAnalyzer:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'posture_classifier.joblib')
        self.classifier = PostureClassifier.load_model(model_path)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def midpoint(self, p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def analyze_video(self, file_path):
        classified_postures = []
        classified_postures2 = ["hello world this is VideoAnalyzer output..................................."]
        cap = cv2.VideoCapture(file_path)

        frame_count = 0
        
        if not cap.isOpened():
            return"Error: File Not Found"
          
        
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            #Process every 30th frame (about 1 frame per second for 30fps video)
            if frame_count % 30 != 0:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = frame.shape
                    
                            
                # left key points
                left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
                left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]

                #right key points
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
                right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
                right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
                right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
                right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]
                            
                    
                    
                # Calculate angles
                   
                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                 #shoulder angles w.r.t elbow and opp shoulder 
                left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, right_shoulder)
                right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, left_shoulder)
                    
                #shoulder angle w.r.t torso
                right_torso_angle = self.calculate_angle(right_elbow, right_shoulder, right_hip)
                left_torso_angle = self.calculate_angle(left_elbow, left_shoulder, left_hip)
                    
                #hip angles (torso-knee angle)
                right_hip_angle=self.calculate_angle(right_shoulder, right_hip, right_knee)
                left_hip_angle=self.calculate_angle(left_shoulder, left_hip, left_knee)
                    
                #knee angles
                right_knee_angle=self.calculate_angle(right_hip, right_knee,right_ankle)
                left_knee_angle=self.calculate_angle(left_hip, left_knee,left_ankle)

                # Compute midpoints
                elbow_mid = self.midpoint(left_elbow, right_elbow)
                wrist_mid = self.midpoint(left_wrist, right_wrist) 
                    
                # Compute Euclidean distance
                elbow_distance = np.linalg.norm(np.array(left_elbow )- np.array(right_elbow))
                wrist_distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
                            
                # Draw landmarks
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                # Get prediction
                angles = {
                    "left_elbow_angle": left_elbow_angle,
                    "right_elbow_angle": right_elbow_angle,
                    "left_shoulder_angle": left_shoulder_angle,
                    "right_shoulder_angle": right_shoulder_angle,
                    "right_torso_angle": right_torso_angle,
                    "left_torso_angle": left_torso_angle,
                    "right_hip_angle": right_hip_angle,
                    "left_hip_angle": left_hip_angle,
                    "right_knee_angle": right_knee_angle,
                    "left_knee_angle": left_knee_angle,
                    "elbow_mid_x":elbow_mid[0],
                    "elbow_mid_y":elbow_mid[1],
                    "wrist_mid_x":wrist_mid[0],
                    "wrist_mid_y":wrist_mid[1],
                    "elbow_distance":elbow_distance,
                    "wrist_distance":wrist_distance
                }

                result = self.classifier.predict(angles)
                classified_postures.append(result['predicted_category'])
            # cv2.imshow('Pose Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Reshape results into 8x15 grid
        reshaped_list = []
        
        
        rows=math.ceil(len(classified_postures)/15)
        for i in range(0, len(classified_postures), 15):
            row = classified_postures[i:i+15]
            if len(row) <= 15 :  # Only add complete rows
                reshaped_list.append(row)
            if len(reshaped_list) == rows:  # Stop after 8 rows
                break

        return reshaped_list


