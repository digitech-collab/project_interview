import cv2
import mediapipe as mp
import numpy as np
import time
import threading

from .posture_classifier import PostureClassifier

class PostureDetector:
    def __init__(self):
        
        self.start_time = time.time()
        self.max_duration = 120
        self.classifier = PostureClassifier.load_model('posture\posture_classifier.joblib')
        self.camera = None
        self.is_running = False
        self.classified_postures=[]
        
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

    def generate_frames(self):
        # classified_postures=[]
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        self.camera = cv2.VideoCapture(0)
        self.is_running = True

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.is_running:
                success, frame = self.camera.read()
                if not success:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    # Extract landmarks and calculate angles (same as your original code)
                    landmarks = results.pose_landmarks.landmark
                    height, width, _ = image.shape
                    
                    # Extract key points (same as original)
                            
                    # left key points
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]

                    #right key points
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]# ... (include all the landmark extractions)
                            
                    
                    
                    # Calculate angles (same as original)
                    # ... (include all angle calculations)
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
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
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
                    if time.time() - self.start_time >= 1:
                        self.classified_postures.append(result['predicted_category'])
                        self.start_time = time.time()

                    # Add prediction text to frame
                    cv2.putText(image, str(result['predicted_category']),
                              (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                              (0, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def stop(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
