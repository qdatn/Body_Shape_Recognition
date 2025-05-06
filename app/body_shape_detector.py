import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class BodyShapeDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.class_names = ['Hourglass', 'Inverted Triangle', 'Rectangle', 'Apple', 'Pear']

    def detect(self, image_path):
        # Đọc ảnh và tiền xử lý
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Dự đoán kiểu dáng cơ thể
        pred = self.model.predict(img_array)
        predicted_class = np.argmax(pred, axis=1)
        return self.class_names[predicted_class[0]]



# import cv2
# import mediapipe as mp
# from .utils import calc_distance

# class BodyShapeDetector:
#     def __init__(self):
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(static_image_mode=True)

#     def detect(self, image_path):
#         image = cv2.imread(image_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = self.pose.process(image_rgb)

#         if not results.pose_landmarks:
#             raise ValueError("No person detected in the image.")

#         landmarks = results.pose_landmarks.landmark

#         left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
#         right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

#         shoulder_width = calc_distance(left_shoulder, right_shoulder)
#         hip_width = calc_distance(left_hip, right_hip)
#         waist_width = (shoulder_width + hip_width) / 2 * 0.8  # estimated waist width

#         return self.classify_body_shape(shoulder_width, waist_width, hip_width)

#     def classify_body_shape(self, shoulder_width, waist_width, hip_width):
#         if abs(shoulder_width - hip_width) < 0.05 and waist_width < shoulder_width * 0.8:
#             return "Hourglass"
#         elif hip_width > shoulder_width:
#             return "Pear"
#         elif shoulder_width > hip_width:
#             return "Inverted Triangle"
#         else:
#             return "Rectangle"
