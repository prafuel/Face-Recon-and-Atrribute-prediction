import numpy as np
from config import config
import queue
import mediapipe as mp
from datetime import datetime
import cv2
import os
from threading import Thread
from keras_vggface.vggface import VGGFace
from deepface import DeepFace
import dlib
import time

from src.generate_face_image_patches import CropImageForStructureInput
from src.anti_spoof_predict import AntiSpoofPredict

from src.utility import (
    time_taken, 
    draw_text_with_background, 
    blur_image, 
    get_cosine_similarity, 
    crop_image,
    parse_model_name
)
# from face_recognition_inference import run_fr, fr_video

class FaceRecognitionMerged():
    def __init__(self):
        self.result_queue = queue.Queue()
        self.processing = False
        self.skip_frames = 30 # Skip every 30 frames
        self.frame_count = 0
        self.last_result = []  # Store face boxes persistently
        self.video_writer = None
        self.features = []
        # self.last_detected_name = "Unknown"

        self.threshold = 0.60

        # Face Vectorization model
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        # Face Detection model
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        # Age and Gender Prediction Model
        self.age_net = cv2.dnn.readNet(config.AGE_MODEL, config.AGE_PROTO)
        self.gender_net = cv2.dnn.readNet(config.GENDER_MODEL, config.GENDER_PROTO)

        # Image structured cropper
        self.image_structure_cropper = CropImageForStructureInput()

        # Liveness Prediction model
        self.retina_facenet = cv2.dnn.readNetFromCaffe(
            config.RETINAFACE_PROTO,
            config.RETINAFACE_MODEL
        )

        self.anti_spoof_predict = AntiSpoofPredict(device_id=0)

        # Drowsyness
        if 'drowsiness' in self.features:
            self.dlib_face_detector = dlib.get_frontal_face_detector()
            self.dlib_face_landmark_predictor = dlib.shape_predictor(config.DLIB_FACE_LANDMARK_PREDICTOR_MODEL_PATH)

            # Constants
            self.EYE_AR_THRESH = 0.25      # Eye aspect ratio threshold for closed eyes
            self.EYE_AR_CONSEC_FRAMES = 48 # Number of consecutive frames for drowsiness detection
            self.MOUTH_AR_THRESH = 0.6     # Mouth aspect ratio threshold for yawning
            self.MOUTH_AR_CONSEC_FRAMES = 30  # Consecutive frames for yawn detection
            self.HEAD_MOVEMENT_THRESH = 5  # Threshold for head movement detection
            self.BLINK_PATTERN_THRESH = 5  # Number of blinks to check for pattern

            # Initialize counters
            self.COUNTER = 0
            self.YAWN_COUNTER = 0
            self.BLINK_COUNTER = 0
            self.DISTRACTION_COUNTER = 0
            self.blink_pattern = []
            self.prev_landmarks = None
            self.total_blinks = 0
            self.last_blink_time = time.time()

        # Face Vectors Database
        npy_database = np.load(config.DETAILED_NPY_FILE, allow_pickle=True).item()
        self.face_vectors = npy_database['face_vectors']
        self.labels = npy_database['labels']
        self.age = npy_database['age']
        self.gender = npy_database['gender']

        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

    def draw_results(self, frame, results):
        """
        Draw bounding boxes and recognition results for multiple faces.
        """
        print(results)
        
        for res in results:
            x, y, w, h = res['face_cord']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_x, text_y = x, y + h + 20

            if res.get("name", False):
                draw_text_with_background(frame, f'NAME: {res["name"]}', (text_x, text_y), font_scale=0.4)
            gap = 20

            for attr in ["age", "gender", "emotion", "is_drowsy", "concentration_level"]:
                if res.get(attr, False):
                    draw_text_with_background(frame, f'{attr.upper()}: {res[attr]}', (text_x, text_y + gap), font_scale=0.4)
                    gap = gap + 20

            if res.get("is_fake_face", False):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # To blur faces
            if res.get("blur", False):
                frame = blur_image(frame, [x, y, w, h])

        return frame

    @time_taken
    def get_face_box(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_faces_in_frame = self.face_detection.process(rgb_frame)

        results = []
        if all_faces_in_frame.detections:
            for face in all_faces_in_frame.detections:
                bboxC = face.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                        int(bboxC.width * w), int(bboxC.height * h))
                
                max_prob, index = get_cosine_similarity(
                    self.model, frame, [x, y, width, height], self.face_vectors
                )

                name = 'unknown' if max_prob < self.threshold else f'{self.labels[index]} ({max_prob})'
                age = False
                gender = False

                # Checks for real or face faces (liveliness prediction)
                liveliness_prediction = self.get_liveliness(
                    frame, [x, y, width, height]
                )

                # Skip Next Processing if fake face detected
                if liveliness_prediction[0] == "Fake_Face":
                    results.append({
                        "face_cord" : [x, y, width, height],
                        "name" : name,
                        "is_fake_face" : True
                    })
                    continue

                if name == 'unknown':
                    if 'age' in self.features:
                        age = self.get_age(frame, [x, y, width, height])
                    if 'gender' in self.features:
                        gender = self.get_gender(frame, [x, y, width, height])
                else:
                    if 'age' in self.features:
                        age = self.age[index]
                    if 'gender' in self.features:
                        gender = self.gender[index]
                
                # Emotion Detection
                deepface_analysis = {}
                cropped_image = crop_image(frame, [x, y, width, height])
                print("cropped image : ", type(cropped_image))
                if type(cropped_image) != None:
                    if 'emotion' in self.features:
                        deepface_analysis = DeepFace.analyze(
                            cropped_image,
                            actions=['emotion'],
                            enforce_detection=False
                        )
                    if isinstance(deepface_analysis, list):
                        deepface_analysis = deepface_analysis[0]

                results.append({
                    "face_cord": [x, y, width, height],
                    "blur": 'blur_face' in self.features,
                    "name" : name,
                    "age" : age, 
                    "gender" : gender,
                    "emotion" : deepface_analysis.get("dominant_emotion", False)
                })
            
        return results if results else self.last_result  # Persist previous face box if detection fails

    def process_source(self, source=0, source_type='webcam', features=[]):
        self.features = features
        if source_type == "image":
            frame = cv2.imread(source)
            if frame is None:
                print(f"Error: Could not read image at {source}")
                return
            results = self.analyze_frame(frame)
            frame = self.draw_results(frame, results)
            cv2.imwrite(f"{os.path.basename(source).split('.')[0]}_output.jpg", frame)
            print("Result saved...")
            return
        elif source_type == 'webcam':
            cap = cv2.VideoCapture(0)
        elif source_type == "video": 
            cap = cv2.VideoCapture(source)
        else:
            print("Source is not supported")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.processing = True
        first_frame = True

        while self.processing:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            face_boxes = self.get_face_box(frame)
            self.last_result = face_boxes  # Persist face boxes

            # Saving frames into video 
            if first_frame:
                self.start_video_writer(frame.shape)
                first_frame = False

            if self.frame_count % self.skip_frames == 0:
                if not self.result_queue.empty():
                    self.last_result = self.result_queue.get()
                # Thread(target=self.get_face_box_async, args=(frame.copy(),)).start()
                continue
            
            frame = self.draw_results(frame, self.last_result)
            cv2.imshow('Face Recognition', frame)

            if self.video_writer:
                self.video_writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.processing = False
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def get_age(self, image, points):
        """Predicts the age of a detected face."""
        x, y, w, h = points

        # Ensure cropping stays within image bounds
        h, w, _ = image.shape
        x, y, w, h = max(0, x), max(0, y), min(w, w - x), min(h, h - y)

        face_img = image[y:y+h, x:x+w].copy()

        if face_img.size == 0:
            print("Warning: Empty cropped face image!")
            return None

        # Convert face image to blob
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                    (78.4263377603, 87.7689143744, 114.895847746), 
                                    swapRB=False)

        # Predict Age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]

        return age
    
    def get_gender(self, image, points):
        """Predicts the gender of a detected face."""
        x, y, w, h = points

        # Ensure cropping stays within image bounds
        h, w, _ = image.shape
        x, y, w, h = max(0, x), max(0, y), min(w, w - x), min(h, h - y)

        face_img = image[y:y+h, x:x+w].copy()

        if face_img.size == 0:
            print("Warning: Empty cropped face image!")
            return None

        # Convert face image to blob
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                    (78.4263377603, 87.7689143744, 114.895847746), 
                                    swapRB=False)

        # Predict Gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        return gender
    
    def get_liveliness(self, frame, points):
        x, y, w, h = points
        predictions = np.zeros((1, 3))

        for model_name in os.listdir(config.ANTI_SPOOF_MODELS_DIR):
            h_input, w_input, model_type, scale = parse_model_name(model_name=model_name)
            params = {
                "org_img": frame,
                "bbox": (x, y, w, h),
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            if scale is None:
                params['crop'] = False

            crop_img = self.image_structure_cropper.crop(**params)
            predictions = predictions + self.anti_spoof_predict.predict(
                crop_img, os.path.join(config.ANTI_SPOOF_MODELS_DIR, model_name)
            )
        
        label = np.argmax(predictions)
        value = predictions[0][label] / 2

        # return f"{'Real Face' if label == 1 else 'Fake Face'} Score: {value:.2f}"
        return 'Real_Face' if label == 1 else 'Fake_Face', round(value, 3)

    def get_drownsiness(self, frame, points):
        face = crop_image(frame, points)
        face = cv2.resize(face, (640, 480))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        pass


    def start_video_writer(self, frame_shape):
        """
            Initializes video writer for saving the output.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join('live_stream_vids', f"output_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 encoding
        frame_width, frame_height = frame_shape[1], frame_shape[0]
        self.video_writer = cv2.VideoWriter(output_filename, fourcc, 10.0, (frame_width, frame_height))
        print(f"Recording started: {output_filename}")


def main():
    detector = FaceRecognitionMerged()
    # choice = input("Enter here choice : ")
    choice = "3"
    
    features = ["age", "gender", "emotion"]

    if choice == "1":
        img_path = input("Enter image path: ")
        detector.process_source(img_path, "image")
    
    elif choice == "2":
        vid_path = input("Enter video path: ")
        detector.features = features
        detector.process_source(vid_path, "video")
    
    elif choice == "3":
        detector.process_source(0, "webcam", features=features)

if __name__ == "__main__":
    main()
