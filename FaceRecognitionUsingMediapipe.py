import cv2
import numpy as np
import os
import time
import queue
from threading import Thread
from datetime import datetime
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

from config import config

from src.utility import draw_text_with_background, blur_image, time_taken, crop_image

class FaceRecognitionUsingMediapipe:
    def __init__(self):
        self.result_queue = queue.Queue()
        self.processing = False
        self.skip_frames = 10  # Process every 30th frame
        self.frame_count = 0
        self.last_face_name = "Unknown"
        self.last_result = None
        self.video_writer = None
        self.features = []  # Must be a List

        # Initialize Mediapipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

        npy_database = np.load(config.FACE_VECTOR_MEDIAPIPE_NPY_FILE, allow_pickle=True).item()
        self.face_vectors = npy_database['face_vectors']
        self.labels = npy_database['labels']

    def start_video_writer(self, frame_shape):
        """
        Initializes video writer for saving the output.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join('live_stream_vids', f"output_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 encoding
        frame_width, frame_height = frame_shape[1], frame_shape[0]
        self.video_writer = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))
        print(f"Recording started: {output_filename}")

    def face_recognition(self, face_vector) -> str:
        # print("face vector" , face_vector)
        # print("face_vectors" , self.face_vectors)
        face_scores = cosine_similarity(face_vector, self.face_vectors)
        max_value = np.max(face_scores)
        max_index = np.argmax(face_scores)

        return f"{self.labels[max_index]} ({max_value})"

    @time_taken
    def analyze_frame(self, image):
        """
        Detect faces and extract landmarks using Mediapipe.
        """
        try:
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            detected_faces = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract 468 facial landmarks as (x, y, z)
                    face_vector = (
                        np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]).flatten()
                        .reshape(1, -1)
                    )

                    # Approximate bounding box from landmarks
                    x_min = int(min([lm.x for lm in face_landmarks.landmark]) * image.shape[1])
                    y_min = int(min([lm.y for lm in face_landmarks.landmark]) * image.shape[0])
                    x_max = int(max([lm.x for lm in face_landmarks.landmark]) * image.shape[1])
                    y_max = int(max([lm.y for lm in face_landmarks.landmark]) * image.shape[0])

                    recognized_face = self.face_recognition(face_vector=face_vector)
                    self.last_face_name = recognized_face

                    detected_faces.append({
                        'face_vector': face_vector,
                        'face_cord': [x_min, y_min, x_max - x_min, y_max - y_min],
                        'blur': 'blur_face' in self.features,
                        'name' : recognized_face
                    }) 

                    print(detected_faces)

            return detected_faces
        
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return []

    def process_frame_async(self, frame):
        """
        Process frame in a separate thread for better performance.
        """
        results = self.analyze_frame(frame)
        self.result_queue.put(results)

    def draw_results(self, frame, results):
        """
        Draw bounding boxes and recognition results for multiple faces.
        """
        for res in results:
            x, y, w, h = res['face_cord']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_x, text_y = x, y + h + 20

            draw_text_with_background(frame, f'NAME: {res.get("name", self.last_face_name)}', (text_x, text_y), font_scale=0.4)
            gap = 20

            for attr in ["age", "dominant_gender", "emotion", "is_drowsy", "concentration_level"]:
                if res.get(attr, False):
                    draw_text_with_background(frame, f'{attr.upper()}: {res[attr]}', (text_x, text_y + gap), font_scale=0.4)
                    gap = gap + 20

            # To blur faces
            if res.get("blur", False):
                frame = blur_image(frame, [x, y, w, h])

        return frame

    def process_source(self, source=0, source_type='webcam', features=[]):
        """
        Process input sources: webcam, video file, or image.
        """
        # update features
        self.features = features

        if source_type == 'webcam':
            cap = cv2.VideoCapture(0)

        elif source_type == 'video':
            cap = cv2.VideoCapture(source)

        elif source_type == 'image':
            frame = cv2.imread(source)
            if frame is None:
                print(f"Error: Could not read image at {source}")
                return
            results = self.analyze_frame(frame)
            frame = self.draw_results(frame, results)
            cv2.imwrite(f"{os.path.basename(source).split('.')[0]}_output.jpg", frame)
            print("Result saved...")
            return

        else:
            print("Invalid source type")
            return

        self.processing = True
        first_frame = True

        while self.processing:
            ret, frame = cap.read()
            if not ret:
                break

            if first_frame:
                self.start_video_writer(frame.shape)
                first_frame = False

            self.frame_count += 1

            if self.frame_count % self.skip_frames == 0:
                if not self.result_queue.empty():
                    self.last_result = self.result_queue.get()
                
                Thread(target=self.process_frame_async, args=(frame.copy(),)).start()
            
            if self.last_result:
                frame = self.draw_results(frame, self.last_result)

            if self.video_writer:
                self.video_writer.write(frame)  # Save frame to video file

            cv2.imshow('Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.processing = False
                break
        
        cap.release()
        if self.video_writer:
            self.video_writer.release()  # Stop video recording
            print("Recording saved...")
        cv2.destroyAllWindows()

def main():
    detector = FaceRecognitionUsingMediapipe()
    choice = '3'  # Change to '1' for image, '2' for video, '3' for webcam

    if choice == "1":
        img_path = "./images/prafull.jpg"
        detector.process_source(img_path, "image")
    
    elif choice == "2":
        vid_path = input("Enter video path: ")
        detector.process_source(vid_path, "video")
    
    elif choice == "3":
        detector.process_source(0, "webcam", features=[])

    else:
        print("Try again...")

if __name__ == "__main__":
    main()
