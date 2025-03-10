import cv2
import numpy as np
import os
import time
# from deepface import DeepFace
from threading import Thread
import queue
from datetime import datetime

from config import config

# face_detection using mtcnn
from face_recognition_inference import run_fr, fr_video
from src.utility import draw_text_with_background, blur_image, time_taken

class FaceRecognitionUsingMTCNN:
    def __init__(self):
        self.result_queue = queue.Queue()
        self.processing = False
        self.skip_frames = 30  # Process every 30th frame
        self.frame_count = 0
        self.last_result = None
        self.video_writer = None  # Video writer for saving output
        self.features = [] # Must be List

        # Load stored face vectors
        npy_database = np.load(config.FACE_VECTOR_MTCNN_NPY_FILE, allow_pickle=True).item()
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

    @time_taken
    def analyze_frame(self, image):
        """
        Analyze multiple faces for recognition, emotion, and drowsiness detection.
        """

        try:
            # face_bb, face_recognised, face_rec_score, similarity_df_list = 
            face_attributes = run_fr(image)
            face_bb = face_attributes.get("face_bb")
            face_recognised = face_attributes.get("face_recognised")
            face_rec_score = face_attributes.get("face_rec_score")
            similarity_df_list = face_attributes.get("similarity_df_list")

            results = []
            for i in range(len(face_bb)):
                x, y, w, h = face_bb[i]
                face_crop = image[y:y + h, x:x + w]

                # Processing only features which asked
                actions = []
                if "age" in self.features:
                    actions.append("age")
                if "gender" in self.features:
                    actions.append("gender")
                if "emotion" in self.features or "is_drowsy" in self.features:
                    actions.append("emotion")

                # If no action is selected then will skip it
                print("actions", actions)
                if len(actions) == 0:
                    results.append({
                        'name': f"{face_recognised[i]} ({face_rec_score[i]:.2f})",
                        'face_cord': [x, y, w, h]
                    })
                    continue

                # Analyze cropped face, predict -> age
                analysis = []
                # analysis = DeepFace.analyze(
                #     face_crop,
                #     actions=actions,
                #     enforce_detection=False,
                #     silent=True
                # )

                emotion = analysis[0].get('dominant_emotion', False)
                is_drowsy = emotion in ['neutral', 'sad']

                # Print Face similarity score
                print("Class: ", face_recognised[i])
                print("Similarity_scores: ", similarity_df_list[i])
                
                # Store result for this face
                results.append({
                    'name': f"{face_recognised[i]} ({face_rec_score[i]:.2f})",
                    'face_cord': [x, y, w, h],
                    'age': analysis[0].get('age', False),
                    'gender': analysis[0].get('dominant_gender', False),
                    'is_drowsy': is_drowsy,
                    'concentration_level': ('Low' if is_drowsy else 'High') if 'concentrantion' in self.features else False,
                    'emotion': emotion if 'emotion' in self.features else False,
                    'blur' : 'blur_face' in self.features
                    # 'warning': is_drowsy
                })

            return results
        
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

            draw_text_with_background(frame, f'NAME: {res["name"]}', (text_x, text_y), font_scale=0.4)
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
            print(features)
            cap = cv2.VideoCapture(0)

        elif source_type == 'video':
            # cap = cv2.VideoCapture(source)
            return fr_video(source)

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

            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.processing = False
                break
        
        cap.release()
        if self.video_writer:
            self.video_writer.release()  # Stop video recording
            print("Recording saved...")
        cv2.destroyAllWindows()

def main():
    detector = FaceRecognitionUsingMTCNN()
    # detector.features = ['age']

    # choice = input("""Choose from above:\n1. Image\n2. Video\n3. Webcam\n""")
    choice = '3'

    if choice == "1":
        img_path = input("Enter image path: ")
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
