from datetime import datetime
import cv2

from face_recognition_inference import run_fr

from utility import draw_text_with_background
from src.deepface_predictions import analyze_face

from pandas import DataFrame

Process_FrameRate = 1
skip_frames = 60

def live_stream(source=0):
    print("src", source)
    current_datetime = datetime.now()
    save_path = "./live_stream_vids/" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S") + "_result.mp4"

    cap = cv2.VideoCapture(source)  # Use source parameter
    if not cap.isOpened():
        return {"Error": "Unable to open video source"}

    read_once = True
    f_cnt = 0
    video_data_list = []
    deepface_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_cnt += 1
        
        if read_once:
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), frame.shape[1], frame.shape[0]
            save_fps = fps / Process_FrameRate if fps else 30
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), save_fps, (w, h))
            read_once = False
        
        if f_cnt % skip_frames == 0:
            vid_writer.write(frame)
            continue

        # face cords, name, confidence score
        face_bb, face_recognised, face_rec_score, similarity_df = run_fr(frame)
        # age, gender, emotion
        # deepface_features.append(analyze_face(frame))

        # print(deepface_features)
        # print(f"type: {type(deepface_features)}")

        # Format results
        face_recognised_list = [face_r if face_r else "Unknown" for face_r in face_recognised]
        scores_list = ["{:.2f}".format(score) for score in face_rec_score]

        video_data_list.append([{"frame": f_cnt, "bbox": face_bb, "recognised_face_names": face_recognised_list, "scores": scores_list}])

        # Draw bounding boxes and labels
        for idx, fbb in enumerate(face_bb):
            x, y, w, h = fbb
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw face bounding box

            # Adjust text placement below the face
            text_x = x - 100 if x - 100 > 0 else 10  # Ensure it doesn't go off-screen
            text_y = y  # Align with face top

            draw_text_with_background(frame, str(face_recognised_list[idx]), (text_x, text_y), font_scale=0.5)
            draw_text_with_background(frame, f"Score: {scores_list[idx]}", (text_x, text_y + 20), font_scale=0.5)

            # draw_text_with_background(frame, f"Age: {deepface_features[idx].get('age')}", (text_x, text_y + 40), font_scale=0.5)
            # draw_text_with_background(frame, f"Gender: {deepface_features[idx].get('gender')}", (text_x, text_y + 60), font_scale=0.5)
            # draw_text_with_background(frame, f"Emotion: {deepface_features[idx].get('emotion')}", (text_x, text_y + 80), font_scale=0.5)


        vid_writer.write(frame)

        # Uncomment for live preview
        cv2.imshow('Live Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if 'vid_writer' in locals():
        vid_writer.release()

    try:
        file_to_save_remote = CR.copy_file_to_server(save_path)
    except Exception as e:
        file_to_save_remote = f"Error: Failed to save - {str(e)}"

    return {
        "Results": video_data_list,
        "video_result_path": file_to_save_remote
    }


if __name__ == "__main__":
    live_stream()

"""
To optimize it
- SSD (Single Shot MultiBox Detector) or YOLO-NAS.

"""