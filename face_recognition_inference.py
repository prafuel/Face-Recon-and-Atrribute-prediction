import cv2
import mtcnn
from pandas import DataFrame
import numpy as np
# from keras.models import Model
from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
import os
from sklearn.metrics.pairwise import cosine_similarity
# import copy_video_to_remote as CR

from src.deepface_predictions import analyze_face
from utility import draw_text_with_background, crop_image, time_taken

from config import config

# from live_stream import live_stream

thresh = 0.60
Process_FrameRate = 1

# Detect faces in the image
detector = mtcnn.MTCNN()

# VGGFace Model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# face_vectors = []
# load trained faces
# faces_loaded = np.load("face_recognition/facevector.npy")
loaded_data = np.load(config.DETAILED_NPY_FILE, allow_pickle=True).item()
# {"face_vectors" : np.array[], "labels" : np.array[]}

# print('loaded_data', loaded_data.shape)
# Retrieve the face vectors and labels from the loaded data
loaded_face_vectors = loaded_data['face_vectors']
loaded_labels = loaded_data['labels']
loaded_age = loaded_data['age']
loaded_gender = loaded_data['gender']

faces_trained = loaded_face_vectors

# faces_trained = np.reshape(loaded_face_vectors, (-1, 2048))
# print("faces_trained", faces_trained.shape)
# print(faces_trained.shape)

@time_taken
def run_fr(image):
    face_rec_score = []
    face_recognised = []
    face_bb = []
    faces = detector.detect_faces(image)
    similarity_df = []

    age_list = []
    gender_list = []

    # deepface_analysis = []

    # Draw the bounding boxes around the faces.
    for face in faces:
        x, y, w, h = face['box']
        # crop image based on roi
        crop_face = crop_image(image, face['box'])

        # create face vector
        face_vector = model.predict(crop_face)
        # print(face_vector.shape)

        print(crop_face)

        # Deepface outputs
        # deepface_analysis.append(
        #     analyze_face(crop_face)
        # )

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # face_vectors.append(face_vector)
        # Check the faces
        face_scores = cosine_similarity(face_vector, faces_trained)

        similarity_df.append(
            DataFrame()
            .assign(
                labels = loaded_labels,
                probs = face_scores[0]
            )
            .sort_values(by=['probs'], ascending=False)
            .head(5)
        )

        max_value = np.max(face_scores)
        max_index = np.argmax(face_scores)

        if max_value > thresh:
            # label_idx = max_index // 5  # 5 images taken for each person
            label = loaded_labels[max_index]
        else:
            # label = np.array(["Unknown"])
            label = "Unknown"
        # print('label', label)

        face_bb.append(face['box'])

        face_recognised.append(label)
        face_rec_score.append(max_value)
        age_list.append(loaded_age[max_index])
        gender_list.append(loaded_gender[max_index])

    # result = [{"bbox": face_bb, "recognised_face_names":face_recognised, "score":face_rec_score}]
    # return face_bb, face_recognised, face_rec_score, similarity_df

    return {
        "face_bb": face_bb,
        "face_recognised" : face_recognised,
        "face_rec_score" : face_rec_score,
        "similartiy_df" : similarity_df,
        "age" : age_list,
        "gender" : gender_list
    }

def fr_video(source = 0, save_path="result.mp4"):
    save_path = os.path.splitext(os.path.basename(source))[0]
    save_path += "_result.mp4"
    video_data_list = []
    # Load the image.
    cap = cv2.VideoCapture(source)
    read_once = True
    f_cnt = 0

    while(True):

        # Capture the video frame
        # by frame
        ret, frame = cap.read()
        if not ret:
            break

        if read_once:
            fps, w, h = 30, frame.shape[1], frame.shape[0]

            fps = cap.get(cv2.CAP_PROP_FPS)
            save_fps = fps / Process_FrameRate
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), save_fps, (w, h))
            read_once = False

        face_bb, face_recognised, face_rec_score, similarity_df = run_fr(frame)
        deepface_features = analyze_face(frame)
        # print('face ', face_recognised)
        face_recognised_list = []
        scores_list = []
        for idx, face_r in enumerate(face_recognised):
            face_recognised_list.append(face_r[0])
            scores_list.append("{:.2f}".format(face_rec_score[idx]))
        # for idx, fbb in enumerate(face_bb):
        #     face_recognised_list = np.concatenate(face_recognised[idx]).tolist()
        video_data_list.append([{"frame": f_cnt, "bbox": face_bb,"recognised_face_names": face_recognised_list, "scores":scores_list}])
        # print('face_bb, face_recognised, face_rec_score ', face_bb, face_recognised, face_rec_score )
        # Draw the bounding boxes around the faces.
        for idx, fbb in enumerate(face_bb):
            x, y, w, h = fbb
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            draw_text_with_background(frame, str(face_recognised[idx]), (x, y - 10))
            draw_text_with_background(frame, f"Score: {face_rec_score[idx]}", (x, y + 15))

            draw_text_with_background(frame, f"Age: {deepface_features.get('age')}", (x, y + 50))
            draw_text_with_background(frame, f"Gender: {deepface_features.get('gender')}", (x, y + 75))
            draw_text_with_background(frame, f"Emotion: {deepface_features.get('emotion')}", (x, y + 100))

        vid_writer.write(frame)
        # Display the output
        # cv2.imshow('frame', frame)

        # press 'q' button to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        f_cnt += 1

        if f_cnt > 30:
            break

    # save face_vector file for Face Training #
    # np.save('facevector.npy', face_vectors)
    # print("saved faces")
    # release camera and close all windows
    cap.release()
    try:
        vid_writer.release()
    except:
        pass
    # vid_writer.release()
    # cv2.destroyAllWindows()
    try:
        file_to_save_remote = CR.copy_file_to_server(save_path)
    except:
        file_to_save_remote = "Error:Failed to save"
    # print("video_data_list", video_data_list)
    return {"path" : save_path}
    # {"Results": video_data_list,
    #         "video_result_path": file_to_save_remote
    #         }

def run_face_recognition(type_of_input, filename):
    if type_of_input == "image":
        # image = imread_from_url(filename)
        image = cv2.imread(filename)
        face_bb, face_recognised, face_rec_score, deepface_features = run_fr(image)
        # deepface_features = analyze_face(image)

        result = [{"bbox": face_bb, "recognised_face_names":face_recognised, "score":face_rec_score}]
        # print(deepface_features)

        # print(result)

        for idx, f_bb in enumerate(face_bb):
            x, y, w, h = f_bb
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_x = x - 100 if x - 100 > 0 else 10  # Ensure it doesn't go off-screen
            text_y = y  # Align with face top

            draw_text_with_background(image, str(face_recognised[idx]), (text_x, text_y), font_scale=0.5)
            draw_text_with_background(image, f"Score: {face_rec_score[idx]}", (text_x, text_y + 20), font_scale=0.5)

            draw_text_with_background(image, f"Age: {deepface_features[idx].get('age')}", (text_x, text_y + 40), font_scale=0.5)
            draw_text_with_background(image, f"Gender: {deepface_features[idx].get('gender')}", (text_x, text_y + 60), font_scale=0.5)
            draw_text_with_background(image, f"Emotion: {deepface_features[idx].get('emotion')}", (text_x, text_y + 80), font_scale=0.5)


            cv2.imwrite(f"{face_recognised[idx]}_output.jpg", image)

    elif type_of_input == "video":
        result = fr_video(filename)
    else:
        result = {"Error": "File cannot be processed"}

    return result


if __name__ == "__main__":
    # res = fr_video("./head-pose-face-detection-male.mp4")

    # im = cv2.imread("./images/prafull_test1.jpg")
    # face_bb, face_recognised, face_rec_score = run_fr(im)

    res = run_face_recognition("video","/home/prafull/Downloads/head-pose-face-detection-male.mp4" )
    print("res", res)

    # print("im", im)
    # print("face_bb",face_bb)
    # print("face_recognition", face_recognised)
    # print("face_rec_score", face_rec_score)


    pass

