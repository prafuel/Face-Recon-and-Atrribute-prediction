import cv2
import numpy as np
import mtcnn
import os

import tensorflow as tf

from config import config

# from keras.models import Model
from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input


# Detect faces in the image.
detector = mtcnn.MTCNN()

# VGGFace Model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Fileename to save training sample face vectors

# face_vectors_npy_filename = "face_recognition/face_vectors.npy"
# - Problem with this approach is it will store multiple entries of same data
face_vectors_npy_filename = config.FACE_VECTOR_MTCNN_NPY_FILE

if not os.path.exists(face_vectors_npy_filename):
    np.save(face_vectors_npy_filename, {'face_vectors': np.array([]), 'labels': np.array([])})

def list_directories(directory):
    directories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            directories.append(entry.name)
    return directories

def get_face_vector(image):
    faces = detector.detect_faces(image)
    if len(faces) > 1 :
        max_area = 0
        largest_rectangle = None

        for face_i in faces:
            xi, yi, wi, hi = face_i['box']

            area = wi * hi
            if area > max_area:
                max_area = area
                largest_rectangle = face_i['box']

        # print("faces and exiting", faces)
    elif len(faces) == 0:
        largest_rectangle = []
        return largest_rectangle
    else:
        for face_i in faces:
            largest_rectangle = face_i['box']


    x, y, w, h = largest_rectangle
    # Draw the bounding boxes around the faces.
    crop_face = image[y:y + h, x:x + w]
    crop_face = cv2.resize(crop_face, (224, 224))

    # cv2.imshow("crop_face", crop_face)
    # cv2.waitKey()

    # crop_face = preprocess_input(crop_face, version=2)
    crop_face = np.expand_dims(crop_face, axis=0)


    # create face vector
    face_vector = model.predict(crop_face)
            # print(face_vector.shape)

    return len(faces), face_vector

def add_images_from_dir_to_npy(training_dir: str):
    """
        Using this function we can put all data from training_dir path into dataset aka face_vectors_collected.npy
        - only thing is expecting is all the images in training_dir must named by there actual persons name eg. prafull_img1.png
        Args: 
            training_dir: str
        Return None: just put data into face_vectors_collected.npy
    """
    # Get all items in training_dir
    training_dir_items = os.listdir(training_dir)

    person_name_list = [os.path.basename(item).split(".")[0] for item in training_dir_items]
    image_urls = [os.path.join(training_dir, item) for item in training_dir_items]

    add_multiple_images_to_npy(person_name_list=person_name_list, image_urls=image_urls)

def add_image_to_npy(person_name, image_url):
    """
        Using this function we can add new image data into dataset aka face_vectors_collected.npy
        Args: person_name: str
              image_url: str
        
        Return None: Directly Updates into file
    """
    image = cv2.imread(image_url)
    num_faces, face_vector_i = get_face_vector(image)  # Ensure this function returns correct shapes

    if face_vector_i.shape[0] == 1:
        # Load existing face vectors and labels
        loaded_data = np.load(face_vectors_npy_filename, allow_pickle=True).item()
        
        # Ensure the loaded face vectors are 2D
        if len(loaded_data['face_vectors'].shape) == 1:
            loaded_data['face_vectors'] = loaded_data['face_vectors'].reshape(-1, face_vector_i.shape[1])

        # Ensure face_vector_i is also 2D
        face_vector_i = face_vector_i.reshape(1, -1)

        # Append new face vector and label

        appended_face_vectors = np.concatenate((loaded_data['face_vectors'], face_vector_i), axis=0)
        appended_labels = np.concatenate((loaded_data['labels'], np.array([person_name])), axis=0)

        # Save the updated data
        appended_data = {'face_vectors': appended_face_vectors, 'labels': appended_labels}
        np.save(face_vectors_npy_filename, appended_data)

        print(f"Data added named : {person_name}")

        return {"Message": "Face added to training data"}
    else:
        return {"Error": "Could not add face to training data"}

def add_multiple_images_to_npy(person_name_list, image_urls):
    """
    Using this function we can put multiple files at once using this function
    Args: person_name_list: list
          image_urls: list
    """
    try:
        for idx, img_url in enumerate(image_urls):
            # add_image_to_training_npy(img_url)
            add_image_to_npy(person_name=person_name_list[idx], image_url=img_url)

        return {"Message": "Faces added to training data"}
    except:
        return {"Error": "Failed to add one or more images"}

def check_last_image_added_to_npy():
    loaded_data = np.load(face_vectors_npy_filename, allow_pickle=True).item()

    # Retrieve the face vectors and labels from the loaded data
    # loaded_face_vectors = loaded_data['face_vectors']
    loaded_labels = loaded_data['labels']

    return {"Last_face_added": loaded_labels[-1][0]}
    # print('loaded_face_vectors', loaded_face_vectors.shape, loaded_labels.shape)
    # print('loaded_labels', loaded_labels[-1])

if __name__ == "__main__":
    # Single image
    # image_url =  "./images/manas.jpg"

    # add_image_to_npy("manas",image_url)

    ##########################################

    # Multi Image
    # person_names = [item.split(".")[0] for item in os.listdir("./images")]
    # image_urls = ["./images/" + item for item in os.listdir("./images")]

    # print(image_urls)

    # add_multiple_images(
    #     person_names, image_urls
    # )

    ##########################################
    
    # From Dir
    add_images_from_dir_to_npy("./images")
    #############################################

    loaded_file = np.load(face_vectors_npy_filename, allow_pickle=True).item()
    print(loaded_file)