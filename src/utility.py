
import cv2
import numpy as np
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

# from config import config

# Time taken for function run
def time_taken(func):
    def inner_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)

        with open(os.path.join("run_log", f"func_log.txt"), "a+") as f:
            f.write(f"time taken for function {func.__name__} : {time.time() - start}\n")
        return result
    return inner_wrapper

# highlights boundary
@time_taken
def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=0.9,
                              text_color=(255, 255, 255),
                              bg_color=(0, 0, 0),
                              thickness=1,
                              padding=5):
    """
    Draw text with a background rectangle for better visibility.
    """
    x, y = position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size[0], text_size[1]

    # Draw filled rectangle as background
    cv2.rectangle(frame, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), bg_color, -1)

    # Put text on top of the rectangle
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

# Recording faces
def face_recording():
    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(0)

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")

    f_cnt = 0
    base_name = "temp_"
    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()

        if ret == True:
            cv2.imshow('Frame',frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey()

            if key == ord('s'):
                filename = os.join(base_name, "frame_", str(f_cnt), ".jpg")
                cv2.imwrite(filename, frame)

            elif key == ord('q'):
                break
            f_cnt+=1
        else:
            break

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()

# Crop images based on provided [x, y, w, h]
# @time_taken
def crop_image(image, points: list, image_shape=(224, 224)):
    x, y, w, h = points

    # Check if image is loaded
    if image is None:
        return None
        # raise ValueError("Image is None, check if it's loaded correctly.")
    
    print("image: ", image.shape)

    # Check if points are within the image bounds
    h_img, w_img, _ = image.shape
    if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
        return None
        # raise ValueError(f"Invalid cropping coordinates: {points}, Image shape: {image.shape}")

    crop_face = image[y:y + h, x:x + w]

    # Ensure crop_face is not empty
    if crop_face.size == 0:
        return None
        # raise ValueError("Cropped face is empty, check the coordinates.")

    # Resize to 224x224
    crop_face = cv2.resize(crop_face, image_shape)

    # Expand dimensions for model input
    # crop_face = np.expand_dims(crop_face, axis=0)

    return crop_face


# Blur image based on ROI (Region Of Interest)
# @time_taken

def blur_image(image, points: list):
    h_img, w_img = image.shape[:2]  # Get image dimensions
    x, y, w, h = points

    # Ensure x, y, w, h are within bounds
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(0, min(w, w_img - x))  # Ensure width doesn't exceed image
    h = max(0, min(h, h_img - y))  # Ensure height doesn't exceed image

    try:
        roi = image[y:y+h, x:x+w]

        # Apply Gaussian Blur
        blurred_roi = cv2.blur(roi, (200, 200), 0)

        # Replace the original region with blurred region
        image[y:y+h, x:x+w] = blurred_roi
        
        return image
    except Exception as e:
        print("Error occurred:", e)
        return cv2.blur(image, (200, 200), 0)
    
def get_cosine_similarity(model, frame, points, trained_face_vectors):
    try:
        crop_face = crop_image(frame, points)
        crop_face = np.expand_dims(crop_face, axis=0)

        face_vector = model.predict(
            crop_face
        )

        face_scores = cosine_similarity(face_vector, trained_face_vectors)
        max_prob = np.max(face_scores)
        index = np.argmax(face_scores)

        return max_prob, index
    except Exception as e:
        print("Error in get_cosine function : ", e)
        return 0, 0
    

def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale

def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


if __name__ == "__main__":
    print(face_recording.__name__)
