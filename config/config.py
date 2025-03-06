# Default Variables

FACE_VECTOR_MTCNN_NPY_FILE = "./db/face_vectors_mtcnn.npy"
FACE_VECTOR_MEDIAPIPE_NPY_FILE = "./db/face_vectors_mediapipe.npy"
DETAILED_NPY_FILE = "./db/detailed_data.npy"

RUN_LOG_FILE = "./run_log"


# Models
AGE_MODEL = "./models/age_model/age_net.caffemodel"
AGE_PROTO = "./models/age_model/deploy_age.prototxt"

GENDER_MODEL = "./models/gender_model/gender_net.caffemodel"
GENDER_PROTO = "./models/gender_model/deploy_gender.prototxt"

RETINAFACE_MODEL = "./models/detection_model/Widerface-RetinaFace.caffemodel"
RETINAFACE_PROTO = "./models/detection_model/deploy.prototxt"

ANTI_SPOOF_MODELS_DIR = "./models/anti_spoof_models"

DLIB_FACE_LANDMARK_PREDICTOR_MODEL_PATH = "./models/shape_predictor_68_face_landmarks.dat"