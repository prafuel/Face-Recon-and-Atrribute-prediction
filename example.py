from fastapi import FastAPI, File, UploadFile, Response
import cv2
import numpy as np
import uvicorn
from threading import Thread
from io import BytesIO
from starlette.responses import StreamingResponse
from FaceRecognitionMerged import FaceRecognitionMerged

app = FastAPI()
detector = FaceRecognitionMerged()

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    features = ["age", "gender", "emotion"]
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # results = detector.get_face_box(image)
    # processed_image = detector.draw_results(image, results)

    processed_image = detector.process_source(source=image, features=features, source_type="image")
    _, img_encoded = cv2.imencode(".jpg", processed_image)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    contents = await file.read()
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(contents)
    detector.process_source(video_path, "video")
    return {"status": "Video processed and saved"}

@app.get("/live_feed")
async def live_feed():
    def video_stream():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = detector.get_face_box(frame)
            processed_frame = detector.draw_results(frame, results)
            _, buffer = cv2.imencode(".jpg", processed_frame)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        cap.release()
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run("face_recognition_api:app", host="0.0.0.0", port=8000, reload=True)
