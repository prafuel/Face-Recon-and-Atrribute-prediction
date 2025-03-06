import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from fastapi.responses import UJSONResponse
from fastapi.middleware.cors import CORSMiddleware

import face_recognition_training as FT
import face_recognition_inference as FRI
from typing import List

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class file_path(BaseModel):
  file_name : str

class imagepaths(BaseModel):
  file_name : str
  person_name: str

class multiple_files(BaseModel):
  training_data: List[dict]

@app.get("/")
def index():
  return {"message": "Hello!!! Welcome to face recognition"}


@app.post("/face_recognition_inference", response_class=UJSONResponse)
def face_rec_infernece(ii:file_path):
  path = ii.file_name

  # Read file handler api , which checks supported files and type of media.
  # and returns true if supported file type and 2nd type of file as "image" or "video".

  url = 'http://172.21.102.212:3000/cv_file_read'
  myobj = {"file_name": path}
  file_read_response = requests.post(url, json=myobj)
  file_read_res = file_read_response.json()

  # ex : file_read_res[0] => True / False, file_read_res[1] => "image"/"video"
  # file_read_res = [True, "video"]

  if file_read_res[0]: #Checks if file supported.
    result = FRI.run_face_recognition(file_read_res[1], path)
  else:
    result = {"output": "File type not supported"}

  # print(result)
  return result


@app.post("/train_face_single", response_class=UJSONResponse)
def face_training(ii:imagepaths):
  path = ii.file_name
  person_name = ii.person_name

  # Read file handler api , which checks supported files and type of media.
  # and returns true if supported file type and 2nd type of file as "image" or "video".

  url = 'http://172.21.102.212:3000/cv_file_read'
  myobj = {"file_name": path}
  file_read_response = requests.post(url, json=myobj)
  file_read_res = file_read_response.json()

  # ex : file_read_res[0] => True / False, file_read_res[1] => "image"/"video"
  # file_read_res = [True, "image"]

  if file_read_res[0] and file_read_res[1] == "image" : #Checks if file supported.
    result = FT.add_image_to_training_npy(person_name, path)
  else:
    result = {"output": "File type not supported"}

  # print(result)
  return result


@app.post("/train_face_multiple", response_class=UJSONResponse)
def face_training_multiple(list_urls: multiple_files):
  print('list_urls', list_urls)
  tr_data = list_urls.training_data
  image_urls = []
  image_labels = []
  image_urls_unsupported = []

  for item in tr_data:
    # print("item ", item)
    path = item['file_name']
    url = 'http://172.21.102.212:3000/cv_file_read'
    myobj = {"file_name": path}
    file_read_response = requests.post(url, json=myobj)

    # file_read_res = [True, "image"]
    file_read_res = file_read_response.json()

    if file_read_res[0] and file_read_res[1] == "image":
      image_urls.append(path)
      image_labels.append(item['person_name'])
    else:
      image_urls_unsupported.append(path)
    files_checked = {"Unsupported_files": image_urls_unsupported}
    result = FT.add_multiple_images(image_labels, image_urls)

  return [result, files_checked]
  # multiple_file_data = list_urls.file_name

  # print("multiple_file_data", multiple_file_data)

  # Read file handler api , which checks supported files and type of media.
  # and returns true if supported file type and 2nd type of file as "image" or "video".

  # url = 'http://172.21.102.212:3000/cv_file_read'
  # myobj = {"file_name": path}
  # file_read_response = requests.post(url, json=myobj)
  # file_read_res = file_read_response.json()

@app.post("/fr_inference_live_stream", response_class=UJSONResponse)

def fr_inference_live_stream(ii: file_path):
  path = ii.file_name

  # Read file handler api , which checks supported files and type of media.
  # and returns true if supported file type and 2nd type of file as "image" or "video".

  # ex : file_read_res[0] => True / False, file_read_res[1] => "image"/"video"
  # file_read_res = [True, "video"]

  result = FRI.live_stream(path)

  # print(result)
  return result

  # print(result)
@app.get("/check_train_face", response_class=UJSONResponse)
def returns_last_face_added():

  result = FT.check_image_added_to_training()
  return result


if __name__ == "__main__":
  uvicorn.run("fast_api:app", host="127.0.0.1", port=3030, reload=True)
