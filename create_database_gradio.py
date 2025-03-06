import gradio as gr
import os
import numpy as np

from config import config
from face_recognition_training import get_face_vector

def put_img_in_db(image, name, age, gender):
    # Placeholder function to process and store data
    if image is None or not name or not age or not gender:
        return "Please provide all details before submitting."
    
    num_faces, face_vectors = get_face_vector(image)
    
    name = name.replace(" ", "_").lower()
    age = int(age)

    if face_vectors.shape[0] == 1:
        loaded_data = np.load(config.DETAILED_NPY_FILE, allow_pickle=True).item()
        
        # Ensure the loaded face vectors are 2D
        if len(loaded_data['face_vectors'].shape) == 1:
            loaded_data['face_vectors'] = loaded_data['face_vectors'].reshape(-1, face_vectors.shape[1])

        # Ensure face_vector_i is also 2D
        face_vectors = face_vectors.reshape(1, -1)

        # Append new face vector and label
        appended_face_vectors = np.concatenate((loaded_data['face_vectors'], face_vectors), axis=0)
        appended_labels = np.concatenate((loaded_data['labels'], np.array([name])), axis=0)
        appended_age = np.concatenate((loaded_data['age'], np.array([age])), axis=0)
        appended_gender = np.concatenate((loaded_data['gender'], np.array([gender])), axis=0)

        # Save the updated data
        appended_data = {'face_vectors': appended_face_vectors, 'labels': appended_labels, 'age' : appended_age, 'gender' : appended_gender}
        np.save(config.DETAILED_NPY_FILE, appended_data)

        print(f"Data added named : {name}")

        return {"Message": "Image and details successfully stored in the database."}
    else:
        return {"Error": "Could not add face to training data"}

def create_gradio_interface():
    with gr.Blocks(title="Data Collection") as interface:
        gr.Markdown("# Capture or Upload Image & Submit Details")
        gr.Markdown("Take your solo picture using webcam or upload it, then enter your details.")
        
        with gr.Tabs():
            with gr.TabItem("Live Capture / Upload"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(sources=["webcam", "upload"], type="numpy", label="Capture or Upload Image")
                        name = gr.Textbox(label="Name")
                        age = gr.Number(label="Age")
                        gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
                        submit_btn = gr.Button("Submit")
                    
                    with gr.Column():
                        output_text = gr.Textbox(label="Status", interactive=False)
                
                submit_btn.click(
                    fn=put_img_in_db,
                    inputs=[input_image, name, age, gender],
                    outputs=[output_text]
                )
                
        gr.Markdown(
            """
            ## Instructions:
            1. Capture your solo image using webcam or upload an image.
            2. Enter your Name, Age, and select Gender.
            3. Click 'Submit' to store the details in the database.
            """
        )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()