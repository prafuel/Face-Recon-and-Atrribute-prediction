import gradio as gr
import os

from pandas import DataFrame

# from FaceRecognitionUsingMtcnn import FaceRecognition as FR
from FaceRecognitionMerged import FaceRecognitionMerged as FR

face_recon = FR()

def analyze_face_image(frame, features):
    if frame is None:
        print(f"Error: Could not read image at {frame}")
        return
    
    face_recon.features = features
    results = face_recon.analyze_frame(frame)
    frame = face_recon.draw_results(frame, results)
    # cv2.imwrite(f"{os.path.basename(image).split('.')[0]}_output.jpg", frame)
    # print("Result saved...")
    
    return frame, DataFrame(results)[features]

def analyze_face_vid(video_path):
    result_video_path = face_recon.process_source(video_path, "video")
    # print(os.path.exists("/home/prafull/Desktop/face_recognition/" + result_video_path['path']))
    # "/home/prafull/Desktop/face_recognition/" + result_video_path['path']
    return result_video_path, result_video_path


def create_gradio_interface():
    # Initialize the face recognition system
    
    # Define available features
    feature_list = [
        "age",
        "gender",
        "emotion",
        "is_drowsy",
        "concentrantion",
        "blur_face"
    ]
    
    # Create the Gradio interface
    with gr.Blocks(title="Face Analysis System") as interface:
        gr.Markdown("# 🧑‍💻 Face Analysis System")
        gr.Markdown("Analyze facial attributes from images, videos, or live webcam.")
        
        with gr.Tabs():
            with gr.TabItem("Image Upload"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(sources=["upload"], type="numpy", label="Upload Image")
                        features = gr.CheckboxGroup(
                            choices=feature_list,
                            label="Select Features",
                            value=["age", "gender"]
                        )
                        analyze_img_btn = gr.Button("Analyze Image")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Processed Image")
                        output_df = gr.Dataframe(label="Analysis Results")
                
                analyze_img_btn.click(
                    fn=analyze_face_image,
                    inputs=[input_image, features],
                    outputs=[output_image, output_df]
                )
            
            with gr.TabItem("Video Upload"):
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="Upload Video")
                        features = gr.CheckboxGroup(
                            choices=feature_list,
                            label="Select Features",
                            value=["age", "gender"]
                        )
                        analyze_vid_btn = gr.Button("Analyze Video")
                    
                    with gr.Column():
                        output_video = gr.Video(label="Processed Video")
                        # output_df = gr.Dataframe(label="Analysis Results")
                        output_text_video = gr.Textbox(label="Resultant Video Path: ", lines=5)
                
                analyze_vid_btn.click(
                    fn=analyze_face_vid,
                    inputs=input_video,
                    outputs=[output_video,output_text_video]
                )

            with gr.TabItem("Live Webcam"):
                webcam_source = gr.Number(value=0, visible=False)  # Hidden but valid input
                webcam_label = gr.Textbox(value="webcam", visible=False)  # Gradio expects UI elements

                with gr.Column():
                    features = gr.CheckboxGroup(
                        choices=feature_list,
                        label="Select Features",
                        value=['age', 'gender']
                    )

                    # On every feature change function will trigger
                    features.change(
                        fn=face_recon.process_source,
                        inputs=[webcam_source, webcam_label, features]
                    )
                
                analyze_webcam_btn = gr.Button("Start live feed")

                analyze_webcam_btn.click(
                    fn=face_recon.process_source,
                    inputs=[webcam_source, webcam_label, features]  # Both are now valid Gradio components
                )
                
        gr.Markdown(
            """
            ## Instructions:
            1. Select the features you want to analyze.
            2. Choose between Image, Video, or Live Webcam analysis.
            3. Upload an image/video or use the webcam.
            4. Click 'Analyze' to process the input.
            5. View results in the output panel.
            """
        )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    app = create_gradio_interface()
    app.launch(share=True)