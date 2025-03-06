from deepface import DeepFace
import cv2
from utility import time_taken

@time_taken
def get_age_confidence(analysis):
    """
    Estimate confidence in age prediction
    """

    confidence = 0.8  # Base confidence

    # This is a simplified confidence calculation
    # You might want to adjust these weights based on your needs
    
    # Reduce confidence if multiple faces are detected
    if isinstance(analysis, list) and len(analysis) > 1:
        confidence *= 0.9
        
    return confidence

@time_taken
def get_age_range(predicted_age, confidence):
    """
    Calculate age range based on predicted age and confidence
    """
    margin = int((1 - confidence) * 10) + 5  # Base margin of 5 years
    return {
        'min_age': max(0, predicted_age - margin),
        'max_age': predicted_age + margin,
        'predicted_age': predicted_age
    }


@time_taken
def analyze_face(img):
    """
    Comprehensive face analysis including age, gender, and emotion
    with improved age detection
    """
    try:
        # Read image
        if type(img) == str:
            img = cv2.imread(img)
            if img is None:
                raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Analyze face with all attributes
        analysis = DeepFace.analyze(img_rgb, 
                            actions=['age', 'gender', 'emotion'],
                            enforce_detection=False)
        
        if isinstance(analysis, list):
            analysis = analysis[0]
            
        # Extract results
        result = {
            'age': analysis.get('age'),
            'gender': analysis.get('dominant_gender', analysis.get('gender')),
            'emotion': analysis.get('dominant_emotion'),
            'emotion_scores': analysis.get('emotion', {})
        }

        if result['age'] is not None:
            # Apply age correction based on detected features
            age_confidence = get_age_confidence(analysis)
            result['age_range'] = get_age_range(result['age'], age_confidence)
        
        return result

    except Exception as e:
        return f"Some error occured : {e}"
    
if __name__ == "__main__":
    res = analyze_face("./images/prafull_img.jpg")
    print(type(res))
    print(res.get("age"))