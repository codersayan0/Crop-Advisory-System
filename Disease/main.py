import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image
import joblib
from gtts import gTTS
import tempfile

# ===================== 1. Load Model & Class Names =====================
# Load trained CNN model
model = keras.models.load_model("crop_disease_model.h5")

# Load class names from joblib file
class_names = joblib.load("class_names.joblib")

# Define image size (must match training size)
IMG_SIZE = (256, 256)   # Change this if your model was trained with a different size

# ===================== 2. Disease Solutions =====================
disease_solutions = {
    "Early_Blight":
        "Remove infected leaves. Apply fungicides with chlorothalonil or copper. "
        "Rotate crops and avoid overhead watering.",
    "Late_Blight":
        "Immediately remove and destroy affected plants. Use fungicides like "
        "mancozeb or metalaxyl. Ensure good air circulation and avoid wet foliage.",
    "Healthy":
        "Your potato plant appears healthy! Continue regular watering and provide "
        "balanced fertilizer. Monitor weekly for early disease signs."
}

# ===================== 3. Prediction Function =====================
def predict(img):
    image = img.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(image)/255., axis=0)
    
    pred = model.predict(arr)[0]
    probs = {name: float(p) for name, p in zip(class_names, pred)}
    
    idx = int(np.argmax(pred))
    disease = class_names[idx]
    solution = disease_solutions.get(disease, "No solution available for this class.")
    
    # Prepare text for voice
    text = f"The detected disease is {disease}. Confidence: {pred[idx]*100:.2f} percent. Recommended solution: {solution}."
    
    # Generate voice output
    tts = gTTS(text=text, lang='en')
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)

    return probs, {
        "Prediction": disease,
        "Confidence": f"{pred[idx]*100:.2f}%",
        "Recommended Solution": solution
    }, tmp_file.name  # Return path to audio file

# ===================== 4. Gradio Interface =====================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Class Probabilities"),
        gr.JSON(label="Diagnosis & Solution"),
        gr.Audio(label="Voice Output")  
    ],
    title="🌾 Crop Disease Detector",
    description="Upload a crop leaf image to identify disease and get treatment suggestions."
)

demo.launch(share=True)
