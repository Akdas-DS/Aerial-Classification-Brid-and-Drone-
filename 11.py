import streamlit as st
from PIL import Image
import numpy as np
import io
import tflite_runtime.interpreter as tflite
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

MODEL_PATH = "model.tflite"     # <---- Your converted TFLite model
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Bird", "Drone"]

@st.cache_resource(show_spinner=False)
def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

# Get input/output tensor details
input_details = model.get_input_details()
output_details = model.get_output_details()


def preprocess_image(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0).astype("float32")
    arr = resnet_preprocess(arr)
    return arr


st.markdown("""
    <style>
    body {background-color:#191c25;}
    .reportview-container, .main {background:#191c25;}
    .stApp {background-color:#191c25;}
    .stButton>button {
        background: linear-gradient(90deg,#3333ff 0,#ff5a36 100%);
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #f3f3f3 !important;
    }
    .prediction {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #00ECD8;
        margin-top: 0.5em;
        text-shadow: 0px 0px 10px #222;
    }
    .confidence {
        color: #FFD700;
        font-size: 20px;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Aerial Object Recognition")
st.subheader("Bird vs. Drone Classifier (TFLite Edition)")

uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, use_column_width=True)

        img_array = preprocess_image(image)

        predict_btn = st.button("Classify Image")

        if predict_btn:
            with st.spinner("Running prediction..."):

                # Set input tensor
                model.set_tensor(input_details[0]["index"], img_array)

                # Run inference
                model.invoke()

                # Get output
                preds = model.get_tensor(output_details[0]["index"])[0]

                pred_idx = int(np.argmax(preds))
                confidence = float(np.max(preds)) * 100

                st.markdown(
                    f'<div class="prediction">{CLASS_NAMES[pred_idx]}</div>'
                    f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
                    unsafe_allow_html=True,
                )

                st.write("Class probabilities:")
                prob_dict = {CLASS_NAMES[i]: float(preds[i]) * 100 for i in range(len(CLASS_NAMES))}
                st.json(prob_dict)

    except Exception:
        st.error("Error processing the image. Please try a valid file.")

st.markdown("---")
st.caption("© 2025 Mohammed Akdas Ansari — Aerial Image Recognition System (TFLite)")
