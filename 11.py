import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

MODEL_PATH = "final_transfer_model.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Bird', 'Drone']

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('final_transfer_model.h5')

model = load_model()

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
    .stFileUploader {border-radius: 8px;}
    h1, h2, h3, h4, h5, h6, p, span, label, .markdown-text-container {
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
    .sidebar .sidebar-content {background:#23243a;}
    </style>
""", unsafe_allow_html=True)

st.title("Aerial Object Recognition")
st.subheader("Bird vs. Drone Classifier")
st.sidebar.header("Project Information")
st.sidebar.write(
    """Use this tool to determine if an aerial image contains a **bird** or a **drone**.
    - State-of-the-art deep learning model.
    - Designed for aerial surveillance and monitoring.
    """
)
st.sidebar.markdown("---")
st.sidebar.write(
    """**Instructions:**
    - Drop your image in the browser window or click to select.
    - Wait for the prediction result to appear below.
    """
)

def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = resnet_preprocess(arr)
    return arr


uploaded_file = st.file_uploader(
    label="Select an aerial image (JPG or PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, use_column_width=True, caption=None)
        img_array = preprocess_image(image)
        predict_button = st.button("Classify Image")
        if predict_button:
            with st.spinner('Running classification...'):
                preds = model.predict(img_array)
                pred_idx = int(np.argmax(preds))
                confidence = float(np.max(preds)) * 100

                st.markdown(
                    f'<div class="prediction">{CLASS_NAMES[pred_idx]}</div>'
                    f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
                    unsafe_allow_html=True
                )

                st.write("Class probabilities:")
                prob_dict = {name: float(preds[0][i]) * 100 for i, name in enumerate(CLASS_NAMES)}
                st.progress(int(confidence))
                st.json(prob_dict)
    except Exception as e:
        st.error(f"Unable to process the image. Make sure the file is a valid JPG or PNG.")

st.markdown("---")
st.caption("© 2025 [Your Team/Institution]. Aerial Image Recognition System.")
