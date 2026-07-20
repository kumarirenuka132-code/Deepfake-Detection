# ==========================================
# Import Libraries
# ==========================================
import streamlit as st
import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import Counter
import tempfile
import os
import gdown

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Deepfake Detection using Vision Transformer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS
# ==========================================
st.markdown("""
<style>
/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}

/* Title */
.main-title{
    text-align:center;
    color:#00E5FF;
    font-size:45px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:18px;
}

/* Cards */
.card{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,229,255,0.3);
    margin-bottom:20px;
}

/* Prediction Box */
.prediction{
    font-size:30px;
    font-weight:bold;
    text-align:center;
    color:#00ff99;
}

/* Footer */
.footer{
    text-align:center;
    color:white;
    font-size:15px;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================
st.markdown('<h1 class="main-title">🛡️ Deepfake Detection using Vision Transformer (ViT)</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an Image or Video to detect whether it is Real or Fake.</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# Device Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Class Names
# ==========================================
classes = ["Fake", "Real"]

# ==========================================
# Image Transformation
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Load Vision Transformer Model
# ==========================================
@st.cache_resource
def load_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    
    MODEL_PATH = "deepfake_vit_model.pth"
    if not os.path.exists(MODEL_PATH):
        file_id = "1O0uOM6P_4nRitUm67FAWwRLV6U24C4DV"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==========================================
# Load Model Execution
# ==========================================
try:
    with st.spinner("Loading Vision Transformer Model..."):
        model = load_model()
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error("❌ Unable to Load Model")
    st.error(e)
    st.stop()

# ==========================================
# Face Detector
# ==========================================
import requests

cascade_filename = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_filename) or os.path.getsize(cascade_filename) == 0:
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    response = requests.get(cascade_url)
    with open(cascade_filename, "wb") as f:
        f.write(response.content)

face_cascade = cv2.CascadeClassifier(cascade_filename)

def detect_face(image):
    # PIL Image → NumPy Array
    image = np.array(image)
    # RGB → BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Gray Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )
    # No Face Found
    if len(faces) == 0:
        return None
    # Largest Face
    largest_face = max(
        faces,
        key=lambda box: box[2] * box[3]
    )
    x, y, w, h = largest_face
    # Face Crop
    face = image[y:y+h, x:x+w]
    # BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face

# ==========================================
# Preprocess Face
# ==========================================
def preprocess_face(face):
    face = Image.fromarray(face)
    face = transform(face)
    face = face.unsqueeze(0)
    face = face.to(device)
    return face

# ==========================================
# Image Prediction Function
# ==========================================
def predict_image(image):
    # Detect Face
    face = detect_face(image)
    # No Face Found
    if face is None:
        return None, None, None
    # Preprocess Face
    face_tensor = preprocess_face(face)
    # Prediction
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    # Label
    predicted_class = classes[prediction.item()]
    # Confidence (%)
    confidence_score = confidence.item() * 100
    return (predicted_class, confidence_score, face)

# ==========================================
# Video Prediction Function
# ==========================================
def predict_video(video_file):
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()
    cap = cv2.VideoCapture(temp_video.name)
    FRAME_INTERVAL = 30
    frame_count = 0
    predictions = []
    confidences = []
    progress = st.progress(0)
    status = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detect_face(Image.fromarray(frame_rgb))
            if face is not None:
                face_tensor = preprocess_face(face)
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, 1)
                    label = classes[pred.item()]
                    predictions.append(label)
                    confidences.append(confidence.item() * 100)
        frame_count += 1
        if total_frames > 0:
            progress.progress(min(frame_count / total_frames, 1.0))
            status.text(f"Processing Frame : {frame_count}")
    cap.release()
    os.remove(temp_video.name)
    progress.empty()
    status.empty()
    if len(predictions) == 0:
        st.error("❌ No Face Detected in Video")
        return None, None
    # Majority Voting
    final_prediction = Counter(predictions).most_common(1)[0][0]
    average_confidence = np.mean(confidences)
    st.subheader("🎥 Video Prediction Result")
    if final_prediction == "Real":
        st.success(f"✅ Final Prediction : {final_prediction}")
    else:
        st.error(f"❌ Final Prediction : {final_prediction}")
    st.write(f"### Confidence : {average_confidence:.2f}%")
    st.progress(int(average_confidence))
    st.write(f"Frames Processed : {frame_count}")
    st.write(f"Frames Predicted : {len(predictions)}")
    return final_prediction, average_confidence

# ==========================================
# History Management
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(file_name, prediction, confidence):
    st.session_state.history.append({
        "File Name": file_name,
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%"
    })

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("🛡️ Deepfake Detection")
st.sidebar.markdown("---")
st.sidebar.header("📌 Project Information")
st.sidebar.write("**Model :** Vision Transformer (ViT)")
st.sidebar.write("**Dataset :** Celeb-DF v2")
st.sidebar.write("**Framework :** PyTorch")
st.sidebar.write("**Application :** Streamlit")
st.sidebar.write("**Classes :** Fake / Real")
st.sidebar.markdown("---")
prediction_type = st.sidebar.radio(
    "Select Prediction Type",
    ("Image Prediction", "Video Prediction")
)
st.sidebar.markdown("---")
st.sidebar.info("""
Upload an Image or Video.
The model will detect whether the content is **Real** or **Fake**.
""")

# ==========================================
# Image Prediction UI
# ==========================================
if prediction_type == "Image Prediction":
    st.header("🖼️ Image Prediction")
    uploaded_image = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Predict Image"):
            with st.spinner("Predicting..."):
                predicted_class, confidence_score, face = predict_image(image)
                if predicted_class is None:
                    st.error("❌ No Face Detected in Image")
                else:
                    st.subheader("🖼️ Image Prediction Result")
                    if predicted_class == "Real":
                        st.success(f"✅ Final Prediction : {predicted_class}")
                    else:
                        st.error(f"❌ Final Prediction : {predicted_class}")
                    st.write(f"### Confidence : {confidence_score:.2f}%")
                    st.progress(int(confidence_score))
                    # Display Extracted Face
                    st.image(face, caption="Detected Face", width=200)
                    # Save to History
                    add_history(
                        uploaded_image.name,
                        predicted_class,
                        confidence_score
                    )

# ==========================================
# Video Prediction UI
# ==========================================
elif prediction_type == "Video Prediction":
    st.header("🎥 Video Prediction")
    uploaded_video = st.file_uploader(
        "Choose a Video",
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("🎯 Predict Video"):
            with st.spinner("Processing Video..."):
                final_pred, avg_conf = predict_video(uploaded_video)
                if final_pred is not None:
                    add_history(
                        uploaded_video.name,
                        final_pred,
                        avg_conf
                    )

# ==========================================
# Prediction History Table
# ==========================================
st.markdown("---")
st.subheader("📜 Prediction History")
if len(st.session_state.history) == 0:
    st.info("No prediction has been made yet.")
else:
    st.table(st.session_state.history)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ==========================================
# About Project
# ==========================================
st.markdown("---")
with st.expander("📖 About Project"):
    st.write("""
    ### Deepfake Detection using Vision Transformer (ViT)
    This project detects whether an uploaded image or video is **Real** or **Fake**.
    
    **Model Used** - Vision Transformer (ViT Base Patch16 224)  
    **Dataset** - Celeb-DF v2  
    **Framework** - PyTorch  
    **Frontend** - Streamlit  
    
    **Features**
    - Image Prediction
    - Video Prediction
    - Face Detection
    - Face Cropping
    - Confidence Score
    - Prediction History
    """)

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:white; font-size:18px;'>
❤️ Developed by Renuka Kumari
</div>
""", unsafe_allow_html=True)
