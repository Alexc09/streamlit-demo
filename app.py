import streamlit as st
import cv2
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_face(PIL_img):
    img = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for face in faces:
        (x, y, width, height) = face
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
    # Render an RGB image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.markdown(f"Detected {len(faces)} faces")


st.header("Face Detector")
st.text("Upload a picture containing faces")

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", 'jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    detect_face(image)