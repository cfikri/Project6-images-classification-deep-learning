import src.mytools as mt
import streamlit as st
import cv2
import mlflow
import boto3
import os

# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://ec2-52-209-5-80.eu-west-1.compute.amazonaws.com:5000"))

# Chargement du modèle depuis MLflow
model_uri = "s3://mlflow-cfikri/937544924322822054/f154eae389294193aed012085329d82b/artifacts/InceptionV3"
model = mlflow.tensorflow.load_model(model_uri)

# Interface Streamlit
st.title("Quelle est la race du chien ?")

# Option pour télécharger une image
uploaded_file = st.file_uploader("Télécharger une image de chien", type=["jpg", "png", "jpeg"])

# Option pour prendre une photo avec la webcam
if st.button("Prendre une photo avec la webcam"):
    st.write("Prendre une photo ...")
    # Utilisez OpenCV pour capturer une photo depuis la webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        img_path = 'webcam_photo.jpg'
        cv2.imwrite(img_path, frame)
        st.image(frame, channels="BGR")
        race = mt.prediction_race(model, img_path)
        st.write(f"La race du chien est : {race}")
    else:
        st.write("Impossible de prendre une photo. Vérifiez votre webcam.")
    cap.release()
    cv2.destroyAllWindows()

# Si un fichier est téléchargé
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    race = mt.prediction_race(model, "temp.jpg")
    st.write(f"La race du chien est : {race}")