import src.mytools as mt
import streamlit as st
import cv2
import mlflow
import os

# Chargement du modèle depuis MLflow
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://ec2-52-209-5-80.eu-west-1.compute.amazonaws.com:5000"))
    model_uri = "s3://mlflow-cfikri/937544924322822054/f154eae389294193aed012085329d82b/artifacts/InceptionV3"
    return mlflow.tensorflow.load_model(model_uri)

model = load_model()

# Dictionnaire des races de chiens
race_dict = {0: "BORZOI",
             1: "MALINOIS",
             2: "BORDER-COLLIE",
             }

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
        race_name = race_dict.get(race.item(), "Race inconnue")
        st.write(f"La race du chien est : {race_name}")
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
    race_name = race_dict.get(race.item(), "Race inconnue")
    st.write(f"La race du chien est : {race_name}")