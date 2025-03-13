from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API está funcionando corretamente!"}
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI()

@app.post("/analyze-face/")
async def analyze_face(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)

    # Simples detecção facial usando OpenCV
    face_detected = detect_face(image)

    return {"message": "Análise concluída", "face_detected": face_detected}

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0  # Retorna True se detectar uma face

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
