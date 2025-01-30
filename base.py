# Directory structure
# furniture-assessment-app/
# |-- frontend/  (Flutter/React Native UI)
# |-- backend/  (FastAPI, AI Model Inference)
# |-- models/  (YOLOv8, CNN, XGBoost training scripts)
# |-- datasets/ (Raw & preprocessed training data)
# |-- scraping/ (Scripts for scraping resale prices)
# |-- utils/ (Helper functions, pre-processing scripts)

# backend/main.py (FastAPI Backend Setup)
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Load YOLOv8 Model for condition analysis
model = YOLO("yolov8n.pt")

@app.post("/analyze")
async def analyze_furniture(file: UploadFile = File(...)):
    image = Image.open(file.file)
    results = model(image)
    return {"detections": results.pandas().xyxy[0].to_dict(orient="records")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# models/train_condition_classifier.py (Train CNN for condition classification)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory("datasets/condition", target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
val_generator = train_datagen.flow_from_directory("datasets/condition", target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)
model.save("models/condition_classifier.h5")

# scraping/scrape_resale_prices.py (Scrape second-hand furniture prices)
from bs4 import BeautifulSoup
import requests

def scrape_olx():
    url = "https://www.olx.in/furniture/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    listings = []
    for item in soup.find_all('li', class_='EIR5N'):
        title = item.find('span', class_='_2tW1I')
        price = item.find('span', class_='_89yzn')
        if title and price:
            listings.append({"title": title.text, "price": price.text})
    return listings

if __name__ == "__main__":
    print(scrape_olx())
