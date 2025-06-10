from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model

# Load models
condition_model = load_model('models/condition_classifier.h5')
price_model = xgb.XGBRegressor()
price_model.load_model('models/price_predictor.json')

app = FastAPI()

class FurnitureMetadata(BaseModel):
    age: float
    wood_type: str
    material_type: str


def get_wear_score(image: Image.Image) -> float:
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = condition_model.predict(arr)
    return float(pred[0][0])


@ app.post('/predict_price')
async def predict_price(meta: FurnitureMetadata, image: UploadFile = File(...)):
    img = Image.open(image.file)
    wear_score = get_wear_score(img)

    df = pd.DataFrame([{
        'age': meta.age,
        'wear_score': wear_score,
        'wood_type': meta.wood_type,
        'material_type': meta.material_type,
    }])

    df = pd.get_dummies(df)
    required_cols = price_model.get_booster().feature_names
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[required_cols]

    price = price_model.predict(df)[0]
    return {
        'estimated_price': float(price),
        'wear_score': wear_score
    }


