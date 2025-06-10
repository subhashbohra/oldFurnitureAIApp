from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model

# Load ML models
condition_model = load_model('models/condition_classifier.h5')
price_model = xgb.XGBRegressor()
price_model.load_model('models/price_predictor.json')

app = FastAPI()
templates = Jinja2Templates(directory='frontend/templates')


def get_wear_score(image: Image.Image) -> float:
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = condition_model.predict(arr)
    return float(pred[0][0])


def predict_price(age: float, wear_score: float, wood_type: str, material_type: str) -> float:
    df = pd.DataFrame([{"age": age,
                       "wear_score": wear_score,
                       "wood_type": wood_type,
                       "material_type": material_type}])
    df = pd.get_dummies(df)
    required = price_model.get_booster().feature_names
    for col in required:
        if col not in df.columns:
            df[col] = 0
    df = df[required]
    return float(price_model.predict(df)[0])


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.post('/analyze', response_class=HTMLResponse)
async def analyze(request: Request,
                  image: UploadFile = File(...),
                  age: float = Form(...),
                  wood_type: str = Form(...),
                  material_type: str = Form(...)):
    img = Image.open(image.file)
    wear_score = get_wear_score(img)
    price = predict_price(age, wear_score, wood_type, material_type)
    result = {"estimated_price": price, "wear_score": wear_score}
    return templates.TemplateResponse('index.html', {
        "request": request,
        "result": result
    })
