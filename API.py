from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np 
import joblib

# Import your custom transformers so joblib can find them in the namespace
from custom_transformers import DateYearExtractor, SimpleRatioTransformer

# --- CRITICAL FIX: Patch __main__ for Jupyter Notebook Pickling ---
import __main__
__main__.DateYearExtractor = DateYearExtractor
__main__.SimpleRatioTransformer = SimpleRatioTransformer
# ------------------------------------------------------------------

app = FastAPI(
    title="King County Housing Price Predictor",
    description="REST API for the King County Random Forest model.",
    version="1.0.0"
)

# Load the model on startup
try:
    model = joblib.load("king_county_housing_model_best_RF.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Strict Input Validation Schema based on the exact columns your pipeline expects
class HouseFeatures(BaseModel):
    date: str = Field(default="2015-2-15", description="The date the house you want to by on (format: YYYYMMDD).")
    bedrooms: int = Field(default=3, description="Number of bedrooms in the house.")
    bathrooms: float = Field(default=2.0, description="Number of bathrooms (e.g., 0.5 accounts for a room with a toilet but no shower).")
    sqft_living: int = Field(default=2000, description="Square footage of the interior living space.")
    sqft_lot: int = Field(default=5000, description="Square footage of the land lot.")
    floors: float = Field(default=1.0, description="Number of floors (levels) in the house.")
    waterfront: int = Field(default=0, description="A dummy variable for whether the apartment overlooks the waterfront (1 = yes, 0 = no).")
    view: int = Field(default=0, description="An index from 0 to 4 rating how good the view of the property is.")
    condition: int = Field(default=3, description="An index from 1 to 5 rating the overall condition of the apartment.")
    sqft_above: int = Field(default=1500, description="The square footage of the interior housing space that is above ground level.")
    sqft_basement: int = Field(default=0, description="The square footage of the interior housing space that is below ground level.")
    yr_built: int = Field(default=1990, description="The year the house was initially built.")
    yr_renovated: int = Field(default=0, description="The year of the house's last renovation (0 if never renovated).")
    lat: float = Field(default=47.5, description="Latitude coordinate of the property.")
    long: float = Field(default=-122.2, description="Longitude coordinate of the property.")
    sqft_living15: int = Field(default=1500, description="The square footage of interior housing living space for the nearest 15 neighbors.")
    sqft_lot15: int = Field(default=5000, description="The square footage of the land lots of the nearest 15 neighbors.")

@app.get("/")
def read_root():
    return {"message": "Housing Price API is running. Go to /docs to test it."}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    try:
        # Convert incoming JSON payload to a DataFrame (1 row)
        input_data = pd.DataFrame([features.model_dump()])
                
        # Make prediction
        prediction = model.predict(input_data)
                
        return {
            "predicted_price": float(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
