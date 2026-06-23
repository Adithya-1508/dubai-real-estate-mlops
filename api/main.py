import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import sys
import time
import jwt
from collections import defaultdict
from dotenv import load_dotenv

# Load .env file
load_dotenv()

sys.path.append(".")
from models.geospatial import add_geospatial_features
from api.database import init_db, log_prediction

# Initialize SQLite database
init_db()

app = FastAPI(
    title="Dubai Apartment Price Prediction API",
    description="Secure FastAPI service to predict apartment prices in Dubai with JWT auth, rate limiting, and database logging.",
    version="1.1.0"
)

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
IS_PROD = os.getenv("ENV") == "production"

if not SECRET_KEY:
    if IS_PROD:
        raise ValueError("CRITICAL SECURITY ERROR: JWT_SECRET_KEY environment variable must be set in production!")
    else:
        SECRET_KEY = "dubai-real-estate-secret-key-default-dev"
        print("WARNING: Using default development SECRET_KEY. Set JWT_SECRET_KEY in production!")

API_ADMIN_USER = os.getenv("API_ADMIN_USER", "admin")
API_ADMIN_PASSWORD = os.getenv("API_ADMIN_PASSWORD")

if not API_ADMIN_PASSWORD:
    if IS_PROD:
        raise ValueError("CRITICAL SECURITY ERROR: API_ADMIN_PASSWORD environment variable must be set in production!")
    else:
        API_ADMIN_PASSWORD = "admin123"
        print("WARNING: Using default development API_ADMIN_PASSWORD. Set API_ADMIN_PASSWORD in production!")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Rate limiting configuration (60 requests per minute per IP)
RATE_LIMIT_STORE = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

# Load model artifacts
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully!")
else:
    model = None
    preprocessor = None
    print("WARNING: Model files not found! FastAPI will start, but predictions will fail.")

class PropertyFeatures(BaseModel):
    beds: int = Field(..., ge=0, description="Number of bedrooms (0 for Studio)", json_schema_extra={"example": 2})
    baths: int = Field(..., ge=0, description="Number of bathrooms", json_schema_extra={"example": 2})
    area: float = Field(..., gt=0, description="Area of the apartment in sqft", json_schema_extra={"example": 1200.0})
    luxury_score: int = Field(0, ge=0, le=5, description="Luxury score based on premium amenities (0 to 5)", json_schema_extra={"example": 3})
    has_view: int = Field(0, ge=0, le=1, description="Indicator if property has a view (0 or 1)", json_schema_extra={"example": 1})
    has_maids_room: int = Field(0, ge=0, le=1, description="Indicator if property has a maid's room (0 or 1)", json_schema_extra={"example": 0})
    is_freehold: int = Field(0, ge=0, le=1, description="Indicator if property is freehold (0 or 1)", json_schema_extra={"example": 1})
    district: str = Field(..., description="District name", json_schema_extra={"example": "Downtown Dubai"})
    furnished: str = Field("Unknown", description="Furnished status", json_schema_extra={"example": "Furnished"})

def create_access_token(data: dict):
    """Creates a JWT access token."""
    to_encode = data.copy()
    expire = time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    """Decodes and validates a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except jwt.PyJWTError:
        return None

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency to secure endpoints using JWT authentication."""
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload

def check_rate_limit(client_ip: str) -> bool:
    """Checks if a client IP has exceeded the rate limit."""
    now = time.time()
    # Filter out timestamps older than the sliding window
    timestamps = [t for t in RATE_LIMIT_STORE[client_ip] if now - t < RATE_LIMIT_WINDOW_SECONDS]
    RATE_LIMIT_STORE[client_ip] = timestamps
    
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    RATE_LIMIT_STORE[client_ip].append(now)
    return True

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Secure Dubai Apartment Price Prediction API!",
        "health_check_url": "/health",
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    if model is None or preprocessor is None:
        return {"status": "unhealthy", "error": "Model files missing"}
    return {"status": "healthy"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint to authenticate users and issue JWT tokens."""
    if form_data.username == API_ADMIN_USER and form_data.password == API_ADMIN_PASSWORD:
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

@app.post("/predict")
def predict(features: PropertyFeatures, request: Request, current_user: dict = Depends(get_current_user)):
    """Predicts property price. Secured with JWT Auth, Rate Limiting, and Database Logging."""
    start_time = time.time()
    
    # Rate Limiting check
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Maximum 60 requests per minute."
        )
        
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Please ensure model.pkl and preprocessor.pkl are present."
        )
    
    input_df = pd.DataFrame([{
        "beds": features.beds,
        "baths": features.baths,
        "area": features.area,
        "luxury_score": features.luxury_score,
        "has_view": features.has_view,
        "has_maids_room": features.has_maids_room,
        "is_freehold": features.is_freehold,
        "district": features.district,
        "furnished": features.furnished
    }])
    
    try:
        # Augment with geospatial features
        input_df = add_geospatial_features(input_df)
        X_proc = preprocessor.transform(input_df)
        pred_log = model.predict(X_proc)
        predicted_price = float(np.expm1(pred_log)[0])
        
        # Round the result
        final_price = round(predicted_price, 2)
        
        # Compute latency in ms
        latency_ms = (time.time() - start_time) * 1000.0
        
        # Database Log (Feedback Loop)
        log_prediction(
            beds=features.beds,
            baths=features.baths,
            area=features.area,
            luxury_score=features.luxury_score,
            has_view=features.has_view,
            has_maids_room=features.has_maids_room,
            is_freehold=features.is_freehold,
            district=features.district,
            furnished=features.furnished,
            predicted_price=final_price,
            client_host=client_ip,
            latency_ms=latency_ms
        )
        
        return {
            "predicted_price": final_price,
            "currency": "AED"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
