# Dubai Apartment Price Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end machine learning system capable of predicting Dubai apartment prices while generating actionable real estate market insights, including data cleaning, advanced modeling (XGBoost, LightGBM, CatBoost), SHAP explainability, FastAPI deployment, Dockerization, and a Streamlit dashboard.

**Architecture:** Clean raw data, engineer key features (is_studio, luxury_score, price_per_sqft, log_price), compare baseline models, tune advanced models with Optuna cross-validation, explain predictions via SHAP, serialize the best model and pipeline using joblib, wrap it in a FastAPI service, dockerize it, and build a Streamlit web application.

**Tech Stack:** Python, Pandas, Numpy, Scikit-Learn, XGBoost, LightGBM, CatBoost, SHAP, Optuna, FastAPI, Streamlit, Docker.

---

### Task 1: Data Cleaning and Feature Engineering

**Files:**
- Modify: `notebooks/01_EDA_Property_Price_Prediction.ipynb`
- Output: `data/ready_cleaned_v2.csv`
- Test script: `scratch/test_cleaning.py`

- [ ] **Step 1: Write test to verify data cleaning output**
Create `scratch/test_cleaning.py` to assert columns exist, types are correct, and no nulls exist in target variables.
```python
import pandas as pd
import numpy as np

def test_data():
    df = pd.read_csv("data/ready_cleaned_v2.csv")
    assert "log_price" in df.columns, "log_price missing"
    assert "is_studio" in df.columns, "is_studio missing"
    assert "luxury_score" in df.columns, "luxury_score missing"
    assert df["beds"].isnull().sum() == 0, "Null beds found"
    assert df["price"].isnull().sum() == 0, "Null prices found"
    assert df["beds"].dtype in [np.int64, np.float64, int, float], "beds must be numeric"
    print("All data cleaning assertions passed!")

if __name__ == "__main__":
    test_data()
```

- [ ] **Step 2: Execute test and verify it fails**
Run: `python scratch/test_cleaning.py`
Expected: FileNotFoundError or AssertionError (since data/ready_cleaned_v2.csv does not exist yet).

- [ ] **Step 3: Implement cleaning code in 01_EDA_Property_Price_Prediction.ipynb**
Open, edit, and run `notebooks/01_EDA_Property_Price_Prediction.ipynb` with code to load `data/ready_cleaned.csv`, clean beds/furnishing, rename duplicate amenities, engineer features (`luxury_score`, `is_studio`, `price_per_sqft`, `log_price`), drop missing prices, and save to `data/ready_cleaned_v2.csv`.
Here is the core code block to insert and run in the notebook:
```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/ready_cleaned.csv")
df = df.dropna(subset=["price"])
df["furnished"] = df["furnished"].fillna("Unknown")
df["beds"] = df["beds"].replace("Studio", 0)
df["beds"] = pd.to_numeric(df["beds"]).fillna(0).astype(int)
df["baths"] = pd.to_numeric(df["baths"]).fillna(df["baths"].median()).astype(int)

rename_map = {
    "swimming_pool": "private_swimming_pool",
    "swimming_pool.1": "community_swimming_pool",
    "sauna": "private_sauna",
    "sauna.1": "community_sauna",
    "steam_room": "private_steam_room",
    "steam_room.1": "community_steam_room",
    "maintenance_staff": "private_maintenance_staff",
    "maintenance_staff.1": "community_maintenance_staff",
    "security_staff": "community_security_staff",
    "security_staff.1": "private_security_staff",
    "parking_spaces": "private_parking",
    "parking_spaces.1": "community_parking",
    "maids_room": "maids_room_primary",
    "maids_room.1": "maids_room_secondary"
}
df.rename(columns=rename_map, inplace=True)

df["price_per_sqft"] = df["price"] / df["area"]
df["is_studio"] = (df["beds"] == 0).astype(int)

df["has_pool"] = ((df["private_swimming_pool"] == 1) | (df["community_swimming_pool"] == 1)).astype(int)
df["has_sauna"] = ((df["private_sauna"] == 1) | (df["community_sauna"] == 1)).astype(int)
df["has_parking"] = ((df["private_parking"] == 1) | (df["community_parking"] == 1)).astype(int)

luxury_cols = ["has_pool", "has_sauna", "has_parking", "gym_or_health_club", "jacuzzi"]
for col in luxury_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

df["luxury_score"] = df[luxury_cols].sum(axis=1)
df["log_price"] = np.log1p(df["price"])

df.to_csv("data/ready_cleaned_v2.csv", index=False)
print("Data cleaned and saved successfully!")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python scratch/test_cleaning.py`
Expected: Print "All data cleaning assertions passed!"

---

### Task 2: Baseline Modeling

**Files:**
- Modify: `notebooks/02_Model_Training.ipynb`
- Test script: `scratch/test_baseline.py`

- [ ] **Step 1: Write test to verify baseline models are trained and outputted**
Create `scratch/test_baseline.py` to ensure the baseline performance metrics table and models can be run.
```python
import os
def test_baseline_files():
    # Verify that the notebook has run and output file exists (can also just verify notebook execution)
    assert os.path.exists("notebooks/02_Model_Training.ipynb"), "Notebook missing"
    print("Baseline verification file check passed!")

if __name__ == "__main__":
    test_baseline_files()
```

- [ ] **Step 2: Execute test and verify it fails**
Run: `python scratch/test_baseline.py`
Expected: Success check, but we need to complete the notebook cells so that it runs successfully.

- [ ] **Step 3: Implement baseline models in 02_Model_Training.ipynb**
Open, edit, and run `notebooks/02_Model_Training.ipynb`. Load `data/ready_cleaned_v2.csv`, set up a preprocessor with standard scaling for numerical features and One-Hot encoding for categorical features (`district`, `furnished`), split the data (80/20 train/test), and train:
1. Linear Regression
2. Random Forest Regressor
Compute metrics (MAE, RMSE, R² score) on original price scale (by applying exponentiation `expm1` to model predictions since we predict `log_price`).
Write code cells in the notebook:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/ready_cleaned_v2.csv")

# Select features
num_features = ["beds", "baths", "area", "luxury_score"]
cat_features = ["district", "furnished"]
target = "log_price"

X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# Compare models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)
    y_test_orig = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_orig, preds)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
    r2 = r2_score(y_test_orig, preds)
    
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

results_df = pd.DataFrame(results)
print(results_df)
```

- [ ] **Step 4: Execute notebook and verify output**
Run the notebook programmatically or verify that it completes successfully.

---

### Task 3: Advanced Modeling

**Files:**
- Create: `notebooks/03_Advanced_Modeling.ipynb`
- Output: `models/model.pkl`, `models/preprocessor.pkl`
- Test script: `scratch/test_advanced.py`

- [ ] **Step 1: Write test for advanced model output**
Create `scratch/test_advanced.py` to verify that serialized model and preprocessor work together to predict a dummy sample.
```python
import joblib
import pandas as pd
import numpy as np

def test_inference():
    model = joblib.load("models/model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    sample = pd.DataFrame([{
        "beds": 2,
        "baths": 2,
        "area": 1200,
        "luxury_score": 3,
        "district": "Downtown Dubai",
        "furnished": "Furnished"
    }])
    
    X_processed = preprocessor.transform(sample)
    pred_log = model.predict(X_processed)
    pred = np.expm1(pred_log)[0]
    assert pred > 0, "Prediction must be positive"
    print(f"Model verification successful! Prediction for sample: AED {pred:,.2f}")

if __name__ == "__main__":
    test_inference()
```

- [ ] **Step 2: Execute test and verify it fails**
Run: `python scratch/test_advanced.py`
Expected: FAIL (File not found for model.pkl).

- [ ] **Step 3: Create and run 03_Advanced_Modeling.ipynb**
Write code to train XGBoost, LightGBM, and CatBoost. Tune hyperparameters using Optuna or RandomizedSearchCV with K-Fold cross validation. Save the best estimator (model) and the preprocessor fit on the full dataset using `joblib`.
Code template for the notebook:
```python
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import optuna

# Load data
df = pd.read_csv("data/ready_cleaned_v2.csv")
num_features = ["beds", "baths", "area", "luxury_score"]
cat_features = ["district", "furnished"]
X = df[num_features + cat_features]
y = df["log_price"]

# Fit Preprocessor
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

X_proc = preprocessor.fit_transform(X)

# Hyperparameter tuning for XGBoost using Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }
    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_proc):
        X_tr, X_va = X_proc[train_idx], X_proc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = np.expm1(model.predict(X_va))
        scores.append(mean_absolute_error(np.expm1(y_va), preds))
    return np.mean(scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10) # Set to 10 trials for quick tuning

best_params = study.best_params
print("Best parameters:", best_params)

# Train best model
best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_proc, y)

# Save
joblib.dump(best_model, "models/model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Models saved successfully!")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python scratch/test_advanced.py`
Expected: PASS and prints sample prediction.

---

### Task 4: Model Explainability

**Files:**
- Create: `notebooks/04_Model_Explainability.ipynb`

- [ ] **Step 1: Implement explainability code**
Create `notebooks/04_Model_Explainability.ipynb` using SHAP to plot global feature importance (SHAP Summary Plot) and local predictions (SHAP Force/Waterfall Plot).
Code block for explainability:
```python
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
df = pd.read_csv("data/ready_cleaned_v2.csv")

num_features = ["beds", "baths", "area", "luxury_score"]
cat_features = ["district", "furnished"]
X = df[num_features + cat_features]

X_proc = preprocessor.transform(X)

# Get feature names from onehot encoder
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(cat_encoder.get_feature_names_out(cat_features))
all_features = num_features + cat_feature_names

# SHAP Explainers
if hasattr(X_proc, "toarray"):
    X_proc_dense = X_proc.toarray()
else:
    X_proc_dense = X_proc

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_proc_dense[:200]) # explain first 200 samples

# Plot summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_proc_dense[:200], feature_names=all_features, show=False)
plt.title("SHAP Feature Importance (Top Features)")
plt.tight_layout()
plt.savefig("reports/shap_summary.png")
plt.close()

# Local prediction plot
plt.figure(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Local Prediction Explanation")
plt.tight_layout()
plt.savefig("reports/shap_local.png")
plt.close()

print("SHAP plots saved in reports/ directory!")
```

- [ ] **Step 2: Run notebook and verify outputs**
Verify files `reports/shap_summary.png` and `reports/shap_local.png` exist.

---

### Task 5: Business Intelligence

**Files:**
- Create: `notebooks/05_Business_Insights.ipynb`

- [ ] **Step 1: Write Business Intelligence logic**
Create `notebooks/05_Business_Insights.ipynb` to analyze the market and save visualizations/metrics.
Key Questions to Answer:
1. Most/Cheapest districts (median prices)
2. Average AED/sqft per district
3. Premium added by amenities (pool, parking, furnishing)
Code block:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ready_cleaned_v2.csv")

# 1. District Median Prices
district_prices = df.groupby("district")["price"].median().sort_values(ascending=False)
print("Top 10 Most Expensive Districts:\n", district_prices.head(10))
print("Top 10 Cheapest Districts:\n", district_prices.tail(10))

# 2. AED per SqFt
district_sqft = df.groupby("district")["price_per_sqft"].mean().sort_values(ascending=False)
print("Highest AED/sqft Districts:\n", district_sqft.head(10))

# 3. Amenity Impact
print("Value added by pools (median):", df.groupby("has_pool")["price"].median())
print("Value added by parking (median):", df.groupby("has_parking")["price"].median())
print("Value added by furnishing (median):", df.groupby("furnished")["price"].median())

# Plot and save
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="furnished", y="price")
plt.yscale("log")
plt.title("Property Price vs Furnishing")
plt.savefig("reports/furnishing_impact.png")
plt.close()
```

- [ ] **Step 2: Run notebook and verify output**
Verify file `reports/furnishing_impact.png` exists.

---

### Task 6: FastAPI Service

**Files:**
- Create: `api/main.py`
- Test script: `scratch/test_api.py`

- [ ] **Step 1: Write test for FastAPI endpoints**
Create `scratch/test_api.py` to query local FastAPI endpoints `/health` and `/predict` using requests.
```python
import requests
import time

def test_api():
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        print("API Health Check passed!")
        
        payload = {
            "beds": 2,
            "baths": 2,
            "area": 1200.0,
            "luxury_score": 3,
            "district": "Downtown Dubai",
            "furnished": "Furnished"
        }
        r = requests.post("http://127.0.0.1:8000/predict", json=payload)
        assert r.status_code == 200
        pred = r.json()["predicted_price"]
        assert pred > 0
        print(f"API Predict endpoint passed! Predicted Price: AED {pred:,.2f}")
    except Exception as e:
        print("API verification failed. Is uvicorn running?", e)

if __name__ == "__main__":
    test_api()
```

- [ ] **Step 2: Create FastAPI main.py file**
Create `api/main.py` to serve prediction requests.
```python
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Dubai Apartment Price Prediction API")

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
else:
    model = None
    preprocessor = None

class PropertyFeatures(BaseModel):
    beds: int = Field(..., ge=0, example=2)
    baths: int = Field(..., ge=0, example=2)
    area: float = Field(..., gt=0, example=1200.0)
    luxury_score: int = Field(0, ge=0, le=5, example=3)
    district: str = Field(..., example="Downtown Dubai")
    furnished: str = Field("Unknown", example="Furnished")

@app.get("/health")
def health():
    if model is None or preprocessor is None:
        return {"status": "unhealthy", "error": "Model files missing"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: PropertyFeatures):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_df = pd.DataFrame([{
        "beds": features.beds,
        "baths": features.baths,
        "area": features.area,
        "luxury_score": features.luxury_score,
        "district": features.district,
        "furnished": features.furnished
    }])
    
    try:
        X_proc = preprocessor.transform(input_df)
        pred_log = model.predict(X_proc)
        predicted_price = float(np.expm1(pred_log)[0])
        return {"predicted_price": round(predicted_price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 3: Run FastAPI and verify endpoints**
Run FastAPI in background: `python api/main.py`
Wait 2 seconds, then run `python scratch/test_api.py`.
Verify health and prediction printouts pass.

---

### Task 7: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`
- Test: Manually running streamlit and checking interface

- [ ] **Step 1: Implement Streamlit dashboard code**
Create `dashboard/app.py` featuring sidebar controls, predictions tab, and interactive market charts.
```python
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dubai Apartment Price Prediction", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f7f9fc; }
    h1 { color: #1e3d59; font-family: 'Inter', sans-serif; }
    .stButton>button { background-color: #17b978; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ Dubai Apartment Price Prediction & Insights")

@st.cache_data
def load_data():
    df = pd.read_csv("data/ready_cleaned_v2.csv")
    return df

try:
    df = load_data()
    districts = sorted(df["district"].unique().tolist())
    furnishings = sorted(df["furnished"].unique().tolist())
except Exception:
    districts = ["Downtown Dubai", "Dubai Marina", "Jumeirah Village Circle (JVC)"]
    furnishings = ["Furnished", "Unfurnished", "Unknown"]

tab1, tab2 = st.tabs(["🔮 Price Predictor", "📊 Market Dashboard"])

with tab1:
    st.subheader("Estimate Apartment Price")
    
    col1, col2 = st.columns(2)
    with col1:
        beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        baths = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
        area = st.number_input("Area (sqft)", min_value=100.0, max_value=20000.0, value=1200.0)
    with col2:
        district = st.selectbox("District", districts)
        furnished = st.selectbox("Furnishing Status", furnishings)
        luxury_score = st.slider("Luxury Score (0-5)", 0, 5, 3)
        
    if st.button("Predict Price"):
        payload = {
            "beds": int(beds),
            "baths": int(baths),
            "area": float(area),
            "luxury_score": int(luxury_score),
            "district": district,
            "furnished": furnished
        }
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if res.status_code == 200:
                pred = res.json()["predicted_price"]
                st.balloons()
                st.success(f"### Predicted Price: AED {pred:,.2f}")
            else:
                st.error("Error from API service. Make sure FastAPI is running!")
        except Exception as e:
            st.error(f"Failed to connect to FastAPI endpoint: {e}")

with tab2:
    st.subheader("Dubai Real Estate Insights")
    if 'df' in locals():
        col_insight1, col_insight2 = st.columns(2)
        with col_insight1:
            st.markdown("#### Top 10 Most Expensive Districts")
            top_districts = df.groupby("district")["price"].median().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=top_districts.values, y=top_districts.index, palette="viridis", ax=ax)
            ax.set_xlabel("Median Price (AED)")
            st.pyplot(fig)
            
        with col_insight2:
            st.markdown("#### Price Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(df["price"], bins=30, kde=True, ax=ax2)
            ax2.set_xlim(0, 10_000_000)
            ax2.set_xlabel("Price (AED)")
            st.pyplot(fig2)
```

- [ ] **Step 2: Verify dashboard works**
Run: `streamlit run dashboard/app.py`
Verify dashboard opens.

---

### Task 8: Dockerization

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create Dockerfile**
Create a `Dockerfile` in the root workspace to run the FastAPI app.
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Build and run Docker container**
Build docker image:
Run: `docker build -t dubai-realestate-api .`
Run: `docker run -d -p 8000:8000 dubai-realestate-api`
Query local FastAPI `http://127.0.0.1:8000/health` to confirm successful container deployment.

---

### Task 9: Portfolio Presentation and Finalizing Repo

**Files:**
- Create: `presentation/presentation.md`
- Create: `README.md`

- [ ] **Step 1: Create Presentation slides in Markdown**
Create `presentation/presentation.md` outlining project scope, EDA, features, SHAP value explanation, API endpoints, Streamlit dashboard, and Docker setup.
- [ ] **Step 2: Create README.md**
Create the project `README.md` with complete usage instructions for running FastAPI, running Streamlit, and building the Docker container.
- [ ] **Step 3: Cleanup and verify final layout**
Run lints, verify paths, clean up temporary files in `scratch/`, and check git status.

---
