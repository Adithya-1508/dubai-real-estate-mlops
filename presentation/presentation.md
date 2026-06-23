# 🏙️ Dubai Apartment Price Prediction
## End-to-End Machine Learning System & Market Insights
*By Data Science & BI Team*

---

## 1. Business Understanding & Problem Statement

### The Problem
- Real estate transactions in Dubai are fast-paced, highly variable, and influenced by a complex mix of geographical, property-specific, and amenity features.
- Buyers and investors lack a transparent, data-driven way to evaluate fair market value.

### Scope & Solution
- Build an end-to-end predictive system targeting apartment prices.
- Generate actionable BI insights highlighting high-value locations and amenities.

### Success Metrics
- Model Performance: R² Score > 85%, Mean Absolute Error (MAE) under 150k AED.
- Speed: Inference latency < 100ms.

---

## 2. Dataset Overview

### Dataset Dimensions
- Raw: 16,777 records across 88 features.
- Cleaned: 16,752 records across 95 features (after resolving target nulls, duplicate amenities, and feature engineering).

### Key Columns
- **Numerical**: `price`, `beds` ( studio = 0 ), `baths`, `area` (sqft).
- **Categorical**: `district`, `furnished` (Furnished, Unfurnished, Unknown).
- **Boolean Amenities**: 80+ features (swimming pool, sauna, parking, security, maids room, lawn, etc.).

---

## 3. Exploratory Data Analysis (EDA) Findings

- **Skewness**: Apartment prices are heavily right-skewed, requiring a Log transformation (`log_price = np.log1p(price)`) for stable modeling.
- **Top Predictor**: Property **Area** exhibits a 0.71 positive correlation with price.
- **District Premiums**: Districts like Palm Jumeirah and Downtown Dubai command major premiums compared to outlying suburbs.
- **Bedroom Valuations**: Median prices increase consistently with bedroom counts.

---

## 4. Feature Engineering

We created features to improve model performance and business interpretability:

1. **`is_studio`**: Binary flag indicating whether the apartment is a studio (beds == 0).
2. **`price_per_sqft`**: price / area (crucial real estate metric).
3. **`luxury_score`**: Sum of five premium amenities: pool, sauna, parking, gym, and jacuzzi.
4. **`log_price`**: Log-transformed target variable to stabilize training variance.

---

## 5. Modeling Results & Comparison

Below is the comparative performance on the test split (prices in AED):

| Model | Mean MAE (AED) | Mean R² Score | Description |
|---|---|---|---|
| **Linear Regression** | ~238,000 | 0.528 | Baseline, struggles with non-linear location encoding. |
| **Random Forest** | ~118,000 | 0.884 | Strong baseline, captures non-linear splits. |
| **XGBoost (Tuned)** | **~98,000** | **0.912** | Best performance, fast inference, optimized parameters. |
| **LightGBM** | ~104,000 | 0.901 | High speed, slightly lower accuracy. |
| **CatBoost** | ~101,000 | 0.906 | Robust categorical handling. |

---

## 6. Explainability: SHAP Insights

Using **SHAP (SHapley Additive exPlanations)**, we opened the "black box" of the tuned XGBoost model:

- **Global Drivers**: Property **Area** is the single most powerful driver. Next are **luxury_score** and the number of **bathrooms**.
- **Location Shifts**: Location in *Palm Jumeirah* or *Downtown Dubai* shifts the model prediction upward by hundreds of thousands of AED.
- **Local Explanations**: For any single property, we can trace exactly how much each feature (e.g. area = 1500, luxury_score = 4) contributed to the final valuation, giving full auditability.

---

## 7. FastAPI Service Overview

We serve our model using a lightweight, high-performance FastAPI service.

### Endpoints
- `GET /health` -> Verification of API readiness and model load status.
- `POST /predict` -> Receives property characteristics and returns price predictions.

### Request Payload Example:
```json
{
  "beds": 2,
  "baths": 2,
  "area": 1200.0,
  "luxury_score": 3,
  "district": "Downtown Dubai",
  "furnished": "Furnished"
}
```

---

## 8. Streamlit Dashboard Demo

The front-end is built using **Streamlit** for interactive exploration and AI search:

1. **🔮 Price Predictor Tab**: Input property features to see the predicted price in AED and get **similar property suggestions** based on spatial and bedroom distance.
2. **📊 Market Dashboard Tab**:
   - Boxplots showing price variations by furnishing status.
   - Bar charts comparing average AED per sqft across districts.
   - Price distribution histograms.
3. **🤖 Smart Real Estate Assistant Tab**:
   - Natural language search box parsing queries like *"2 bed under 3.5m in Downtown Dubai"* and returning direct matches.


---

## 9. Business Impact & Investment Strategy

1. **Buy-to-Let Investors**: Furnished properties command a **15% median price premium** over unfurnished properties, showing strong rental setup demand.
2. **Value-Add Renovations**: Increasing a property's `luxury_score` (adding pool access or high-end parking spaces) increases value by a median of **8.5% per luxury unit**.
3. **District Selection**: **Palm Jumeirah** yields the highest pricing density, while **JVC** offers the lowest entry price per sqft for yield optimization.
