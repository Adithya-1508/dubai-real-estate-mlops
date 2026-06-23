# Dubai Apartment Price Prediction System 🏙️

An end-to-end machine learning and business intelligence system built to predict apartment prices in Dubai and extract actionable real estate market insights.

---

## 📁 Repository Structure

```
Real-estate/
├── data/
│   ├── ready_cleaned.csv         # Raw real estate dataset
│   └── ready_cleaned_v2.csv      # Cleaned and feature engineered dataset
├── notebooks/
│   ├── 01_EDA_Property_Price_Prediction.ipynb  # Data exploration & cleaning
│   ├── 02_Model_Training.ipynb                 # Baseline modeling (Linear Regression, RF)
│   ├── 03_Advanced_Modeling.ipynb              # Advanced models (XGBoost, LightGBM, CatBoost)
│   ├── 04_Model_Explainability.ipynb           # SHAP value feature explanations
│   └── 05_Business_Insights.ipynb              # Valuation impact of amenities and location
├── api/
│   └── main.py                   # FastAPI service code
├── dashboard/
│   └── app.py                    # Streamlit front-end dashboard
├── models/
│   ├── model.pkl                 # Tuned XGBoost model artifact
│   └── preprocessor.pkl          # ColumnTransformer preprocessing pipeline
├── reports/
│   ├── eda_report.md             # Written market report
│   ├── furnishing_impact.png     # Price premium by furnishing type
│   ├── district_impact.png       # Boxplot of top premium locations
│   ├── shap_summary.png          # SHAP global feature importances
│   └── shap_local.png            # SHAP individual prediction waterfall
├── presentation/
│   └── presentation.md           # Markdown portfolio presentation deck
├── Dockerfile                    # Container configuration file
├── requirements.txt              # Project dependencies list
└── README.md                     # Project documentation (this file)
```

---

## 🚀 Quick Start & Installation

### 1. Set Up Environment
Ensure you have Python 3.10+ installed. Clone or navigate to the workspace, create a virtual environment, and install dependencies:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks
The notebooks are located under `notebooks/` and should be executed in order:
1.  `01_EDA_Property_Price_Prediction.ipynb`: Cleans raw dataset and outputs `ready_cleaned_v2.csv`.
2.  `02_Model_Training.ipynb`: Establishes baselines with Linear Regression and Random Forest.
3.  `03_Advanced_Modeling.ipynb`: Compares boosting regressors, tunes XGBoost via Optuna, and exports `models/`.
4.  `04_Model_Explainability.ipynb`: Calculates SHAP values and exports explainability charts.
5.  `05_Business_Insights.ipynb`: Analyzes price margins for districts and premium amenities.

---

## 🔌 FastAPI Service

Start the FastAPI local web server:
```bash
python api/main.py
```
By default, the server runs on `http://127.0.0.1:8000`. 

### Endpoints
*   **GET `/health`**: Returns status `healthy` if model artifacts are loaded.
*   **POST `/predict`**: Returns predicted apartment price in AED.

#### Sample Request Payload:
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

#### Sample Response:
```json
{
  "predicted_price": 3343942.0,
  "currency": "AED"
}
```

---

## 📊 Streamlit Front-End Dashboard

Start the front-end interface:
```bash
streamlit run dashboard/app.py
```
The dashboard runs on `http://localhost:8501` and features five interactive views:
1.  **🔮 Property Price Predictor**: Input bedrooms, bathrooms, area, location, and amenity level to get an instant valuation, plus a listing of the **top 3 most similar properties** in the database.
2.  **📊 Dubai Real Estate Insights**: Displays boxplots, distribution histograms, and average AED/sqft trends across districts.
3.  **🤖 Smart Real Estate Assistant**: A conversational search box that parses queries like *"2 bed under 3.5m in Downtown Dubai"* or *"studio in JVC under 1.2M"* to filter the database.
4.  **🗺️ 3D Investment Map**: A 3D geospatial map tracking property density and absolute/district price deviations.
5.  **⚙️ ML Ops & SQL Monitoring**: A dedicated developer panel to run direct SQL audit queries on incoming requests, trace endpoint latency, and track feature drift in real-time.

---

## 🐳 Dockerization

To run the API in a containerized environment:

### Build Image
```bash
docker build -t dubai-realestate-api .
```

### Run Container
```bash
docker run -d -p 8000:8000 dubai-realestate-api
```
Once started, access the API at `http://127.0.0.1:8000/health`.

---

## 📈 Key Market Takeaways

1.  **Area Impact**: Property area is the most critical numeric predictor, showing a strong positive correlation with property price (0.71).
2.  **Premium Locations**: Palm Jumeirah and Downtown Dubai remain the highest valued districts per square foot.
3.  **Amenity Value**: Access to private/community swimming pools adds a **24% price premium** to median apartment values, while dedicated parking spots command a **14% premium**.
