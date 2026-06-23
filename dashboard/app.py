import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


sys.path.append(".")
import pydeck as pdk
from models.geospatial import DISTRICT_COORDINATES

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def get_jwt_token():
    """Fetches a JWT token from the FastAPI auth endpoint and stores it in session state."""
    if "jwt_token" not in st.session_state:
        st.session_state["jwt_token"] = None
        
    if st.session_state["jwt_token"] is None:
        try:
            username = os.getenv("API_ADMIN_USER", "admin")
            password = os.getenv("API_ADMIN_PASSWORD", "admin123")
            res = requests.post(
                f"{API_URL}/token",
                data={"username": username, "password": password},
                timeout=3
            )
            if res.status_code == 200:
                st.session_state["jwt_token"] = res.json()["access_token"]
        except Exception:
            pass
    return st.session_state["jwt_token"]

def predict_price_api(payload):
    """Sends a POST request to predict endpoint with JWT token, retrying on 401."""
    token = get_jwt_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, headers=headers, timeout=5)
        # If token is invalid/expired (401), clear and retry once
        if res.status_code == 401:
            st.session_state["jwt_token"] = None
            token = get_jwt_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            res = requests.post(f"{API_URL}/predict", json=payload, headers=headers, timeout=5)
        return res
    except Exception as e:
        raise e

st.set_page_config(page_title="Dubai Real Estate Intelligence", layout="wide")

# Custom premium styling (Glassmorphism & SLEEK Dark Mode)
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    h1 { color: #38bdf8; font-family: 'Outfit', 'Inter', sans-serif; font-weight: 800; text-align: center; }
    h2, h3, h4 { color: #f1f5f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] { background-color: #38bdf8; color: #0f172a !important; font-weight: bold; }
    .stButton>button {
        background-color: #38bdf8;
        color: #0f172a;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #0ea5e9; color: #f1f5f9; transform: translateY(-2px); }
    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-title { color: #38bdf8; font-size: 1.2rem; font-weight: 700; }
    .card-metric { font-size: 1.5rem; font-weight: 800; color: #34d399; }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ Dubai Apartment Price Prediction & Intelligent Assistant")
st.markdown("<p style='text-align: center; color: #94a3b8;'>End-to-End ML valuations, similarity recommendations, and conversational search assistant</p>", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    if os.path.exists("data/ready_cleaned_v2.csv"):
        return pd.read_csv("data/ready_cleaned_v2.csv")
    return None

df = load_data()

# Fallback values
if df is not None:
    districts = sorted(df["district"].unique().tolist())
    furnishings = sorted(df["furnished"].unique().tolist())
else:
    districts = ["Downtown Dubai", "Dubai Marina", "Jumeirah Village Circle (JVC)", "Palm Jumeirah", "Business Bay"]
    furnishings = ["Furnished", "Unfurnished", "Unknown"]

# Recommendation Engine Logic
def get_recommendations(user_input, df, top_n=3):
    if df is None:
        return pd.DataFrame()
    
    # 1. Filter by district if possible
    df_dist = df[df["district"] == user_input["district"]]
    if df_dist.empty:
        df_dist = df.copy()
        
    # 2. Calculate feature distance (Beds, Baths, Area normalized, View, Maids, Freehold)
    df_dist = df_dist.copy()
    area_std = df["area"].std() if df["area"].std() > 0 else 1.0
    
    df_dist["distance"] = (
        abs(df_dist["beds"] - user_input["beds"]) * 2.0 + 
        abs(df_dist["baths"] - user_input["baths"]) * 1.0 +
        abs(df_dist["area"] - user_input["area"]) / area_std * 5.0 +
        abs(df_dist["has_view"] - user_input["has_view"]) * 1.5 +
        abs(df_dist["has_maids_room"] - user_input["has_maids_room"]) * 1.0 +
        abs(df_dist["is_freehold"] - user_input["is_freehold"]) * 1.0
    )
    
    # Sort and return top N
    recs = df_dist.sort_values("distance").head(top_n)
    return recs[["price", "beds", "baths", "area", "furnished", "luxury_score", "has_view", "has_maids_room", "is_freehold", "property_name"]]

# Query parsing helper for LLM Assistant tab
def parse_query(q, df):
    q = q.lower()
    
    # Abbreviations mapping
    abbrev_map = {
        "jvc": "jumeirah village circle (jvc)",
        "jbr": "jumeirah beach residence (jbr)",
        "jlt": "jumeirah lake towers (jlt)",
        "palm": "palm jumeirah",
        "downtown": "downtown dubai",
        "marina": "dubai marina",
        "business bay": "business bay"
    }
    
    # Match district abbreviations first
    matched_district = None
    for abbrev, full_name in abbrev_map.items():
        if abbrev in q:
            # Confirm full name matches in dataset
            districts_list = df["district"].unique().tolist()
            for d in districts_list:
                if d.lower() == full_name.lower():
                    matched_district = d
                    break
            break
            
    # Match direct district names in dataset
    if not matched_district:
        districts_list = df["district"].unique().tolist()
        for d in districts_list:
            if d.lower() in q:
                matched_district = d
                break
                
    # Extract bedrooms (allowing optional spaces or hyphens, e.g. "2-bed", "2 bed", "2bedroom")
    beds = None
    if "studio" in q:
        beds = 0
    else:
        bed_match = re.search(r'(\d+)\s*[-]?\s*(?:bed|bedroom|br)', q)
        if bed_match:
            beds = int(bed_match.group(1))
            
    # Extract max price (e.g. under 2.5m, under 1.5 million, below 800k)
    max_price = None
    price_match = re.search(r'(?:under|below|less than|budget of)\s*(?:aed)?\s*([\d\.]+)\s*(m|million|k)?', q)
    if price_match:
        val = float(price_match.group(1))
        unit = price_match.group(2)
        if unit in ['m', 'million']:
            max_price = val * 1_000_000
        elif unit == 'k':
            max_price = val * 1_000
        else:
            if val < 50: # Assume millions if value is small
                max_price = val * 1_000_000
            else:
                max_price = val
                
    # Extract binary features
    has_view = None
    if "view" in q or "sea" in q or "canal" in q:
        has_view = 1
        
    has_maids_room = None
    if "maid" in q:
        has_maids_room = 1
        
    is_freehold = None
    if "freehold" in q:
        is_freehold = 1
                
    return beds, max_price, matched_district, has_view, has_maids_room, is_freehold


# App Tabs Setup
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Property Price Predictor", "📊 Dubai Real Estate Insights", "🤖 Smart Assistant", "🗺️ 3D Investment Map", "⚙️ ML Ops & SQL Monitoring"])

# Tab 1: Predictor
with tab1:
    st.subheader("🔮 Property Valuation Estimator")
    st.write("Fill in the property details below to receive a predicted price from our machine learning model.")
    
    col1, col2 = st.columns(2)
    with col1:
        beds = st.number_input("Bedrooms (0 for Studio)", min_value=0, max_value=10, value=2, step=1, key="p_beds")
        baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1, key="p_baths")
        area = st.number_input("Area (SqFt)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0, key="p_area")
    with col2:
        district = st.selectbox("District", districts, key="p_dist")
        furnished = st.selectbox("Furnishing Status", furnishings, key="p_furn")
        luxury_score = st.slider("Luxury Score (0 = basic, 5 = ultra-luxury amenities)", 0, 5, 3, key="p_lux")
        
        st.markdown("**Property Characteristics:**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            has_view = st.checkbox("Sea/Canal View", value=False, key="p_view")
        with col_c2:
            has_maids_room = st.checkbox("Maid's Room", value=False, key="p_maids")
        with col_c3:
            is_freehold = st.checkbox("Freehold", value=True, key="p_freehold")

    if st.button("Predict Valuation"):
        payload = {
            "beds": int(beds),
            "baths": int(baths),
            "area": float(area),
            "luxury_score": int(luxury_score),
            "has_view": int(has_view),
            "has_maids_room": int(has_maids_room),
            "is_freehold": int(is_freehold),
            "district": district,
            "furnished": furnished
        }
        
        pred_success = False
        pred_price = None
        
        try:
            res = predict_price_api(payload)
            if res.status_code == 200:
                pred_price = res.json()["predicted_price"]
                pred_success = True
            elif res.status_code == 429:
                st.error("API rate limit exceeded. Please wait a minute and try again.")
            else:
                # Triggers fallback
                raise Exception(f"API status {res.status_code}")
        except Exception:
            # Fallback to local joblib
            model_path = "models/model.pkl"
            prep_path = "models/preprocessor.pkl"
            if os.path.exists(model_path) and os.path.exists(prep_path):
                try:
                    model = joblib.load(model_path)
                    preprocessor = joblib.load(prep_path)
                    
                    from models.geospatial import add_geospatial_features
                    input_df = pd.DataFrame([payload])
                    input_df = add_geospatial_features(input_df)
                    X_proc = preprocessor.transform(input_df)
                    pred_log = model.predict(X_proc)
                    pred_price = float(np.expm1(pred_log)[0])
                    pred_success = True
                except Exception as e:
                    st.error(f"Fallback prediction failed: {e}")
            else:
                st.error("API is offline and local model files were not found.")
                
        if pred_success and pred_price is not None:
            st.balloons()
            st.markdown(f"""
            <div class="card">
                <div class="card-title">💵 Estimated Valuation Result</div>
                <div class="card-metric">AED {pred_price:,.2f}</div>
                <p style="color: #94a3b8; margin-top: 8px;">Price per SqFt: <b>{pred_price/area:,.2f} AED/sqft</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation output
            if df is not None:
                st.markdown("### 🔍 Similar Properties in our database")
                recs = get_recommendations(payload, df)
                
                rec_cols = st.columns(3)
                for idx, (_, row) in enumerate(recs.iterrows()):
                    prop_title = row['property_name']
                    if prop_title == row['district'] or prop_title == "Apartment":
                        prop_title = f"Apartment in {row['district']}"
                        
                    with rec_cols[idx]:
                        st.markdown(f"""
                        <div class="card">
                            <div class="card-title">{prop_title}</div>
                            <div class="card-metric" style="color: #38bdf8; font-size: 1.25rem;">AED {row['price']:,.0f}</div>
                            <p style="margin: 4px 0; color: #94a3b8; font-size: 0.85rem;">📍 {row['district']}</p>
                            <p style="margin: 4px 0; color: #cbd5e1;">📐 <b>{row['area']:,.0f} sqft</b></p>
                            <p style="margin: 4px 0; color: #cbd5e1;">🛏️ {row['beds']} Beds | 🛁 {row['baths']} Baths</p>
                            <p style="margin: 4px 0; color: #cbd5e1;">🛋️ {row['furnished']}</p>
                            <p style="margin: 4px 0; color: #34d399;">✨ Luxury Score: {row['luxury_score']}/5</p>
                            <p style="margin: 4px 0; color: #cbd5e1;">🌅 View: {"Yes" if row['has_view'] == 1 else "No"} | 🧹 Maids: {"Yes" if row['has_maids_room'] == 1 else "No"}</p>
                            <p style="margin: 4px 0; color: #cbd5e1;">🔑 Freehold: {"Yes" if row['is_freehold'] == 1 else "No"}</p>
                        </div>
                        """, unsafe_allow_html=True)

# Tab 2: Dashboard Insights
with tab2:
    st.subheader("📊 Dubai Real Estate Market Dashboard")
    if df is not None:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Top 10 Most Expensive Districts")
            top_districts = df.groupby("district")["price"].median().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            sns.barplot(x=top_districts.values, y=top_districts.index, hue=top_districts.index, palette="viridis", legend=False, ax=ax)
            ax.tick_params(colors='#f1f5f9')
            ax.xaxis.label.set_color('#f1f5f9')
            ax.yaxis.label.set_color('#f1f5f9')
            ax.set_xlabel("Median Price (AED)")
            ax.set_ylabel("District")
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_m2:
            st.markdown("#### Price Distribution in Dubai")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#1e293b')
            ax2.set_facecolor('#1e293b')
            sns.histplot(df["price"], bins=30, kde=True, ax=ax2, color="#38bdf8")
            ax2.tick_params(colors='#f1f5f9')
            ax2.xaxis.label.set_color('#f1f5f9')
            ax2.yaxis.label.set_color('#f1f5f9')
            ax2.set_xlim(0, 10_000_000)
            ax2.set_xlabel("Price (AED)")
            plt.tight_layout()
            st.pyplot(fig2)
            
        st.markdown("---")
        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.markdown("#### Average Price per SqFt by District (Top 10)")
            top_sqft = df.groupby("district")["price_per_sqft"].mean().sort_values(ascending=False).head(10)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            fig3.patch.set_facecolor('#1e293b')
            ax3.set_facecolor('#1e293b')
            sns.barplot(x=top_sqft.values, y=top_sqft.index, hue=top_sqft.index, palette="magma", legend=False, ax=ax3)
            ax3.tick_params(colors='#f1f5f9')
            ax3.xaxis.label.set_color('#f1f5f9')
            ax3.yaxis.label.set_color('#f1f5f9')
            ax3.set_xlabel("Average Price / SqFt (AED)")
            plt.tight_layout()
            st.pyplot(fig3)
            
        with col_m4:
            st.markdown("#### Price Premium by Furnishing Status")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            fig4.patch.set_facecolor('#1e293b')
            ax4.set_facecolor('#1e293b')
            sns.boxplot(data=df, x="furnished", y="price", hue="furnished", palette="Set2", legend=False, ax=ax4)
            ax4.tick_params(colors='#f1f5f9')
            ax4.xaxis.label.set_color('#f1f5f9')
            ax4.yaxis.label.set_color('#f1f5f9')
            ax4.set_yscale("log")
            ax4.set_ylabel("Price (AED) - Log Scale")
            ax4.set_xlabel("Furnishing")
            plt.tight_layout()
            st.pyplot(fig4)
    else:
        st.warning("Cleaned dataset not found. Please ensure data/ready_cleaned_v2.csv exists to view the dashboard.")

# Tab 3: Smart Assistant
with tab3:
    st.subheader("🤖 Smart Real Estate Assistant")
    st.write("Ask the assistant to search the database. Example queries:")
    st.markdown("- *Show me 2-bedroom apartments in Downtown Dubai under 3.5 million*")
    st.markdown("- *Find studios in JVC under 1.2M*")
    st.markdown("- *Show me 3 bed properties in Dubai Marina*")
    
    query = st.text_input("Type your real estate search request here:", placeholder="e.g. 2-bed in Dubai Marina under 2.5 million")
    
    if query:
        if df is not None:
            beds, max_price, district, has_view, has_maids_room, is_freehold = parse_query(query, df)
            
            # Execute search
            results_df = df.copy()
            
            summary_parts = []
            if district:
                results_df = results_df[results_df["district"] == district]
                summary_parts.append(f"in **{district}**")
            if beds is not None:
                results_df = results_df[results_df["beds"] == beds]
                summary_parts.append(f"with **{beds} bedrooms**" if beds > 0 else "which are **studios**")
            if max_price is not None:
                results_df = results_df[results_df["price"] <= max_price]
                summary_parts.append(f"under **AED {max_price:,.0f}**")
            if has_view is not None:
                results_df = results_df[results_df["has_view"] == has_view]
                summary_parts.append("with a **view**")
            if has_maids_room is not None:
                results_df = results_df[results_df["has_maids_room"] == has_maids_room]
                summary_parts.append("with a **maid's room**")
            if is_freehold is not None:
                results_df = results_df[results_df["is_freehold"] == is_freehold]
                summary_parts.append("which is **freehold**")
                
            st.markdown(f"🤖 **Assistant search results** " + " ".join(summary_parts) + ":")
            
            if not results_df.empty:
                st.markdown(f"Found **{len(results_df)}** matching listings. Showing top 5:")
                
                # Show top 5 sorted by price
                display_recs = results_df.sort_values("price", ascending=True).head(5)
                
                for idx, (_, row) in enumerate(display_recs.iterrows()):
                    prop_title = row['property_name']
                    if prop_title == row['district'] or prop_title == "Apartment":
                        prop_title = f"Apartment in {row['district']}"
                        
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid #38bdf8;">
                        <span style="float: right; font-weight: bold; color: #34d399; font-size: 1.15rem;">AED {row['price']:,.2f}</span>
                        <div style="font-weight: bold; font-size: 1.1rem; color: #38bdf8;">{prop_title}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">📍 {row['district']}</div>
                        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                            📐 {row['area']:,.0f} sqft | 🛏️ {row['beds']} Beds | 🛁 {row['baths']} Baths | {row['furnished']} | 🌅 View: {"Yes" if row['has_view'] == 1 else "No"} | 🧹 Maid's: {"Yes" if row['has_maids_room'] == 1 else "No"} | 🔑 Freehold: {"Yes" if row['is_freehold'] == 1 else "No"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("No exact matching properties found. Please try expanding your filters or search query!")
        else:
            st.warning("Database unavailable for parsing.")

# Tab 4: 3D Investment Map
with tab4:
    st.subheader("🗺️ Dubai Real Estate 3D Investment Map")
    st.write(
        "Interact with Dubai properties in 3D. Column height represents **Price (AED)**, "
        "and color represents **Value/Price-Density (AED/sqft)** classification."
    )
    
    if df is not None:
        # Create map filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            unique_districts = sorted(df["district"].unique().tolist())
            selected_districts = st.multiselect("Filter Districts", unique_districts, default=[], placeholder="All Districts", key="map_dist")
        with col_f2:
            unique_beds = sorted(df["beds"].unique().tolist())
            selected_beds = st.multiselect("Filter Bedrooms", unique_beds, default=unique_beds, key="map_beds")
        with col_f3:
            color_mode = st.selectbox(
                "Value Color-Coding Mode",
                ["Local Value (Relative to District Median)", "Absolute Price-Density (AED/sqft)"],
                key="map_color_mode"
            )
            
        col_f4, col_f5 = st.columns(2)
        with col_f4:
            price_min_val = float(df["price"].min())
            price_max_val = float(df["price"].max())
            selected_price = st.slider("Filter Price Range (AED)", price_min_val, price_max_val, (price_min_val, price_max_val), format="AED %d", key="map_price")
        with col_f5:
            elevation_scale = st.slider("3D Column Height Multiplier", 0.1, 5.0, 1.0, step=0.1, key="map_elev")

        # Apply filters to map data
        map_df = df.copy()
        if selected_districts:
            map_df = map_df[map_df["district"].isin(selected_districts)]
        if selected_beds:
            map_df = map_df[map_df["beds"].isin(selected_beds)]
        map_df = map_df[(map_df["price"] >= selected_price[0]) & (map_df["price"] <= selected_price[1])]

        if not map_df.empty:
            # Map coordinates
            default_coord = DISTRICT_COORDINATES.get('Unknown', (25.2048, 55.2708))
            map_df["latitude_base"] = map_df["district"].apply(lambda d: DISTRICT_COORDINATES.get(d, default_coord)[0])
            map_df["longitude_base"] = map_df["district"].apply(lambda d: DISTRICT_COORDINATES.get(d, default_coord)[1])
            
            # Add ±0.0015 coordinate jitter for separation
            np.random.seed(42)
            map_df["latitude"] = map_df["latitude_base"] + np.random.uniform(-0.0015, 0.0015, len(map_df))
            map_df["longitude"] = map_df["longitude_base"] + np.random.uniform(-0.0015, 0.0015, len(map_df))
            
            # Classify price density for coloring
            # Green: Undervalued, Blue: Average, Red: Premium/Luxury
            if color_mode == "Local Value (Relative to District Median)":
                district_medians = df.groupby("district")["price_per_sqft"].median().to_dict()
                def get_local_color(row):
                    dist = row["district"]
                    price_per_sqft = row["price_per_sqft"]
                    median = district_medians.get(dist, 1762.0)
                    if price_per_sqft < 0.9 * median:
                        return [46, 204, 113]  # Green (Undervalued)
                    elif price_per_sqft > 1.1 * median:
                        return [231, 76, 60]   # Red (Premium)
                    else:
                        return [52, 152, 219]  # Blue (Average)
                colors = map_df.apply(get_local_color, axis=1)
            else:
                # Absolute thresholds
                def get_absolute_color(row):
                    price_per_sqft = row["price_per_sqft"]
                    if price_per_sqft < 1250:
                        return [46, 204, 113]  # Green
                    elif price_per_sqft > 2500:
                        return [231, 76, 60]   # Red
                    else:
                        return [52, 152, 219]  # Blue
                colors = map_df.apply(get_absolute_color, axis=1)
                
            map_df["color_r"] = [c[0] for c in colors]
            map_df["color_g"] = [c[1] for c in colors]
            map_df["color_b"] = [c[2] for c in colors]
            
            # Map elevation = price * scale * 0.0001
            map_df["elevation"] = map_df["price"] * (elevation_scale * 0.00015)
            
            # Pre-format columns for Pydeck's client-side JS tooltip rendering
            map_df["price_formatted"] = map_df["price"].apply(lambda p: f"AED {p:,.2f}")
            map_df["area_formatted"] = map_df["area"].apply(lambda a: f"{a:,.0f} sqft")
            map_df["price_per_sqft_formatted"] = map_df["price_per_sqft"].apply(lambda p: f"{p:,.2f} AED/sqft")
            
            # Display layer
            layer = pdk.Layer(
                "ColumnLayer",
                data=map_df,
                get_position="[longitude, latitude]",
                get_elevation="elevation",
                elevation_scale=1.0,
                radius=100,
                get_fill_color="[color_r, color_g, color_b, 180]",
                pickable=True,
                auto_highlight=True,
            )
            
            view_state = pdk.ViewState(
                latitude=25.0794,
                longitude=55.1368,
                zoom=10.5,
                pitch=55,
                bearing=-35
            )
            
            tooltip = {
                "html": """
                    <b>Property:</b> {property_name}<br/>
                    <b>District:</b> {district}<br/>
                    <b>Price:</b> {price_formatted}<br/>
                    <b>Area:</b> {area_formatted}<br/>
                    <b>Price/SqFt:</b> {price_per_sqft_formatted}<br/>
                    <b>Beds/Baths:</b> {beds} / {baths}
                """,
                "style": {"backgroundColor": "#1e293b", "color": "#f1f5f9", "fontSize": "13px"}
            }
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style=pdk.map_styles.CARTO_DARK
            ))
            
            # Map metrics / summary
            st.markdown("### 📈 Map Distribution Summary")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                green_count = sum(1 for c in colors if c == [46, 204, 113])
                st.metric("Undervalued Listings (Green)", f"{green_count} ({green_count/len(map_df)*100:.1f}%)")
            with col_m2:
                blue_count = sum(1 for c in colors if c == [52, 152, 219])
                st.metric("Fair Value Listings (Blue)", f"{blue_count} ({blue_count/len(map_df)*100:.1f}%)")
            with col_m3:
                red_count = sum(1 for c in colors if c == [231, 76, 60])
                st.metric("Premium Listings (Red)", f"{red_count} ({red_count/len(map_df)*100:.1f}%)")
        else:
            st.error("No listings match the selected map filters.")
    else:
        st.warning("Cleaned dataset not found. Please ensure data/ready_cleaned_v2.csv exists.")

# Tab 5: ML Ops & SQL Monitoring
with tab5:
    st.subheader("⚙️ Production ML Operations & Database Monitoring")
    st.write(
        "Monitor live ML endpoint performance, track request latency, run direct SQL audits, "
        "and monitor feature drift against the training dataset."
    )
    
    # Helper to load logs
    database_url = os.getenv("DATABASE_URL")
    db_path = os.getenv("DB_PATH", "data/predictions.db")
    
    def get_db_connection():
        if database_url:
            import psycopg2
            return psycopg2.connect(database_url, connect_timeout=3)
        else:
            import sqlite3
            return sqlite3.connect(db_path, timeout=3.0)
        
    @st.cache_data(ttl=60)
    def load_logs():
        if not database_url and not os.path.exists(db_path):
            return pd.DataFrame()
        try:
            conn = get_db_connection()
            df_l = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC", conn)
            conn.close()
            return df_l
        except Exception as e:
            # Silently fail here to prevent blocking UI loading; user is notified in the UI.
            return pd.DataFrame()

    df_logs = load_logs()
    
    if df_logs.empty:
        st.info("No prediction logs found in database. Once predictions are run through the API or Predictor tab, telemetry will appear here.")
        if database_url:
            st.warning("⚠️ Note: A external DATABASE_URL is configured, but the dashboard was unable to connect. Please ensure your database accepts public incoming traffic and that the connection credentials are correct (connection timeout is 3s).")
        
        # Option to log sample mock data for demonstration purposes
        if st.button("Generate Telemetry Mock Data"):
            import random
            from datetime import datetime, timedelta
            conn = get_db_connection()
            cursor = conn.cursor()
            if database_url:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp VARCHAR(50),
                        beds INTEGER,
                        baths INTEGER,
                        area DOUBLE PRECISION,
                        luxury_score INTEGER,
                        has_view INTEGER,
                        has_maids_room INTEGER,
                        is_freehold INTEGER,
                        district VARCHAR(255),
                        furnished VARCHAR(100),
                        predicted_price DOUBLE PRECISION,
                        client_host VARCHAR(100),
                        latency_ms DOUBLE PRECISION DEFAULT 0.0
                    )
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        beds INTEGER,
                        baths INTEGER,
                        area REAL,
                        luxury_score INTEGER,
                        has_view INTEGER,
                        has_maids_room INTEGER,
                        is_freehold INTEGER,
                        district TEXT,
                        furnished TEXT,
                        predicted_price REAL,
                        client_host TEXT,
                        latency_ms REAL DEFAULT 0.0
                    )
                """)
            
            placeholder = "%s" if database_url else "?"
            query = f"""
                INSERT INTO prediction_logs (
                    timestamp, beds, baths, area, luxury_score, has_view, has_maids_room, 
                    is_freehold, district, furnished, predicted_price, client_host, latency_ms
                ) VALUES ({', '.join([placeholder]*13)})
            """
            
            sample_districts = ["Downtown Dubai", "Dubai Marina", "Jumeirah Village Circle (JVC)", "Palm Jumeirah", "Business Bay"]
            now = datetime.utcnow()
            for i in range(100):
                timestamp = (now - timedelta(minutes=random.randint(1, 1440))).isoformat()
                beds = random.choice([0, 1, 2, 3])
                baths = beds + random.choice([1, 2])
                area = float(beds * 600 + random.randint(400, 1000))
                luxury_score = random.randint(0, 5)
                has_view = random.choice([0, 1])
                has_maids_room = random.choice([0, 1])
                is_freehold = 1
                district = random.choice(sample_districts)
                furnished = random.choice(["Furnished", "Unfurnished"])
                predicted_price = float(area * 1500 + random.randint(-100000, 500000))
                client_host = f"192.168.1.{random.randint(10, 50)}"
                latency_ms = float(12.5 + random.uniform(2.5, 45.0))
                
                cursor.execute(query, (timestamp, beds, baths, area, luxury_score, has_view, has_maids_room, 
                      is_freehold, district, furnished, predicted_price, client_host, latency_ms))
            conn.commit()
            conn.close()
            st.success("Successfully logged 100 mock prediction queries!")
            st.rerun()
    else:
        # We have data! Let's display monitoring tabs
        mon_tab1, mon_tab2, mon_tab3 = st.tabs(["📊 Performance Telemetry", "🔍 SQL Query Console", "⚖️ Feature Drift Detection"])
        
        with mon_tab1:
            st.markdown("### 📈 Endpoint Performance Metrics")
            
            # KPI Cards
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                st.metric("Total API Requests", len(df_logs))
            with col_k2:
                avg_lat = df_logs["latency_ms"].mean() if "latency_ms" in df_logs.columns else 0.0
                st.metric("Avg Latency", f"{avg_lat:.2f} ms")
            with col_k3:
                unique_ips = df_logs["client_host"].nunique()
                st.metric("Active Clients (IPs)", unique_ips)
                
            st.markdown("---")
            
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.markdown("#### Latency Distribution over Time")
                if "latency_ms" in df_logs.columns and "timestamp" in df_logs.columns:
                    # Convert to datetime and sort
                    df_logs["parsed_time"] = pd.to_datetime(df_logs["timestamp"])
                    lat_df = df_logs.sort_values("parsed_time")
                    st.line_chart(lat_df.set_index("parsed_time")["latency_ms"])
                else:
                    st.warning("Latency data unavailable.")
                    
            with col_l2:
                st.markdown("#### API Request Load by District")
                dist_counts = df_logs["district"].value_counts()
                st.bar_chart(dist_counts)
                
            st.markdown("#### Recent API Requests Logs")
            st.dataframe(df_logs.head(10)[["timestamp", "beds", "baths", "area", "district", "predicted_price", "client_host", "latency_ms"]])
            
        with mon_tab2:
            st.markdown("### 🔍 SQL Query Console")
            st.write("Write and execute standard SQL queries directly against the `prediction_logs` SQLite table to audit system data.")
            
            # Simple pre-defined query selector
            preset_query = st.selectbox(
                "Or load a preset SQL audit query:",
                [
                    "Custom Query",
                    "SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 5;",
                    "SELECT district, count(*), avg(predicted_price) FROM prediction_logs GROUP BY district ORDER BY count(*) DESC;",
                    "SELECT beds, count(*), avg(latency_ms) FROM prediction_logs GROUP BY beds;",
                    "SELECT client_host, count(*), max(latency_ms) FROM prediction_logs GROUP BY client_host;"
                ]
            )
            
            query_input_val = "" if preset_query == "Custom Query" else preset_query
            sql_query = st.text_area("SQL Editor", value=query_input_val, height=120, placeholder="SELECT * FROM prediction_logs LIMIT 10;")
            
            if st.button("Run Query", key="run_sql"):
                if sql_query.strip():
                    try:
                        conn = get_db_connection()
                        query_df = pd.read_sql_query(sql_query, conn)
                        conn.close()
                        st.success("Query executed successfully!")
                        st.dataframe(query_df)
                    except Exception as ex:
                        st.error(f"SQL Error: {ex}")
                else:
                    st.warning("Please enter a SQL query to execute.")
                    
        with mon_tab3:
            st.markdown("### ⚖️ Live Data Drift Monitor")
            st.write(
                "Detects differences in the distribution of user queries (production data) "
                "relative to the original property dataset (training baseline) to monitor model accuracy degradation."
            )
            
            if df is not None:
                # We compare Median Area and Median Price
                med_area_train = df["area"].median()
                med_area_prod = df_logs["area"].median()
                
                area_diff_pct = abs(med_area_prod - med_area_train) / med_area_train * 100
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("#### Area (SqFt) Distribution Comparison")
                    st.metric("Training Baseline Median", f"{med_area_train:,.1f} sqft")
                    st.metric("Production Logs Median", f"{med_area_prod:,.1f} sqft", delta=f"{med_area_prod - med_area_train:+,.1f} sqft ({area_diff_pct:.1f}%)")
                    
                with col_d2:
                    st.markdown("#### Drift Status")
                    if area_diff_pct > 25:
                        st.error("⚠️ DRIFT DETECTED\nProduction queries differ significantly in property size (area) compared to training data. Consider retraining the model.")
                    elif area_diff_pct > 10:
                        st.warning("⚠️ MILD DRIFT\nSmall shifts in size profiles detected. Keep an eye on system behavior.")
                    else:
                        st.success("✅ STABLE\nRequest properties are statistically similar to the model's training baseline.")
                        
                st.markdown("---")
                st.markdown("#### Bed Distribution comparison (Train vs Prod)")
                
                train_beds = df["beds"].value_counts(normalize=True).sort_index()
                prod_beds = df_logs["beds"].value_counts(normalize=True).sort_index()
                
                comp_df = pd.DataFrame({
                    "Training Baseline (%)": train_beds * 100,
                    "Production Requests (%)": prod_beds * 100
                }).fillna(0.0)
                
                st.bar_chart(comp_df)
            else:
                st.warning("Training baseline dataset (data/ready_cleaned_v2.csv) is unavailable to compare drift statistics.")

