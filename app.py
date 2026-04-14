import streamlit as st
import pandas as pd
import numpy as np

# ====================== KEEP YOUR ORIGINAL THEME ======================
st.set_page_config(page_title="SupplySyncAI – MLOps UI", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #EDEDED; }
    .block-container { padding-top: 0rem !important; margin-top: -5.5rem !important; }
    .main { background-color: #f0f2f6 !important; }
    div.stRadio > div { background-color: #00D05E; padding: 16px 0px; border-radius: 8px; width: 100%; }
    div[data-baseweb="radio-group"] { display: flex !important; justify-content: center !important; gap: 50px; }
    div[data-baseweb="radio"] label { font-size: 18px !important; font-weight: 800 !important; color: #FFFFFF !important; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 6px 0 10px 0; }
    .summary-card { border: 2px solid #6B7280; border-radius: 2px; background-color: #F8FAFC; text-align: center; }
    .summary-title { background-color:#1F3A5F; color: #ffffff; font-size: 14px; font-weight: 700; padding: 8px 6px; }
    .summary-value { font-size: 22px; font-weight: 600; color: #000000; padding: 1px 0; }
    .quality-card { background-color: #FFFFFF; border-radius: 14px; padding: 18px 20px; margin-bottom: 22px; box-shadow: 0px 2px 8px rgba(0,0,0,0.06); border-left: 6px solid #2F75B5; }
    .quality-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER (EXACT SAME) ======================
st.markdown(
    """
    <div style="background-color:#0B2C5D;padding:35px;border-radius:12px;color:white;text-align:center;margin:0 0 20px 0;">
        <h1 style="margin:0 0 8px 0;">Stage 4: AI-Optimized Logistics & Routing</h1>
        <p style="font-size:17px;margin-top:15px;">Real-time VRP/CVRP optimization • Truck utilization • Fuel & time prediction • Perishable priority</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:25px;">
    This module takes <b> demand forecasts</b> and converts them into <b>optimal delivery routes</b>, load allocation, and real-time monitoring — solving inefficient transportation, half-empty trucks, delayed replenishment, and spoilage of perishable goods.
    </div>
    """,
    unsafe_allow_html=True
)

# ====================== 1. DATA COLLECTION LAYER ======================
st.markdown("""
<div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;margin-top:30px;margin-bottom:12px;">
    <h3 style="margin:0;"> Data Collection Layer</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:25px;">
    Upload your data file to begin.<br>
</div>
""", unsafe_allow_html=True)

# --------------------- FILE UPLOAD ONLY ---------------------
st.markdown("### Upload Your Data File")

uploaded_file = st.file_uploader("Upload Demand CSV File", type=["csv"], help="Upload your consolidated.csv or any demand file")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"File uploaded successfully: **{uploaded_file.name}**")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Show preview only if file is uploaded
if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("#### Data Preview")
    st.dataframe(df.head(15), use_container_width=True)
    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
else:
    st.warning("Please upload your data file to continue.")

# ====================== 2. DATA PROCESSING & PREPARATION ======================
if "df" in st.session_state and st.session_state.df is not None:
    st.markdown("""
    <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;margin-top:40px;margin-bottom:12px;">
        <h3 style="margin:0;">Data Processing & Preparation</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#2F75B5;padding:24px;border-radius:12px;color:white;font-size:16px;line-height:1.7;margin-bottom:20px;">
        • Clean and validate delivery requests<br>
        • Convert addresses to coordinates (Geocoding)<br>
        • Create distance matrix between warehouse and stores<br>
        • Define constraints (capacity, time windows, priority)
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Data Preprocessing", type="primary", use_container_width=True):
        with st.spinner("Cleaning, validating and preparing data..."):
            df = st.session_state.df.copy()
            
            # 1. Clean and validate
            df = df.drop_duplicates()
            df = df.fillna("Unknown")
            
            # 2. Keep only positive quantities
            if "quantity_sold" in df.columns:
                df = df[df["quantity_sold"] > 0]
            
            # 3. Add mock geolocation (latitude & longitude) if not present
            if "latitude" not in df.columns or "longitude" not in df.columns:
                np.random.seed(42)
                df["latitude"] = np.random.uniform(10.95, 11.08, len(df))
                df["longitude"] = np.random.uniform(76.90, 77.05, len(df))
            
            # Save processed data
            st.session_state.df = df
            
            st.success("Data successfully cleaned, validated, and prepared for routing!")

    # Show processed data
    st.markdown("#### Processed Data (Ready for Next Step)")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

# ====================== FOOTER (EXACT SAME) ======================
st.markdown("""
    <br><br>
    <div style="background-color:#2E86C1;padding:12px;text-align:center;color:white;border-radius:6px;font-size:14px;">
        © 2026 SupplySyncAI – Inventory Intelligence & Analytics Platform
    </div>
""", unsafe_allow_html=True)