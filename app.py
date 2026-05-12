import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector

# ====================== RENDER HTML TABLE FUNCTION ======================
def render_html_table(df, max_height=300):
    if df is None or df.empty:
        st.warning("No data to display")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(f"""
    <div style="max-height:{max_height}px; overflow:auto; border:1px solid #ddd; border-radius:8px;">
        {html}
    </div>
    """, unsafe_allow_html=True)

# ====================== MYSQL LOADER FUNCTION ======================
@st.cache_data
def load_data_from_mysql():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="De3@deedee",
        database="logistics_dim"
    )
    df = pd.read_sql("SELECT * FROM final_data", conn)
    conn.close()
    return df

# ====================== APP CONFIG & STYLING ======================
st.set_page_config(page_title="SupplySyncAI - Stage 4", layout="wide")

# ====================== STREAMLIT CONFIG & IMPROVED THEME ======================
st.set_page_config(page_title="SupplySyncAI - Stage 4", layout="wide")
st.markdown("""
<style>
    /* Main Background */
    .stApp { 
        background-color: #EDEDED; 
    }

    /* Important: Reduce negative margin to show header properly */
    .block-container {
        padding-top: 1rem !important;
        margin-top: -2rem !important;     /* Reduced from -6rem */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Full width content */
    section.main > div {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    [data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
    }

    /* Header should be clearly visible */
    header[data-testid="stHeader"] {
        background-color: #EDEDED !important;
        z-index: 1000;
        position: relative;
    }

    /* Button Styling */
    div.stButton > button {
        background-color: #0B2C5D;
        color: white;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #08306B;
    }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown(
    """
    <div style="background-color:#0B2C5D;padding:35px;border-radius:12px;color:white;text-align:center;margin:0 0 20px 0;">
        <h1 style="margin:0 0 8px 0;">SupplySync AI - AI-Optimized Logistics & Routing</h1>
        <p style="font-size:17px;margin-top:15px;">Real-time VRP/CVRP optimization • Truck utilization • Fuel & time prediction • Perishable priority</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ====================== APP DESCRIPTION ======================
st.markdown(
    """
    <div style="background-color:#2F75B5;padding:32px;border-radius:12px;color:white;font-size:17px;line-height:1.8;text-align:justify;">
        This application is an AI-Optimized Logistics & Routing System (Stage 4). 
        It takes your demand forecast data, automatically cleans and prepares the data, 
        and then intelligently generates the most efficient delivery routes and load 
        allocation for your trucks using advanced VRP/CVRP algorithms.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 1. DATA COLLECTION LAYER
# ============================================================
st.markdown(
    """
    <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;margin-top:20px;margin-bottom:10px;">
        <h3 style="margin:0;">1. Data Collection Layer</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background-color:#2F75B5;padding:20px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
    Click below to load data directly from <b>final_data</b> table.
    </div>
    """,
    unsafe_allow_html=True
)

# Left Aligned Small Load Button
if st.button("LOAD DATA", use_container_width=False):
    import time
    start_time = time.time()
    
    try:
        with st.spinner("Loading data from MySQL..."):
            df = load_data_from_mysql()
            st.session_state.df = df
            
            end_time = time.time()
            load_time = round(end_time - start_time, 2)
            
            # Left Aligned Success Box
            st.markdown(f"""
            <div style="background-color:#2F75B5;padding:18px 25px;border-radius:12px;color:white;margin-top:10px;text-align:left;">
                <b>Data loaded successfully from final_data table!</b><br>
                Loaded <b>{df.shape[0]}</b> rows × <b>{df.shape[1]}</b> columns<br>
                Time taken: <b>{load_time} seconds</b>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load data: {e}")

# ====================== DATA PREVIEW ======================
if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    st.markdown(
        """
        <div style="background-color:#0B2C5D;padding:15px 25px;border-radius:10px;color:white;margin-top:30px;">
            <h4 style="margin:0;">Data Preview</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    render_html_table(df.head(20), max_height=280)
    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

   
        # ====================== 2. DATA PRE-PROCESSING ======================
    st.markdown(
        """
        <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;margin-top:40px;margin-bottom:12px;">
            <h3 style="margin:0;">2. Data Pre-Processing</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    step = st.radio("Select Pre-Processing Step", 
                    ["Remove Duplicate Rows", "Remove Outliers", "Replace Missing Values"],
                    horizontal=True, label_visibility="collapsed")

    # ====================== REMOVE DUPLICATE ROWS ======================
    if step == "Remove Duplicate Rows":
        
        
        if st.button("Apply Duplicate Row Removal"):
            before = len(df)
            df_clean = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df_clean)
            
            st.session_state.df = df_clean
            
            # Single Row Table Summary
            st.markdown("#### Summary of Changes")
            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; margin:15px 0;">
                <tr style="background-color:#1F3A5F; color:white;">
                    <th style="padding:12px; text-align:center; border:1px solid #ddd;">Rows Before</th>
                    <th style="padding:12px; text-align:center; border:1px solid #ddd;">Rows After</th>
                    <th style="padding:12px; text-align:center; border:1px solid #ddd;">Duplicates Removed</th>
                </tr>
                <tr style="background-color:#F8FAFC;">
                    <td style="padding:15px; text-align:center; font-size:18px; font-weight:600; border:1px solid #ddd;">{before}</td>
                    <td style="padding:15px; text-align:center; font-size:18px; font-weight:600; border:1px solid #ddd;">{len(df_clean)}</td>
                    <td style="padding:15px; text-align:center; font-size:18px; font-weight:600; color:#FF4B4B; border:1px solid #ddd;">{removed}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

            st.success(f"Duplicates removed successfully!")

    # ====================== REMOVE OUTLIERS ======================
    elif step == "Remove Outliers":
        if st.button("Apply Outlier Removal"):
            before = len(df)
            # Outlier handling logic
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
            
            st.session_state.df = df
            st.success("Outliers handled successfully!")

    # ====================== REPLACE MISSING VALUES ======================
    elif step == "Replace Missing Values":
        if st.button("Apply NULL Replacement"):
            df = df.fillna("Unknown")
            st.session_state.df = df
            st.success("Missing values replaced with 'Unknown'")

    # ====================== PROCESSED DATA PREVIEW ======================
    st.markdown("#### Processed Data Preview")
    render_html_table(st.session_state.df.head(15), max_height=300)
       
else:
    st.info("Click **LOAD DATA** to start.")

############################################
                # Footer
############################################
st.markdown("""
    <br><br>
    <div style="background-color:#2E86C1;padding:12px;text-align:center;color:white;border-radius:6px;font-size:14px;">
        © 2026 SupplySyncAI – AI-Optimized Logistics & Routing
    </div>
""", unsafe_allow_html=True)