import streamlit as st
from supabase import create_client
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from utils.html_table import render_html_table
import altair as alt
from streamlit_option_menu import option_menu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time



# ====================== SUPABASE LOADER FUNCTION ======================
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_data(ttl=600)
def load_data_from_supabase():
    try:
        client = init_connection()
        response = client.table("final_data").select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"Database Connection Failed: {e}")
        return None
    

# ====================== STREAMLIT CONFIG & IMPROVED THEME ======================

st.set_page_config(
    page_title="SupplySyncAI",
    layout="wide"
)

st.markdown("""
<style>
div[data-testid="stAlert"] {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #EDEDED;
    margin: 0;
    padding: 0;
}

/* Remove top spacing completely */
.block-container {
    padding-top: 0rem !important;
    margin-top: -5.5rem !important;
}

/* keep app background */
.main {
    background-color: #f0f2f6 !important;
}

/* Remove main section spacing */
section.main > div:first-child {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}
            
/* Remove extra container padding */
[data-testid="stAppViewContainer"] {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/*  REMOVE TOP GAP COMPLETELY */
[data-testid="stAppViewContainer"] {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/*  REMOVE TOP SPACER DIV */
[data-testid="stAppViewContainer"] > div:first-child {
    margin-top: 0rem !important;
    padding-top: 0rem !important;
} 
              
/* Header fix */
header[data-testid="stHeader"] {
    position: relative;
    background-color: #EDEDED !important;
}
            
header[data-testid="stHeader"] * {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Block container — single source of truth */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

section.main > div {
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

[data-testid="stAppViewContainer"] {
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    overflow-x: hidden !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* =========================================
   RADIO CONTAINER – FULL WIDTH
   ========================================= */
div.element-container:has(div.stRadio) {
    width: 100% !important;
}

/* =========================================
   Teal WRAP BOX – FULL PAGE WIDTH
   ========================================= */
div.stRadio > div {
    background-color:  #00D05E;
    padding: 16px 0px;
    border-radius: 8px;
    width: 100%;
    box-sizing: border-box;
    display: flex;
    justify-content: center;
}

/* =========================================
   RADIO GROUP ALIGNMENT
   ========================================= */
div[data-baseweb="radio-group"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center;
    gap: 50px;
    width: 100%;
    margin: 0 auto;
}
            
div[data-baseweb="radio"] {
    display: flex;
    align-items: center;
    justify-content: center;
}

/* =========================================
   RADIO OPTION TEXT
   ========================================= */
/* RADIO LABEL TEXT – FORCE WHITE */
div[data-baseweb="radio"] label,
div[data-baseweb="radio"] label span {
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    white-space: nowrap;
}


/* =========================================
   SPACE BETWEEN OPTIONS
   ========================================= */
div[data-baseweb="radio"] {
    margin-right: 28px;
}

          

</style>
""", unsafe_allow_html=True)

st.markdown(""" 
 <style> /* Expander outer card */ 
    div[data-testid="stExpander"]
        { background-color: #2F75B5;
        border-radius: 20px; 
        border: 1px solid #9EDAD0; 
        overflow: hidden; /* 🔑 fixes unfinished edges */ }
    /* Hide expander header completely */
    div[data-testid="stExpander"]:nth-of-type(1)
             summary { display: none; }
    /* Inner content padding fix */
     div[data-testid="stExpander"]:nth-of-type(1) > 
            div { padding: 22px 18px; } 
            </style> """, unsafe_allow_html=True)


st.markdown(
    """
    <style>
        /* Dark blue themed button */
        div.stButton > button {
            background-color: #0B2C5D;   /* Dark blue from your header */
            color: #FFFFFF;
            border-radius: 8px;
            padding: 8px 18px;
            border: none;
            font-weight: 600;
        }

        div.stButton > button:hover {
            background-color: #08306B;   /* Slightly darker on hover */
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>

/* =========================================
   SUMMARY GRID (CENTERED, SMALL, EQUAL BOXES)
   ========================================= */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin: 6px 0 10px 0;
    justify-content: center;
    
}

/* =========================================
   SUMMARY CARD (TABLE CONTAINER)
   ========================================= */
.summary-card {
    border: 2px solid #6B7280;
    border-radius: 2px;
    background-color: #F8FAFC;
    overflow: hidden;
    text-align: center;
}

/* =========================================
   HEADER ROW (NO WRAP, SAME HEIGHT)
   ========================================= */
.summary-title {
    background-color:#1F3A5F;
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    padding: 8px 6px;
    border-bottom: 1px solid #6B7280;

    white-space: nowrap;       /* 🔥 stop wrapping */
    overflow: hidden;
    text-overflow: ellipsis;
}

/* =========================================
   VALUE CELL (COMPACT)
   ========================================= */
.summary-value {
    font-size: 22px;
    font-weight: 600;
    color: #000000;
    padding: 1px 0;
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

st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:25px;
    ">
    <p>
    <strong>SupplySync AI</strong> is an intelligent AI-powered Logistics & Routing Optimization System 
    built for retail businesses.
    </p>
    <p>
    It automatically processes demand forecasts, cleans and prepares the data, and intelligently generates 
    the most efficient delivery routes and optimal load allocation for your trucks using advanced VRP/CVRP algorithms.
    </p>
    
    <h4 style="margin-top:22px;">Key Capabilities</h4>
    
    <ul>
        <li>Advanced route optimization to reduce travel distance and fuel costs</li>
        <li>Smart load allocation for maximum truck utilization</li>
        <li>Automated scheduling for timely store replenishment</li>
        <li>Minimizes spoilage of perishable goods through faster deliveries</li>
        <li>Significant reduction in overall transportation costs and inefficiencies</li>
    </ul>
    
    <p style="margin-top:15px;">
        <b>The result:</b> More efficient deliveries, lower operational costs, better resource utilization, 
        and improved profitability across your retail supply chain.
    </p>
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
            df = load_data_from_supabase()
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
                    horizontal=True, 
                    label_visibility="collapsed",
                    index=None,           # ← No default selection
                    key="preprocess_step")

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


# ============================================================
# STEP 3 – EDA (LOCKED UNTIL PREPROCESSING)
# ============================================================

if "df" not in st.session_state or st.session_state.df is None:
    st.stop()

df = st.session_state.df

if "eda_completed" not in st.session_state:
    st.session_state.eda_completed = False

# ---------------- EDA HEADER ----------------
st.markdown(
    """
    <div style="
        background-color:#0B2C5D;
        padding:18px 25px;
        border-radius:10px;
        color:white;
        margin-top:20px;
        margin-bottom:10px;
    ">
        <h3 style="margin:0;">Exploratory Data Analysis (EDA)</h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")
st.info(f"Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.write("")

# ---------------- EDA INTRO CARD ----------------
st.markdown(
    """
    <div style="
        background-color:#2F75B5;
        padding:28px;
        border-radius:12px;
        color:white;
        font-size:16px;
        line-height:1.6;
        margin-bottom:20px;
    ">
    <b>Exploratory Data Analysis (EDA)</b><br><br>
    Provides <b>high-level insights</b> to understand data behavior before model engineering.<br><br>
    <b>Key Insights Generated:</b>
    <ul>
        <li>Delivery performance and delay analysis across routes and vehicles</li>
        <li>Fuel consumption, efficiency, and cost trends</li>
        <li>Truck capacity utilization and load optimization</li>
        <li>Driver behavior and idle time monitoring</li>
        <li>Route-level distance, travel time, and stop analysis</li>
        <li>Warehouse dispatching and logistics cost breakdown</li>
        <li>Temperature and humidity threshold breach tracking</li>
        <li>Store-wise and region-wise delivery performance comparison</li>
    </ul>
    This section focuses on <b>interpretability</b>, not deep statistical modeling.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# COLUMN MAPPING
# ============================================================

def map_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_date         = map_col(["date", "Dispatch_Time", "Delivery_Time"])
col_order        = map_col(["Order_ID"])
col_delivery     = map_col(["Delivery_Order_ID"])
col_store        = map_col(["Store_ID"])
col_warehouse    = map_col(["Warehouse_ID"])
col_sku          = map_col(["SKU_ID"])
col_qty          = map_col(["Quantity"])
col_weight       = map_col(["Shipment_Weight", "Loaded_Weight"])
col_volume       = map_col(["Shipment_Volume", "Loaded_Volume"])
col_priority     = map_col(["Priority_Level"])
col_delay        = map_col(["Delay_Minutes", "Delivery_Delay_Minutes"])
col_order_value  = map_col(["Order_Value"])
col_route        = map_col(["Route_ID"])
col_vehicle      = map_col(["Vehicle_ID"])
col_driver       = map_col(["Driver_ID"])
col_fuel_cost    = map_col(["Fuel_Cost"])
col_fuel_used    = map_col(["Fuel_Consumed"])
col_fuel_eff     = map_col(["Fuel_Efficiency"])
col_distance     = map_col(["Distance_Traveled", "Actual_Distance", "distance_km"])
col_status       = map_col(["Delivery_Status_[0]", "Route_Status"])
col_cap_util     = map_col(["Capacity_Utilization_Percentage"])
col_idle         = map_col(["Idle_Time", "Idle_Time_[0]"])
col_total_cost   = map_col(["Total_Logistics_Cost"])
col_labor_cost   = map_col(["Labor_Cost"])
col_temp         = map_col(["Temperature"])
col_humidity     = map_col(["Humidity"])
col_breach       = map_col(["Threshold_Breach_Flag"])
col_region       = map_col(["region"])
col_city         = map_col(["city"])
col_vehicle_type = map_col(["vehicle_type"])
col_fuel_type    = map_col(["fuel_type"])
col_stop_type    = map_col(["stop_type"])
col_travel_time  = map_col(["Actual_Travel_Time", "avg_travel_time_hrs"])
col_est_time     = map_col(["Estimated_Travel_Time"])
col_stops        = map_col(["Total_Delivery_Stops"])
col_failed_stops = map_col(["Failed_Stops"])
col_speed        = map_col(["Speed"])
col_is_holiday   = map_col(["is_holiday"])
col_is_weekend   = map_col(["is_weekend"])
col_weekday      = map_col(["weekday_name"])
col_month        = map_col(["month"])

num_df = df.select_dtypes(include=np.number)

# ============================================================
# EDA NAVIGATION
# ============================================================

st.markdown("<h3 style='color:black;'>List of Analytics</h3>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

if "eda_option" not in st.session_state:
    st.session_state.eda_option = None


def nav_button(label, value):
    if st.session_state.eda_option == value:
        st.markdown(
            f"""
            <div style="
                background-color:#4F97EE;
                color:white;
                padding:14px;
                border-radius:10px;
                font-weight:600;
                text-align:center;
                margin-bottom:12px;
            ">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.button(label, use_container_width=True):
            st.session_state.eda_option = value
            st.rerun()


with st.expander(" ", expanded=True):
    row1 = st.columns(5)
    row2 = st.columns(4)

    with row1[0]:
        nav_button("Data Quality Overview", "Data Quality Overview")
    with row1[1]:
        nav_button("Delivery Performance", "Delivery Performance")
    with row1[2]:
        nav_button("Fuel & Cost Analysis", "Fuel & Cost Analysis")
    with row1[3]:
        nav_button("Capacity & Load Analysis", "Capacity & Load Analysis")
    with row1[4]:
        nav_button("Route & Stop Analysis", "Route & Stop Analysis")

    with row2[0]:
        nav_button("Driver & Vehicle Analysis", "Driver & Vehicle Analysis")
    with row2[1]:
        nav_button("Temperature & Breach Analysis", "Temperature & Breach Analysis")
    with row2[2]:
        nav_button("Store & Region Analysis", "Store & Region Analysis")
    with row2[3]:
        nav_button("Summary Report", "Summary Report")


eda_option = st.session_state.eda_option
if eda_option is not None:
    st.session_state.eda_completed = True

st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

if eda_option is None:
    st.info("Select an analysis to view insights.")

# ============================================================
# EDA ROUTER
# ============================================================

# ============================================================
# 1. DATA QUALITY OVERVIEW
# ============================================================
if eda_option == "Data Quality Overview":

    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:28px;
            border-radius:12px;
            color:white;
            font-size:16px;
            line-height:1.6;
            margin-bottom:20px;
        ">
        <b>What this section does:</b><br><br>
        Provides a <b>high-level health check</b> of the logistics dataset before any modeling or optimization is attempted.<br><br>
        It evaluates:
        <ul>
            <li>Missing values across all logistics fields</li>
            <li>Duplicate delivery records</li>
            <li>Data type consistency</li>
            <li>Overall row and column completeness</li>
        </ul>
        <b>Why this matters:</b><br>
        Route optimization and delivery forecasting models are highly sensitive to <b>poor data quality</b>.
        Missing weights, volumes, or timestamps can distort load planning and delay predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    rows_count   = df.shape[0]
    cols_count   = df.shape[1]
    dup_count    = df.duplicated().sum()
    dtype_counts = df.dtypes.value_counts()
    mv           = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)

    # Date range
    date_min = date_max = "N/A"
    if col_date and col_date in df.columns:
        try:
            parsed = pd.to_datetime(df[col_date], errors="coerce")
            if not parsed.isna().all():
                date_min = parsed.min().date()
                date_max = parsed.max().date()
        except Exception:
            pass

    import streamlit.components.v1 as components

    dup_impact = "⚠️ Risk of inflated delivery counts" if dup_count > 0 else "✅ Clean – no duplicates found"

    # ── shared CSS (injected once) ──────────────────────────────
    COMMON_CSS = """
    <style>
      body { margin:0; padding:0; font-family: sans-serif; background:transparent; }
      .box {
        overflow-y: auto;
        border: 2px solid #1F3A5F;
        border-radius: 10px;
        background-color: #F4F7FB;
        scrollbar-width: thin;
        scrollbar-color: #2F75B5 #dce6f0;
      }
      .box::-webkit-scrollbar { width: 6px; }
      .box::-webkit-scrollbar-track { background: #dce6f0; border-radius: 10px; }
      .box::-webkit-scrollbar-thumb { background-color: #2F75B5; border-radius: 10px; }
      .box-title {
        background-color: #1F3A5F;
        color: #fff;
        font-size: 13px;
        font-weight: 700;
        padding: 9px 16px;
        position: sticky;
        top: 0;
        z-index: 2;
        letter-spacing: 0.4px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      th {
        background-color: #2F75B5;
        color: #fff;
        padding: 7px 16px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #1F3A5F;
      }
      td {
        padding: 6px 16px;
        color: #1a1a2e;
        border-bottom: 1px solid #dce6f0;
      }
      tr:nth-child(even) td { background-color: #eaf1fb; }
      tr:hover td { background-color: #d0e4f7; }
    </style>
    """

    # ── BOX 1 : Dataset Shape ────────────────────────────────────
    components.html(COMMON_CSS + """
    <div class="box" style="height:160px;">
      <div class="box-title">📐 Dataset Shape</div>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Delivery Records</td><td>{rc}</td></tr>
        <tr><td>Total Features</td><td>{cc}</td></tr>
        <tr><td>Date Range</td><td>{dmin} → {dmax}</td></tr>
      </table>
    </div>
    """.format(rc=f"{rows_count:,}", cc=cols_count, dmin=date_min, dmax=date_max),
    height=168, scrolling=False)

    # ── BOX 2 : Missing Value Analysis ──────────────────────────
    mv_rows = "".join([
        "<tr><td>{}</td><td>{}{:.2f}%</td></tr>".format(
            c, " " if v > 0 else " ", v
        )
        for c, v in mv.items()
    ])
    components.html(COMMON_CSS + """
    <div class="box" style="height:220px;">
      <div class="box-title">🔍 Missing Value Analysis (%)</div>
      <table>
        <tr><th>Column Name</th><th>Missing (%)</th></tr>
        {mv}
      </table>
    </div>
    """.format(mv=mv_rows),
    height=228, scrolling=False)

    # ── BOX 3 : Duplicate Analysis ──────────────────────────────
    components.html(COMMON_CSS + """
    <div class="box" style="height:120px;">
      <div class="box-title">♻️ Duplicate Analysis</div>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Duplicate Rows</td><td>{dc}</td></tr>
        <tr><td>Impact</td><td>{di}</td></tr>
      </table>
    </div>
    """.format(dc=f"{dup_count:,}", di=dup_impact),
    height=128, scrolling=False)

    # ── BOX 4 : Data Types Summary ───────────────────────────────
    dtype_rows = "".join([
        "<tr><td>{}</td><td>{}</td></tr>".format(d, c)
        for d, c in dtype_counts.items()
    ])
    components.html(COMMON_CSS + """
    <div class="box" style="height:160px;">
      <div class="box-title">🗂️ Data Types Summary</div>
      <table>
        <tr><th>Data Type</th><th>Column Count</th></tr>
        {dt}
      </table>
    </div>
    """.format(dt=dtype_rows),
    height=168, scrolling=False)

# ============================================================
# 2. DELIVERY PERFORMANCE
# ============================================================
elif eda_option == "Delivery Performance":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Delivery Performance Analysis</b><br><br>
        Evaluates on-time vs delayed deliveries, average delay minutes, priority-level distribution,
        and failed stop rates to identify bottlenecks in last-mile logistics.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Delivery status breakdown
    if col_status and col_status in df.columns:
        status_counts = df[col_status].value_counts()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Delivery Status Breakdown</div>
                <table class="clean-table">
                    <tr><th>Status</th><th>Count</th></tr>
                    {''.join([f"<tr><td>{s}</td><td>{c:,}</td></tr>" for s, c in status_counts.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Delay stats
    if col_delay and col_delay in df.columns:
        delay_series = pd.to_numeric(df[col_delay], errors="coerce").dropna()
        avg_delay  = round(delay_series.mean(), 2)
        max_delay  = round(delay_series.max(), 2)
        delayed_pct = round((delay_series > 0).mean() * 100, 2)
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Delay Minutes Summary</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Delay (mins)</td><td>{avg_delay}</td></tr>
                    <tr><td>Maximum Delay (mins)</td><td>{max_delay}</td></tr>
                    <tr><td>% Delayed Deliveries</td><td>{delayed_pct}%</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Delay distribution chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(delay_series[delay_series > 0], bins=30, color="#2F75B5", edgecolor="white")
        ax.set_title("Distribution of Delay Minutes (Delayed Only)", color="black")
        ax.set_xlabel("Delay Minutes")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Priority level distribution
    if col_priority and col_priority in df.columns:
        priority_counts = df[col_priority].value_counts()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Priority Level Distribution</div>
                <table class="clean-table">
                    <tr><th>Priority</th><th>Orders</th></tr>
                    {''.join([f"<tr><td>{p}</td><td>{c:,}</td></tr>" for p, c in priority_counts.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Failed stops
    if col_failed_stops and col_failed_stops in df.columns:
        failed = pd.to_numeric(df[col_failed_stops], errors="coerce")
        total_failed = int(failed.sum())
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Failed Stops Summary</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Failed Stops</td><td>{total_failed:,}</td></tr>
                    <tr><td>Avg Failed Stops per Route</td><td>{round(failed.mean(), 2)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 3. FUEL & COST ANALYSIS
# ============================================================
elif eda_option == "Fuel & Cost Analysis":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Fuel & Cost Analysis</b><br><br>
        Breaks down fuel consumption, efficiency, and total logistics costs including labor,
        maintenance, tolls, and parking — helping identify high-cost routes and vehicles.
        </div>
        """,
        unsafe_allow_html=True
    )

    cost_cols = {
        "Fuel Cost":        map_col(["Fuel_Cost"]),
        "Labor Cost":       map_col(["Labor_Cost"]),
        "Maintenance Cost": map_col(["Maintenance_Cost"]),
        "Toll Cost":        map_col(["Toll_Cost"]),
        "Parking Cost":     map_col(["Parking_Cost"]),
        "Total Logistics Cost": map_col(["Total_Logistics_Cost"]),
    }

    rows = ""
    for label, col in cost_cols.items():
        if col and col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            rows += f"<tr><td>{label}</td><td>{round(series.sum(), 2):,}</td><td>{round(series.mean(), 2)}</td></tr>"

    st.markdown(
        f"""
        <div class="quality-card">
            <div class="quality-title">Logistics Cost Summary</div>
            <table class="clean-table">
                <tr><th>Cost Type</th><th>Total</th><th>Average per Record</th></tr>
                {rows}
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    if col_fuel_used and col_fuel_used in df.columns:
        fuel_series = pd.to_numeric(df[col_fuel_used], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Fuel Consumption Summary</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Fuel Consumed</td><td>{round(fuel_series.sum(), 2):,}</td></tr>
                    <tr><td>Average per Trip</td><td>{round(fuel_series.mean(), 2)}</td></tr>
                    <tr><td>Max Single Trip</td><td>{round(fuel_series.max(), 2)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_fuel_eff and col_fuel_eff in df.columns:
        eff_series = pd.to_numeric(df[col_fuel_eff], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(eff_series, bins=30, color="#00D05E", edgecolor="white")
        ax.set_title("Fuel Efficiency Distribution", color="black")
        ax.set_xlabel("Fuel Efficiency")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    if col_fuel_type and col_fuel_type in df.columns:
        ft_counts = df[col_fuel_type].value_counts()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Fleet Fuel Type Breakdown</div>
                <table class="clean-table">
                    <tr><th>Fuel Type</th><th>Vehicle Count</th></tr>
                    {''.join([f"<tr><td>{f}</td><td>{c:,}</td></tr>" for f, c in ft_counts.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 4. CAPACITY & LOAD ANALYSIS
# ============================================================
elif eda_option == "Capacity & Load Analysis":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Capacity & Load Analysis</b><br><br>
        Monitors truck weight/volume utilization, identifies under-loaded and over-loaded vehicles,
        and supports smarter load consolidation to reduce trips and costs.
        </div>
        """,
        unsafe_allow_html=True
    )

    if col_cap_util and col_cap_util in df.columns:
        cap_series = pd.to_numeric(df[col_cap_util], errors="coerce").dropna()
        under = (cap_series < 50).sum()
        optimal = ((cap_series >= 50) & (cap_series <= 90)).sum()
        over = (cap_series > 90).sum()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Capacity Utilization Summary (%)</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Utilization</td><td>{round(cap_series.mean(), 2)}%</td></tr>
                    <tr><td>Under-loaded (&lt;50%)</td><td>{under:,} records</td></tr>
                    <tr><td>Optimal (50–90%)</td><td>{optimal:,} records</td></tr>
                    <tr><td>Over-loaded (&gt;90%)</td><td>{over:,} records</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(cap_series, bins=30, color="#2F75B5", edgecolor="white")
        ax.axvline(50, color="orange", linestyle="--", label="50% threshold")
        ax.axvline(90, color="red", linestyle="--", label="90% threshold")
        ax.set_title("Capacity Utilization Distribution", color="black")
        ax.set_xlabel("Utilization %")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    for label, col in [("Shipment Weight (kg)", col_weight), ("Shipment Volume (m³)", col_volume)]:
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            st.markdown(
                f"""
                <div class="quality-card">
                    <div class="quality-title">{label}</div>
                    <table class="clean-table">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total</td><td>{round(s.sum(), 2):,}</td></tr>
                        <tr><td>Average per Shipment</td><td>{round(s.mean(), 2)}</td></tr>
                        <tr><td>Max</td><td>{round(s.max(), 2)}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# 5. ROUTE & STOP ANALYSIS
# ============================================================
elif eda_option == "Route & Stop Analysis":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Route & Stop Analysis</b><br><br>
        Evaluates estimated vs actual travel times, distance per route, stop counts,
        and stop type distributions to optimize future route planning.
        </div>
        """,
        unsafe_allow_html=True
    )

    if col_distance and col_distance in df.columns:
        dist = pd.to_numeric(df[col_distance], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Distance Traveled Summary (km)</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Distance</td><td>{round(dist.sum(), 2):,}</td></tr>
                    <tr><td>Average per Route</td><td>{round(dist.mean(), 2)}</td></tr>
                    <tr><td>Max Route Distance</td><td>{round(dist.max(), 2)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_travel_time and col_est_time and col_travel_time in df.columns and col_est_time in df.columns:
        actual = pd.to_numeric(df[col_travel_time], errors="coerce").dropna()
        est    = pd.to_numeric(df[col_est_time], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Travel Time: Estimated vs Actual (hrs)</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Estimated</th><th>Actual</th></tr>
                    <tr><td>Average</td><td>{round(est.mean(), 2)}</td><td>{round(actual.mean(), 2)}</td></tr>
                    <tr><td>Max</td><td>{round(est.max(), 2)}</td><td>{round(actual.max(), 2)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_stops and col_stops in df.columns:
        stops = pd.to_numeric(df[col_stops], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Delivery Stops Summary</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Stops per Route</td><td>{round(stops.mean(), 2)}</td></tr>
                    <tr><td>Max Stops</td><td>{int(stops.max())}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_stop_type and col_stop_type in df.columns:
        stype = df[col_stop_type].value_counts()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Stop Type Distribution</div>
                <table class="clean-table">
                    <tr><th>Stop Type</th><th>Count</th></tr>
                    {''.join([f"<tr><td>{s}</td><td>{c:,}</td></tr>" for s, c in stype.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 6. DRIVER & VEHICLE ANALYSIS
# ============================================================
elif eda_option == "Driver & Vehicle Analysis":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Driver & Vehicle Analysis</b><br><br>
        Monitors idle time, speed behavior, vehicle type distribution, and driver activity
        to improve fleet utilization and reduce operational inefficiencies.
        </div>
        """,
        unsafe_allow_html=True
    )

    if col_vehicle_type and col_vehicle_type in df.columns:
        vt = df[col_vehicle_type].value_counts()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Vehicle Type Distribution</div>
                <table class="clean-table">
                    <tr><th>Vehicle Type</th><th>Count</th></tr>
                    {''.join([f"<tr><td>{v}</td><td>{c:,}</td></tr>" for v, c in vt.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_idle and col_idle in df.columns:
        idle = pd.to_numeric(df[col_idle], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Idle Time Summary (mins)</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Idle Time</td><td>{round(idle.mean(), 2)}</td></tr>
                    <tr><td>Max Idle Time</td><td>{round(idle.max(), 2)}</td></tr>
                    <tr><td>Total Idle Time</td><td>{round(idle.sum(), 2):,}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_speed and col_speed in df.columns:
        speed = pd.to_numeric(df[col_speed], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(speed, bins=30, color="#0B2C5D", edgecolor="white")
        ax.set_title("Vehicle Speed Distribution", color="black")
        ax.set_xlabel("Speed")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# 7. TEMPERATURE & BREACH ANALYSIS
# ============================================================
elif eda_option == "Temperature & Breach Analysis":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Temperature & Breach Analysis</b><br><br>
        Tracks cold-chain compliance by monitoring temperature and humidity sensor readings,
        threshold breach frequency, and breach duration — critical for perishable goods.
        </div>
        """,
        unsafe_allow_html=True
    )

    if col_temp and col_temp in df.columns:
        temp = pd.to_numeric(df[col_temp], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Temperature Summary (°C)</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Temperature</td><td>{round(temp.mean(), 2)}</td></tr>
                    <tr><td>Min Temperature</td><td>{round(temp.min(), 2)}</td></tr>
                    <tr><td>Max Temperature</td><td>{round(temp.max(), 2)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(temp, bins=30, color="#2F75B5", edgecolor="white")
        ax.set_title("Temperature Distribution", color="black")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    if col_humidity and col_humidity in df.columns:
        hum = pd.to_numeric(df[col_humidity], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Humidity Summary (%)</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average Humidity</td><td>{round(hum.mean(), 2)}%</td></tr>
                    <tr><td>Min</td><td>{round(hum.min(), 2)}%</td></tr>
                    <tr><td>Max</td><td>{round(hum.max(), 2)}%</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_breach and col_breach in df.columns:
        breach = pd.to_numeric(df[col_breach], errors="coerce").dropna()
        breach_count = int(breach.sum())
        total        = len(breach)
        breach_pct   = round(breach_count / total * 100, 2) if total > 0 else 0
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Threshold Breach Summary</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Breaches</td><td>{breach_count:,}</td></tr>
                    <tr><td>Breach Rate</td><td>{breach_pct}%</td></tr>
                    <tr><td>No Breach Records</td><td>{total - breach_count:,}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 8. STORE & REGION ANALYSIS
# ============================================================
elif eda_option == "Store & Region Analysis":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Store & Region Analysis</b><br><br>
        Compares delivery volumes, order values, and delay patterns across stores, cities,
        regions, and zones to identify high-demand and underserved areas.
        </div>
        """,
        unsafe_allow_html=True
    )

    if col_store and col_store in df.columns:
        top_stores = df[col_store].value_counts().head(10)
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Top 10 Stores by Delivery Volume</div>
                <table class="clean-table">
                    <tr><th>Store ID</th><th>Deliveries</th></tr>
                    {''.join([f"<tr><td>{s}</td><td>{c:,}</td></tr>" for s, c in top_stores.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_region and col_region in df.columns:
        region_counts = df[col_region].value_counts()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Deliveries by Region</div>
                <table class="clean-table">
                    <tr><th>Region</th><th>Count</th></tr>
                    {''.join([f"<tr><td>{r}</td><td>{c:,}</td></tr>" for r, c in region_counts.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_city and col_city in df.columns:
        top_cities = df[col_city].value_counts().head(10)
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Top 10 Cities by Delivery Volume</div>
                <table class="clean-table">
                    <tr><th>City</th><th>Deliveries</th></tr>
                    {''.join([f"<tr><td>{c}</td><td>{v:,}</td></tr>" for c, v in top_cities.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    if col_order_value and col_order_value in df.columns:
        ov = pd.to_numeric(df[col_order_value], errors="coerce").dropna()
        st.markdown(
            f"""
            <div class="quality-card">
                <div class="quality-title">Order Value Summary</div>
                <table class="clean-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Order Value</td><td>{round(ov.sum(), 2):,}</td></tr>
                    <tr><td>Average Order Value</td><td>{round(ov.mean(), 2)}</td></tr>
                    <tr><td>Max Order Value</td><td>{round(ov.max(), 2)}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 9. SUMMARY REPORT
# ============================================================
elif eda_option == "Summary Report":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Summary Report</b><br><br>
        A consolidated snapshot of key logistics KPIs across all analysis dimensions —
        ready for management reporting or downstream model input.
        </div>
        """,
        unsafe_allow_html=True
    )

    summary_rows = []

    if col_delivery and col_delivery in df.columns:
        summary_rows.append(("Total Delivery Records", f"{df[col_delivery].nunique():,}"))
    if col_order and col_order in df.columns:
        summary_rows.append(("Unique Orders", f"{df[col_order].nunique():,}"))
    if col_store and col_store in df.columns:
        summary_rows.append(("Unique Stores", f"{df[col_store].nunique():,}"))
    if col_vehicle and col_vehicle in df.columns:
        summary_rows.append(("Unique Vehicles", f"{df[col_vehicle].nunique():,}"))
    if col_driver and col_driver in df.columns:
        summary_rows.append(("Unique Drivers", f"{df[col_driver].nunique():,}"))
    if col_route and col_route in df.columns:
        summary_rows.append(("Unique Routes", f"{df[col_route].nunique():,}"))
    if col_delay and col_delay in df.columns:
        delay_s = pd.to_numeric(df[col_delay], errors="coerce").dropna()
        summary_rows.append(("Avg Delay (mins)", f"{round(delay_s.mean(), 2)}"))
        summary_rows.append(("% Delayed", f"{round((delay_s > 0).mean() * 100, 2)}%"))
    if col_total_cost and col_total_cost in df.columns:
        tc = pd.to_numeric(df[col_total_cost], errors="coerce").dropna()
        summary_rows.append(("Total Logistics Cost", f"{round(tc.sum(), 2):,}"))
    if col_cap_util and col_cap_util in df.columns:
        cu = pd.to_numeric(df[col_cap_util], errors="coerce").dropna()
        summary_rows.append(("Avg Capacity Utilization", f"{round(cu.mean(), 2)}%"))
    if col_breach and col_breach in df.columns:
        b = pd.to_numeric(df[col_breach], errors="coerce").dropna()
        summary_rows.append(("Total Temp Breaches", f"{int(b.sum()):,}"))

    st.markdown(
        f"""
        <div class="quality-card">
            <div class="quality-title">Logistics KPI Summary</div>
            <table class="clean-table">
                <tr><th>KPI</th><th>Value</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary_rows])}
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )
    

############################################
                # Footer
############################################
st.markdown("""
    <br><br>
    <div style="background-color:#2E86C1;padding:12px;text-align:center;color:white;border-radius:6px;font-size:14px;">
        © 2026 SupplySyncAI – AI-Optimized Logistics & Routing
    </div>
""", unsafe_allow_html=True)