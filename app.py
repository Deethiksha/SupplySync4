import streamlit as st
from supabase import create_client
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils.html_table import render_html_table
from streamlit_option_menu import option_menu


# ====================== CHART THEME CONSTANTS ======================
NAVY        = '#0a2240'
NAVY2       = '#1a3a5c'
NAVY3       = '#0d2e4a'
RED         = '#e74c3c'
GREEN_BG    = '#2ecc71'
GREEN_DK    = '#27ae60'
BLUE_HDR    = '#3a6fa8'
WHITE70     = (1, 1, 1, 0.7)

# Pie/Doughnut color palettes
PIE_2       = ['#1a6b3c', '#f39c12', RED]
COST_PIE    = ['#e67e22', '#2980b9', '#8e44ad', RED]
REGION_BAR  = [NAVY, NAVY2, NAVY3, RED]


def apply_chart_style(ax, xlabel='', ylabel='', title=''):
    """Apply the green-background navy style to any matplotlib Axes."""
    ax.set_facecolor(GREEN_BG)
    ax.figure.patch.set_facecolor(GREEN_BG)
    ax.tick_params(colors='white', labelsize=9)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor((0, 0, 0, 0))          # transparent — use tuple, not CSS string
    ax.grid(True, color=(0, 0, 0, 0.15), linewidth=0.5)   # semi-transparent black grid
    ax.set_xlabel(xlabel, fontsize=9, color='white')
    ax.set_ylabel(ylabel, fontsize=9, color='white')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=8)


def section_header(title, sublabel=''):
    """Renders the blue header + darker-green sublabel band."""
    st.markdown(
        f"""
        <div style="background-color:{BLUE_HDR};padding:11px 18px;
             border-radius:7px 7px 0 0;color:#fff;font-size:14px;font-weight:600;">
            {title}
        </div>
        <div style="background-color:{GREEN_DK};padding:12px 16px 8px 16px;
             font-size:13px;font-weight:500;color:rgba(255,255,255,0.9);
             border-radius:0 0 0 0;">
            {sublabel}
        </div>
        <div style="background-color:{GREEN_BG};border-radius:0 0 7px 7px;padding:14px;">
        """,
        unsafe_allow_html=True
    )


def section_footer():
    st.markdown("</div>", unsafe_allow_html=True)


def metric_row(metrics: list):
    """
    metrics: list of dicts with keys: label, value, delta
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        col.markdown(
            f"""
            <div style="background:rgba(0,0,0,0.18);border-radius:5px;padding:10px 12px;margin-bottom:10px;">
                <div style="font-size:11px;color:rgba(255,255,255,0.75);margin-bottom:3px;">{m['label']}</div>
                <div style="font-size:20px;font-weight:600;color:#fff;line-height:1;">{m['value']}</div>
                <div style="font-size:10px;margin-top:3px;color:rgba(255,255,255,0.8);">{m['delta']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def chart_card(fig, title=''):
    """Wrap a matplotlib figure in a styled card inside the green section."""
    if title:
        st.markdown(
            f'<div style="font-size:12px;color:#fff;font-weight:600;margin-bottom:4px;">{title}</div>',
            unsafe_allow_html=True
        )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ====================== SUPABASE LOADER ======================
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


# ====================== STREAMLIT CONFIG ======================
st.set_page_config(page_title="SupplySyncAI", layout="wide")

st.markdown("""
<style>
div[data-testid="stAlert"] { color: #000000 !important; }
.stApp { background-color: #EDEDED; margin:0; padding:0; }
.block-container { padding-top:1rem !important; margin-top:0rem !important; }
.main { background-color:#f0f2f6 !important; }
section.main > div:first-child { padding-top:0rem !important; margin-top:0rem !important; }
[data-testid="stAppViewContainer"] { padding-top:3rem !important; margin-top:0rem !important; }
header[data-testid="stHeader"] { position:sticky; top:0; z-index:999;
    background-color:#EDEDED !important; height:auto !important; }
header[data-testid="stHeader"] * { color:#000000 !important; }
.block-container { padding-left:1rem !important; padding-right:1rem !important;
    max-width:100% !important; overflow-x:hidden !important; }
section.main > div { padding-left:0rem !important; padding-right:0rem !important;
    max-width:100% !important; overflow-x:hidden !important; }
[data-testid="stAppViewContainer"] { padding-left:0rem !important;
    padding-right:0rem !important; overflow-x:hidden !important; }

div.element-container:has(div.stRadio) { width:100% !important; }
div.stRadio > div { background-color:#00D05E; padding:16px 0px; border-radius:8px;
    width:100%; box-sizing:border-box; display:flex; justify-content:center; }
div[data-baseweb="radio-group"] { display:flex !important; justify-content:center !important;
    align-items:center; gap:50px; width:100%; margin:0 auto; }
div[data-baseweb="radio"] { display:flex; align-items:center; justify-content:center; }
div[data-baseweb="radio"] label, div[data-baseweb="radio"] label span {
    font-size:18px !important; font-weight:800 !important;
    color:#FFFFFF !important; white-space:nowrap; }
div[data-baseweb="radio"] { margin-right:28px; }

div[data-testid="stExpander"] { background-color:#2F75B5; border-radius:20px;
    border:1px solid #9EDAD0; overflow:hidden; }
div[data-testid="stExpander"]:nth-of-type(1) summary { display:none; }
div[data-testid="stExpander"]:nth-of-type(1) > div { padding:22px 18px; }

div.stButton > button { background-color:#0B2C5D; color:#FFFFFF; border-radius:8px;
    padding:8px 18px; border:none; font-weight:600; }
div.stButton > button:hover { background-color:#08306B; color:#FFFFFF; }

.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:14px; margin:6px 0 10px 0; justify-content:center; }
.summary-card { border:2px solid #6B7280; border-radius:2px;
    background-color:#F8FAFC; overflow:hidden; text-align:center; }
.summary-title { background-color:#1F3A5F; color:#ffffff; font-size:14px;
    font-weight:700; padding:8px 6px; border-bottom:1px solid #6B7280;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.summary-value { font-size:22px; font-weight:600; color:#000000; padding:1px 0; }

/* Legend dot helper */
.leg { display:inline-block; width:10px; height:10px; border-radius:2px;
    margin-right:4px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div style="background-color:#0B2C5D;padding:35px;border-radius:12px;color:white;
     text-align:center;margin:0 0 20px 0;">
    <h1 style="margin:0 0 8px 0;">SupplySync AI - AI-Optimized Logistics & Routing</h1>
    <p style="font-size:17px;margin-top:15px;">Real-time VRP/CVRP optimization • Truck utilization
    • Fuel & time prediction • Perishable priority</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#2F75B5;padding:32px;border-radius:12px;color:white;
     font-size:17px;line-height:1.8;text-align:justify;">
    This application is an AI-Optimized Logistics & Routing System (Stage 4).
    It takes your demand forecast data, automatically cleans and prepares the data,
    and then intelligently generates the most efficient delivery routes and load
    allocation for your trucks using advanced VRP/CVRP algorithms.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. DATA COLLECTION LAYER
# ============================================================
st.markdown("""
<div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;
     margin-top:20px;margin-bottom:10px;">
    <h3 style="margin:0;">1. Data Collection Layer</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#2F75B5;padding:20px;border-radius:12px;color:white;
     font-size:16px;line-height:1.6;margin-bottom:20px;">
Click below to load data directly from <b>final_data</b> table.
</div>
""", unsafe_allow_html=True)

if st.button("LOAD DATA", use_container_width=False):
    import time
    start_time = time.time()
    try:
        with st.spinner("Loading data from Supabase..."):
            df = load_data_from_supabase()
            st.session_state.df = df
            load_time = round(time.time() - start_time, 2)
            st.markdown(f"""
            <div style="background-color:#2F75B5;padding:18px 25px;border-radius:12px;
                 color:white;margin-top:10px;text-align:left;">
                <b>Data loaded successfully from final_data table!</b><br>
                Loaded <b>{df.shape[0]}</b> rows × <b>{df.shape[1]}</b> columns<br>
                Time taken: <b>{load_time} seconds</b>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load data: {e}")

if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("""
    <div style="background-color:#0B2C5D;padding:15px 25px;border-radius:10px;
         color:white;margin-top:30px;">
        <h4 style="margin:0;">Data Preview</h4>
    </div>
    """, unsafe_allow_html=True)
    render_html_table(df.head(20), max_height=280)
    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ====================== 2. DATA PRE-PROCESSING ======================
    st.markdown("""
    <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;
         color:white;margin-top:40px;margin-bottom:12px;">
        <h3 style="margin:0;">2. Data Pre-Processing</h3>
    </div>
    """, unsafe_allow_html=True)

    step = st.radio("Select Pre-Processing Step",
                    ["Remove Duplicate Rows", "Remove Outliers", "Replace Missing Values"],
                    horizontal=True, label_visibility="collapsed",
                    index=None, key="preprocess_step")

    if step == "Remove Duplicate Rows":
        if st.button("Apply Duplicate Row Removal"):
            before = len(df)
            df_clean = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df_clean)
            st.session_state.df = df_clean
            st.markdown("#### Summary of Changes")
            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;margin:15px 0;">
                <tr style="background-color:#1F3A5F;color:white;">
                    <th style="padding:12px;text-align:center;border:1px solid #ddd;">Rows Before</th>
                    <th style="padding:12px;text-align:center;border:1px solid #ddd;">Rows After</th>
                    <th style="padding:12px;text-align:center;border:1px solid #ddd;">Duplicates Removed</th>
                </tr>
                <tr style="background-color:#F8FAFC;">
                    <td style="padding:15px;text-align:center;font-size:18px;font-weight:600;border:1px solid #ddd;">{before}</td>
                    <td style="padding:15px;text-align:center;font-size:18px;font-weight:600;border:1px solid #ddd;">{len(df_clean)}</td>
                    <td style="padding:15px;text-align:center;font-size:18px;font-weight:600;color:#FF4B4B;border:1px solid #ddd;">{removed}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
            st.success("Duplicates removed successfully!")

    elif step == "Remove Outliers":
        if st.button("Apply Outlier Removal"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            st.session_state.df = df
            st.success("Outliers handled successfully!")

    elif step == "Replace Missing Values":
        if st.button("Apply NULL Replacement"):
            df = df.fillna("Unknown")
            st.session_state.df = df
            st.success("Missing values replaced with 'Unknown'")

    st.markdown("#### Processed Data Preview")
    render_html_table(st.session_state.df.head(15), max_height=300)

else:
    st.info("Click **LOAD DATA** to start.")

# ============================================================
# STEP 3 – EDA (LOCKED UNTIL PREPROCESSING)
# ============================================================

# --- SESSION STATE INITIALIZATION ---
if "preprocessing_completed" not in st.session_state:
    st.session_state.preprocessing_completed = False

if "eda_option" not in st.session_state:
    st.session_state.eda_option = None

if "df" not in st.session_state:
    st.session_state.df = None

# Mark preprocessing as completed if user has interacted
if "df" in st.session_state and st.session_state.df is not None:
    st.session_state.preprocessing_completed = True

df = st.session_state.get("df", None)

if df is None:
    st.warning("⚠ No dataset available. Please load and preprocess data first.")
    st.stop()

# ---------------- EDA HEADER ----------------
st.markdown(
    """
    <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;margin-top:20px;margin-bottom:10px;">
        <h3 style="margin:0;">Exploratory Data Analysis (EDA)</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(f"Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

# ---------------- EDA INTRO CARD ----------------
st.markdown(
    """
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Exploratory Data Analysis (EDA)</b><br><br>
        Provides <b>high-level insights</b> to understand logistics data behavior before model engineering.<br><br>
        <b>Key Insights Generated:</b>
        <ul>
            <li>Delivery performance and on-time rate trends</li>
            <li>Truck utilization and load efficiency patterns</li>
            <li>Route cost and distance distribution</li>
            <li>Store replenishment frequency and delay analysis</li>
            <li>Fuel consumption patterns across routes</li>
            <li>Perishable goods spoilage risk by delivery time</li>
        </ul>
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

col_delivery_id     = map_col(["delivery_id", "shipment_id", "order_id"])
col_route_id        = map_col(["route_id", "route"])
col_truck_id        = map_col(["truck_id", "vehicle_id"])
col_store_id        = map_col(["store_id", "destination_store"])
col_driver_id       = map_col(["driver_id"])
col_date            = map_col(["delivery_date", "shipment_date", "date", "created_at"])
col_distance        = map_col(["distance_km", "total_distance", "route_distance"])
col_fuel            = map_col(["fuel_cost", "fuel_consumed", "fuel_consumption"])
col_transport_cost  = map_col(["transport_cost", "delivery_cost", "total_cost"])
col_load            = map_col(["load_weight", "cargo_weight", "load_kg"])
col_capacity        = map_col(["truck_capacity", "max_capacity", "vehicle_capacity"])
col_utilization     = map_col(["truck_utilization", "load_utilization", "utilization_pct"])
col_delivery_time   = map_col(["delivery_time_hrs", "delivery_duration", "time_taken"])
col_on_time         = map_col(["on_time_delivery", "is_on_time", "on_time"])
col_delay           = map_col(["delay_minutes", "delay_hours", "delay"])
col_perishable      = map_col(["is_perishable", "perishable_flag", "perishable"])
col_spoilage        = map_col(["spoilage_flag", "spoilage", "goods_spoiled"])
col_stops           = map_col(["num_stops", "delivery_stops", "stops"])
col_replenishment   = map_col(["replenishment_frequency", "replenishment_days"])
col_priority        = map_col(["priority", "delivery_priority"])
col_region          = map_col(["region", "zone", "area"])

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
            f"""<div style="background-color:#4F97EE;color:white;padding:14px;
                border-radius:10px;font-weight:600;text-align:center;
                margin-bottom:12px;">{label}</div>""",
            unsafe_allow_html=True
        )
    else:
        if st.button(label, use_container_width=True):
            st.session_state.eda_option = value
            st.rerun()


with st.expander(" ", expanded=True):
    row1 = st.columns(4)
    row2 = st.columns(4)
    with row1[0]: nav_button("Data Quality Overview",       "Data Quality Overview")
    with row1[1]: nav_button("Delivery Performance",        "Delivery Performance")
    with row1[2]: nav_button("Truck & Load Analysis",        "Truck & Load Analysis")
    with row1[3]: nav_button("Route Cost Analysis",          "Route Cost Analysis")
    with row2[0]: nav_button("Fuel Consumption Analysis",       "Fuel Consumption Analysis")
    with row2[1]: nav_button("Store Replenishment Analysis",   "Store Replenishment Analysis")
    with row2[2]: nav_button("Perishable & Spoilage Risk",     "Perishable & Spoilage Risk")
    with row2[3]: nav_button("Summary Report",              "Summary Report")

eda_option = st.session_state.eda_option
if eda_option is not None:
    st.session_state.eda_completed = True

st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

if eda_option is None:
    st.info("Select an analysis to view insights.")



# ============================================================
# EDA ROUTER
# ============================================================

# ----------------------------------------------------------------
# 1. DATA QUALITY OVERVIEW
# ----------------------------------------------------------------
if eda_option == "Data Quality Overview":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Data Quality Overview</b><br><br>
        Provides a high-level health check of the logistics dataset before modeling.
        </div>
        """,
        unsafe_allow_html=True
    )

    rows_count = df.shape[0]
    cols_count = df.shape[1]

    # Dataset Shape
    st.markdown(
        f"""
        <div style="background-color:#0B2C5D;padding:20px;border-radius:10px;color:white;margin-bottom:20px;">
            <h4>Dataset Overview</h4>
            <strong>Total Rows:</strong> {rows_count:,} &nbsp;&nbsp;&nbsp;&nbsp;
            <strong>Total Columns:</strong> {cols_count}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Missing Values Table
    mv = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)
    mv = mv[mv > 0]

    st.subheader("Missing Value Analysis (%)")
    if not mv.empty:
        st.markdown(
            f"""
            <table style="width:100%;border-collapse:collapse;margin:15px 0;">
                <tr style="background-color:#1F3A5F;color:white;">
                    <th style="padding:12px;text-align:left;border:1px solid #ddd;">Column Name</th>
                    <th style="padding:12px;text-align:center;border:1px solid #ddd;">Missing (%)</th>
                </tr>
                {''.join([f"<tr style='background-color:#F8FAFC;'><td style='padding:10px;border:1px solid #ddd;'>{c}</td><td style='padding:10px;text-align:center;border:1px solid #ddd;'>{v}%</td></tr>" for c, v in mv.items()])}
            </table>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success("✅ No missing values found in the dataset!")

    # Data Types Summary - Styled like Preprocessing
    st.subheader("Data Types Summary")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["Data Type", "Column Count"]

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;margin:15px 0;">
            <tr style="background-color:#1F3A5F;color:white;">
                <th style="padding:12px;text-align:left;border:1px solid #ddd;">Data Type</th>
                <th style="padding:12px;text-align:center;border:1px solid #ddd;">Column Count</th>
            </tr>
            {''.join([f"<tr style='background-color:#F8FAFC;'><td style='padding:10px;border:1px solid #ddd;'>{d}</td><td style='padding:10px;text-align:center;border:1px solid #ddd;'>{c}</td></tr>" for d, c in zip(dtype_counts["Data Type"], dtype_counts["Column Count"])])}
        </table>
        """,
        unsafe_allow_html=True
    )

    # Overall Quality
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_score = max(0, 100 - missing_pct)

    st.markdown(
        f"""
        <div style="background-color:#0B2C5D;padding:25px;border-radius:10px;color:white;text-align:center;margin-top:20px;">
            <h4>Overall Data Quality Score</h4>
            <h2 style="color:#00D05E;margin:10px 0;">{quality_score:.1f} / 100</h2>
            <p>The dataset is ready for route optimization modeling.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------
# 2. DELIVERY PERFORMANCE (Final Fixed Version)
# ----------------------------------------------------------------
elif eda_option == "Delivery Performance":

    # Local color definitions
    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"

    def blue_title(title):
        st.markdown(
            f"""
            <div style="background-color:#2F75B5;padding:14px;border-radius:8px;
                 font-size:16px;color:white;margin:15px 0 8px 0;
                 text-align:center;font-weight:600;">{title}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Delivery Performance Analysis</b><br><br>
        Evaluates on-time performance, delay patterns, and delivery efficiency.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== KPI METRICS ====================
    total_deliveries = len(df)

    # Exact column names from your dataset
    on_time_col = "Delivery_Status_[0]"
    delay_col = "Delay_Minutes"

    # Convert delay to numeric safely
    df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')

    # Calculate metrics
    on_time_rate = (df[on_time_col].astype(str).str.contains('On Time', case=False, na=False).mean() * 100)
    avg_delay = df[delay_col].mean() if not df[delay_col].isna().all() else 0
    avg_delivery_time = 0  # You can add Actual_Travel_Time later

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Deliveries</div>
                <div class="summary-value">{total_deliveries:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">On-Time Rate</div>
                <div class="summary-value">{on_time_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Delay</div>
                <div class="summary-value">{avg_delay:.1f} min</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Delivery Time</div>
                <div class="summary-value">N/A</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ==================== CHARTS ====================
    col1, col2 = st.columns(2)

    with col1:
        blue_title("On-Time Performance by Route (Top 15)")
        if "Route_ID" in df.columns:
            df['is_on_time'] = df[on_time_col].astype(str).str.contains('On Time', case=False, na=False).astype(int)
            route_otd = df.groupby("Route_ID")['is_on_time'].mean().sort_values(ascending=False).head(15) * 100

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.patch.set_facecolor(GREEN_BG)
            ax1.set_facecolor(GREEN_BG)
            ax1.bar(route_otd.index.astype(str), route_otd.values, color=BAR_BLUE)
            ax1.axhline(85, color='orange', linestyle='--', linewidth=2, label='85% Target')
            ax1.set_xlabel("Route ID")
            ax1.set_ylabel("On-Time Rate (%)")
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("Route_ID column not found")

    with col2:
        blue_title("Average Delay by Store (Top 15)")
        if "Store_ID" in df.columns:
            store_delay = df.groupby("Store_ID")[delay_col].mean().sort_values(ascending=False).head(15)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(GREEN_BG)
            ax2.set_facecolor(GREEN_BG)
            ax2.bar(store_delay.index.astype(str), store_delay.values, color='#EF4444')
            ax2.set_xlabel("Store ID")
            ax2.set_ylabel("Avg Delay (minutes)")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Store_ID column not found")

    # Additional Charts
    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Monthly Delivery Volume")
        if "Dispatch_Time" in df.columns:
            df_temp = df.copy()
            df_temp["Dispatch_Time"] = pd.to_datetime(df_temp["Dispatch_Time"], errors='coerce')
            monthly = df_temp.groupby(df_temp["Dispatch_Time"].dt.to_period('M')).size()

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)
            ax3.bar(monthly.index.astype(str), monthly.values, color=BAR_BLUE)
            ax3.set_xlabel("Month")
            ax3.set_ylabel("Number of Deliveries")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        blue_title("Delay Distribution")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        ax4.hist(df[delay_col].dropna(), bins=25, color=BAR_BLUE, edgecolor='white')
        ax4.axvline(df[delay_col].mean(), color='orange', linestyle='--', 
                   label=f'Mean: {df[delay_col].mean():.1f} min')
        ax4.set_xlabel("Delay (minutes)")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        st.pyplot(fig4)
        plt.close(fig4)


# ----------------------------------------------------------------
# 3. TRUCK & LOAD ANALYSIS (Matplotlib Style - Consistent with Delivery Performance)
# ----------------------------------------------------------------
elif eda_option == "Truck & Load Analysis":

    # Colors (consistent with your theme)
    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"

    def blue_title(title):
        st.markdown(
            f"""
            <div style="background-color:#2F75B5;padding:14px;border-radius:8px;
                 font-size:16px;color:white;margin:15px 0 8px 0;
                 text-align:center;font-weight:600;">{title}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Truck & Load Analysis</b><br><br>
        Analyzes truck utilization, load efficiency, and capacity usage — critical for 
        effective VRP/CVRP optimization and cost reduction.
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI Summary
    util_col = "Capacity_Utilization_Percentage" if "Capacity_Utilization_Percentage" in df.columns else None
    load_col = next((c for c in ["Loaded_Weight", "Shipment_Weight"] if c in df.columns), None)
    vehicle_col = "Vehicle_ID" if "Vehicle_ID" in df.columns else None

    avg_util = float(df[util_col].mean()) if util_col else 0
    avg_load = float(df[load_col].mean()) if load_col else 0
    total_trucks = df[vehicle_col].nunique() if vehicle_col else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Avg Truck Utilization</div>
                <div class="summary-value">{avg_util:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Load Weight</div>
                <div class="summary-value">{avg_load:,.0f} kg</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Total Trucks</div>
                <div class="summary-value">{total_trucks}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    # Graph 1: Truck Utilization Rate (with 70% target line)
    with col1:
        blue_title("Truck Utilization Rate by Vehicle (Top 15)")
        if vehicle_col and util_col:
            truck_util = df.groupby(vehicle_col)[util_col].mean().sort_values(ascending=False).head(15)

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.patch.set_facecolor(GREEN_BG)
            ax1.set_facecolor(GREEN_BG)
            
            colors = [BAR_BLUE if v >= 70 else "#EF4444" for v in truck_util.values]
            ax1.bar(truck_util.index.astype(str), truck_util.values, color=colors)
            
            ax1.axhline(70, color='orange', linestyle='--', linewidth=2, label='70% Target')
            ax1.set_xlabel("Vehicle ID")
            ax1.set_ylabel("Utilization Rate (%)")
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("Vehicle_ID or Utilization column not found")

    # Graph 2: Utilization Distribution
    with col2:
        blue_title("Utilization Rate Distribution Across Fleet")
        if util_col:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(GREEN_BG)
            ax2.set_facecolor(GREEN_BG)
            
            ax2.hist(df[util_col].dropna(), bins=20, color=BAR_BLUE, edgecolor='white')
            ax2.axvline(df[util_col].mean(), color='orange', linestyle='--', 
                       linewidth=2, label=f'Mean: {df[util_col].mean():.1f}%')
            ax2.axvline(70, color='#EF4444', linestyle='--', linewidth=2, label='70% Target')
            ax2.set_xlabel("Utilization Rate (%)")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            st.pyplot(fig2)
            plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    # Graph 3: Deliveries per Truck
    with col3:
        blue_title("Number of Deliveries per Truck (Top 15)")
        if vehicle_col:
            deliveries = df[vehicle_col].value_counts().head(15)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)
            ax3.bar(deliveries.index.astype(str), deliveries.values, color=BAR_BLUE)
            ax3.set_xlabel("Vehicle ID")
            ax3.set_ylabel("Number of Deliveries")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)
            plt.close(fig3)

    # Graph 4: Load Weight Analysis
    with col4:
        blue_title("Average Load Weight by Truck (Top 15)")
        if vehicle_col and load_col:
            load_data = df.groupby(vehicle_col)[load_col].mean().sort_values(ascending=False).head(15)

            fig4, ax4 = plt.subplots(figsize=(8, 5))
            fig4.patch.set_facecolor(GREEN_BG)
            ax4.set_facecolor(GREEN_BG)
            ax4.bar(load_data.index.astype(str), load_data.values, color="#3B6E8E")
            ax4.set_xlabel("Vehicle ID")
            ax4.set_ylabel("Average Load Weight (kg)")
            ax4.tick_params(axis='x', rotation=45)
            st.pyplot(fig4)
            plt.close(fig4)


# ----------------------------------------------------------------
# 4. ROUTE COST ANALYSIS 
# ----------------------------------------------------------------
elif eda_option == "Route Cost Analysis":

    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"

    def blue_title(title):
        st.markdown(
            f"""
            <div style="background-color:#2F75B5;padding:14px;border-radius:8px;
                 font-size:16px;color:white;margin:15px 0 8px 0;
                 text-align:center;font-weight:600;">{title}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Route Cost Analysis</b><br><br>
        Identifies expensive routes, cost per km, and cost efficiency patterns — 
        critical for route optimization and reducing logistics spend.
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI Metrics
    cost_col = next((c for c in ["Total_Logistics_Cost", "transport_cost", "Total_Cost"] if c in df.columns), None)
    distance_col = next((c for c in ["distance_km", "Actual_Distance", "Distance_Traveled"] if c in df.columns), None)
    route_col = "Route_ID" if "Route_ID" in df.columns else None

    total_cost = float(df[cost_col].sum()) if cost_col else 0
    avg_cost = float(df[cost_col].mean()) if cost_col else 0
    avg_distance = float(df[distance_col].mean()) if distance_col else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Transport Cost</div>
                <div class="summary-value">${total_cost:,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Cost per Delivery</div>
                <div class="summary-value">${avg_cost:,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Route Distance</div>
                <div class="summary-value">{avg_distance:.1f} km</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Total Cost by Route (Top 15)")
        if route_col and cost_col:
            route_cost = df.groupby(route_col)[cost_col].sum().sort_values(ascending=False).head(15)

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.patch.set_facecolor(GREEN_BG)
            ax1.set_facecolor(GREEN_BG)
            ax1.bar(route_cost.index.astype(str), route_cost.values, color=BAR_BLUE)
            ax1.set_xlabel("Route ID")
            ax1.set_ylabel("Total Cost ($)")
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
            plt.close(fig1)

    with col2:
        blue_title("Cost per KM by Route (Top 15)")
        if route_col and cost_col and distance_col:
            cpk = df.groupby(route_col).agg(
                total_cost=(cost_col, 'sum'),
                total_dist=(distance_col, 'sum')
            )
            cpk['cost_per_km'] = cpk['total_cost'] / (cpk['total_dist'] + 1e-6)
            cpk = cpk.sort_values('cost_per_km', ascending=False).head(15)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(GREEN_BG)
            ax2.set_facecolor(GREEN_BG)
            ax2.bar(cpk.index.astype(str), cpk['cost_per_km'], color='#E67E22')
            ax2.set_xlabel("Route ID")
            ax2.set_ylabel("Cost per KM ($/km)")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)
            plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Distance vs Transport Cost")
        if distance_col and cost_col:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)
            ax3.scatter(df[distance_col], df[cost_col], alpha=0.6, color=BAR_BLUE, edgecolors='white')
            ax3.set_xlabel("Distance (km)")
            ax3.set_ylabel("Transport Cost ($)")
            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        blue_title("Number of Stops vs Avg Cost")
        stops_col = next((c for c in ["Total_Delivery_Stops", "num_stops"] if c in df.columns), None)
        if stops_col and cost_col:
            stops_cost = df.groupby(stops_col)[cost_col].mean()

            fig4, ax4 = plt.subplots(figsize=(8, 5))
            fig4.patch.set_facecolor(GREEN_BG)
            ax4.set_facecolor(GREEN_BG)
            ax4.bar(stops_cost.index.astype(str), stops_cost.values, color=BAR_BLUE)
            ax4.set_xlabel("Number of Stops")
            ax4.set_ylabel("Average Cost ($)")
            st.pyplot(fig4)
            plt.close(fig4)


# ----------------------------------------------------------------
# 5. FUEL CONSUMPTION ANALYSIS (Highly Recommended)
# ----------------------------------------------------------------
elif eda_option == "Fuel Consumption Analysis":

    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"

    def blue_title(title):
        st.markdown(
            f"""
            <div style="background-color:#2F75B5;padding:14px;border-radius:8px;
                 font-size:16px;color:white;margin:15px 0 8px 0;
                 text-align:center;font-weight:600;">{title}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Fuel Consumption Analysis</b><br><br>
        Examines fuel usage patterns across routes, trucks, and load levels — 
        critical for reducing operational costs and improving fuel efficiency.
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI Metrics
    fuel_col = next((c for c in ["Fuel_Cost", "fuel_cost", "Fuel_Cost_[0]"] if c in df.columns), None)
    distance_col = next((c for c in ["distance_km", "Actual_Distance", "Distance_Traveled"] if c in df.columns), None)

    total_fuel = float(df[fuel_col].sum()) if fuel_col else 0
    avg_fuel = float(df[fuel_col].mean()) if fuel_col else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Fuel Cost</div>
                <div class="summary-value">${total_fuel:,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Fuel Cost per Delivery</div>
                <div class="summary-value">${avg_fuel:,.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Fuel Cost by Route (Top 15)")
        if "Route_ID" in df.columns and fuel_col:
            route_fuel = df.groupby("Route_ID")[fuel_col].sum().sort_values(ascending=False).head(15)

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.patch.set_facecolor(GREEN_BG)
            ax1.set_facecolor(GREEN_BG)
            ax1.bar(route_fuel.index.astype(str), route_fuel.values, color=BAR_BLUE)
            ax1.set_xlabel("Route ID")
            ax1.set_ylabel("Total Fuel Cost ($)")
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
            plt.close(fig1)

    with col2:
        blue_title("Fuel Cost by Truck (Top 15)")
        if "Vehicle_ID" in df.columns and fuel_col:
            truck_fuel = df.groupby("Vehicle_ID")[fuel_col].sum().sort_values(ascending=False).head(15)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(GREEN_BG)
            ax2.set_facecolor(GREEN_BG)
            ax2.bar(truck_fuel.index.astype(str), truck_fuel.values, color=BAR_BLUE)
            ax2.set_xlabel("Vehicle ID")
            ax2.set_ylabel("Total Fuel Cost ($)")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)
            plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Load Weight vs Fuel Cost")
        if "Loaded_Weight" in df.columns or "Shipment_Weight" in df.columns and fuel_col:
            load_col = "Loaded_Weight" if "Loaded_Weight" in df.columns else "Shipment_Weight"
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)
            ax3.scatter(df[load_col], df[fuel_col], alpha=0.6, color=BAR_BLUE, edgecolors='white')
            ax3.set_xlabel("Load Weight (kg)")
            ax3.set_ylabel("Fuel Cost ($)")
            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        blue_title("Fuel Cost Trend Over Time")
        date_col = "Dispatch_Time" if "Dispatch_Time" in df.columns else None
        if date_col and fuel_col:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            fuel_trend = df_temp.groupby(df_temp[date_col].dt.to_period('M'))[fuel_col].sum()

            fig4, ax4 = plt.subplots(figsize=(8, 5))
            fig4.patch.set_facecolor(GREEN_BG)
            ax4.set_facecolor(GREEN_BG)
            ax4.bar(fuel_trend.index.astype(str), fuel_trend.values, color=BAR_BLUE)
            ax4.set_xlabel("Month")
            ax4.set_ylabel("Total Fuel Cost ($)")
            ax4.tick_params(axis='x', rotation=45)
            st.pyplot(fig4)
            plt.close(fig4)


# ----------------------------------------------------------------
# 6. STORE REPLENISHMENT ANALYSIS
# ----------------------------------------------------------------
elif eda_option == "Store Replenishment Analysis":

    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"

    def blue_title(title):
        st.markdown(
            f"""
            <div style="background-color:#2F75B5;padding:14px;border-radius:8px;
                 font-size:16px;color:white;margin:15px 0 8px 0;
                 text-align:center;font-weight:600;">{title}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Store Replenishment Analysis</b><br><br>
        Analyzes how frequently stores receive deliveries and identifies stores with 
        chronic delays — crucial for improving stock availability and customer satisfaction.
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI Summary
    store_col = "Store_ID" if "Store_ID" in df.columns else None
    delay_col = "Delay_Minutes" if "Delay_Minutes" in df.columns else None

    total_stores = df[store_col].nunique() if store_col else 0
    avg_deliveries_per_store = df[store_col].value_counts().mean() if store_col else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Stores</div>
                <div class="summary-value">{total_stores}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Deliveries per Store</div>
                <div class="summary-value">{avg_deliveries_per_store:.1f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Replenishment Frequency by Store (Top 15)")
        if store_col:
            store_freq = df[store_col].value_counts().head(15)

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.patch.set_facecolor(GREEN_BG)
            ax1.set_facecolor(GREEN_BG)
            ax1.bar(store_freq.index.astype(str), store_freq.values, color=BAR_BLUE)
            ax1.set_xlabel("Store ID")
            ax1.set_ylabel("Number of Deliveries")
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
            plt.close(fig1)

    with col2:
        blue_title("Stores with Highest Average Delay")
        if store_col and delay_col:
            df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')
            store_delay = df.groupby(store_col)[delay_col].mean().sort_values(ascending=False).head(15)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(GREEN_BG)
            ax2.set_facecolor(GREEN_BG)
            ax2.bar(store_delay.index.astype(str), store_delay.values, color="#EF4444")
            ax2.set_xlabel("Store ID")
            ax2.set_ylabel("Avg Delay (minutes)")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)
            plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Delivery Volume vs Avg Delay by Store")
        if store_col and delay_col:
            store_metrics = df.groupby(store_col).agg(
                delivery_count=('Store_ID', 'count'),
                avg_delay=(delay_col, 'mean')
            ).sort_values('avg_delay', ascending=False).head(15)

            x = np.arange(len(store_metrics))
            width = 0.35

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)

            ax3.bar(x - width/2, store_metrics['delivery_count'], width, label="Deliveries", color=BAR_BLUE)
            ax4 = ax3.twinx()
            ax4.bar(x + width/2, store_metrics['avg_delay'], width, label="Avg Delay", color="#EF4444")

            ax3.set_xlabel("Store ID")
            ax3.set_ylabel("Number of Deliveries")
            ax4.set_ylabel("Avg Delay (minutes)")
            ax3.set_xticks(x)
            ax3.set_xticklabels(store_metrics.index.astype(str), rotation=45)
            ax3.legend(loc="upper left")
            ax4.legend(loc="upper right")
            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        blue_title("Top Stores by Transport Cost")
        cost_col = next((c for c in ["Total_Logistics_Cost", "transport_cost"] if c in df.columns), None)
        if store_col and cost_col:
            store_cost = df.groupby(store_col)[cost_col].sum().sort_values(ascending=False).head(15)

            fig4, ax4 = plt.subplots(figsize=(8, 5))
            fig4.patch.set_facecolor(GREEN_BG)
            ax4.set_facecolor(GREEN_BG)
            ax4.bar(store_cost.index.astype(str), store_cost.values, color="#E67E22")
            ax4.set_xlabel("Store ID")
            ax4.set_ylabel("Total Transport Cost ($)")
            ax4.tick_params(axis='x', rotation=45)
            st.pyplot(fig4)
            plt.close(fig4)





# ----------------------------------------------------------------
# 8. PERISHABLE & SPOILAGE RISK
# ----------------------------------------------------------------
elif eda_option == "Perishable & Spoilage Risk":

    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"

    def blue_title(title):
        st.markdown(
            f"""
            <div style="background-color:#2F75B5;padding:14px;border-radius:8px;
                 font-size:16px;color:white;margin:15px 0 8px 0;
                 text-align:center;font-weight:600;">{title}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Perishable & Spoilage Risk Analysis</b><br><br>
        Identifies high spoilage risk areas due to delays and temperature breaches — 
        critical for prioritizing perishable goods in routing.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Safe Column Detection & Conversion
    breach_col = next((c for c in ["Threshold_Breach_Flag", "breach_flag"] if c in df.columns), None)
    temp_col = "Temperature" if "Temperature" in df.columns else None
    delay_col = "Delay_Minutes" if "Delay_Minutes" in df.columns else None

    # Convert breach flag to numeric safely
    if breach_col:
        df['breach_numeric'] = pd.to_numeric(df[breach_col], errors='coerce').fillna(0)
        breach_rate = (df['breach_numeric'].mean() * 100)
    else:
        breach_rate = 0

    avg_temp = float(df[temp_col].mean()) if temp_col and pd.api.types.is_numeric_dtype(df[temp_col]) else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Temperature Breach Rate</div>
                <div class="summary-value">{breach_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Temperature</div>
                <div class="summary-value">{avg_temp:.1f}°C</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ============================================================
    # FULL WIDTH - Routes with Highest Delay Risk
    # ============================================================
    blue_title("Routes with Highest Delay Risk")

    if "Route_ID" in df.columns and delay_col:
        df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')
        route_delay = (
            df.groupby("Route_ID")[delay_col]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )

        fig2, ax2 = plt.subplots(figsize=(16, 5))
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)

        ax2.bar(
            route_delay.index.astype(str),
            route_delay.values,
            color="#EF4444"
        )

        ax2.set_xlabel("Route ID")
        ax2.set_ylabel("Avg Delay (minutes)")
        ax2.tick_params(axis='x', rotation=45)

        st.pyplot(fig2)
        plt.close(fig2)

    st.write("---")

    # ============================================================
    # TWO HALF-WIDTH CHARTS BELOW
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        blue_title("Spoilage Risk by Route (Top 15)")

        if "Route_ID" in df.columns and breach_col:
            route_risk = (
                df.groupby("Route_ID")['breach_numeric']
                .mean()
                .sort_values(ascending=False)
                .head(15) * 100
            )

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.patch.set_facecolor(GREEN_BG)
            ax1.set_facecolor(GREEN_BG)

            colors = [
                "#EF4444" if v > 20 else BAR_BLUE
                for v in route_risk.values
            ]

            ax1.bar(
                route_risk.index.astype(str),
                route_risk.values,
                color=colors
            )

            ax1.axhline(
                20,
                color='orange',
                linestyle='--',
                linewidth=2,
                label='20% Risk Threshold'
            )

            ax1.set_xlabel("Route ID")
            ax1.set_ylabel("Breach Rate (%)")
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()

            st.pyplot(fig1)
            plt.close(fig1)

    with col2:
        blue_title("Temperature Distribution")

        if temp_col:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)

            ax3.hist(
                df[temp_col].dropna(),
                bins=25,
                color=BAR_BLUE,
                edgecolor='white'
            )

            ax3.axvline(
                df[temp_col].mean(),
                color='orange',
                linestyle='--',
                label=f'Mean: {df[temp_col].mean():.1f}°C'
            )

            ax3.set_xlabel("Temperature (°C)")
            ax3.set_ylabel("Frequency")
            ax3.legend()

            st.pyplot(fig3)
            plt.close(fig3)


# ----------------------------------------------------------------
# 7. EDA SUMMARY REPORT
# ----------------------------------------------------------------



# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <br><br>
    <div style="background-color:#2E86C1;padding:12px;text-align:center;color:white;
         border-radius:6px;font-size:14px;">
        © 2026 SupplySyncAI – AI-Optimized Logistics & Routing
    </div>
""", unsafe_allow_html=True)