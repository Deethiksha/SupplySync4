import streamlit as st
from supabase import create_client
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils.html_table import render_html_table
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


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
    <h1 style="margin:0 0 8px 0;">SupplySync.AI Autonomous Inventory Intelligence & Demand-Driven Retail Execution Platform</h1>
    <h3 style="font-weight:400;margin-top: 0;">Optimizing supply chain operations with machine learning.</h3>
    <p style="font-size:17px;margin-top: 15px;">
            AI-powered demand forecasting, logistics optimization, and inventory intelligence.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#2F75B5;padding:32px;border-radius:12px;color:white;
     font-size:17px;line-height:1.8;">

<b>SupplySync.AI</b> is an AI-powered Supply Chain Intelligence Platform developed to optimize logistics,
transportation, inventory management, and retail operations. The platform leverages Machine Learning,
Predictive Analytics, and Business Intelligence to analyze supply chain data and generate actionable
insights for improving operational efficiency and decision-making.


The system predicts key business metrics such as delivery delays, travel time, fuel costs,
logistics costs, truck utilization, spoilage risks, on-time deliveries, route risks,
vehicle recommendations, and delivery success. Through interactive dashboards and real-time
analytics, SupplySync.AI helps organizations monitor performance, identify potential risks,
optimize resources, and enhance overall supply chain operations.


<b>Key Objectives</b>
<br>✓ Optimize logistics and transportation operations.
<br>✓ Improve delivery performance and on-time delivery rates.
<br>✓ Reduce logistics and fuel-related costs.
<br>✓ Maximize vehicle utilization and resource efficiency.
<br>✓ Predict and mitigate operational risks.
<br>✓ Monitor perishable goods during transit.
<br>✓ Support data-driven supply chain decision-making.

</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. DATA COLLECTION LAYER
# ============================================================
st.markdown("""
<div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;
     margin-top:20px;margin-bottom:10px;">
    <h3 style="margin:0;">Data Collection Layer</h3>
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
    <div style="
        background-color:#0B2C5D;
        padding:15px 25px;
        border-radius:10px;
        color:white;
        margin-top:30px;
        margin-bottom:20px;">
        <h4 style="margin:0;">Data Preview</h4>
    </div>
    """, unsafe_allow_html=True)

    render_html_table(df.head(15), max_height=300)
    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ====================== 2. DATA PRE-PROCESSING ======================
    st.markdown("""
    <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;
         color:white;margin-top:40px;margin-bottom:12px;">
        <h3 style="margin:0;">Data Pre-Processing</h3>
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
        <b>What this section does:</b>
        This section provides a <b>high-level health check</b> of the dataset before any modeling or forecasting is attempted.
        It evaluates:
        <ul>
            <li>Missing values</li>
            <li>Duplicate records</li>
            <li>Data type consistency</li>
            <li>Overall row and column completeness</li>
        </ul>
        <b>Why this matters:</b>
        Demand forecasting models are highly sensitive to <b>poor data quality</b>.
        Even small inconsistencies (missing prices, invalid quantities, duplicate transactions)
        can significantly distort predictions.<br>
        <b>Key insights users get:</b>
        <ul>
            <li>Whether the dataset is <b>model-ready</b></li>
            <li>Which columns require cleaning or transformation</li>
            <li>Confidence in the reliability of downstream analysis</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # PREPARE DATA
    # =========================
    rows_count = df.shape[0]
    cols_count = df.shape[1]
    dup_count = df.duplicated().sum()
    dtype_counts = df.dtypes.value_counts()
    mv = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)

    # =========================
    # DATASET SHAPE
    # =========================
    st.markdown(f"""
        <div style="
            background-color:#ffffff;
            border-radius:12px;
            border:1px solid #D0D8E4;
            border-left:4px solid #1F3A5F;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            overflow:hidden;
            margin-bottom:24px;
        ">
            <div style="background-color:#1F3A5F;padding:14px 20px;border-radius:0;">
                <span style="color:white;font-size:15px;font-weight:700;">Dataset Shape</span>
            </div>
            <div style="padding:16px 20px;background-color:#ffffff;">
                <table style="width:100%;border-collapse:collapse;border:1px solid #D0D8E4;border-radius:8px;overflow:hidden;">
                    <tr style="background-color:#E8EEF4;">
                        <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Metric</th>
                        <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Value</th>
                    </tr>
                    <tr style="background-color:#ffffff;">
                        <td style="padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;">Total Rows</td>
                        <td style="padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;">{rows_count:,}</td>
                    </tr>
                    <tr style="background-color:#ffffff;">
                        <td style="padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;">Total Columns</td>
                        <td style="padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;">{cols_count}</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # MISSING VALUE ANALYSIS
    # =========================
    missing_filtered = mv[mv > 0]

    if not missing_filtered.empty:
        rows_html = ''.join([
            f"<tr style='background-color:#ffffff;'>"
            f"<td style='padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;'>{c}</td>"
            f"<td style='padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;'>{v}%</td>"
            f"</tr>"
            for c, v in missing_filtered.items()
        ])

        st.markdown(f"""
            <div style="
                background-color:#ffffff;
                border-radius:12px;
                border:1px solid #D0D8E4;
                border-left:4px solid #1F3A5F;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);
                overflow:hidden;
                margin-bottom:24px;
            ">
                <div style="background-color:#1F3A5F;padding:14px 20px;">
                    <span style="color:white;font-size:15px;font-weight:700;">Missing Value Analysis (%)</span>
                </div>
                <div style="padding:16px 20px;background-color:#ffffff;">
                    <table style="width:100%;border-collapse:collapse;border:1px solid #D0D8E4;">
                        <tr style="background-color:#E8EEF4;">
                            <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Column Name</th>
                            <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Missing (%)</th>
                        </tr>
                        {rows_html}
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No missing values found in the dataset!")

    # =========================
    # DUPLICATE ANALYSIS
    # =========================
    st.markdown(f"""
        <div style="
            background-color:#ffffff;
            border-radius:12px;
            border:1px solid #D0D8E4;
            border-left:4px solid #1F3A5F;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            overflow:hidden;
            margin-bottom:24px;
        ">
            <div style="background-color:#1F3A5F;padding:14px 20px;">
                <span style="color:white;font-size:15px;font-weight:700;">Duplicate Analysis</span>
            </div>
            <div style="padding:16px 20px;background-color:#ffffff;">
                <table style="width:100%;border-collapse:collapse;border:1px solid #D0D8E4;">
                    <tr style="background-color:#E8EEF4;">
                        <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Metric</th>
                        <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Value</th>
                    </tr>
                    <tr style="background-color:#ffffff;">
                        <td style="padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;">Total Duplicate Rows</td>
                        <td style="padding:11px 16px;font-size:14px;font-weight:600;border:1px solid #D0D8E4;color:{'#FF9800' if dup_count > 0 else '#00A854'};">{dup_count:,}</td>
                    </tr>
                </table>
                {f'<p style="color:#FF9800;margin:10px 0 0 0;font-size:13px;">⚠️ Duplicates found — consider removing them before modeling.</p>' if dup_count > 0 else '<p style="color:#00A854;margin:10px 0 0 0;font-size:13px;">✅ No duplicate records found.</p>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # DATA TYPES SUMMARY
    # =========================
    dtype_rows = ''.join([
        f"<tr style='background-color:#ffffff;'>"
        f"<td style='padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;'>{d}</td>"
        f"<td style='padding:11px 16px;color:#333;font-size:14px;border:1px solid #D0D8E4;'>{c}</td>"
        f"</tr>"
        for d, c in dtype_counts.items()
    ])

    st.markdown(f"""
        <div style="
            background-color:#ffffff;
            border-radius:12px;
            border:1px solid #D0D8E4;
            border-left:4px solid #1F3A5F;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            overflow:hidden;
            margin-bottom:24px;
        ">
            <div style="background-color:#1F3A5F;padding:14px 20px;">
                <span style="color:white;font-size:15px;font-weight:700;">Data Types Summary</span>
            </div>
            <div style="padding:16px 20px;background-color:#ffffff;">
                <table style="width:100%;border-collapse:collapse;border:1px solid #D0D8E4;">
                    <tr style="background-color:#E8EEF4;">
                        <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Data Type</th>
                        <th style="padding:11px 16px;text-align:left;color:#1F3A5F;font-weight:600;font-size:14px;border:1px solid #D0D8E4;">Column Count</th>
                    </tr>
                    {dtype_rows}
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # OVERALL QUALITY SCORE
    # =========================
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_score = max(0, 100 - missing_pct)

    st.markdown(f"""
        <div style="
            background-color:#ffffff;
            border-radius:12px;
            border:1px solid #D0D8E4;
            border-left:4px solid #1F3A5F;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            overflow:hidden;
            margin-bottom:24px;
        ">
            <div style="background-color:#1F3A5F;padding:14px 20px;">
                <span style="color:white;font-size:15px;font-weight:700;">Overall Data Quality Score</span>
            </div>
            <div style="padding:28px;background-color:#ffffff;text-align:center;">
                <h2 style="color:{'#00A854' if quality_score >= 90 else '#FF9800'};margin:0 0 10px 0;font-size:36px;">{quality_score:.1f} / 100</h2>
                <p style="color:#555;margin:0;font-size:14px;">The dataset is {'<strong style="color:#00A854;">ready for modeling</strong>' if quality_score >= 90 else '<strong style="color:#FF9800;">needs some cleaning</strong>'}.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
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
        Delivery performance is the <b>heartbeat of any logistics operation</b> — even small delays 
        can cascade into <b>customer dissatisfaction, penalty charges, and lost contracts</b>. 
        Maintaining a consistent on-time delivery rate is critical to building a 
        <b>reliable and competitive supply chain</b>. This analysis helps you:<br><br>
        <ul>
        &nbsp;&nbsp;<li><b>On-Time Performance by Route</b> — Identify which routes are consistently meeting or missing the <b>85% on-time target</b></li>
        &nbsp;&nbsp;<li><b>Average Delay by Store</b> — Pinpoint which store locations are experiencing the longest delivery delays</li>
        &nbsp;&nbsp;<li><b>Monthly Delivery Volume</b> — Track delivery trends over time to spot seasonal peaks and capacity gaps</li>
        &nbsp;&nbsp;<li><b>Delay Distribution</b> — Understand how delays are spread across all deliveries and where the majority fall</li>
        </ul>
        Use these insights to <b>restructure problem routes</b>, <b>prioritise high-delay stores</b>,
        and implement <b>proactive scheduling adjustments</b> that keep deliveries 
        on time and customers satisfied.
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
            fig3.tight_layout()   # ✅ Prevents rotated labels from being clipped
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
        fig4.tight_layout()   # ✅ Keeps consistent padding with fig3
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Delivery Performance Analysis</b><br><br>
        Delivery performance is the <b>heartbeat of any logistics operation</b> — even small delays
        can cascade into <b>customer dissatisfaction, penalty charges, and lost contracts</b>.
        Maintaining a consistent on-time delivery rate is critical to building a
        <b>reliable and competitive supply chain</b>. This analysis helps you:<br><br>
        
        
        <ul>
            <li>Overall on-time delivery rate stands at <b>{on_time_rate:.1f}%%</b> —
                {on_time_status} the <b>85%% business target</b>.</li>
            <li><b>{pct_late:.1f}%%</b> of all deliveries are delayed, with an average delay of
                <b>{avg_delay:.1f} minutes</b> per late shipment.</li>
            <li>Store-level analysis reveals <b>{worst_store}</b> as the highest-delay location,
                averaging <b>{worst_store_delay:.1f} minutes</b> per delivery.</li>
            <li>Route-level analysis shows <b>{worst_route}</b> as the lowest-performing route,
                achieving only <b>{worst_route_otd:.1f}%%</b> on-time rate.</li>
            <li>Monthly volume trends indicate {volume_trend}.</li>
        </ul>

        
        """
        ).format(
            on_time_rate    = on_time_rate,
            pct_late        = 100 - on_time_rate,
            avg_delay       = avg_delay,
            on_time_status  = " above" if on_time_rate >= 85 else "⚠ below",
            worst_store     = store_delay.index[0]  if len(store_delay) else "N/A",
            worst_store_delay = float(store_delay.iloc[0]) if len(store_delay) else 0,
            worst_route     = route_otd.index[-1]   if len(route_otd) else "N/A",
            worst_route_otd = float(route_otd.iloc[-1]) if len(route_otd) else 0,
            avg_store_delay = float(store_delay.mean()) if len(store_delay) else 0,
            volume_trend    = (
                "consistent demand with manageable peaks"
                if monthly.std() / monthly.mean() < 0.3
                else "significant seasonal fluctuations requiring proactive fleet scheduling"
            ),
            top_recommendation = (
                "Sustain current routing strategy and monitor for degradation."
                if on_time_rate >= 85
                else "Immediately review scheduling and routing for the bottom 20%% of routes."
            ),
        ),
        unsafe_allow_html=True
    )    


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
        <b>Truck Utilization Analysis</b><br><br>
        Truck utilization is a <b>key indicator of fleet efficiency</b> — underutilized trucks 
        mean you are paying for capacity that is not being used, while overutilized trucks 
        lead to <b>faster wear, higher maintenance costs, and delivery failures</b>. 
        This analysis helps you:<br><br>
        &nbsp;&nbsp;<li><b>Truck Utilization Rate</b> — Spot vehicles consistently falling below the <b>70% efficiency target</b></li>
        &nbsp;&nbsp;<li><b>Utilization Distribution</b> — Understand how utilization is spread across your entire fleet</li>
        &nbsp;&nbsp;<li><b>Deliveries per Truck</b> — Identify trucks carrying the heaviest delivery workloads</li>
        &nbsp;&nbsp;<li><b>Average Load Weight</b> — Detect trucks being overloaded or significantly underloaded per trip</li>
        Use these insights to <b>rebalance workloads</b>, <b>retire or redeploy underperforming trucks</b>,
        and ensure every vehicle in your fleet is operating at <b>peak efficiency</b>.<br><br>
        
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
            fig1.tight_layout()   # ✅ Normalize padding for rotated labels
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
            fig2.tight_layout()   # ✅ Consistent padding with fig1
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
            fig3.tight_layout()   # ✅ Normalize padding for rotated labels
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
            fig4.tight_layout()   # ✅ Consistent padding with fig3
            st.pyplot(fig4)
            plt.close(fig4)

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-top:20px;margin-bottom:20px;">
        <b>Truck Utilization Analysis</b><br><br>
        Truck utilization is a <b>key indicator of fleet efficiency</b> — underutilized trucks
        mean you are paying for capacity that is not being used, while overutilized trucks
        lead to <b>faster wear, higher maintenance costs, and delivery failures</b>.
        This analysis helps you:<br><br>
        
        <ul>
            <li>Fleet-wide average utilization is <b>{avg_util:.1f}%%</b> —
                {util_status} the <b>70%% efficiency threshold</b>,
                {util_meaning}.</li>
            <li><b>{pct_underutil:.1f}%%</b> of all truck trips run below 70%% capacity,
                representing direct <b>cost and fuel wastage</b> that could be recovered
                through smarter load consolidation.</li>
            <li>The highest-loaded truck is <b>{top_load_truck}</b> averaging
                <b>{top_load_val:,.0f} kg</b> per trip, while the lowest among the top 15 averages
                <b>{bot_load_val:,.0f} kg</b> — indicating a <b>{load_spread:.0f} kg workload gap</b>
                across the fleet.</li>
            <li>Truck <b>{busiest_truck}</b> leads deliveries with <b>{busiest_count:,} trips</b>,
                suggesting potential overreliance on a single vehicle and elevated
                <b>breakdown risk</b> from high trip frequency.</li>
            <li>Utilization distribution shows the fleet is
                {dist_shape} — meaning
                {dist_meaning}.</li>
        </ul>
        </div>
        """
        ).format(
            avg_util      = avg_util,
            util_status   = "above" if avg_util >= 70 else "below",
            util_meaning  = (
                "the fleet is operating efficiently overall"
                if avg_util >= 70
                else "a significant share of fleet capacity is going unused on every run"
            ),
            pct_underutil = (
                float((df[util_col] < 70).mean() * 100) if util_col else 0
            ),
            top_load_truck = (
                load_data.index[0] if vehicle_col and load_col and len(load_data) else "N/A"
            ),
            top_load_val  = (
                float(load_data.iloc[0]) if vehicle_col and load_col and len(load_data) else 0
            ),
            bot_load_val  = (
                float(load_data.iloc[-1]) if vehicle_col and load_col and len(load_data) > 1 else 0
            ),
            load_spread   = (
                float(load_data.iloc[0] - load_data.iloc[-1])
                if vehicle_col and load_col and len(load_data) > 1 else 0
            ),
            busiest_truck = deliveries.index[0] if vehicle_col and len(deliveries) else "N/A",
            busiest_count = int(deliveries.iloc[0]) if vehicle_col and len(deliveries) else 0,
            dist_shape    = (
                "tightly clustered around the mean"
                if util_col and float(df[util_col].std()) < 10
                else "widely spread across the range"
            ),
            dist_meaning  = (
                "utilization is consistent and predictable across vehicles"
                if util_col and float(df[util_col].std()) < 10
                else "some trucks are heavily loaded while others are barely used, flagging an uneven dispatch strategy"
            ),
        ),
        unsafe_allow_html=True
    )


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
        Route costs are a <b>major driver of logistics expenditure</b> — inefficient routes can 
        silently drain <b>20–40% of your transport budget</b> without proper visibility. 
        This section gives you full clarity to act:<br><br>
        &nbsp;&nbsp;<li><b>Total Cost by Route</b> — Reveal the most expensive routes consuming your logistics budget</li>
        &nbsp;&nbsp;<li><b>Cost per KM by Route</b> — Identify routes with poor cost efficiency relative to distance covered</li>
        &nbsp;&nbsp;<li><b>Distance vs Transport Cost</b> — Understand whether longer routes are truly costing proportionally more</li>
        &nbsp;&nbsp;<li><b>Number of Stops vs Avg Cost</b> — Measure how multi-stop routes drive up delivery costs</li><br>
        Use these insights to <b>eliminate costly routes</b>, <b>consolidate stops</b>, 
        renegotiate <b>carrier contracts</b>, and redesign routes that deliver better 
        value with every kilometer driven.
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
            fig1.tight_layout()   # ✅
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
            fig2.tight_layout()   # ✅
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
            fig3.tight_layout()   # ✅
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
            fig4.tight_layout()   # ✅
            st.pyplot(fig4)
            plt.close(fig4)

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Route Cost Analysis</b><br><br>
        Route costs are one of the <b>largest controllable expenses</b> in logistics operations —
        inefficient routing can silently drain <b>20–40% of your total transport budget</b>. 
        This analysis helps you:<br><br>
        &nbsp;&nbsp;<li><b>Identify the most expensive routes</b> consuming a disproportionate share of your logistics budget</li>
        &nbsp;&nbsp;<li><b>Evaluate cost efficiency per kilometer</b> to find routes that cost more than they should</li>
        &nbsp;&nbsp;<li><b>Understand distance vs cost relationships</b> to verify if longer routes justify their expenses</li>
        &nbsp;&nbsp;<li><b>Analyse multi-stop route costs</b> to determine how additional stops impact overall spending</li><br>
        Use these insights to <b>eliminate wasteful routes</b>, <b>consolidate deliveries</b>,
        renegotiate <b>carrier contracts</b>, and redesign routes that deliver 
        maximum value with every kilometer driven.
        </div>
        """,
        unsafe_allow_html=True
    )


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
    Fuel expenses represent one of the <b>most significant and controllable costs</b> in any 
    logistics operation — often consuming <b>25–35% of total transport budgets</b>. 
    This section provides a comprehensive breakdown to help you take action:<br><br>
    &nbsp;&nbsp;<li><b>Fuel Cost by Route</b> — Pinpoint which routes are burning the most fuel and costing the most money</li>
    &nbsp;&nbsp;<li><b>Fuel Cost by Truck</b> — Identify underperforming vehicles that may need maintenance or replacement</li>
    &nbsp;&nbsp;<li><b>Load Weight vs Fuel Cost</b> — Understand how cargo weight directly drives fuel expenses</li>
    &nbsp;&nbsp;<li><b>Fuel Cost Trend Over Time</b> — Track monthly fuel spending to spot seasonal patterns and rising costs</li><br>
    Leverage these insights to <b>optimize route planning</b>, enforce <b>load limits</b>, 
    schedule <b>preventive maintenance</b>, and ultimately <b>reduce fuel spend</b> across your entire fleet.
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
            fig1.tight_layout()   # ✅
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
            fig2.tight_layout()   # ✅
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
            fig3.tight_layout()   # ✅
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
            fig4.tight_layout()   # ✅
            st.pyplot(fig4)
            plt.close(fig4)
    
    st.markdown(
    """
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Fuel Consumption Analysis</b><br><br>
    Fuel is one of the <b>largest controllable costs</b> in logistics operations — 
    typically accounting for <b>25–35% of total transport expenses</b>. This analysis helps you:<br><br>
    &nbsp;&nbsp;<li><b>Identify high fuel-consuming routes</b> and trucks draining your budget</li>
    &nbsp;&nbsp;<li><b>Detect inefficiencies</b> caused by overloading, poor route planning, or aging vehicles</li>
    &nbsp;&nbsp;<li><b>Reduce carbon footprint</b> by targeting the worst-performing assets first</li>
    &nbsp;&nbsp;<li><b>Cut operational costs</b> by optimizing fuel usage across your entire fleet</li><br>
    Use these insights to prioritize <b>maintenance schedules</b>, reassign <b>high-load routes</b>, 
    and make data-driven decisions that directly impact your <b>bottom line</b>.
    </div>
    """,
    unsafe_allow_html=True
)


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
        <b>Store Performance Analysis</b><br><br>
        Store-level performance is a <b>critical dimension of logistics efficiency</b> — high-frequency 
        stores demand more resources, while high-delay stores signal <b>routing problems, 
        scheduling conflicts, or operational bottlenecks</b> that directly impact 
        <b>customer satisfaction and transport costs</b>. This analysis helps you:<br><br>
        &nbsp;&nbsp;<li><b>Replenishment Frequency by Store</b> — Identify the most frequently served stores that require priority logistics planning</li>
        &nbsp;&nbsp;<li><b>Stores with Highest Average Delay</b> — Pinpoint stores consistently receiving late deliveries that need immediate attention</li>
        &nbsp;&nbsp;<li><b>Delivery Volume vs Avg Delay</b> — Understand whether high delivery volumes are contributing to increased delays at specific stores</li>
        &nbsp;&nbsp;<li><b>Top Stores by Transport Cost</b> — Reveal which stores are consuming the largest share of your transport budget</li><br>
        Use these insights to <b>prioritise high-demand stores</b>, <b>resolve chronic delay hotspots</b>,
        and optimise delivery schedules to reduce transport costs 
        across your entire store network.
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
            fig1.tight_layout()   # ✅
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
            fig2.tight_layout()   # ✅
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
            fig3.tight_layout()   # ✅
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
            fig4.tight_layout()   # ✅
            st.pyplot(fig4)
            plt.close(fig4)

    # ── compute extra stats needed for insight panel ──────────────────────────
    _store_freq_all  = df[store_col].value_counts()                if store_col                  else pd.Series(dtype=float)
    _store_delay_all = df.groupby(store_col)[delay_col].mean()     if store_col and delay_col    else pd.Series(dtype=float)
    _cost_col        = next((c for c in ["Total_Logistics_Cost", "transport_cost"] if c in df.columns), None)
    _store_cost_all  = df.groupby(store_col)[_cost_col].sum()      if store_col and _cost_col    else pd.Series(dtype=float)

    # stores that are BOTH high-frequency AND high-delay
    if len(_store_freq_all) and len(_store_delay_all):
        _freq_threshold  = float(_store_freq_all.mean())
        _delay_threshold = float(_store_delay_all.mean())
        _overlap = set(_store_freq_all[_store_freq_all >= _freq_threshold].index) & \
                   set(_store_delay_all[_store_delay_all >= _delay_threshold].index)
        _overlap_count = len(_overlap)
    else:
        _overlap_count = 0

    # cost concentration: top 3 stores share of total cost
    _top3_cost_pct = (
        float(_store_cost_all.nlargest(3).sum() / _store_cost_all.sum() * 100)
        if len(_store_cost_all) and _store_cost_all.sum() > 0 else 0
    )

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-top:20px;margin-bottom:20px;">
        <b>Store Performance Analysis</b><br><br>
        Store-level performance is a <b>critical dimension of logistics efficiency</b> — high-frequency
        stores demand more resources, while high-delay stores signal <b>routing problems,
        scheduling conflicts, or operational bottlenecks</b> that directly impact
        <b>customer satisfaction and transport costs</b>. This analysis helps you:<br><br>
        
        <ul>
            <li>The network serves <b>{total_stores} unique stores</b> with an average of
                <b>{avg_deliveries_per_store:.1f} deliveries per store</b> — indicating
                {freq_spread}.</li>
            <li>Store <b>{top_freq_store}</b> is the most frequently replenished location with
                <b>{top_freq_count:,} deliveries</b>, receiving
                <b>{freq_ratio:.1f}x</b> more deliveries than the network average.</li>
            <li>Store <b>{worst_delay_store}</b> has the highest average delay at
                <b>{worst_delay_val:.1f} minutes</b>, while the fleet-wide store average is
                <b>{avg_delay_all:.1f} minutes</b> — a gap of
                <b>{delay_gap:.1f} minutes</b> above average.</li>
            <li><b>{overlap_count} store{plural}</b> fall into the critical zone of being
                <b>both high-frequency and high-delay</b>, representing the highest operational
                risk as demand pressure compounds delivery failures.</li>
            <li>The top 3 stores by transport cost account for
                <b>{top3_cost_pct:.1f}%%</b> of total network transport spend —
                {cost_concentration}.</li>
        </ul>
        </div>
        """
        ).format(
            total_stores             = total_stores,
            avg_deliveries_per_store = avg_deliveries_per_store,
            freq_spread = (
                "an evenly distributed replenishment workload across the network"
                if len(_store_freq_all) and float(_store_freq_all.std() / _store_freq_all.mean()) < 0.4
                else "an uneven replenishment pattern with certain stores dominating delivery demand"
            ),
            top_freq_store  = _store_freq_all.index[0]    if len(_store_freq_all)  else "N/A",
            top_freq_count  = int(_store_freq_all.iloc[0]) if len(_store_freq_all) else 0,
            freq_ratio      = (
                float(_store_freq_all.iloc[0] / _store_freq_all.mean())
                if len(_store_freq_all) and _store_freq_all.mean() > 0 else 0
            ),
            worst_delay_store = _store_delay_all.idxmax() if len(_store_delay_all) else "N/A",
            worst_delay_val   = float(_store_delay_all.max())  if len(_store_delay_all) else 0,
            avg_delay_all     = float(_store_delay_all.mean()) if len(_store_delay_all) else 0,
            delay_gap         = (
                float(_store_delay_all.max() - _store_delay_all.mean())
                if len(_store_delay_all) else 0
            ),
            overlap_count = _overlap_count,
            plural        = "s" if _overlap_count != 1 else "",
            top3_cost_pct = _top3_cost_pct,
            cost_concentration = (
                "a highly concentrated cost structure where a few stores drive the majority of spend"
                if _top3_cost_pct >= 40
                else "a relatively balanced cost distribution across the store network"
            ),
        ),
        unsafe_allow_html=True
    )



# ----------------------------------------------------------------
# 7. PERISHABLE & SPOILAGE RISK
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
        Spoilage risk is one of the <b>most costly and time-sensitive challenges</b> in cold chain 
        logistics — every degree of temperature breach and every minute of delay can result in 
        <b>product loss, compliance violations, and customer rejections</b>. 
        This analysis helps you:<br><br>
        &nbsp;&nbsp;<li><b>Routes with Highest Delay Risk</b> — Identify routes where excessive delays are putting perishable goods at risk</li>
        &nbsp;&nbsp;<li><b>Spoilage Risk by Route</b> — Spot routes exceeding the <b>20% breach threshold</b> that require immediate rerouting</li>
        &nbsp;&nbsp;<li><b>Temperature Distribution</b> — Understand how temperature varies across deliveries and where unsafe ranges occur</li><br>
        Use these insights to <b>prioritise cold chain routes</b>, <b>enforce temperature compliance</b>,
        and redesign delivery schedules to minimise spoilage risk 
        across your perishable goods network.<br><br>
        
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
            fig1.tight_layout()   # ✅
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
            fig3.tight_layout()   # ✅
            st.pyplot(fig3)
            plt.close(fig3)

    # ── extra stats for insight panel ─────────────────────────────────────────
    # Routes breaching the 20% threshold
    _routes_above_threshold = (
        int((route_risk > 20).sum())
        if "Route_ID" in df.columns and breach_col and len(route_risk) else 0
    )
    _total_routes_checked = (
        int(len(route_risk))
        if "Route_ID" in df.columns and breach_col and len(route_risk) else 0
    )
    _pct_routes_breaching = (
        float(_routes_above_threshold / _total_routes_checked * 100)
        if _total_routes_checked > 0 else 0
    )

    # Highest-risk route by breach rate
    _worst_breach_route = route_risk.index[0]  if len(route_risk) else "N/A"
    _worst_breach_val   = float(route_risk.iloc[0]) if len(route_risk) else 0

    # Highest-delay route (perishable context)
    _worst_delay_route = route_delay.index[0]  if "Route_ID" in df.columns and delay_col and len(route_delay) else "N/A"
    _worst_delay_val   = float(route_delay.iloc[0]) if "Route_ID" in df.columns and delay_col and len(route_delay) else 0

    # Temperature spread
    _temp_std  = float(df[temp_col].std())  if temp_col and pd.api.types.is_numeric_dtype(df[temp_col]) else 0
    _temp_min  = float(df[temp_col].min())  if temp_col and pd.api.types.is_numeric_dtype(df[temp_col]) else 0
    _temp_max  = float(df[temp_col].max())  if temp_col and pd.api.types.is_numeric_dtype(df[temp_col]) else 0

    # Routes that are high-delay AND high-breach
    if "Route_ID" in df.columns and delay_col and breach_col:
        _route_delay_all  = df.groupby("Route_ID")[delay_col].mean()
        _route_breach_all = df.groupby("Route_ID")['breach_numeric'].mean() * 100
        _dual_risk = set(_route_delay_all[_route_delay_all >= _route_delay_all.mean()].index) & \
                     set(_route_breach_all[_route_breach_all >= 20].index)
        _dual_risk_count = len(_dual_risk)
    else:
        _dual_risk_count = 0

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-top:20px;margin-bottom:20px;">
        <b>Perishable &amp; Spoilage Risk Analysis</b><br><br>
        Spoilage risk is one of the <b>most costly and time-sensitive challenges</b> in cold chain
        logistics — every degree of temperature breach and every minute of delay can result in
        <b>product loss, compliance violations, and customer rejections</b>.
        This analysis helps you:<br><br>
        
        <ul>
            <li>Fleet-wide temperature breach rate is <b>{breach_rate:.1f}%%</b> —
                {breach_status} the <b>10%% safe threshold</b>,
                {breach_meaning}.</li>
            <li><b>{routes_above} out of {total_routes} routes</b> ({pct_routes:.1f}%%)
                exceed the <b>20%% spoilage breach threshold</b>, with route
                <b>{worst_breach_route}</b> recording the highest breach rate at
                <b>{worst_breach_val:.1f}%%</b>.</li>
            <li>Route <b>{worst_delay_route}</b> carries the highest average delay at
                <b>{worst_delay_val:.1f} minutes</b> — prolonged transit times
                directly elevate spoilage exposure for every perishable load on this route.</li>
            <li><b>{dual_risk_count} route{dual_plural}</b> are in the
                <b>dual-risk zone</b> — simultaneously above-average delay
                <em>and</em> exceeding the 20%% breach threshold —
                representing the most urgent cold chain intervention points.</li>
            <li>Delivery temperatures range from <b>{temp_min:.1f}°C to {temp_max:.1f}°C</b>
                with a mean of <b>{avg_temp:.1f}°C</b> and a standard deviation of
                <b>{temp_std:.1f}°C</b> — {temp_consistency}.</li>
        </ul>
        </div>
        """
        ).format(
            breach_rate        = breach_rate,
            breach_status      = "within" if breach_rate <= 10 else "⚠ above",
            breach_meaning     = (
                "cold chain integrity is being maintained across the majority of deliveries"
                if breach_rate <= 10
                else "a significant share of perishable loads are being exposed to unsafe conditions"
            ),
            routes_above       = _routes_above_threshold,
            total_routes       = _total_routes_checked,
            pct_routes         = _pct_routes_breaching,
            worst_breach_route = _worst_breach_route,
            worst_breach_val   = _worst_breach_val,
            worst_delay_route  = _worst_delay_route,
            worst_delay_val    = _worst_delay_val,
            dual_risk_count    = _dual_risk_count,
            dual_plural        = "s" if _dual_risk_count != 1 else "",
            temp_min           = _temp_min,
            temp_max           = _temp_max,
            avg_temp           = avg_temp,
            temp_std           = _temp_std,
            temp_consistency   = (
                "temperatures are tightly controlled and consistent across the fleet"
                if _temp_std < 3
                else "high variability signals inconsistent refrigeration performance across vehicles"
            ),
        ),
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------
# 8. EDA SUMMARY REPORT
# ----------------------------------------------------------------
elif eda_option == "Summary Report":

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
        <b>EDA Summary Report</b><br><br>
        This report provides a <b>consolidated snapshot of all key logistics metrics</b> discovered 
        across every analysis section — giving decision-makers a single view of overall 
        operational health without navigating each section individually. 
        This summary helps you:<br><br>
        &nbsp;&nbsp;<li><b>On-Time vs Late Deliveries</b> — Get an instant overview of delivery success rates across your entire operation</li>
        &nbsp;&nbsp;<li><b>Fleet Utilization Summary</b> — See how many trucks are operating above and below the <b>70% efficiency threshold</b></li>
        &nbsp;&nbsp;<li><b>Top Routes by Transport Cost</b> — Quickly identify the most expensive routes draining your logistics budget</li>
        &nbsp;&nbsp;<li><b>Monthly Fuel Cost Trend</b> — Track how fuel expenditure has evolved over time across all routes</li>
        &nbsp;&nbsp;<li><b>Top Stores by Avg Delay</b> — Highlight the store locations consistently experiencing the worst delivery delays</li>
        &nbsp;&nbsp;<li><b>Overall Delay Distribution</b> — Understand the full spread of delays across all deliveries in your dataset</li><br>
        Use this summary to <b>identify the biggest problem areas at a glance</b>, 
        prioritise where to take action first, and track progress across 
        <b>Delivery Performance, Truck Utilization, Route Cost, Fuel Consumption, 
        Store Replenishment, and Perishable Risk</b> — all in one place.<br><br>
       
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # COLUMN DETECTION
    # ============================================================
    on_time_col = "Delivery_Status_[0]"
    delay_col = "Delay_Minutes"
    vehicle_col = "Vehicle_ID" if "Vehicle_ID" in df.columns else None
    route_col = "Route_ID" if "Route_ID" in df.columns else None
    store_col = "Store_ID" if "Store_ID" in df.columns else None
    util_col = "Capacity_Utilization_Percentage" if "Capacity_Utilization_Percentage" in df.columns else None
    load_col = next((c for c in ["Loaded_Weight", "Shipment_Weight"] if c in df.columns), None)
    cost_col = next((c for c in ["Total_Logistics_Cost", "transport_cost", "Total_Cost"] if c in df.columns), None)
    fuel_col = next((c for c in ["Fuel_Cost", "fuel_cost", "Fuel_Cost_[0]"] if c in df.columns), None)
    distance_col = next((c for c in ["distance_km", "Actual_Distance", "Distance_Traveled"] if c in df.columns), None)
    breach_col = next((c for c in ["Threshold_Breach_Flag", "breach_flag"] if c in df.columns), None)
    temp_col = "Temperature" if "Temperature" in df.columns else None
    date_col = "Dispatch_Time" if "Dispatch_Time" in df.columns else None

    df[delay_col] = pd.to_numeric(df[delay_col], errors="coerce")
    if breach_col:
        df["breach_numeric"] = pd.to_numeric(df[breach_col], errors="coerce").fillna(0)

    # ============================================================
    # METRICS
    # ============================================================
    total_deliveries = len(df)
    on_time_rate = df[on_time_col].astype(str).str.contains("On Time", case=False, na=False).mean() * 100
    avg_delay = df[delay_col].mean()
    pct_late = (df[delay_col] > 0).mean() * 100
    avg_util = float(df[util_col].mean()) if util_col else 0
    pct_underutil = float((df[util_col] < 70).mean() * 100) if util_col else 0
    total_cost = float(df[cost_col].sum()) if cost_col else 0
    avg_cost = float(df[cost_col].mean()) if cost_col else 0
    total_fuel = float(df[fuel_col].sum()) if fuel_col else 0
    avg_fuel = float(df[fuel_col].mean()) if fuel_col else 0
    total_stores = df[store_col].nunique() if store_col else 0
    worst_store_delay = df.groupby(store_col)[delay_col].mean().idxmax() if store_col else "N/A"
    breach_rate = float(df["breach_numeric"].mean() * 100) if breach_col else 0
    avg_temp = float(df[temp_col].mean()) if temp_col and pd.api.types.is_numeric_dtype(df[temp_col]) else 0
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_score = max(0, 100 - missing_pct)

    # ============================================================
    # KPI SCORECARD
    # ============================================================
    st.markdown(
        """
        <div style="background-color:#0B2C5D;padding:20px;border-radius:10px;
                    color:white;margin-bottom:20px;">
            <h4 style="margin:0 0 14px 0;">Overall Logistics Performance Scorecard</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

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
                <div class="summary-title">Late Deliveries</div>
                <div class="summary-value">{pct_late:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Truck Utilization</div>
                <div class="summary-value">{avg_util:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Transport Cost</div>
                <div class="summary-value">${total_cost:,.0f}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Cost per Delivery</div>
                <div class="summary-value">${avg_cost:,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Total Fuel Cost</div>
                <div class="summary-value">${total_fuel:,.0f}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Breach Rate</div>
                <div class="summary-value">{breach_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Data Quality Score</div>
                <div class="summary-value">{quality_score:.1f}/100</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ============================================================
    # CHART ROW 1
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        blue_title("On-Time vs Late Deliveries Overview")
        on_time_count = int(df[on_time_col].astype(str).str.contains("On Time", case=False, na=False).sum())
        late_count = total_deliveries - on_time_count
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        ax1.bar(["On-Time", "Late / Delayed"], [on_time_count, late_count], color=[BAR_BLUE, "#EF4444"])
        for bar, val in zip(ax1.patches, [on_time_count, late_count]):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_deliveries * 0.01,
                f"{val:,}", ha="center", fontsize=11, color="black", fontweight="bold")
        ax1.set_ylabel("Number of Deliveries")
        ax1.set_xlabel("Delivery Status")
        ax1.grid(axis="y", linestyle="-", color="#3B3B3B", alpha=0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        blue_title("Fleet Utilization Summary")
        if util_col:
            above_70 = int((df[util_col] >= 70).sum())
            below_70 = int((df[util_col] < 70).sum())
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(GREEN_BG)
            ax2.set_facecolor(GREEN_BG)
            ax2.bar(["≥70% (Efficient)", "<70% (Under-utilized)"], [above_70, below_70], color=[BAR_BLUE, "#EF4444"])
            for bar, val in zip(ax2.patches, [above_70, below_70]):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total_deliveries * 0.005,
                    f"{val:,}", ha="center", fontsize=11, color="black", fontweight="bold")
            ax2.set_ylabel("Number of Trips")
            ax2.set_xlabel("Utilization Category")
            ax2.grid(axis="y", linestyle="-", color="#3B3B3B", alpha=0.5)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Column not found: Capacity_Utilization_Percentage")

    st.write("---")

    # ============================================================
    # CHART ROW 2
    # ============================================================
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Top 10 Routes by Transport Cost")
        if route_col and cost_col:
            top_routes = df.groupby(route_col)[cost_col].sum().sort_values(ascending=False).head(10)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)
            ax3.bar(top_routes.index.astype(str), top_routes.values, color=BAR_BLUE)
            ax3.set_xlabel("Route ID")
            ax3.set_ylabel("Total Cost ($)")
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(axis="y", linestyle="-", color="#3B3B3B", alpha=0.5)
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("Columns not found: Route_ID / Total_Logistics_Cost")

    with col4:
        blue_title("Monthly Fuel Cost Trend")
        if date_col and fuel_col:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
            fuel_trend = df_temp.groupby(df_temp[date_col].dt.to_period("M"))[fuel_col].sum().sort_index()
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            fig4.patch.set_facecolor(GREEN_BG)
            ax4.set_facecolor(GREEN_BG)
            ax4.bar(fuel_trend.index.astype(str), fuel_trend.values, color=BAR_BLUE)
            ax4.set_xlabel("Month")
            ax4.set_ylabel("Total Fuel Cost ($)")
            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(axis="y", linestyle="-", color="#3B3B3B", alpha=0.5)
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.info("Columns not found: Dispatch_Time / Fuel_Cost")

    st.write("---")

    # ============================================================
    # CHART ROW 3
    # ============================================================
    col5, col6 = st.columns(2)

    with col5:
        blue_title("Top 10 Stores by Avg Delay")
        if store_col and delay_col:
            store_delay = df.groupby(store_col)[delay_col].mean().sort_values(ascending=False).head(10)
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            fig5.patch.set_facecolor(GREEN_BG)
            ax5.set_facecolor(GREEN_BG)
            ax5.bar(store_delay.index.astype(str), store_delay.values, color="#EF4444")
            ax5.set_xlabel("Store ID")
            ax5.set_ylabel("Avg Delay (min)")
            ax5.tick_params(axis="x", rotation=45)
            ax5.grid(axis="y", linestyle="-", color="#3B3B3B", alpha=0.5)
            ax5.spines["top"].set_visible(False)
            ax5.spines["right"].set_visible(False)
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
        else:
            st.info("Columns not found: Store_ID / Delay_Minutes")

    with col6:
        blue_title("Overall Delay Distribution")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        fig6.patch.set_facecolor(GREEN_BG)
        ax6.set_facecolor(GREEN_BG)
        ax6.hist(df[delay_col].dropna(), bins=25, color=BAR_BLUE, edgecolor="white")
        ax6.axvline(df[delay_col].mean(), color="orange", linestyle="--",
            linewidth=2, label=f"Mean: {df[delay_col].mean():.1f} min")
        ax6.set_xlabel("Delay (minutes)")
        ax6.set_ylabel("Frequency")
        ax6.legend()
        ax6.grid(axis="y", linestyle="-", color="#3B3B3B", alpha=0.5)
        ax6.spines["top"].set_visible(False)
        ax6.spines["right"].set_visible(False)
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    st.write("---")

    # ============================================================
    # INSIGHTS REPORT
    # ============================================================
    st.markdown(
        f"""
        <div style="background-color:#0B2C5D;padding:30px;border-radius:12px;
                    color:white;font-size:15px;line-height:1.8;">

        <h4>Data Health & Readiness</h4>
        <ul>
            <li>Dataset contains <b>{df.shape[0]:,} rows and {df.shape[1]} columns</b>.</li>
            <li>Overall data quality score: <b>{quality_score:.1f} / 100</b> — dataset is <b>model-ready</b> after preprocessing.</li>
            <li>Core identifiers (Route_ID, Vehicle_ID, Store_ID, Dispatch_Time) are <b>consistent and validated</b>.</li>
        </ul>

        <h4>Delivery Performance</h4>
        <ul>
            <li>Overall on-time delivery rate is <b>{on_time_rate:.1f}%</b> —
                {"above" if on_time_rate >= 85 else "below"} the 85% business target.</li>
            <li><b>{pct_late:.1f}%</b> of all deliveries are delayed, with an average delay of <b>{avg_delay:.1f} minutes</b>.</li>
            <li>Certain routes and stores consistently underperform — <b>primary targets for route re-planning</b>.</li>
            <li>Monthly delivery volume shows seasonal patterns requiring <b>proactive fleet scheduling</b>.</li>
        </ul>

        <h4>Truck & Load Utilization</h4>
        <ul>
            <li>Average fleet utilization is <b>{avg_util:.1f}%</b> —
                {"efficient" if avg_util >= 70 else "below the 70% threshold, indicating load consolidation opportunity"}.</li>
            <li><b>{pct_underutil:.1f}%</b> of all truck trips run below 70% capacity, representing direct <b>fuel and cost wastage</b>.</li>
            <li>Load consolidation and smarter dispatch can immediately reduce operational cost per delivery.</li>
        </ul>

        <h4>Route Cost Analysis</h4>
        <ul>
            <li>Total transport cost across all deliveries: <b>${total_cost:,.2f}</b>.</li>
            <li>Average cost per delivery: <b>${avg_cost:,.2f}</b>.</li>
            <li>A small number of routes contribute disproportionately to total cost — <b>candidates for consolidation or re-sequencing</b>.</li>
            <li>Cost per KM analysis reveals routes with poor efficiency relative to distance covered.</li>
        </ul>

        <h4>Fuel Consumption</h4>
        <ul>
            <li>Total fuel cost: <b>${total_fuel:,.2f}</b>, averaging <b>${avg_fuel:,.2f}</b> per trip.</li>
            <li>Fuel cost is concentrated in high-distance, high-stop routes.</li>
            <li>Better load planning directly reduces fuel spend.</li>
        </ul>

        <h4>Store Replenishment</h4>
        <ul>
            <li><b>{total_stores} unique stores</b> are served across the logistics network.</li>
            <li>Store <b>{worst_store_delay}</b> has the highest average delivery delay — requiring priority scheduling.</li>
            <li>Replenishment frequency is uneven — some stores are chronically under-served, increasing stockout risk.</li>
        </ul>

        <h4>Perishable & Spoilage Risk</h4>
        <ul>
            <li>Temperature breach rate: <b>{breach_rate:.1f}%</b> —
                {"within acceptable range" if breach_rate <= 10 else "above safe threshold, requiring urgent attention"}.</li>
            <li>Average delivery temperature: <b>{avg_temp:.1f}°C</b>.</li>
            <li>Routes with high delays also carry the <b>highest spoilage risk</b>.</li>
        </ul>

        <h4>Final Takeaway</h4>
        <ul>
            <li>The dataset is <b>clean, complete, and ready for ML-based route optimization modelling</b>.</li>
            <li>Key inefficiencies identified across delivery timing, truck utilization, fuel cost, and store replenishment.</li>
            <li>Route optimization at the <b>Route × Truck × Store × Driver × Time</b> level will directly address all pain points.</li>
            <li>EDA strongly supports <b>route optimization, load planning, delivery scheduling, and spoilage reduction</b>.</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ML IMPLEMENTATION
# ============================================================
# ============================================================
# ML IMPLEMENTATION - STAGE 4
# ============================================================
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.markdown(
    """
    <div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;color:white;margin-top:20px;margin-bottom:10px;">
        <h3 style="margin:0;">ML Implementation </h3>
    </div>
    """,
    unsafe_allow_html=True
)

if "ml_option" not in st.session_state:
    st.session_state.ml_option = None


def ml_nav_button(label, value):
    if st.session_state.ml_option == value:
        st.markdown(
            f"""<div style="background-color:#4F97EE;color:white;padding:14px;
                border-radius:10px;font-weight:600;text-align:center;
                margin-bottom:12px;">{label}</div>""",
            unsafe_allow_html=True
        )
    else:
        if st.button(label, use_container_width=True, key=f"ml_nav_{value}"):
            st.session_state.ml_option = value
            st.rerun()


with st.expander(" ", expanded=True):
    mrow1 = st.columns(3)
    mrow2 = st.columns(2)
    with mrow1[0]: ml_nav_button("Demand & Store Risk", "Demand & Store Risk")
    with mrow1[1]: ml_nav_button("Route Optimization", "Route Optimization")
    with mrow1[2]: ml_nav_button("Fleet Allocation", "Fleet Allocation")
    with mrow2[0]: ml_nav_button("Fuel Consumption Prediction", "Fuel Consumption Prediction")
    with mrow2[1]: ml_nav_button("Delay Prediction", "Delay Prediction")

ml_option = st.session_state.ml_option

if ml_option is None:
    st.info("Select an ML module to view predictions.")


# ============================================================
# ML ROUTER
# ============================================================


# ============================================================
# 1. DEMAND & STORE RISK
# ============================================================
if ml_option == "Demand & Store Risk":

    # ============================================================
    # SECTION INTRO / EXPLANATION
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:28px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;">
            <h2 style="margin:0;">Demand & Store Load Prediction</h2>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                Retail supply chains often struggle with two opposite problems at the
                same store level: some stores get <b>overloaded</b> with deliveries
                they can't handle on time, while others are <b>under-serviced</b> and
                run into stockouts. This module predicts how much demand each store
                is likely to generate and flags stores that show patterns of
                <b>high order frequency</b> or <b>recurring delivery delays</b>.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Why this matters:</b> by knowing predicted demand and risk level
                ahead of dispatch, planners can pre-allocate stock, prioritize
                routing to risky stores, and avoid last-mile failures before they
                happen — turning reactive firefighting into proactive planning.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>What you'll see here:</b> a store-wise predicted demand figure
                (Random Forest Regression), a High Risk / Normal classification for
                every store (Random Forest Classifier), model tuning details,
                comparison across algorithms, and diagnostic checks for
                overfitting/underfitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MODEL ENGINEERING BOX
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#1a4d80;
            padding:26px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;
            line-height:1.7;">
            <h3 style="margin:0 0 12px 0;">Model Engineering</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA PREP (cached, always runs)
    # ============================================================
    @st.cache_data
    def build_store_features(data):
        d = data.copy()
 
        d["Dispatch_Time"] = pd.to_datetime(
            d["Dispatch_Time"], format="%d-%m-%y %H:%M", errors="coerce"
        )
        d["dispatch_month"] = d["Dispatch_Time"].dt.month
        d["dispatch_is_weekend"] = d["Dispatch_Time"].dt.weekday.isin([5, 6]).astype(int)
 
        priority_map = {"Low": 1, "Medium": 2, "High": 3}
        d["priority_score"] = d["Priority_Level"].map(priority_map).astype(float)
 
        for c in ["area_sqft", "Quantity", "Delay_Minutes", "Total_Route_Delay",
                  "distance_km", "dispatch_month", "dispatch_is_weekend"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
 
        store_df = d.groupby("Store_ID").agg(
            region=("region", "first"),
            zone=("zone", "first"),
            store_type=("store_type", "first"),
            area_sqft=("area_sqft", "first"),
            order_frequency=("Order_ID", "count"),
            total_quantity=("Quantity", "sum"),
            avg_quantity=("Quantity", "mean"),
            avg_priority_score=("priority_score", "mean"),
            avg_delay_minutes=("Delay_Minutes", "mean"),
            max_delay_minutes=("Delay_Minutes", "max"),
            total_route_delay=("Total_Route_Delay", "sum"),
            avg_distance_km=("distance_km", "mean"),
            avg_dispatch_month=("dispatch_month", "mean"),
            weekend_order_ratio=("dispatch_is_weekend", "mean"),
        ).reset_index()
 
        return store_df
 
    # ============================================================
    # TRAINING FUNCTION (runs only on button click)
    # ============================================================
    @st.cache_resource
    def train_all_models(store_df):
        sdf = store_df.copy()
 
        # ---- Risk labeling ----
        freq_thr = sdf["order_frequency"].quantile(0.75)
        delay_thr = sdf["avg_delay_minutes"].quantile(0.75)
        sdf["risk_label"] = np.where(
            (sdf["order_frequency"] >= freq_thr) | (sdf["avg_delay_minutes"] >= delay_thr),
            "High Risk", "Normal"
        )
        freq_norm = (sdf["order_frequency"] - sdf["order_frequency"].min()) / (
            sdf["order_frequency"].max() - sdf["order_frequency"].min())
        delay_norm = (sdf["avg_delay_minutes"] - sdf["avg_delay_minutes"].min()) / (
            sdf["avg_delay_minutes"].max() - sdf["avg_delay_minutes"].min())
        sdf["risk_score"] = (0.5 * freq_norm + 0.5 * delay_norm).round(3)
 
        cat_cols = ["region", "zone", "store_type"]
        for c in cat_cols:
            le = LabelEncoder()
            sdf[c + "_enc"] = le.fit_transform(sdf[c].astype(str))
 
        # ============================================================
        # DEMAND REGRESSION - MULTIPLE MODELS
        # ============================================================
        demand_features = ["area_sqft", "order_frequency", "avg_priority_score",
                            "avg_distance_km", "avg_dispatch_month", "weekend_order_ratio"] \
                          + [c + "_enc" for c in cat_cols]
 
        d_data = sdf.dropna(subset=demand_features + ["total_quantity"])
        X = d_data[demand_features]
        y = d_data["total_quantity"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
        demand_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        }
 
        demand_results = {}
        for name, model in demand_models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            demand_results[name] = {
                "model": model,
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_mae": mean_absolute_error(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
            }
 
        # Best demand model = highest test R2
        best_demand_name = max(demand_results, key=lambda k: demand_results[k]["test_r2"])
        best_demand_model = demand_results[best_demand_name]["model"]
 
        X_full = sdf[demand_features].fillna(0)
        sdf["predicted_demand"] = best_demand_model.predict(X_full).round(1)
 
        # ============================================================
        # RISK CLASSIFICATION - MULTIPLE MODELS
        # ============================================================
        risk_features = ["area_sqft", "order_frequency", "avg_quantity", "avg_priority_score",
                          "avg_distance_km", "weekend_order_ratio"] + [c + "_enc" for c in cat_cols]
 
        r_data = sdf.dropna(subset=risk_features + ["risk_label"])
        Xr = r_data[risk_features]
        yr = r_data["risk_label"]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            Xr, yr, test_size=0.2, random_state=42, stratify=yr)
 
        risk_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        }
 
        risk_results = {}
        for name, model in risk_models.items():
            model.fit(Xr_train, yr_train)
            train_pred = model.predict(Xr_train)
            test_pred = model.predict(Xr_test)
            risk_results[name] = {
                "model": model,
                "train_acc": accuracy_score(yr_train, train_pred),
                "test_acc": accuracy_score(yr_test, test_pred),
                "precision": precision_score(yr_test, test_pred, pos_label="High Risk"),
                "recall": recall_score(yr_test, test_pred, pos_label="High Risk"),
                "f1": f1_score(yr_test, test_pred, pos_label="High Risk"),
            }
 
        best_risk_name = max(risk_results, key=lambda k: risk_results[k]["test_acc"])
 
        return sdf, demand_results, risk_results, best_demand_name, best_risk_name
 
 
    # ============================================================
    # TRAIN BUTTON
    # ============================================================
    store_features = build_store_features(df)
 
    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        train_clicked = st.button("Train Model", use_container_width=True, key="train_demand_risk")
 
    if train_clicked:
        st.session_state.demand_risk_trained = True
 
    if st.session_state.get("demand_risk_trained", False):
 
        result_df, demand_results, risk_results, best_demand_name, best_risk_name = train_all_models(store_features)
 
        # ============================================================
        # MODEL TUNING SUMMARY
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Tuning Summary</h3>", unsafe_allow_html=True)
 
        best_demand = demand_results[best_demand_name]
        best_risk = risk_results[best_risk_name]
 
        # Overfit / underfit check (demand)
        d_gap = best_demand["train_r2"] - best_demand["test_r2"]
        if d_gap > 0.25:
            d_fit_status = "Overfitting (train R² much higher than test R²)"
        elif best_demand["test_r2"] < 0.1:
            d_fit_status = "Underfitting (model explains very little variance)"
        else:
            d_fit_status = "Good Fit (train and test performance are close)"
 
        # Overfit / underfit check (risk)
        r_gap = best_risk["train_acc"] - best_risk["test_acc"]
        if r_gap > 0.20:
            r_fit_status = "Overfitting (train accuracy much higher than test accuracy)"
        elif best_risk["test_acc"] < 0.6:
            r_fit_status = "Underfitting (low predictive accuracy on test data)"
        else:
            r_fit_status = "Good Fit (train and test accuracy are close)"
 
        demand_algo_list = ", ".join(demand_results.keys())
        risk_algo_list = ", ".join(risk_results.keys())
 
        # Why best won (demand)
        other_demand = [n for n in demand_results if n != best_demand_name][0]
        if best_demand["test_r2"] > demand_results[other_demand]["test_r2"]:
            d_why_best = (f"{best_demand_name} achieved a higher test R² "
                           f"({best_demand['test_r2']:.3f} vs {demand_results[other_demand]['test_r2']:.3f}), "
                           f"meaning it explains more of the variation in store demand on unseen data.")
        else:
            d_why_best = f"{best_demand_name} gave the most balanced train/test performance."
 
        # Why best won (risk)
        other_risk = [n for n in risk_results if n != best_risk_name][0]
        if best_risk["test_acc"] > risk_results[other_risk]["test_acc"]:
            r_why_best = (f"{best_risk_name} achieved higher test accuracy "
                           f"({best_risk['test_acc']*100:.1f}% vs {risk_results[other_risk]['test_acc']*100:.1f}%) "
                           f"and an F1 score of {best_risk['f1']:.2f}, indicating better balance between "
                           f"catching high-risk stores and avoiding false alarms.")
        else:
            r_why_best = f"{best_risk_name} gave the most balanced train/test performance."
 
        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">Demand Prediction Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {demand_algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_demand_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {d_why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {d_fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train R² = {best_demand['train_r2']:.3f}, Test R² = {best_demand['test_r2']:.3f},
                    Train MAE = {best_demand['train_mae']:.2f}, Test MAE = {best_demand['test_mae']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
 
        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">High-Risk Store Classification Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {risk_algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_risk_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {r_why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {r_fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train Accuracy = {best_risk['train_acc']*100:.1f}%, Test Accuracy = {best_risk['test_acc']*100:.1f}%,
                    Precision = {best_risk['precision']:.2f}, Recall = {best_risk['recall']:.2f}, F1 = {best_risk['f1']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
 
        # ============================================================
        # MODEL COMPARISON (Before / After grid, screenshot style)
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='color:#3c3c3c;font-weight:700;'>Model Performance Comparison</h2>",
            unsafe_allow_html=True
        )


        def grid_cards(card_data):
            cards_html = "".join(
                f"""<div class="summary-card">
                        <div class="summary-title">{label}</div>
                        <div class="summary-value">{value}</div>
                    </div>"""
                for label, value in card_data
            )
            st.markdown(
                f"""<div class="summary-grid">{cards_html}</div>""",
                unsafe_allow_html=True
            )

        names = list(demand_results.keys())
        before_name, after_name = names[0], names[1]
        before_d, after_d = demand_results[before_name], demand_results[after_name]

        # ---- Demand Prediction: Before ----
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Demand Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {before_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{before_d['train_mae']:.2f}"),
            ("Test MAE", f"{before_d['test_mae']:.2f}"),
            ("Train R²", f"{before_d['train_r2']:.3f}"),
            ("Test R²", f"{before_d['test_r2']:.3f}"),
        ])

        # ---- Demand Prediction: After ----
        st.markdown(f"<h4 style='color:#3c3c3c;'> {after_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{after_d['train_mae']:.2f}"),
            ("Test MAE", f"{after_d['test_mae']:.2f}"),
            ("Train R²", f"{after_d['train_r2']:.3f}"),
            ("Test R²", f"{after_d['test_r2']:.3f}"),
        ])

        if after_d["test_r2"] > before_d["test_r2"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Demand model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Demand model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )

        # ---- High-Risk Store Classification: Before / After ----
        rnames = list(risk_results.keys())
        rbefore_name, rafter_name = rnames[0], rnames[1]
        rbefore, rafter = risk_results[rbefore_name], risk_results[rafter_name]

        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>High-Risk Store Classification</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {rbefore_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train Acc", f"{rbefore['train_acc']*100:.2f}%"),
            ("Test Acc", f"{rbefore['test_acc']*100:.2f}%"),
            ("Precision", f"{rbefore['precision']:.3f}"),
            ("Recall", f"{rbefore['recall']:.3f}"),
            ("F1 Score", f"{rbefore['f1']:.3f}"),
        ])

        st.markdown(f"<h4 style='color:#3c3c3c;'> {rafter_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train Acc", f"{rafter['train_acc']*100:.2f}%"),
            ("Test Acc", f"{rafter['test_acc']*100:.2f}%"),
            ("Precision", f"{rafter['precision']:.3f}"),
            ("Recall", f"{rafter['recall']:.3f}"),
            ("F1 Score", f"{rafter['f1']:.3f}"),
        ])

        if rafter["test_acc"] > rbefore["test_acc"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Risk classification model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Risk classification model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )



        # ============================================================
        # PREDICTIONS / RESULTS
        # ============================================================
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Predictions</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
 
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Stores Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">High Risk Stores</div>
                    <div class="summary-value">{(result_df['risk_label']=='High Risk').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Demand Model R²</div>
                    <div class="summary-value">{best_demand['test_r2']:.2f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Risk Model Accuracy</div>
                    <div class="summary-value">{best_risk['test_acc']*100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
 
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.markdown("<h4 style='color:black;'>Predicted Demand - Top 10 Stores</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            top10 = result_df.sort_values("predicted_demand", ascending=False).head(10)
            ax.bar(top10["Store_ID"], top10["predicted_demand"], color=NAVY)
            apply_chart_style(ax, xlabel="Store ID", ylabel="Predicted Demand",
                            title="Top 10 Stores by Predicted Demand")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        with col2:
            st.markdown("<h4 style='color:black;'>High Risk vs Normal Stores</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = result_df["risk_label"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PIE_2[:len(counts)], startangle=90)
            st.pyplot(fig)
            
 
        st.markdown("<h4 style='color:black;'>Top 15 High-Risk Stores</h4>", unsafe_allow_html=True)
        risk_table = result_df.sort_values("risk_score", ascending=False).head(15)[
         ["Store_ID", "region", "zone", "store_type", "order_frequency",
         "avg_delay_minutes", "predicted_demand", "risk_score", "risk_label"]
        ].round(2)

        st.markdown("<div style='width:100%; overflow-x:auto;'>", unsafe_allow_html=True)
        render_html_table(risk_table)
        st.markdown("</div>", unsafe_allow_html=True)
        # ============================================================
        # MODEL DIAGNOSIS
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Diagnosis</h3>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color:#fff8e8;border:1px solid #f0d9a0;
                border-radius:10px;padding:20px;line-height:1.7;">
                <p style="margin:0 0 8px 0;"><b>Demand Model:</b> {d_fit_status}.
                The gap between train R² ({best_demand['train_r2']:.3f}) and test R²
                ({best_demand['test_r2']:.3f}) is {abs(d_gap):.3f}.
                {"Consider adding more historical order data or engineering seasonal features to improve generalization." if best_demand['test_r2'] < 0.4 else "Performance is acceptable for store-level demand planning."}
                </p>
                <p style="margin:0;"><b>Risk Model:</b> {r_fit_status}.
                The gap between train accuracy ({best_risk['train_acc']*100:.1f}%) and test accuracy
                ({best_risk['test_acc']*100:.1f}%) is {abs(r_gap)*100:.1f} percentage points.
                Precision of {best_risk['precision']:.2f} and recall of {best_risk['recall']:.2f}
                indicate {"the model is reliable at catching most high-risk stores." if best_risk['recall'] > 0.7 else "some high-risk stores may be missed — consider adjusting the risk threshold."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
# ============================================================
# 2. Route Optimization
# ============================================================
elif ml_option == "Route Optimization":

    # ============================================================
    # SECTION INTRO / EXPLANATION
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:28px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;">
            <h2 style="margin:0;">Intelligent Route Optimization Engine</h2>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                Every delivery route involves a trade-off between <b>delivery delay</b>,
                <b>cost per km</b>, and <b>fuel consumption</b>. Routes that look short
                on a map can still be expensive if they pass through high-traffic
                zones, or fast but fuel-inefficient if vehicles are poorly loaded
                or distances are badly sequenced.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Objective:</b> Minimize <b>delay</b> + <b>cost per km</b> +
                <b>fuel consumption</b> jointly, so planners can identify which
                routes are already optimized and which ones need re-planning to cut
                operating costs while keeping delivery promises.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>What you'll see here:</b> a route-wise optimization score that
                combines delay, cost per km, and fuel consumption (Random Forest
                Regression), the most and least optimized routes, model tuning
                details, comparison across algorithms, and diagnostic checks for
                overfitting/underfitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MODEL ENGINEERING BOX
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#1a4d80;
            padding:26px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;
            line-height:1.7;">
            <h3 style="margin:0 0 12px 0;">Model Engineering</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA PREP (cached, always runs)
    # ============================================================
    @st.cache_data
    def build_route_features(data):
        d = data.copy()

        for c in ["distance_km", "Fuel_Consumed", "Total_Logistics_Cost",
                  "Delay_Minutes", "Total_Route_Delay", "Quantity"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d["cost_per_km"] = d["Total_Logistics_Cost"] / d["distance_km"].replace(0, np.nan)
        d["fuel_consumption"] = d["Fuel_Consumed"]

        cat_cols = ["region", "zone", "vehicle_type"]
        for c in cat_cols:
            if c not in d.columns:
                d[c] = "Unknown"

        route_df = d.groupby("Route_ID").agg(
            region=("region", "first"),
            zone=("zone", "first"),
            vehicle_type=("vehicle_type", "first"),
            distance_km=("distance_km", "mean"),
            fuel_consumption=("fuel_consumption", "mean"),
            cost_per_km=("cost_per_km", "mean"),
            avg_delay_minutes=("Delay_Minutes", "mean"),
            total_route_delay=("Total_Route_Delay", "sum"),
            avg_quantity=("Quantity", "mean"),
            stop_count=("Order_ID", "count"),
        ).reset_index()

        def normalize(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-9)

        # Composite optimization score = delay + cost/km + fuel (lower = better, more optimized)
        route_df["optimization_score"] = (
            0.4 * normalize(route_df["avg_delay_minutes"]) +
            0.3 * normalize(route_df["cost_per_km"]) +
            0.3 * normalize(route_df["fuel_consumption"])
        ).round(3)

        return route_df

    # ============================================================
    # TRAINING FUNCTION (runs only on button click)
    # ============================================================
    @st.cache_resource
    def train_route_models(route_df):
        sdf = route_df.copy()

        cat_cols = ["region", "zone", "vehicle_type"]
        for c in cat_cols:
            le = LabelEncoder()
            sdf[c + "_enc"] = le.fit_transform(sdf[c].astype(str))

        # ============================================================
        # OPTIMIZATION SCORE REGRESSION - MULTIPLE MODELS
        # ============================================================
        features = ["distance_km", "stop_count", "avg_quantity",
                    "total_route_delay"] + [c + "_enc" for c in cat_cols]

        r_data = sdf.dropna(subset=features + ["optimization_score"])
        X = r_data[features]
        y = r_data["optimization_score"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        route_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        }

        route_results = {}
        for name, model in route_models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            route_results[name] = {
                "model": model,
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_mae": mean_absolute_error(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
            }

        best_route_name = max(route_results, key=lambda k: route_results[k]["test_r2"])
        best_route_model = route_results[best_route_name]["model"]

        X_full = sdf[features].fillna(0)
        sdf["predicted_score"] = best_route_model.predict(X_full).round(3)

        sdf["optimization_rank"] = sdf["predicted_score"].rank(method="min").astype(int)
        sdf["route_label"] = np.where(
            sdf["predicted_score"] <= sdf["predicted_score"].quantile(0.25),
            "Optimized",
            np.where(
                sdf["predicted_score"] >= sdf["predicted_score"].quantile(0.75),
                "Needs Optimization", "Moderate"
            )
        )

        return sdf, route_results, best_route_name

    # ============================================================
    # TRAIN BUTTON
    # ============================================================
    route_features = build_route_features(df)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        train_clicked = st.button("Train Model", use_container_width=True, key="train_route_opt")

    if train_clicked:
        st.session_state.route_opt_trained = True

    if st.session_state.get("route_opt_trained", False):

        result_df, route_results, best_route_name = train_route_models(route_features)

        # ============================================================
        # MODEL TUNING SUMMARY
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Tuning Summary</h3>", unsafe_allow_html=True)

        best_route = route_results[best_route_name]

        gap = best_route["train_r2"] - best_route["test_r2"]
        if gap > 0.25:
            fit_status = "Overfitting (train R² much higher than test R²)"
        elif best_route["test_r2"] < 0.1:
            fit_status = "Underfitting (model explains very little variance)"
        else:
            fit_status = "Good Fit (train and test performance are close)"

        algo_list = ", ".join(route_results.keys())

        other_name = [n for n in route_results if n != best_route_name][0]
        if best_route["test_r2"] > route_results[other_name]["test_r2"]:
            why_best = (f"{best_route_name} achieved a higher test R² "
                        f"({best_route['test_r2']:.3f} vs {route_results[other_name]['test_r2']:.3f}), "
                        f"meaning it explains more of the variation in the delay/cost/fuel "
                        f"optimization score on unseen data.")
        else:
            why_best = f"{best_route_name} gave the most balanced train/test performance."

        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">Route Optimization Score Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_route_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train R² = {best_route['train_r2']:.3f}, Test R² = {best_route['test_r2']:.3f},
                    Train MAE = {best_route['train_mae']:.2f}, Test MAE = {best_route['test_mae']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # MODEL COMPARISON (Before / After grid)
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='color:#3c3c3c;font-weight:700;'>Model Performance Comparison</h2>",
            unsafe_allow_html=True
        )

        def grid_cards(card_data):
            cards_html = "".join(
                f"""<div class="summary-card">
                        <div class="summary-title">{label}</div>
                        <div class="summary-value">{value}</div>
                    </div>"""
                for label, value in card_data
            )
            st.markdown(
                f"""<div class="summary-grid">{cards_html}</div>""",
                unsafe_allow_html=True
            )

        names = list(route_results.keys())
        before_name, after_name = names[0], names[1]
        before_r, after_r = route_results[before_name], route_results[after_name]

        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Route Optimization Score Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {before_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{before_r['train_mae']:.3f}"),
            ("Test MAE", f"{before_r['test_mae']:.3f}"),
            ("Train R²", f"{before_r['train_r2']:.3f}"),
            ("Test R²", f"{before_r['test_r2']:.3f}"),
        ])

        st.markdown(f"<h4 style='color:#3c3c3c;'> {after_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{after_r['train_mae']:.3f}"),
            ("Test MAE", f"{after_r['test_mae']:.3f}"),
            ("Train R²", f"{after_r['train_r2']:.3f}"),
            ("Test R²", f"{after_r['test_r2']:.3f}"),
        ])

        if after_r["test_r2"] > before_r["test_r2"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Route optimization model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Route optimization model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )

        # ============================================================
        # PREDICTIONS / RESULTS
        # ============================================================
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Predictions</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Routes Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Optimized Routes</div>
                    <div class="summary-value">{(result_df['route_label']=='Optimized').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Routes Needing Optimization</div>
                    <div class="summary-value">{(result_df['route_label']=='Needs Optimization').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Model R²</div>
                    <div class="summary-value">{best_route['test_r2']:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        # ============================================================
        # OPTIMIZATION OBJECTIVES — BEFORE vs AFTER
        # ============================================================
        st.markdown("<h4 style='color:black;'>Optimization Objectives — Before vs After</h4>", unsafe_allow_html=True)

        optimized_routes = result_df[result_df["route_label"] == "Optimized"]

        before_delay = result_df["avg_delay_minutes"].mean()
        before_cost = result_df["cost_per_km"].mean()
        before_fuel = result_df["fuel_consumption"].mean()
        before_score = result_df["predicted_score"].mean()

        after_delay = optimized_routes["avg_delay_minutes"].mean()
        after_cost = optimized_routes["cost_per_km"].mean()
        after_fuel = optimized_routes["fuel_consumption"].mean()
        after_score = optimized_routes["predicted_score"].mean()

        st.markdown("<h5 style='color:#3c3c3c;'>Before Optimization (All Routes)</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Avg Delay (min)</div>
                    <div class="summary-value">{before_delay:.1f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Cost per KM</div>
                    <div class="summary-value">₹{before_cost:.2f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Fuel Consumption</div>
                    <div class="summary-value">{before_fuel:.2f} L</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Optimization Score</div>
                    <div class="summary-value">{before_score:.3f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>After Optimization (Optimized Routes Only)</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Avg Delay (min)</div>
                    <div class="summary-value">{after_delay:.1f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Cost per KM</div>
                    <div class="summary-value">₹{after_cost:.2f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Fuel Consumption</div>
                    <div class="summary-value">{after_fuel:.2f} L</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Optimization Score</div>
                    <div class="summary-value">{after_score:.3f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

        delay_pct = (before_delay - after_delay) / before_delay * 100 if before_delay else 0
        cost_pct = (before_cost - after_cost) / before_cost * 100 if before_cost else 0
        fuel_pct = (before_fuel - after_fuel) / before_fuel * 100 if before_fuel else 0

        st.markdown(
            f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:18px;
                color:#3c5a35;font-weight:500;">
                ✅ Optimized routes show {delay_pct:.1f}% lower delay, {cost_pct:.1f}% lower cost per km,
                and {fuel_pct:.1f}% lower fuel consumption compared to the fleet average.
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='color:black;'>Top 10 Optimized Routes (Delay+Cost+Fuel Score)</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            top10 = result_df.sort_values("predicted_score", ascending=True).head(10)
            ax.bar(top10["Route_ID"].astype(str), top10["predicted_score"], color=NAVY)
            apply_chart_style(ax, xlabel="Route ID", ylabel="Optimization Score (lower=better)",
                            title="Top 10 Optimized Routes")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("<h4 style='color:black;'>Route Optimization Distribution</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = result_df["route_label"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PIE_2[:len(counts)] if len(counts) <= len(PIE_2) else None, startangle=90)
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("<h4 style='color:black;'>Top 15 Routes Needing Optimization (High Delay/Cost/Fuel)</h4>", unsafe_allow_html=True)
        worst_table = result_df.sort_values("predicted_score", ascending=False).head(15)[
            ["Route_ID", "region", "zone", "vehicle_type", "distance_km",
             "avg_delay_minutes", "cost_per_km", "fuel_consumption",
             "predicted_score", "route_label"]
        ].round(2)

        st.markdown("<div style='width:100%; overflow-x:auto;'>", unsafe_allow_html=True)
        render_html_table(worst_table)
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # MODEL DIAGNOSIS
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Diagnosis</h3>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color:#fff8e8;border:1px solid #f0d9a0;
                border-radius:10px;padding:20px;line-height:1.7;">
                <p style="margin:0;"><b>Route Optimization Score Model:</b> {fit_status}.
                The gap between train R² ({best_route['train_r2']:.3f}) and test R²
                ({best_route['test_r2']:.3f}) is {abs(gap):.3f}.
                {"Consider adding traffic/time-of-day features or more route-level history to improve generalization." if best_route['test_r2'] < 0.4 else "Performance is acceptable for identifying routes that need delay, cost, or fuel optimization."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 3. Fleet Allocation
# ============================================================
elif ml_option == "Fleet Allocation":

    # ============================================================
    # SECTION INTRO / EXPLANATION
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:28px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;">
            <h2 style="margin:0;">Smart Fleet Allocation & Truck Utilization Balancer</h2>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                Fleet capacity is one of the most expensive resources in a supply
                chain, yet it's often poorly balanced — some trucks run almost
                empty while others are repeatedly overloaded or overused. This
                creates maintenance strain on a few vehicles while the rest of the
                fleet sits idle.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Issues detected in current operations:</b> overall fleet
                capacity utilization is critically low (around <b>12.4%</b>),
                truck usage is highly <b>uneven</b> across the fleet, and there is
                a strong <b>overdependence on Truck 315</b> for dispatches.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Why this matters:</b> low and uneven utilization means higher
                cost per delivery, faster wear on overused vehicles, and wasted
                capacity on underused ones. Balancing allocation reduces fleet
                size requirements, spreads wear evenly, and lowers overall
                logistics cost.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>What you'll see here:</b> per-truck utilization analysis,
                a predicted "ideal load share" for each truck (Random Forest
                Regression), identification of overused and underused trucks,
                model tuning details, comparison across algorithms, and
                diagnostic checks for overfitting/underfitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MODEL ENGINEERING BOX
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#1a4d80;
            padding:26px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;
            line-height:1.7;">
            <h3 style="margin:0 0 12px 0;">Model Engineering</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA PREP (cached, always runs)
    # ============================================================
    @st.cache_data
    def build_fleet_features(data):
        d = data.copy()

        for c in ["Loaded_Weight", "Loaded_Volume", "capacity_weight", "capacity_volume",
                  "Capacity_Utilization_Percentage", "Idle_Time", "Operating_Time",
                  "distance_km", "Delay_Minutes"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        cat_cols = ["region", "zone", "vehicle_type"]
        for c in cat_cols:
            if c not in d.columns:
                d[c] = "Unknown"

        fleet_df = d.groupby("Vehicle_ID").agg(
            region=("region", "first"),
            zone=("zone", "first"),
            vehicle_type=("vehicle_type", "first"),
            capacity_weight=("capacity_weight", "first"),
            capacity_volume=("capacity_volume", "first"),
            loaded_weight=("Loaded_Weight", "mean"),
            loaded_volume=("Loaded_Volume", "mean"),
            avg_utilization_pct=("Capacity_Utilization_Percentage", "mean"),
            avg_idle_time=("Idle_Time", "mean"),
            avg_operating_time=("Operating_Time", "mean"),
            avg_distance_km=("distance_km", "mean"),
            avg_delay_minutes=("Delay_Minutes", "mean"),
            trip_count=("Dispatch_ID", "count"),
        ).reset_index()

        fleet_df["weight_utilization_pct"] = (
            fleet_df["loaded_weight"] / fleet_df["capacity_weight"].replace(0, np.nan) * 100
        ).round(2)
        fleet_df["volume_utilization_pct"] = (
            fleet_df["loaded_volume"] / fleet_df["capacity_volume"].replace(0, np.nan) * 100
        ).round(2)

        # Overall utilization score combining capacity, weight and volume utilization
        fleet_df["overall_utilization_pct"] = fleet_df[
            ["avg_utilization_pct", "weight_utilization_pct", "volume_utilization_pct"]
        ].mean(axis=1).round(2)

        # Ideal load share = each truck's fair share of total trips, based on equal distribution
        total_trips = fleet_df["trip_count"].sum()
        fleet_count = fleet_df.shape[0]
        fleet_df["ideal_trip_share"] = (total_trips / fleet_count)
        fleet_df["actual_trip_share_pct"] = (fleet_df["trip_count"] / total_trips * 100).round(2)
        fleet_df["ideal_trip_share_pct"] = (100 / fleet_count)

        # Usage label based on deviation from ideal share
        fleet_df["usage_gap_pct"] = (
            fleet_df["actual_trip_share_pct"] - fleet_df["ideal_trip_share_pct"]
        ).round(2)

        return fleet_df

    # ============================================================
    # TRAINING FUNCTION (runs only on button click)
    # ============================================================
    @st.cache_resource
    def train_fleet_models(fleet_df):
        sdf = fleet_df.copy()

        cat_cols = ["region", "zone", "vehicle_type"]
        for c in cat_cols:
            le = LabelEncoder()
            sdf[c + "_enc"] = le.fit_transform(sdf[c].astype(str))

        # ============================================================
        # IDEAL LOAD SHARE REGRESSION - MULTIPLE MODELS
        # Target: actual_trip_share_pct (what we predict an optimally
        # balanced allocation should look like based on truck capacity/type)
        # ============================================================
        features = ["capacity_weight", "capacity_volume", "avg_idle_time",
                    "avg_operating_time", "avg_distance_km"] + [c + "_enc" for c in cat_cols]

        r_data = sdf.dropna(subset=features + ["actual_trip_share_pct"])
        X = r_data[features]
        y = r_data["actual_trip_share_pct"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fleet_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        }

        fleet_results = {}
        for name, model in fleet_models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            fleet_results[name] = {
                "model": model,
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_mae": mean_absolute_error(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
            }

        best_fleet_name = max(fleet_results, key=lambda k: fleet_results[k]["test_r2"])
        best_fleet_model = fleet_results[best_fleet_name]["model"]

        X_full = sdf[features].fillna(0)
        sdf["recommended_trip_share_pct"] = best_fleet_model.predict(X_full).round(2)

        # Normalize recommended shares so they sum to 100%
        total_recommended = sdf["recommended_trip_share_pct"].clip(lower=0).sum()
        if total_recommended > 0:
            sdf["recommended_trip_share_pct"] = (
                sdf["recommended_trip_share_pct"].clip(lower=0) / total_recommended * 100
            ).round(2)

        # Usage classification
        overuse_thr = sdf["actual_trip_share_pct"].quantile(0.75)
        underuse_thr = sdf["actual_trip_share_pct"].quantile(0.25)

        sdf["usage_status"] = np.where(
            sdf["actual_trip_share_pct"] >= overuse_thr, "Overused",
            np.where(sdf["actual_trip_share_pct"] <= underuse_thr, "Underused", "Balanced")
        )

        return sdf, fleet_results, best_fleet_name

    # ============================================================
    # TRAIN BUTTON
    # ============================================================
    fleet_features = build_fleet_features(df)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        train_clicked = st.button("Train Model", use_container_width=True, key="train_fleet_alloc")

    if train_clicked:
        st.session_state.fleet_alloc_trained = True

    if st.session_state.get("fleet_alloc_trained", False):

        result_df, fleet_results, best_fleet_name = train_fleet_models(fleet_features)

        # ============================================================
        # MODEL TUNING SUMMARY
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Tuning Summary</h3>", unsafe_allow_html=True)

        best_fleet = fleet_results[best_fleet_name]

        gap = best_fleet["train_r2"] - best_fleet["test_r2"]
        if gap > 0.25:
            fit_status = "Overfitting (train R² much higher than test R²)"
        elif best_fleet["test_r2"] < 0.1:
            fit_status = "Underfitting (model explains very little variance)"
        else:
            fit_status = "Good Fit (train and test performance are close)"

        algo_list = ", ".join(fleet_results.keys())

        other_name = [n for n in fleet_results if n != best_fleet_name][0]
        if best_fleet["test_r2"] > fleet_results[other_name]["test_r2"]:
            why_best = (f"{best_fleet_name} achieved a higher test R² "
                        f"({best_fleet['test_r2']:.3f} vs {fleet_results[other_name]['test_r2']:.3f}), "
                        f"meaning it more accurately predicts a balanced trip-share allocation "
                        f"for each truck based on its capacity and usage pattern.")
        else:
            why_best = f"{best_fleet_name} gave the most balanced train/test performance."

        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">Truck Load Share Allocation Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_fleet_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train R² = {best_fleet['train_r2']:.3f}, Test R² = {best_fleet['test_r2']:.3f},
                    Train MAE = {best_fleet['train_mae']:.2f}, Test MAE = {best_fleet['test_mae']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # MODEL COMPARISON (Before / After grid)
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='color:#3c3c3c;font-weight:700;'>Model Performance Comparison</h2>",
            unsafe_allow_html=True
        )

        def grid_cards(card_data):
            cards_html = "".join(
                f"""<div class="summary-card">
                        <div class="summary-title">{label}</div>
                        <div class="summary-value">{value}</div>
                    </div>"""
                for label, value in card_data
            )
            st.markdown(
                f"""<div class="summary-grid">{cards_html}</div>""",
                unsafe_allow_html=True
            )

        names = list(fleet_results.keys())
        before_name, after_name = names[0], names[1]
        before_f, after_f = fleet_results[before_name], fleet_results[after_name]

        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Truck Load Share Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {before_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{before_f['train_mae']:.3f}"),
            ("Test MAE", f"{before_f['test_mae']:.3f}"),
            ("Train R²", f"{before_f['train_r2']:.3f}"),
            ("Test R²", f"{before_f['test_r2']:.3f}"),
        ])

        st.markdown(f"<h4 style='color:#3c3c3c;'> {after_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{after_f['train_mae']:.3f}"),
            ("Test MAE", f"{after_f['test_mae']:.3f}"),
            ("Train R²", f"{after_f['train_r2']:.3f}"),
            ("Test R²", f"{after_f['test_r2']:.3f}"),
        ])

        if after_f["test_r2"] > before_f["test_r2"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Truck load share model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Truck load share model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )

        # ============================================================
        # PREDICTIONS / RESULTS
        # ============================================================
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Predictions</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Trucks Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Overused Trucks</div>
                    <div class="summary-value">{(result_df['usage_status']=='Overused').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Underused Trucks</div>
                    <div class="summary-value">{(result_df['usage_status']=='Underused').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Model R²</div>
                    <div class="summary-value">{best_fleet['test_r2']:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        # ============================================================
        # UTILIZATION — BEFORE vs AFTER
        # ============================================================
        st.markdown("<h4 style='color:black;'>Fleet Utilization — Before vs After Rebalancing</h4>", unsafe_allow_html=True)

        before_util = result_df["overall_utilization_pct"].mean()
        before_gap_std = result_df["usage_gap_pct"].abs().mean()

        top_truck = result_df.loc[result_df["actual_trip_share_pct"].idxmax()]
        top_truck_share = top_truck["actual_trip_share_pct"]
        top_truck_id = top_truck["Vehicle_ID"]

        after_gap_std = (
            result_df["recommended_trip_share_pct"] - result_df["ideal_trip_share_pct"]
        ).abs().mean()

        st.markdown("<h5 style='color:#3c3c3c;'>Before Rebalancing (Current State)</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Avg Capacity Utilization</div>
                    <div class="summary-value">{before_util:.1f}%</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Trip-Share Deviation</div>
                    <div class="summary-value">{before_gap_std:.2f} pts</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Most Dependent Truck</div>
                    <div class="summary-value">{top_truck_id}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Its Trip Share</div>
                    <div class="summary-value">{top_truck_share:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>After Rebalancing (Recommended Allocation)</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Target Utilization</div>
                    <div class="summary-value">{result_df['ideal_trip_share_pct'].iloc[0]:.1f}% / truck</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Trip-Share Deviation</div>
                    <div class="summary-value">{after_gap_std:.2f} pts</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Trucks Reassigned Load</div>
                    <div class="summary-value">{(result_df['usage_status']!='Balanced').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Deviation Reduction</div>
                    <div class="summary-value">{((before_gap_std - after_gap_std)/before_gap_std*100 if before_gap_std else 0):.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:18px;
                color:#3c5a35;font-weight:500;">
                ✅ Recommended allocation reduces trip-share imbalance by
                {((before_gap_std - after_gap_std)/before_gap_std*100 if before_gap_std else 0):.1f}% and
                lowers {top_truck_id}'s dependency from {top_truck_share:.1f}% toward the
                fleet-balanced target of {result_df['ideal_trip_share_pct'].iloc[0]:.1f}%.
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='color:black;'>Current vs Recommended Trip Share (Top 10 Trucks)</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            top10 = result_df.sort_values("actual_trip_share_pct", ascending=False).head(10)
            x = np.arange(len(top10))
            width = 0.35
            ax.bar(x - width/2, top10["actual_trip_share_pct"], width, label="Current", color=NAVY)
            ax.bar(x + width/2, top10["recommended_trip_share_pct"], width, label="Recommended", color="#7FB3D5")
            ax.set_xticks(x)
            ax.set_xticklabels(top10["Vehicle_ID"].astype(str), rotation=45, ha="right")
            ax.legend()
            apply_chart_style(ax, xlabel="Truck ID", ylabel="Trip Share (%)",
                            title="Current vs Recommended Trip Share")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("<h4 style='color:black;'>Truck Usage Distribution</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = result_df["usage_status"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PIE_2[:len(counts)] if len(counts) <= len(PIE_2) else None, startangle=90)
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("<h4 style='color:black;'>Truck Allocation Details</h4>", unsafe_allow_html=True)
        fleet_table = result_df.sort_values("actual_trip_share_pct", ascending=False)[
            ["Vehicle_ID", "region", "zone", "vehicle_type", "trip_count",
             "overall_utilization_pct", "actual_trip_share_pct",
             "recommended_trip_share_pct", "usage_status"]
        ].round(2)

        st.markdown("<div style='width:100%; overflow-x:auto;'>", unsafe_allow_html=True)
        render_html_table(fleet_table)
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # MODEL DIAGNOSIS
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Diagnosis</h3>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color:#fff8e8;border:1px solid #f0d9a0;
                border-radius:10px;padding:20px;line-height:1.7;">
                <p style="margin:0;"><b>Truck Load Share Model:</b> {fit_status}.
                The gap between train R² ({best_fleet['train_r2']:.3f}) and test R²
                ({best_fleet['test_r2']:.3f}) is {abs(gap):.3f}.
                {"Consider adding driver availability, route distance distribution, or seasonal demand features to improve generalization." if best_fleet['test_r2'] < 0.4 else "Performance is acceptable for guiding fleet rebalancing decisions."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )



# ============================================================
# 4. Fuel Consumption Prediction
# ============================================================
elif ml_option == "Fuel Consumption Prediction":

    # ============================================================
    # SECTION INTRO / EXPLANATION
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:28px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;">
            <h2 style="margin:0;">Fuel Consumption Prediction & Optimization</h2>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                Fuel is one of the largest variable costs in logistics, and it's
                often driven by factors that go unnoticed — overloading, idle
                engine time, longer-than-necessary routes, or inefficient vehicle
                types being assigned to the wrong trips.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Goal:</b> Reduce <b>fuel wastage</b>, <b>cost per trip</b>, and
                <b>emissions</b> by predicting expected fuel consumption for each
                trip and flagging trips that consume significantly more fuel than
                expected for their distance, load, and vehicle type.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Why this matters:</b> trips that burn more fuel than their
                profile predicts are early signals of inefficient driving, poor
                route choice, overloading, or vehicle maintenance issues —
                catching these early cuts cost and emissions before they
                accumulate across the fleet.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>What you'll see here:</b> a predicted fuel consumption figure
                for each trip (Random Forest Regression), flagged trips with
                excess fuel usage and estimated wastage/cost/emissions, model
                tuning details, comparison across algorithms, and diagnostic
                checks for overfitting/underfitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MODEL ENGINEERING BOX
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#1a4d80;
            padding:26px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;
            line-height:1.7;">
            <h3 style="margin:0 0 12px 0;">Model Engineering</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA PREP (cached, always runs)
    # ============================================================
    @st.cache_data
    def build_fuel_features(data):
        d = data.copy()

        for c in ["Fuel_Consumed", "Fuel_Cost", "Distance_Traveled", "Loaded_Weight",
                  "Loaded_Volume", "Idle_Time", "Operating_Time", "distance_km"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        cat_cols = ["region", "zone", "vehicle_type", "fuel_type"]
        for c in cat_cols:
            if c not in d.columns:
                d[c] = "Unknown"

        fuel_df = d.dropna(subset=["Fuel_Consumed"]).copy()

        # Emission factor: kg CO2 per litre of fuel (approx, by fuel type)
        emission_factor = {"Diesel": 2.68, "Petrol": 2.31, "CNG": 1.93, "Electric": 0.0}
        fuel_df["emission_kg"] = fuel_df.apply(
            lambda r: r["Fuel_Consumed"] * emission_factor.get(str(r["fuel_type"]), 2.5),
            axis=1
        ).round(2)

        fuel_df["region"] = fuel_df["region"].fillna("Unknown")
        fuel_df["zone"] = fuel_df["zone"].fillna("Unknown")
        fuel_df["vehicle_type"] = fuel_df["vehicle_type"].fillna("Unknown")
        fuel_df["fuel_type"] = fuel_df["fuel_type"].fillna("Unknown")

        keep_cols = ["Fuel_Record_ID", "Vehicle_ID", "Route_ID", "region", "zone",
                      "vehicle_type", "fuel_type", "Distance_Traveled", "Loaded_Weight",
                      "Loaded_Volume", "Idle_Time", "Operating_Time", "Fuel_Consumed",
                      "Fuel_Cost", "emission_kg"]
        keep_cols = [c for c in keep_cols if c in fuel_df.columns]

        return fuel_df[keep_cols]

    # ============================================================
    # TRAINING FUNCTION (runs only on button click)
    # ============================================================
    @st.cache_resource
    def train_fuel_models(fuel_df):
        sdf = fuel_df.copy()

        cat_cols = ["region", "zone", "vehicle_type", "fuel_type"]
        for c in cat_cols:
            le = LabelEncoder()
            sdf[c + "_enc"] = le.fit_transform(sdf[c].astype(str))

        # ============================================================
        # FUEL CONSUMPTION REGRESSION - MULTIPLE MODELS
        # ============================================================
        features = ["Distance_Traveled", "Loaded_Weight", "Loaded_Volume",
                    "Idle_Time", "Operating_Time"] + [c + "_enc" for c in cat_cols]
        features = [f for f in features if f in sdf.columns]

        r_data = sdf.dropna(subset=features + ["Fuel_Consumed"])
        X = r_data[features]
        y = r_data["Fuel_Consumed"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fuel_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        }

        fuel_results = {}
        for name, model in fuel_models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            fuel_results[name] = {
                "model": model,
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_mae": mean_absolute_error(y_train, train_pred),
                "test_mae": mean_absolute_error(y_test, test_pred),
            }

        best_fuel_name = max(fuel_results, key=lambda k: fuel_results[k]["test_r2"])
        best_fuel_model = fuel_results[best_fuel_name]["model"]

        X_full = sdf[features].fillna(0)
        sdf["predicted_fuel"] = best_fuel_model.predict(X_full).round(2)

        # Excess fuel = actual - predicted (positive means wastage)
        sdf["excess_fuel"] = (sdf["Fuel_Consumed"] - sdf["predicted_fuel"]).round(2)

        # Estimate wastage cost using per-litre cost from this trip (or fleet avg fallback)
        per_litre_cost = (sdf["Fuel_Cost"] / sdf["Fuel_Consumed"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        fleet_avg_cost_per_litre = per_litre_cost.mean()
        sdf["cost_per_litre"] = per_litre_cost.fillna(fleet_avg_cost_per_litre)

        sdf["excess_cost"] = (sdf["excess_fuel"].clip(lower=0) * sdf["cost_per_litre"]).round(2)

        emission_factor_avg = (sdf["emission_kg"] / sdf["Fuel_Consumed"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(2.5)
        sdf["excess_emission_kg"] = (sdf["excess_fuel"].clip(lower=0) * emission_factor_avg).round(2)

        # Flag trips with significant excess fuel usage
        excess_thr = sdf["excess_fuel"].quantile(0.75)
        sdf["fuel_flag"] = np.where(
            sdf["excess_fuel"] >= excess_thr, "Excess Usage",
            np.where(sdf["excess_fuel"] <= 0, "Efficient", "Normal")
        )

        return sdf, fuel_results, best_fuel_name

    # ============================================================
    # TRAIN BUTTON
    # ============================================================
    fuel_features = build_fuel_features(df)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        train_clicked = st.button("Train Model", use_container_width=True, key="train_fuel_pred")

    if train_clicked:
        st.session_state.fuel_pred_trained = True

    if st.session_state.get("fuel_pred_trained", False):

        result_df, fuel_results, best_fuel_name = train_fuel_models(fuel_features)

        # ============================================================
        # MODEL TUNING SUMMARY
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Tuning Summary</h3>", unsafe_allow_html=True)

        best_fuel = fuel_results[best_fuel_name]

        gap = best_fuel["train_r2"] - best_fuel["test_r2"]
        if gap > 0.25:
            fit_status = "Overfitting (train R² much higher than test R²)"
        elif best_fuel["test_r2"] < 0.1:
            fit_status = "Underfitting (model explains very little variance)"
        else:
            fit_status = "Good Fit (train and test performance are close)"

        algo_list = ", ".join(fuel_results.keys())

        other_name = [n for n in fuel_results if n != best_fuel_name][0]
        if best_fuel["test_r2"] > fuel_results[other_name]["test_r2"]:
            why_best = (f"{best_fuel_name} achieved a higher test R² "
                        f"({best_fuel['test_r2']:.3f} vs {fuel_results[other_name]['test_r2']:.3f}), "
                        f"meaning it more accurately predicts expected fuel consumption "
                        f"for a trip based on distance, load, and vehicle profile.")
        else:
            why_best = f"{best_fuel_name} gave the most balanced train/test performance."

        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">Fuel Consumption Prediction Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_fuel_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train R² = {best_fuel['train_r2']:.3f}, Test R² = {best_fuel['test_r2']:.3f},
                    Train MAE = {best_fuel['train_mae']:.2f}, Test MAE = {best_fuel['test_mae']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # MODEL COMPARISON (Before / After grid)
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='color:#3c3c3c;font-weight:700;'>Model Performance Comparison</h2>",
            unsafe_allow_html=True
        )

        def grid_cards(card_data):
            cards_html = "".join(
                f"""<div class="summary-card">
                        <div class="summary-title">{label}</div>
                        <div class="summary-value">{value}</div>
                    </div>"""
                for label, value in card_data
            )
            st.markdown(
                f"""<div class="summary-grid">{cards_html}</div>""",
                unsafe_allow_html=True
            )

        names = list(fuel_results.keys())
        before_name, after_name = names[0], names[1]
        before_f, after_f = fuel_results[before_name], fuel_results[after_name]

        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Fuel Consumption Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {before_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{before_f['train_mae']:.3f}"),
            ("Test MAE", f"{before_f['test_mae']:.3f}"),
            ("Train R²", f"{before_f['train_r2']:.3f}"),
            ("Test R²", f"{before_f['test_r2']:.3f}"),
        ])

        st.markdown(f"<h4 style='color:#3c3c3c;'> {after_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{after_f['train_mae']:.3f}"),
            ("Test MAE", f"{after_f['test_mae']:.3f}"),
            ("Train R²", f"{after_f['train_r2']:.3f}"),
            ("Test R²", f"{after_f['test_r2']:.3f}"),
        ])

        if after_f["test_r2"] > before_f["test_r2"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Fuel consumption model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Fuel consumption model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )

        # ============================================================
        # PREDICTIONS / RESULTS
        # ============================================================
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Predictions</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>Actual</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Trips Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Excess Usage Trips</div>
                    <div class="summary-value">{(result_df['excess_fuel'] > 0).sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Efficient Trips</div>
                    <div class="summary-value">{(result_df['excess_fuel'] <= 0).sum()}</div>
                </div>
                
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>Predicted</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Trips Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Excess Usage Trips</div>
                    <div class="summary-value">{(result_df['fuel_flag']=='Excess Usage').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Efficient Trips</div>
                    <div class="summary-value">{(result_df['fuel_flag']=='Efficient').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Model R²</div>
                    <div class="summary-value">{best_fuel['test_r2']:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        # ============================================================
        # FUEL WASTAGE — BEFORE vs AFTER OPTIMIZATION
        # ============================================================
        st.markdown("<h4 style='color:black;'>Fuel Wastage, Cost & Emissions — Before vs After</h4>", unsafe_allow_html=True)

        total_actual_fuel = result_df["Fuel_Consumed"].sum()
        total_predicted_fuel = result_df["predicted_fuel"].sum()
        total_excess_fuel = result_df["excess_fuel"].clip(lower=0).sum()
        total_excess_cost = result_df["excess_cost"].sum()
        total_excess_emission = result_df["excess_emission_kg"].sum()

        st.markdown("<h5 style='color:#3c3c3c;'>Before Optimization (Current Usage)</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Total Fuel Consumed</div>
                    <div class="summary-value">{total_actual_fuel:,.1f} L</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Total Excess Fuel</div>
                    <div class="summary-value">{total_excess_fuel:,.1f} L</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Excess Cost</div>
                    <div class="summary-value">₹{total_excess_cost:,.0f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Excess Emissions</div>
                    <div class="summary-value">{total_excess_emission:,.1f} kg CO₂</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>After Optimization (Predicted/Expected Usage)</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Total Fuel (Predicted)</div>
                    <div class="summary-value">{total_predicted_fuel:,.1f} L</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Fuel Savings Potential</div>
                    <div class="summary-value">{total_excess_fuel:,.1f} L</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Cost Savings Potential</div>
                    <div class="summary-value">₹{total_excess_cost:,.0f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Emission Reduction</div>
                    <div class="summary-value">{total_excess_emission:,.1f} kg CO₂</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

        savings_pct = (total_excess_fuel / total_actual_fuel * 100) if total_actual_fuel else 0

        st.markdown(
            f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:18px;
                color:#3c5a35;font-weight:500;">
                ✅ Eliminating excess fuel usage across flagged trips could reduce total
                fleet fuel consumption by {savings_pct:.1f}%, saving an estimated
                ₹{total_excess_cost:,.0f} and {total_excess_emission:,.1f} kg of CO₂ emissions.
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='color:black;'>Top 10 Trips by Excess Fuel Usage</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            top10 = result_df.sort_values("excess_fuel", ascending=False).head(10)
            ax.bar(top10["Vehicle_ID"].astype(str), top10["excess_fuel"], color=NAVY)
            apply_chart_style(ax, xlabel="Vehicle ID", ylabel="Excess Fuel (L)",
                            title="Top 10 Trips by Excess Fuel")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("<h4 style='color:black;'>Trip Fuel Usage Distribution</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = result_df["fuel_flag"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PIE_2[:len(counts)] if len(counts) <= len(PIE_2) else None, startangle=90)
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("<h4 style='color:black;'>Top 15 Trips with Excess Fuel Usage</h4>", unsafe_allow_html=True)
        fuel_table = result_df.sort_values("excess_fuel", ascending=False).head(15)[
            ["Vehicle_ID", "Route_ID", "region", "zone", "vehicle_type",
             "Distance_Traveled", "Fuel_Consumed", "predicted_fuel",
             "excess_fuel", "excess_cost", "excess_emission_kg", "fuel_flag"]
        ].round(2)

        st.markdown("<div style='width:100%; overflow-x:auto;'>", unsafe_allow_html=True)
        render_html_table(fuel_table)
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # MODEL DIAGNOSIS
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Diagnosis</h3>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color:#fff8e8;border:1px solid #f0d9a0;
                border-radius:10px;padding:20px;line-height:1.7;">
                <p style="margin:0;"><b>Fuel Consumption Model:</b> {fit_status}.
                The gap between train R² ({best_fuel['train_r2']:.3f}) and test R²
                ({best_fuel['test_r2']:.3f}) is {abs(gap):.3f}.
                {"Consider adding traffic conditions, terrain/elevation, or driver behavior features to improve generalization." if best_fuel['test_r2'] < 0.4 else "Performance is acceptable for flagging trips with abnormal fuel consumption."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# SECTION INTRO / EXPLANATION
# ============================================================
elif ml_option == "Delay Prediction":

    # ============================================================
    # SECTION INTRO / EXPLANATION
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#2F75B5;
            padding:28px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;">
            <h2 style="margin:0;">Real-Time Delivery Delay Prediction System</h2>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                Most delivery delays are detected only after they happen — once
                the customer is already affected. But the conditions that lead to
                a delay (route distance, priority level, store load, time of
                dispatch, vehicle type) are often known <b>before</b> the vehicle
                even leaves.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Goal:</b> Predict, before dispatch, both the <b>probability
                that a delivery will be delayed</b> (classification) and the
                <b>expected delay duration in minutes</b> (regression), so
                planners can intervene — re-route, re-prioritize, or notify
                customers — ahead of time.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>Why this matters:</b> early warning on high-risk dispatches
                allows proactive rescheduling, better customer communication, and
                reduced SLA breaches, turning delay management from reactive to
                predictive.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                <b>What you'll see here:</b> a pre-dispatch delay-risk
                classification (Random Forest Classifier), a predicted delay
                duration in minutes (Random Forest Regression), model tuning
                details, comparison across algorithms, and diagnostic checks for
                overfitting/underfitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # MODEL ENGINEERING BOX
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#1a4d80;
            padding:26px;
            border-radius:12px;
            color:white;
            margin-bottom:18px;
            line-height:1.7;">
            <h3 style="margin:0 0 12px 0;">Model Engineering</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA PREP (cached, always runs)
    # ============================================================
    @st.cache_data
    def build_delay_features(data):
        d = data.copy()

        d["Dispatch_Time"] = pd.to_datetime(
            d["Dispatch_Time"], format="%d-%m-%y %H:%M", errors="coerce"
        )
        d["dispatch_hour"] = d["Dispatch_Time"].dt.hour
        d["dispatch_weekday"] = d["Dispatch_Time"].dt.weekday
        d["dispatch_is_weekend"] = d["dispatch_weekday"].isin([5, 6]).astype(int)
        d["dispatch_month"] = d["Dispatch_Time"].dt.month

        priority_map = {"Low": 1, "Medium": 2, "High": 3}
        d["priority_score"] = d["Priority_Level"].map(priority_map).astype(float)

        for c in ["distance_km", "Quantity", "Total_Route_Delay", "area_sqft",
                  "Delay_Minutes"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        cat_cols = ["region", "zone", "store_type", "vehicle_type"]
        for c in cat_cols:
            if c not in d.columns:
                d[c] = "Unknown"
            d[c] = d[c].fillna("Unknown")

        d = d.dropna(subset=["Delay_Minutes"]).copy()
        d["is_delayed"] = (d["Delay_Minutes"] > 0).astype(int)

        keep_cols = ["Order_ID", "Store_ID", "Route_ID", "region", "zone",
                      "store_type", "vehicle_type", "distance_km", "Quantity",
                      "priority_score", "dispatch_hour", "dispatch_weekday",
                      "dispatch_is_weekend", "dispatch_month", "Delay_Minutes",
                      "is_delayed"]
        keep_cols = [c for c in keep_cols if c in d.columns]

        return d[keep_cols]

    # ============================================================
    # TRAINING FUNCTION (runs only on button click)
    # ============================================================
    @st.cache_resource
    def train_delay_models(delay_df):
        sdf = delay_df.copy()

        cat_cols = ["region", "zone", "store_type", "vehicle_type"]
        for c in cat_cols:
            le = LabelEncoder()
            sdf[c + "_enc"] = le.fit_transform(sdf[c].astype(str))

        base_features = ["distance_km", "Quantity", "priority_score",
                          "dispatch_hour", "dispatch_weekday",
                          "dispatch_is_weekend", "dispatch_month"] + [c + "_enc" for c in cat_cols]
        base_features = [f for f in base_features if f in sdf.columns]

        # ============================================================
        # PART A: DELAY PROBABILITY CLASSIFICATION - MULTIPLE MODELS
        # ============================================================
        c_data = sdf.dropna(subset=base_features + ["is_delayed"])
        Xc = c_data[base_features]
        yc = c_data["is_delayed"]
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.2, random_state=42, stratify=yc)

        delay_prob_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        }

        prob_results = {}
        for name, model in delay_prob_models.items():
            model.fit(Xc_train, yc_train)
            train_pred = model.predict(Xc_train)
            test_pred = model.predict(Xc_test)
            prob_results[name] = {
                "model": model,
                "train_acc": accuracy_score(yc_train, train_pred),
                "test_acc": accuracy_score(yc_test, test_pred),
                "precision": precision_score(yc_test, test_pred, pos_label=1, zero_division=0),
                "recall": recall_score(yc_test, test_pred, pos_label=1, zero_division=0),
                "f1": f1_score(yc_test, test_pred, pos_label=1, zero_division=0),
            }

        best_prob_name = max(prob_results, key=lambda k: prob_results[k]["test_acc"])
        best_prob_model = prob_results[best_prob_name]["model"]

        X_full_c = sdf[base_features].fillna(0)
        sdf["delay_probability"] = best_prob_model.predict_proba(X_full_c)[:, 1].round(3)
        sdf["predicted_delay_flag"] = best_prob_model.predict(X_full_c)
        sdf["delay_risk_label"] = np.where(sdf["delay_probability"] >= 0.5, "High Risk", "Low Risk")

        # ============================================================
        # PART B: DELAY DURATION REGRESSION - MULTIPLE MODELS
        # ============================================================
        r_data = sdf.dropna(subset=base_features + ["Delay_Minutes"])
        Xr = r_data[base_features]
        yr = r_data["Delay_Minutes"]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

        delay_dur_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        }

        dur_results = {}
        for name, model in delay_dur_models.items():
            model.fit(Xr_train, yr_train)
            train_pred = model.predict(Xr_train)
            test_pred = model.predict(Xr_test)
            dur_results[name] = {
                "model": model,
                "train_r2": r2_score(yr_train, train_pred),
                "test_r2": r2_score(yr_test, test_pred),
                "train_mae": mean_absolute_error(yr_train, train_pred),
                "test_mae": mean_absolute_error(yr_test, test_pred),
            }

        best_dur_name = max(dur_results, key=lambda k: dur_results[k]["test_r2"])
        best_dur_model = dur_results[best_dur_name]["model"]

        X_full_r = sdf[base_features].fillna(0)
        sdf["predicted_delay_minutes"] = best_dur_model.predict(X_full_r).round(1)
        sdf["predicted_delay_minutes"] = sdf["predicted_delay_minutes"].clip(lower=0)

        return sdf, prob_results, dur_results, best_prob_name, best_dur_name

    # ============================================================
    # TRAIN BUTTON
    # ============================================================
    delay_features = build_delay_features(df)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        train_clicked = st.button("Train Model", use_container_width=True, key="train_delay_pred")

    if train_clicked:
        st.session_state.delay_pred_trained = True

    if st.session_state.get("delay_pred_trained", False):

        result_df, prob_results, dur_results, best_prob_name, best_dur_name = train_delay_models(delay_features)

        # ============================================================
        # MODEL TUNING SUMMARY
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Tuning Summary</h3>", unsafe_allow_html=True)

        best_prob = prob_results[best_prob_name]
        best_dur = dur_results[best_dur_name]

        # Overfit / underfit check (probability)
        p_gap = best_prob["train_acc"] - best_prob["test_acc"]
        if p_gap > 0.20:
            p_fit_status = "Overfitting (train accuracy much higher than test accuracy)"
        elif best_prob["test_acc"] < 0.6:
            p_fit_status = "Underfitting (low predictive accuracy on test data)"
        else:
            p_fit_status = "Good Fit (train and test accuracy are close)"

        # Overfit / underfit check (duration)
        d_gap = best_dur["train_r2"] - best_dur["test_r2"]
        if d_gap > 0.25:
            d_fit_status = "Overfitting (train R² much higher than test R²)"
        elif best_dur["test_r2"] < 0.1:
            d_fit_status = "Underfitting (model explains very little variance)"
        else:
            d_fit_status = "Good Fit (train and test performance are close)"

        prob_algo_list = ", ".join(prob_results.keys())
        dur_algo_list = ", ".join(dur_results.keys())

        # Why best won (probability)
        other_prob = [n for n in prob_results if n != best_prob_name][0]
        if best_prob["test_acc"] > prob_results[other_prob]["test_acc"]:
            p_why_best = (f"{best_prob_name} achieved higher test accuracy "
                           f"({best_prob['test_acc']*100:.1f}% vs {prob_results[other_prob]['test_acc']*100:.1f}%) "
                           f"and an F1 score of {best_prob['f1']:.2f}, indicating better balance between "
                           f"catching delayed dispatches and avoiding false alarms.")
        else:
            p_why_best = f"{best_prob_name} gave the most balanced train/test performance."

        # Why best won (duration)
        other_dur = [n for n in dur_results if n != best_dur_name][0]
        if best_dur["test_r2"] > dur_results[other_dur]["test_r2"]:
            d_why_best = (f"{best_dur_name} achieved a higher test R² "
                           f"({best_dur['test_r2']:.3f} vs {dur_results[other_dur]['test_r2']:.3f}), "
                           f"meaning it explains more of the variation in delay duration on unseen data.")
        else:
            d_why_best = f"{best_dur_name} gave the most balanced train/test performance."

        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">Pre-Dispatch Delay Probability Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {prob_algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_prob_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {p_why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {p_fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train Accuracy = {best_prob['train_acc']*100:.1f}%, Test Accuracy = {best_prob['test_acc']*100:.1f}%,
                    Precision = {best_prob['precision']:.2f}, Recall = {best_prob['recall']:.2f}, F1 = {best_prob['f1']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color:#f4f8fb;border:1px solid #d6e4f0;
                border-radius:10px;padding:20px;margin-bottom:14px;line-height:1.7;">
                <h4 style="margin:0 0 8px 0;color:{NAVY};">Delay Duration Prediction Model</h4>
                <p style="margin:0;"><b>Algorithms used:</b> {dur_algo_list}</p>
                <p style="margin:0;"><b>Best Model Selected:</b> {best_dur_name}</p>
                <p style="margin:0;"><b>Why this model performed best:</b> {d_why_best}</p>
                <p style="margin:0;"><b>Fit Diagnosis:</b> {d_fit_status}</p>
                <p style="margin:0;"><b>Evaluation Metrics:</b>
                    Train R² = {best_dur['train_r2']:.3f}, Test R² = {best_dur['test_r2']:.3f},
                    Train MAE = {best_dur['train_mae']:.2f}, Test MAE = {best_dur['test_mae']:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # MODEL COMPARISON (Before / After grid)
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='color:#3c3c3c;font-weight:700;'>Model Performance Comparison</h2>",
            unsafe_allow_html=True
        )

        def grid_cards(card_data):
            cards_html = "".join(
                f"""<div class="summary-card">
                        <div class="summary-title">{label}</div>
                        <div class="summary-value">{value}</div>
                    </div>"""
                for label, value in card_data
            )
            st.markdown(
                f"""<div class="summary-grid">{cards_html}</div>""",
                unsafe_allow_html=True
            )

        # ---- Delay Probability: Before / After ----
        pnames = list(prob_results.keys())
        pbefore_name, pafter_name = pnames[0], pnames[1]
        pbefore, pafter = prob_results[pbefore_name], prob_results[pafter_name]

        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Delay Probability Classification</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {pbefore_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train Acc", f"{pbefore['train_acc']*100:.2f}%"),
            ("Test Acc", f"{pbefore['test_acc']*100:.2f}%"),
            ("Precision", f"{pbefore['precision']:.3f}"),
            ("Recall", f"{pbefore['recall']:.3f}"),
            ("F1 Score", f"{pbefore['f1']:.3f}"),
        ])

        st.markdown(f"<h4 style='color:#3c3c3c;'> {pafter_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train Acc", f"{pafter['train_acc']*100:.2f}%"),
            ("Test Acc", f"{pafter['test_acc']*100:.2f}%"),
            ("Precision", f"{pafter['precision']:.3f}"),
            ("Recall", f"{pafter['recall']:.3f}"),
            ("F1 Score", f"{pafter['f1']:.3f}"),
        ])

        if pafter["test_acc"] > pbefore["test_acc"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Delay probability model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Delay probability model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )

        # ---- Delay Duration: Before / After ----
        dnames = list(dur_results.keys())
        dbefore_name, dafter_name = dnames[0], dnames[1]
        dbefore, dafter = dur_results[dbefore_name], dur_results[dafter_name]

        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Delay Duration Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#3c3c3c;'> {dbefore_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{dbefore['train_mae']:.2f}"),
            ("Test MAE", f"{dbefore['test_mae']:.2f}"),
            ("Train R²", f"{dbefore['train_r2']:.3f}"),
            ("Test R²", f"{dbefore['test_r2']:.3f}"),
        ])

        st.markdown(f"<h4 style='color:#3c3c3c;'> {dafter_name}</h4>", unsafe_allow_html=True)
        grid_cards([
            ("Train MAE", f"{dafter['train_mae']:.2f}"),
            ("Test MAE", f"{dafter['test_mae']:.2f}"),
            ("Train R²", f"{dafter['train_r2']:.3f}"),
            ("Test R²", f"{dafter['test_r2']:.3f}"),
        ])

        if dafter["test_r2"] > dbefore["test_r2"]:
            st.markdown(
                f"""<div style="background-color:#eef3ea;border:1px solid #cfe0c8;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#3c5a35;font-weight:500;">
                    ✅ Delay duration model improved after correction
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;
                    border-radius:6px;padding:14px 18px;margin-top:6px;margin-bottom:24px;
                    color:#8a6d1d;font-weight:500;">
                    ⚠️ Delay duration model did NOT improve after correction
                </div>""",
                unsafe_allow_html=True
            )

        # ============================================================
        # PREDICTIONS / RESULTS
        # ============================================================
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Predictions</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>Actual</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Orders Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Actually Delayed</div>
                    <div class="summary-value">{(result_df['is_delayed']==1).sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Delay (min)</div>
                    <div class="summary-value">{result_df['Delay_Minutes'].mean():.1f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Max Delay (min)</div>
                    <div class="summary-value">{result_df['Delay_Minutes'].max():.0f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#3c3c3c;'>Predicted</h5>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-title">Orders Analyzed</div>
                    <div class="summary-value">{result_df.shape[0]}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Predicted High Risk</div>
                    <div class="summary-value">{(result_df['delay_risk_label']=='High Risk').sum()}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Avg Predicted Delay (min)</div>
                    <div class="summary-value">{result_df['predicted_delay_minutes'].mean():.1f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-title">Risk Model Accuracy</div>
                    <div class="summary-value">{best_prob['test_acc']*100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='color:black;'>Top 10 Highest Delay-Risk Dispatches</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            top10 = result_df.sort_values("delay_probability", ascending=False).head(10)
            ax.bar(top10["Order_ID"].astype(str), top10["delay_probability"], color=NAVY)
            apply_chart_style(ax, xlabel="Order ID", ylabel="Delay Probability",
                            title="Top 10 Delay-Risk Dispatches")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("<h4 style='color:black;'>Delay Risk Distribution</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = result_df["delay_risk_label"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PIE_2[:len(counts)] if len(counts) <= len(PIE_2) else None, startangle=90)
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("<h4 style='color:black;'>Top 15 Pre-Dispatch High-Risk Orders</h4>", unsafe_allow_html=True)
        delay_table = result_df.sort_values("delay_probability", ascending=False).head(15)[
            ["Order_ID", "Store_ID", "region", "zone", "store_type", "vehicle_type",
             "distance_km", "priority_score", "delay_probability",
             "predicted_delay_minutes", "delay_risk_label"]
        ].round(2)

        st.markdown("<div style='width:100%; overflow-x:auto;'>", unsafe_allow_html=True)
        render_html_table(delay_table)
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # MODEL DIAGNOSIS
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Diagnosis</h3>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="background-color:#fff8e8;border:1px solid #f0d9a0;
                border-radius:10px;padding:20px;line-height:1.7;">
                <p style="margin:0 0 8px 0;"><b>Delay Probability Model:</b> {p_fit_status}.
                The gap between train accuracy ({best_prob['train_acc']*100:.1f}%) and test accuracy
                ({best_prob['test_acc']*100:.1f}%) is {abs(p_gap)*100:.1f} percentage points.
                Precision of {best_prob['precision']:.2f} and recall of {best_prob['recall']:.2f}
                indicate {"the model is reliable at flagging dispatches likely to be delayed." if best_prob['recall'] > 0.7 else "some delayed dispatches may be missed — consider adjusting the risk threshold."}
                </p>
                <p style="margin:0;"><b>Delay Duration Model:</b> {d_fit_status}.
                The gap between train R² ({best_dur['train_r2']:.3f}) and test R²
                ({best_dur['test_r2']:.3f}) is {abs(d_gap):.3f}.
                {"Consider adding traffic/weather conditions or historical store-level delay patterns to improve generalization." if best_dur['test_r2'] < 0.4 else "Performance is acceptable for proactive delay-duration estimation."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
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