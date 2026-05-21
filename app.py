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
# STEP 3 – EDA
# ============================================================
if "df" not in st.session_state or st.session_state.df is None:
    st.stop()

df = st.session_state.df

if "eda_completed" not in st.session_state:
    st.session_state.eda_completed = False

# ── EDA HEADER ──
st.markdown("""
<div style="background-color:#0B2C5D;padding:18px 25px;border-radius:10px;
     color:white;margin-top:20px;margin-bottom:10px;">
    <h3 style="margin:0;">Exploratory Data Analysis (EDA)</h3>
</div>
""", unsafe_allow_html=True)
st.write("")
st.info(f"Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.write("")

st.markdown("""
<div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
     font-size:16px;line-height:1.6;margin-bottom:20px;">
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
</div>
""", unsafe_allow_html=True)

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
    row1 = st.columns(5)
    row2 = st.columns(4)
    with row1[0]: nav_button("Data Quality Overview",       "Data Quality Overview")
    with row1[1]: nav_button("Delivery Performance",        "Delivery Performance")
    with row1[2]: nav_button("Fuel & Cost Analysis",        "Fuel & Cost Analysis")
    with row1[3]: nav_button("Capacity & Load Analysis",    "Capacity & Load Analysis")
    with row1[4]: nav_button("Route & Stop Analysis",       "Route & Stop Analysis")
    with row2[0]: nav_button("Driver & Vehicle Analysis",   "Driver & Vehicle Analysis")
    with row2[1]: nav_button("Temperature & Breach Analysis","Temperature & Breach Analysis")
    with row2[2]: nav_button("Store & Region Analysis",     "Store & Region Analysis")
    with row2[3]: nav_button("Summary Report",              "Summary Report")

eda_option = st.session_state.eda_option
if eda_option is not None:
    st.session_state.eda_completed = True

st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

if eda_option is None:
    st.info("Select an analysis to view insights.")


# ─────────────────────────────────────────────────────────────
# HELPER: render a styled legend row above a chart
# ─────────────────────────────────────────────────────────────
def render_legend(items):
    """items: list of (color_hex, label)"""
    parts = "".join(
        f'<span><span class="leg" style="background:{c};"></span>{l}</span>&nbsp;&nbsp;'
        for c, l in items
    )
    st.markdown(
        f'<div style="font-size:10px;color:rgba(255,255,255,0.85);margin-bottom:6px;">{parts}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# HELPER: two-column chart layout inside the green section
# ─────────────────────────────────────────────────────────────
# CHART CARD TITLE & LEGEND — use existing blue theme
# ─────────────────────────────────────────────────────────────
CHART_TITLE_STYLE = (
    'font-size:13px;font-weight:700;margin-bottom:8px;margin-top:4px;'
    'color:#ffffff;'
    'background-color:#0B2C5D;'
    'padding:6px 12px;border-radius:6px 6px 0 0;display:block;'
    'letter-spacing:0.3px;'
)
CHART_CARD_STYLE  = (
    'background-color:#2F75B5;border-radius:8px;padding:0px;'
    'margin-bottom:12px;border:1px solid #1a5a9a;overflow:hidden;'
)
CHART_INNER_STYLE = 'padding:10px;background-color:#2F75B5;'
LEGEND_ITEM_STYLE = (
    'display:inline-flex;align-items:center;gap:5px;margin-right:14px;'
    'font-size:11px;font-weight:600;color:#ffffff;'
)


def render_legend(items):
    """items: list of (color_hex, label)"""
    parts = "".join(
        f'<span style="{LEGEND_ITEM_STYLE}">'
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;'
        f'background-color:{c};flex-shrink:0;border:1px solid rgba(255,255,255,0.3);"></span>'
        f'{l}</span>'
        for c, l in items
    )
    st.markdown(
        f'<div style="background-color:#1F5C9A;padding:5px 10px;border-radius:4px;'
        f'margin-bottom:8px;flex-wrap:wrap;display:flex;gap:2px;">{parts}</div>',
        unsafe_allow_html=True
    )


def two_charts(fig_left, title_left, legend_left,
               fig_right, title_right, legend_right):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div style="{CHART_CARD_STYLE}">'
            f'<div style="{CHART_TITLE_STYLE}">{title_left}</div>'
            f'<div style="{CHART_INNER_STYLE}">',
            unsafe_allow_html=True
        )
        if legend_left:
            render_legend(legend_left)
        st.pyplot(fig_left, use_container_width=True)
        plt.close(fig_left)
        st.markdown("</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div style="{CHART_CARD_STYLE}">'
            f'<div style="{CHART_TITLE_STYLE}">{title_right}</div>'
            f'<div style="{CHART_INNER_STYLE}">',
            unsafe_allow_html=True
        )
        if legend_right:
            render_legend(legend_right)
        st.pyplot(fig_right, use_container_width=True)
        plt.close(fig_right)
        st.markdown("</div></div>", unsafe_allow_html=True)


def full_chart(fig, title, legend=None):
    st.markdown(
        f'<div style="{CHART_CARD_STYLE}">'
        f'<div style="{CHART_TITLE_STYLE}">{title}</div>'
        f'<div style="{CHART_INNER_STYLE}">',
        unsafe_allow_html=True
    )
    if legend:
        render_legend(legend)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("</div></div>", unsafe_allow_html=True)
    if legend:
        render_legend(legend)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 1. DATA QUALITY OVERVIEW  — kept exactly as original (HTML tables)
# ============================================================
if eda_option == "Data Quality Overview":

    import streamlit.components.v1 as components

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>What this section does:</b><br><br>
    Provides a <b>high-level health check</b> of the logistics dataset before any modeling
    or optimization is attempted.<br><br>
    It evaluates:
    <ul>
        <li>Missing values across all logistics fields</li>
        <li>Duplicate delivery records</li>
        <li>Data type consistency</li>
        <li>Overall row and column completeness</li>
    </ul>
    <b>Why this matters:</b><br>
    Route optimization and delivery forecasting models are highly sensitive to
    <b>poor data quality</b>. Missing weights, volumes, or timestamps can distort
    load planning and delay predictions.
    </div>
    """, unsafe_allow_html=True)

    rows_count   = df.shape[0]
    cols_count   = df.shape[1]
    dup_count    = df.duplicated().sum()
    dtype_counts = df.dtypes.value_counts()
    mv           = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)

    date_min = date_max = "N/A"
    if col_date and col_date in df.columns:
        try:
            parsed = pd.to_datetime(df[col_date], errors="coerce")
            if not parsed.isna().all():
                date_min = parsed.min().date()
                date_max = parsed.max().date()
        except Exception:
            pass

    dup_impact = "⚠️ Risk of inflated delivery counts" if dup_count > 0 else "✅ Clean – no duplicates found"

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

    mv_rows = "".join([
        "<tr><td>{}</td><td>{:.2f}%</td></tr>".format(c, v)
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

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Delivery Performance Analysis</b><br><br>
    Evaluates on-time vs delayed deliveries, average delay minutes, priority-level
    distribution, and failed stop rates.
    </div>
    """, unsafe_allow_html=True)

    # ── metrics ──
    total_trips  = df.shape[0]
    delay_series = pd.to_numeric(df[col_delay], errors="coerce").dropna() if col_delay else pd.Series(dtype=float)
    avg_delay    = round(delay_series.mean(), 1) if not delay_series.empty else "N/A"
    delayed_pct  = round((delay_series > 0).mean() * 100, 1) if not delay_series.empty else "N/A"
    failed_total = int(pd.to_numeric(df[col_failed_stops], errors="coerce").sum()) if col_failed_stops else 0

    section_header("Delivery Performance", "On-time rate · delay distribution · priority split")
    metric_row([
        {"label": "Total Trips",         "value": f"{total_trips:,}",  "delta": "All records loaded"},
        {"label": "Avg Delay (min)",      "value": str(avg_delay),      "delta": "Across all routes"},
        {"label": "% Delayed",            "value": f"{delayed_pct}%",   "delta": "Deliveries > 0 delay"},
        {"label": "Total Failed Stops",   "value": f"{failed_total:,}", "delta": "Across all routes"},
    ])

    # ── Chart 1: Delivery status bar ──
    if col_status and col_status in df.columns:
        status_counts = df[col_status].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        colors_s = [NAVY if i == 0 else NAVY2 if i == 1 else RED for i in range(len(status_counts))]
        ax1.bar(status_counts.index.astype(str), status_counts.values, color=colors_s)
        apply_chart_style(ax1, xlabel='Delivery Status', ylabel='Number of Deliveries',
                          title='Delivery Status Breakdown')
        for i, v in enumerate(status_counts.values):
            ax1.text(i, v + max(status_counts.values) * 0.01, f'{v:,}',
                     ha='center', color='white', fontsize=9)
        fig1.tight_layout()
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.text(0.5, 0.5, 'Status column not found', ha='center', va='center',
                 color='white', transform=ax1.transAxes)
        apply_chart_style(ax1)
        fig1.tight_layout()

    # ── Chart 2: Delay distribution histogram ──
    if not delay_series.empty:
        delayed_only = delay_series[delay_series > 0]
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(delayed_only, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax2, xlabel='Delay (minutes)', ylabel='Number of Deliveries',
                          title='Delay Distribution (Delayed Only)')
        fig2.tight_layout()
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.text(0.5, 0.5, 'Delay column not found', ha='center', va='center',
                 color='white', transform=ax2.transAxes)
        apply_chart_style(ax2)
        fig2.tight_layout()

    two_charts(fig1, "Delivery Status Breakdown",
               [(NAVY, 'Primary'), (NAVY2, 'Secondary'), (RED, 'Failed')],
               fig2, "Delay Minutes Distribution", [(NAVY, 'Delayed trips')])

    # ── Chart 3: Priority pie — smaller size ──
    if col_priority and col_priority in df.columns:
        prio = df[col_priority].value_counts()
        pie_colors = ['#1a6b3c', '#f39c12', RED, '#2980b9', '#8e44ad']
        # Use columns to constrain pie to half width
        col_pie, col_empty = st.columns([1, 1])
        with col_pie:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            ax3.set_facecolor(GREEN_BG)
            fig3.patch.set_facecolor(GREEN_BG)
            wedges, texts, autotexts = ax3.pie(
                prio.values, labels=prio.index.astype(str),
                autopct='%1.1f%%', colors=pie_colors[:len(prio)],
                startangle=90,
                wedgeprops=dict(width=0.55),
                pctdistance=0.75,
                labeldistance=1.08,
            )
            for t in texts + autotexts:
                t.set_color('white')
                t.set_fontsize(8)
            ax3.set_title('Priority Level Distribution', color='white', fontsize=10, fontweight='bold')
            fig3.tight_layout()
            st.markdown(
                f'<div style="{CHART_CARD_STYLE}">'
                f'<div style="{CHART_TITLE_STYLE}">Order Priority Split</div>'
                f'<div style="{CHART_INNER_STYLE}">',
                unsafe_allow_html=True
            )
            render_legend(list(zip(pie_colors[:len(prio)], prio.index.astype(str))))
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
            st.markdown("</div></div>", unsafe_allow_html=True)
    section_footer()


# ============================================================
# 3. FUEL & COST ANALYSIS
# ============================================================
elif eda_option == "Fuel & Cost Analysis":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Fuel & Cost Analysis</b><br><br>
    Breaks down fuel consumption, efficiency, and total logistics costs including
    labor, maintenance, tolls, and parking.
    </div>
    """, unsafe_allow_html=True)

    cost_map = {
        "Fuel Cost":        map_col(["Fuel_Cost"]),
        "Labor Cost":       map_col(["Labor_Cost"]),
        "Maintenance":      map_col(["Maintenance_Cost"]),
        "Toll Cost":        map_col(["Toll_Cost"]),
        "Parking Cost":     map_col(["Parking_Cost"]),
        "Total Logistics":  map_col(["Total_Logistics_Cost"]),
    }
    cost_totals = {
        k: round(pd.to_numeric(df[v], errors="coerce").sum(), 2)
        for k, v in cost_map.items() if v and v in df.columns
    }

    total_cost_val = round(pd.to_numeric(df[col_total_cost], errors="coerce").sum(), 2) if col_total_cost else 0
    avg_cost_val   = round(pd.to_numeric(df[col_total_cost], errors="coerce").mean(), 2) if col_total_cost else 0
    fuel_total     = round(pd.to_numeric(df[col_fuel_used], errors="coerce").sum(), 2) if col_fuel_used else 0
    fuel_avg       = round(pd.to_numeric(df[col_fuel_used], errors="coerce").mean(), 2) if col_fuel_used else 0

    section_header("Fuel & Cost Analysis", "Cost breakdown · fuel efficiency · fleet fuel type")
    metric_row([
        {"label": "Total Logistics Cost", "value": f"{total_cost_val:,.0f}", "delta": "All routes combined"},
        {"label": "Avg Cost / Trip",      "value": f"{avg_cost_val:,.1f}",   "delta": "Per delivery record"},
        {"label": "Total Fuel Consumed",  "value": f"{fuel_total:,.0f}",     "delta": "All vehicles"},
        {"label": "Avg Fuel / Trip",      "value": f"{fuel_avg:,.2f}",       "delta": "Litres per record"},
    ])

    # ── Chart 1: Cost breakdown pie ──
    if cost_totals:
        labels  = [k for k in cost_totals if k != "Total Logistics"]
        values  = [cost_totals[k] for k in labels]
        pie_c   = ['#e67e22', '#2980b9', '#8e44ad', RED, '#1a6b3c']
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.set_facecolor(GREEN_BG)
        fig1.patch.set_facecolor(GREEN_BG)
        wedges, texts, autotexts = ax1.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=pie_c[:len(labels)], startangle=90,
            wedgeprops=dict(width=0.6)
        )
        for t in texts + autotexts:
            t.set_color('white'); t.set_fontsize(9)
        ax1.set_title('Cost Breakdown by Category', color='white', fontsize=11, fontweight='bold')
        leg1 = list(zip(pie_c[:len(labels)], labels))
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.text(0.5, 0.5, 'Cost columns not found', ha='center', va='center',
                 color='white', transform=ax1.transAxes)
        apply_chart_style(ax1)
        leg1 = None
    fig1.tight_layout()

    # ── Chart 2: Fuel efficiency histogram ──
    if col_fuel_eff and col_fuel_eff in df.columns:
        eff = pd.to_numeric(df[col_fuel_eff], errors="coerce").dropna()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.hist(eff, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax2, xlabel='Fuel Efficiency (km/L)', ylabel='Number of Records',
                          title='Fuel Efficiency Distribution')
        fig2.tight_layout()
        leg2 = [(NAVY, 'Efficiency records')]
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.text(0.5, 0.5, 'Fuel efficiency column not found',
                 ha='center', va='center', color='white', transform=ax2.transAxes)
        apply_chart_style(ax2)
        fig2.tight_layout()
        leg2 = None

    two_charts(fig1, "Cost Breakdown by Category", leg1,
               fig2, "Fuel Efficiency Distribution", leg2)

    # ── Chart 3: Fuel type bar ──
    if col_fuel_type and col_fuel_type in df.columns:
        ft = df[col_fuel_type].value_counts()
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        bar_c = [NAVY, NAVY2, NAVY3, RED]
        ax3.bar(ft.index.astype(str), ft.values,
                color=[bar_c[i % len(bar_c)] for i in range(len(ft))])
        apply_chart_style(ax3, xlabel='Fuel Type', ylabel='Number of Vehicles',
                          title='Fleet Fuel Type Breakdown')
        for i, v in enumerate(ft.values):
            ax3.text(i, v + max(ft.values) * 0.01, f'{v:,}',
                     ha='center', color='white', fontsize=9)
        fig3.tight_layout()
        full_chart(fig3, "Fleet Fuel Type Distribution")
    section_footer()


# ============================================================
# 4. CAPACITY & LOAD ANALYSIS
# ============================================================
elif eda_option == "Capacity & Load Analysis":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Capacity & Load Analysis</b><br><br>
    Monitors truck weight/volume utilization, identifies under-loaded and over-loaded
    vehicles, and supports smarter load consolidation.
    </div>
    """, unsafe_allow_html=True)

    cap_series = pd.to_numeric(df[col_cap_util], errors="coerce").dropna() if col_cap_util else pd.Series(dtype=float)
    avg_cap    = round(cap_series.mean(), 1) if not cap_series.empty else "N/A"
    under      = int((cap_series < 50).sum()) if not cap_series.empty else 0
    optimal    = int(((cap_series >= 50) & (cap_series <= 90)).sum()) if not cap_series.empty else 0
    over       = int((cap_series > 90).sum()) if not cap_series.empty else 0

    section_header("Capacity & Load Analysis", "Utilisation bands · weight & volume distribution")
    metric_row([
        {"label": "Avg Utilisation",   "value": f"{avg_cap}%",      "delta": "Fleet average"},
        {"label": "Under-loaded <50%", "value": f"{under:,}",       "delta": "Records below 50%"},
        {"label": "Optimal 50–90%",    "value": f"{optimal:,}",     "delta": "Well-utilised trips"},
        {"label": "Over-loaded >90%",  "value": f"{over:,}",        "delta": "Risk of damage"},
    ])

    # ── Chart 1: Capacity utilisation histogram ──
    if not cap_series.empty:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.hist(cap_series, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        ax1.axvline(50, color='#f39c12', linestyle='--', linewidth=1.5, label='50%')
        ax1.axvline(90, color=RED,       linestyle='--', linewidth=1.5, label='90%')
        apply_chart_style(ax1, xlabel='Utilisation (%)', ylabel='Number of Trips',
                          title='Capacity Utilisation Distribution')
        ax1.legend(facecolor='none', labelcolor='white', fontsize=8)
        fig1.tight_layout()
        leg1 = [(NAVY, 'Trips'), ('#f39c12', '50% threshold'), (RED, '90% threshold')]
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.text(0.5, 0.5, 'Capacity utilisation column not found',
                 ha='center', va='center', color='white', transform=ax1.transAxes)
        apply_chart_style(ax1); fig1.tight_layout()
        leg1 = None

    # ── Chart 2: Load band — horizontal bar (no overlap possible) ──
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.set_facecolor(GREEN_BG); fig2.patch.set_facecolor(GREEN_BG)
    if not cap_series.empty:
        band_vals  = [under, optimal, over]
        band_lbls  = ['Under-loaded\n(<50%)', 'Optimal\n(50–90%)', 'Over-loaded\n(>90%)']
        band_cols  = ['#2980b9', '#1a6b3c', RED]
        bars2 = ax2.barh(band_lbls, band_vals, color=band_cols, height=0.5)
        apply_chart_style(ax2, xlabel='Number of Trips', ylabel='', title='Load Band Distribution')
        x_max = max(band_vals) if max(band_vals) > 0 else 1
        ax2.set_xlim(0, x_max * 1.22)
        for bar, v in zip(bars2, band_vals):
            ax2.text(bar.get_width() + x_max * 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{v:,}', va='center', color='white', fontsize=9, fontweight='600')
        leg2 = list(zip(band_cols, ['Under-loaded', 'Optimal', 'Over-loaded']))
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center',
                 color='white', transform=ax2.transAxes)
        apply_chart_style(ax2)
        leg2 = None
    fig2.tight_layout()

    two_charts(fig1, "Capacity Utilisation Histogram", leg1,
               fig2, "Load Band Distribution", leg2)

    # ── Charts 3 & 4: Weight and Volume ──
    figs_wv = []
    for label, col in [("Shipment Weight (kg)", col_weight), ("Shipment Volume (m³)", col_volume)]:
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(s, bins=30, color=NAVY2, edgecolor='white', linewidth=0.5)
            apply_chart_style(ax, xlabel=label, ylabel='Frequency', title=f'{label} Distribution')
            fig.tight_layout()
            figs_wv.append((fig, label))

    if len(figs_wv) == 2:
        two_charts(figs_wv[0][0], figs_wv[0][1], [(NAVY2, 'Shipments')],
                   figs_wv[1][0], figs_wv[1][1], [(NAVY2, 'Shipments')])
    elif len(figs_wv) == 1:
        full_chart(figs_wv[0][0], figs_wv[0][1])
    section_footer()


# ============================================================
# 5. ROUTE & STOP ANALYSIS
# ============================================================
elif eda_option == "Route & Stop Analysis":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Route & Stop Analysis</b><br><br>
    Evaluates estimated vs actual travel times, distance per route, stop counts,
    and stop type distributions to optimise future route planning.
    </div>
    """, unsafe_allow_html=True)

    dist_s   = pd.to_numeric(df[col_distance],    errors="coerce").dropna() if col_distance    else pd.Series(dtype=float)
    stops_s  = pd.to_numeric(df[col_stops],       errors="coerce").dropna() if col_stops       else pd.Series(dtype=float)
    actual_s = pd.to_numeric(df[col_travel_time], errors="coerce").dropna() if col_travel_time else pd.Series(dtype=float)
    est_s    = pd.to_numeric(df[col_est_time],    errors="coerce").dropna() if col_est_time    else pd.Series(dtype=float)

    section_header("Route & Stop Analysis", "Distance · travel time · stops per route")
    metric_row([
        {"label": "Total Distance (km)",   "value": f"{dist_s.sum():,.0f}"   if not dist_s.empty   else "N/A", "delta": "All routes"},
        {"label": "Avg Distance / Route",  "value": f"{dist_s.mean():,.1f}"  if not dist_s.empty   else "N/A", "delta": "km per trip"},
        {"label": "Avg Stops / Route",     "value": f"{stops_s.mean():,.1f}" if not stops_s.empty  else "N/A", "delta": "delivery stops"},
        {"label": "Avg Actual Time (hrs)", "value": f"{actual_s.mean():,.2f}" if not actual_s.empty else "N/A", "delta": "vs estimated"},
    ])

    # ── Chart 1: Distance histogram ──
    if not dist_s.empty:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.hist(dist_s, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax1, xlabel='Distance (km)', ylabel='Number of Routes',
                          title='Route Distance Distribution')
        fig1.tight_layout()
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.text(0.5, 0.5, 'Distance column not found', ha='center', va='center',
                 color='white', transform=ax1.transAxes)
        apply_chart_style(ax1); fig1.tight_layout()

    # ── Chart 2: Estimated vs Actual travel time grouped bar ──
    if not actual_s.empty and not est_s.empty:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        x = np.arange(2)
        ax2.bar(x, [est_s.mean(), actual_s.mean()], color=[NAVY, NAVY2],
                width=0.4, edgecolor='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Estimated', 'Actual'])
        apply_chart_style(ax2, xlabel='Time Type', ylabel='Average Hours',
                          title='Estimated vs Actual Travel Time (avg hrs)')
        fig2.tight_layout()
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.text(0.5, 0.5, 'Travel time columns not found', ha='center', va='center',
                 color='white', transform=ax2.transAxes)
        apply_chart_style(ax2); fig2.tight_layout()

    two_charts(fig1, "Route Distance Distribution", [(NAVY, 'Routes')],
               fig2, "Estimated vs Actual Travel Time", [(NAVY, 'Estimated'), (NAVY2, 'Actual')])

    # ── Chart 3: Stops histogram ──
    if not stops_s.empty:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.hist(stops_s, bins=20, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax3, xlabel='Number of Stops', ylabel='Number of Routes',
                          title='Stops per Route Distribution')
        fig3.tight_layout()
    else:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.text(0.5, 0.5, 'Stops column not found', ha='center', va='center',
                 color='white', transform=ax3.transAxes)
        apply_chart_style(ax3); fig3.tight_layout()

    # ── Chart 4: Stop type pie ──
    if col_stop_type and col_stop_type in df.columns:
        stype = df[col_stop_type].value_counts()
        pie_c = ['#1a6b3c', '#f39c12', RED, '#2980b9', '#8e44ad']
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.set_facecolor(GREEN_BG); fig4.patch.set_facecolor(GREEN_BG)
        wedges, texts, autotexts = ax4.pie(
            stype.values, labels=stype.index.astype(str),
            autopct='%1.1f%%', colors=pie_c[:len(stype)],
            startangle=90, wedgeprops=dict(width=0.6)
        )
        for t in texts + autotexts:
            t.set_color('white'); t.set_fontsize(9)
        ax4.set_title('Stop Type Distribution', color='white', fontsize=11, fontweight='bold')
        fig4.tight_layout()
        leg4 = list(zip(pie_c[:len(stype)], stype.index.astype(str)))
    else:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.text(0.5, 0.5, 'Stop type column not found', ha='center', va='center',
                 color='white', transform=ax4.transAxes)
        apply_chart_style(ax4); fig4.tight_layout()
        leg4 = None

    two_charts(fig3, "Stops per Route Distribution", [(NAVY, 'Routes')],
               fig4, "Stop Type Split", leg4)
    section_footer()


# ============================================================
# 6. DRIVER & VEHICLE ANALYSIS
# ============================================================
elif eda_option == "Driver & Vehicle Analysis":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Driver & Vehicle Analysis</b><br><br>
    Monitors idle time, speed behaviour, vehicle type distribution, and driver
    activity to improve fleet utilisation and reduce inefficiencies.
    </div>
    """, unsafe_allow_html=True)

    idle_s  = pd.to_numeric(df[col_idle],  errors="coerce").dropna() if col_idle  else pd.Series(dtype=float)
    speed_s = pd.to_numeric(df[col_speed], errors="coerce").dropna() if col_speed else pd.Series(dtype=float)
    n_drivers  = df[col_driver].nunique()  if col_driver  else "N/A"
    n_vehicles = df[col_vehicle].nunique() if col_vehicle else "N/A"

    section_header("Driver & Vehicle Analysis", "Fleet composition · idle time · speed distribution")
    metric_row([
        {"label": "Active Drivers",       "value": str(n_drivers),                                   "delta": "Unique driver IDs"},
        {"label": "Active Vehicles",      "value": str(n_vehicles),                                  "delta": "Unique vehicle IDs"},
        {"label": "Avg Idle Time (min)",  "value": f"{idle_s.mean():,.1f}" if not idle_s.empty else "N/A", "delta": "Per vehicle record"},
        {"label": "Avg Speed",            "value": f"{speed_s.mean():,.1f}" if not speed_s.empty else "N/A", "delta": "km/h fleet average"},
    ])

    # ── Chart 1: Vehicle type bar ──
    if col_vehicle_type and col_vehicle_type in df.columns:
        vt = df[col_vehicle_type].value_counts()
        bar_c = [NAVY, NAVY2, NAVY3, RED]
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.bar(vt.index.astype(str), vt.values,
                color=[bar_c[i % len(bar_c)] for i in range(len(vt))])
        apply_chart_style(ax1, xlabel='Vehicle Type', ylabel='Number of Vehicles',
                          title='Vehicle Type Distribution')
        for i, v in enumerate(vt.values):
            ax1.text(i, v + max(vt.values) * 0.01, f'{v:,}',
                     ha='center', color='white', fontsize=8)
        fig1.tight_layout()
        leg1 = [(bar_c[i % len(bar_c)], l) for i, l in enumerate(vt.index.astype(str))]
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.text(0.5, 0.5, 'Vehicle type column not found',
                 ha='center', va='center', color='white', transform=ax1.transAxes)
        apply_chart_style(ax1); fig1.tight_layout(); leg1 = None

    # ── Chart 2: Idle time histogram ──
    if not idle_s.empty:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(idle_s, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax2, xlabel='Idle Time (minutes)', ylabel='Number of Records',
                          title='Idle Time Distribution')
        fig2.tight_layout()
        leg2 = [(NAVY, 'Idle records')]
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.text(0.5, 0.5, 'Idle time column not found',
                 ha='center', va='center', color='white', transform=ax2.transAxes)
        apply_chart_style(ax2); fig2.tight_layout(); leg2 = None

    two_charts(fig1, "Vehicle Type Distribution", leg1,
               fig2, "Idle Time Distribution", leg2)

    # ── Chart 3: Speed histogram ──
    if not speed_s.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        ax3.hist(speed_s, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax3, xlabel='Speed (km/h)', ylabel='Number of Records',
                          title='Vehicle Speed Distribution')
        fig3.tight_layout()
        full_chart(fig3, "Vehicle Speed Distribution", [(NAVY, 'Speed records')])
    section_footer()


# ============================================================
# 7. TEMPERATURE & BREACH ANALYSIS
# ============================================================
elif eda_option == "Temperature & Breach Analysis":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Temperature & Breach Analysis</b><br><br>
    Tracks cold-chain compliance by monitoring temperature and humidity readings,
    breach frequency, and duration — critical for perishable goods.
    </div>
    """, unsafe_allow_html=True)

    temp_s  = pd.to_numeric(df[col_temp],     errors="coerce").dropna() if col_temp     else pd.Series(dtype=float)
    hum_s   = pd.to_numeric(df[col_humidity], errors="coerce").dropna() if col_humidity else pd.Series(dtype=float)
    breach_s = pd.to_numeric(df[col_breach],  errors="coerce").dropna() if col_breach   else pd.Series(dtype=float)
    breach_count = int(breach_s.sum()) if not breach_s.empty else 0
    breach_pct   = round(breach_count / len(breach_s) * 100, 2) if not breach_s.empty else 0

    section_header("Temperature & Breach Analysis", "Temperature · humidity · cold-chain breaches")
    metric_row([
        {"label": "Avg Temperature (°C)", "value": f"{temp_s.mean():,.1f}"  if not temp_s.empty  else "N/A", "delta": "Fleet average"},
        {"label": "Avg Humidity (%)",     "value": f"{hum_s.mean():,.1f}"   if not hum_s.empty   else "N/A", "delta": "Fleet average"},
        {"label": "Total Breaches",       "value": f"{breach_count:,}",                                       "delta": "Threshold violations"},
        {"label": "Breach Rate",          "value": f"{breach_pct}%",                                          "delta": "Of all records"},
    ])

    # ── Chart 1: Temperature histogram ──
    if not temp_s.empty:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.hist(temp_s, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax1, xlabel='Temperature (°C)', ylabel='Number of Records',
                          title='Temperature Distribution')
        fig1.tight_layout()
        leg1 = [(NAVY, 'Temperature records')]
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.text(0.5, 0.5, 'Temperature column not found',
                 ha='center', va='center', color='white', transform=ax1.transAxes)
        apply_chart_style(ax1); fig1.tight_layout(); leg1 = None

    # ── Chart 2: Humidity histogram ──
    if not hum_s.empty:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(hum_s, bins=30, color=NAVY2, edgecolor='white', linewidth=0.5)
        apply_chart_style(ax2, xlabel='Humidity (%)', ylabel='Number of Records',
                          title='Humidity Distribution')
        fig2.tight_layout()
        leg2 = [(NAVY2, 'Humidity records')]
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.text(0.5, 0.5, 'Humidity column not found',
                 ha='center', va='center', color='white', transform=ax2.transAxes)
        apply_chart_style(ax2); fig2.tight_layout(); leg2 = None

    two_charts(fig1, "Temperature Distribution", leg1,
               fig2, "Humidity Distribution", leg2)

    # ── Chart 3: Breach pie — smaller, no label overlap ──
    if not breach_s.empty:
        no_breach = len(breach_s) - breach_count
        pie_c = ['#1a6b3c', RED]
        col_pie2, col_empty2 = st.columns([1, 1])
        with col_pie2:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            ax3.set_facecolor(GREEN_BG); fig3.patch.set_facecolor(GREEN_BG)
            wedges, texts, autotexts = ax3.pie(
                [no_breach, breach_count],
                autopct='%1.1f%%',
                colors=pie_c,
                startangle=90,
                wedgeprops=dict(width=0.55),
                pctdistance=0.78,
            )
            for t in autotexts:
                t.set_color('white'); t.set_fontsize(8)
            for t in texts:
                t.set_text('')
            ax3.set_title('Threshold Breach Rate', color='white', fontsize=10, fontweight='bold')
            ax3.legend(wedges, ['No Breach', 'Breach'],
                       loc='lower center', bbox_to_anchor=(0.5, -0.1),
                       ncol=2, fontsize=8, frameon=False, labelcolor='white')
            fig3.tight_layout()
            st.markdown(
                f'<div style="{CHART_CARD_STYLE}">'
                f'<div style="{CHART_TITLE_STYLE}">Breach vs No-Breach Split</div>'
                f'<div style="{CHART_INNER_STYLE}">',
                unsafe_allow_html=True
            )
            render_legend([('#1a6b3c', 'No Breach'), (RED, 'Breach')])
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
            st.markdown("</div></div>", unsafe_allow_html=True)
    section_footer()


# ============================================================
# 8. STORE & REGION ANALYSIS
# ============================================================
elif eda_option == "Store & Region Analysis":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Store & Region Analysis</b><br><br>
    Compares delivery volumes, order values, and delay patterns across stores,
    cities, and regions to identify high-demand and underserved areas.
    </div>
    """, unsafe_allow_html=True)

    n_stores  = df[col_store].nunique()  if col_store  else "N/A"
    n_regions = df[col_region].nunique() if col_region else "N/A"
    n_cities  = df[col_city].nunique()   if col_city   else "N/A"
    ov_s      = pd.to_numeric(df[col_order_value], errors="coerce").dropna() if col_order_value else pd.Series(dtype=float)

    section_header("Store & Region Analysis", "Top stores · regional delivery volumes · city breakdown")
    metric_row([
        {"label": "Unique Stores",       "value": str(n_stores),                                        "delta": "Stores served"},
        {"label": "Regions",             "value": str(n_regions),                                       "delta": "Geographic zones"},
        {"label": "Cities",              "value": str(n_cities),                                        "delta": "Cities covered"},
        {"label": "Avg Order Value",     "value": f"{ov_s.mean():,.1f}" if not ov_s.empty else "N/A",  "delta": "Per order"},
    ])

    # ── Chart 1: Top 10 stores — styled horizontal bar with value badges ──
    if col_store and col_store in df.columns:
        top_s = df[col_store].value_counts().head(10)
        fig1, ax1 = plt.subplots(figsize=(5, 4.5))
        bar_colors = [NAVY if i % 2 == 0 else NAVY2 for i in range(len(top_s))]
        bars1 = ax1.barh(top_s.index.astype(str)[::-1], top_s.values[::-1],
                         color=bar_colors[::-1], height=0.6, edgecolor='white', linewidth=0.4)
        apply_chart_style(ax1, xlabel='Number of Deliveries', ylabel='Store ID',
                          title='Top 10 Stores by Delivery Volume')
        x_max1 = max(top_s.values) if len(top_s) > 0 else 1
        ax1.set_xlim(0, x_max1 * 1.25)
        for bar, v in zip(bars1, top_s.values[::-1]):
            ax1.text(bar.get_width() + x_max1 * 0.015,
                     bar.get_y() + bar.get_height() / 2,
                     f'{v:,}', va='center', color='white', fontsize=8, fontweight='600')
        fig1.tight_layout()
        leg1 = [(NAVY, 'Store (even rank)'), (NAVY2, 'Store (odd rank)')]
    else:
        fig1, ax1 = plt.subplots(figsize=(5, 4.5))
        ax1.text(0.5, 0.5, 'Store column not found',
                 ha='center', va='center', color='white', transform=ax1.transAxes)
        apply_chart_style(ax1); fig1.tight_layout(); leg1 = None

    # ── Chart 2: Region delivery — pie chart (clean, no overlap) ──
    if col_region and col_region in df.columns:
        reg = df[col_region].value_counts()
        pie_c_reg = ['#2980b9', '#1a6b3c', '#e67e22', '#8e44ad', RED, '#16a085']
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        ax2.set_facecolor(GREEN_BG); fig2.patch.set_facecolor(GREEN_BG)
        wedges, texts, autotexts = ax2.pie(
            reg.values,
            autopct='%1.1f%%',
            colors=pie_c_reg[:len(reg)],
            startangle=140,
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=1.2),
            pctdistance=0.78,
        )
        for t in autotexts:
            t.set_color('white'); t.set_fontsize(9); t.set_fontweight('bold')
        for t in texts:
            t.set_text('')
        ax2.set_title('Deliveries by Region', color='white', fontsize=11, fontweight='bold')
        ax2.legend(wedges, reg.index.astype(str),
                   loc='lower center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(3, len(reg)), fontsize=8, frameon=False, labelcolor='white')
        fig2.tight_layout()
        leg2 = list(zip(pie_c_reg[:len(reg)], reg.index.astype(str)))
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        ax2.text(0.5, 0.5, 'Region column not found',
                 ha='center', va='center', color='white', transform=ax2.transAxes)
        apply_chart_style(ax2); fig2.tight_layout(); leg2 = None

    two_charts(fig1, "Top 10 Stores by Deliveries", leg1,
               fig2, "Deliveries by Region", leg2)

    # ── Chart 3: Top 10 cities — styled horizontal bar with gradient colors ──
    if col_city and col_city in df.columns:
        top_c = df[col_city].value_counts().head(10)
        cmap_colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(top_c)))[::-1]
        fig3, ax3 = plt.subplots(figsize=(10, max(3.5, len(top_c) * 0.45)))
        bars3 = ax3.barh(top_c.index.astype(str)[::-1], top_c.values[::-1],
                         color=cmap_colors, height=0.6, edgecolor='white', linewidth=0.4)
        apply_chart_style(ax3, xlabel='Number of Deliveries', ylabel='City',
                          title='Top 10 Cities by Delivery Volume')
        x_max3 = max(top_c.values) if len(top_c) > 0 else 1
        ax3.set_xlim(0, x_max3 * 1.22)
        for bar, v in zip(bars3, top_c.values[::-1]):
            ax3.text(bar.get_width() + x_max3 * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{v:,}', va='center', color='white', fontsize=9, fontweight='600')
        fig3.tight_layout()
        full_chart(fig3, "Top 10 Cities by Delivery Volume", [(NAVY, 'Cities (Blues gradient)')])

    # ── Chart 4: Order value distribution ──
    if col_order_value and col_order_value in df.columns:
        ov_full = pd.to_numeric(df[col_order_value], errors="coerce").dropna()
        if not ov_full.empty:
            fig4, ax4 = plt.subplots(figsize=(10, 3.5))
            n, bins, patches = ax4.hist(ov_full, bins=35, edgecolor='white', linewidth=0.4)
            # Color bars by value: low=navy, mid=navy2, high=red
            for patch, left in zip(patches, bins[:-1]):
                pct = (left - bins[0]) / (bins[-1] - bins[0] + 1e-9)
                patch.set_facecolor(NAVY if pct < 0.4 else NAVY2 if pct < 0.75 else RED)
            apply_chart_style(ax4, xlabel='Order Value (₹)', ylabel='Number of Orders',
                              title='Order Value Distribution')
            fig4.tight_layout()
            full_chart(fig4, "Order Value Distribution",
                       [(NAVY, 'Low value'), (NAVY2, 'Mid value'), (RED, 'High value')])
    section_footer()


# ============================================================
# 9. SUMMARY REPORT
# ============================================================
elif eda_option == "Summary Report":

    st.markdown("""
    <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
         font-size:16px;line-height:1.6;margin-bottom:20px;">
    <b>Summary Report</b><br><br>
    A consolidated snapshot of key logistics KPIs across all analysis dimensions —
    ready for management reporting or downstream model input.
    </div>
    """, unsafe_allow_html=True)

    section_header("Summary Report", "Monthly performance overview")

    # ── KPI summary bar chart ──
    kpi_labels, kpi_values = [], []
    if col_delivery and col_delivery in df.columns:
        kpi_labels.append("Total Deliveries"); kpi_values.append(df[col_delivery].nunique())
    if col_order and col_order in df.columns:
        kpi_labels.append("Unique Orders"); kpi_values.append(df[col_order].nunique())
    if col_store and col_store in df.columns:
        kpi_labels.append("Stores Served"); kpi_values.append(df[col_store].nunique())
    if col_vehicle and col_vehicle in df.columns:
        kpi_labels.append("Vehicles"); kpi_values.append(df[col_vehicle].nunique())
    if col_driver and col_driver in df.columns:
        kpi_labels.append("Drivers"); kpi_values.append(df[col_driver].nunique())
    if col_route and col_route in df.columns:
        kpi_labels.append("Routes"); kpi_values.append(df[col_route].nunique())

    if kpi_labels:
        fig1, ax1 = plt.subplots(figsize=(10, 3.5))
        bar_c = [NAVY, NAVY2, NAVY3, RED, NAVY, NAVY2]
        ax1.bar(kpi_labels, kpi_values,
                color=[bar_c[i % len(bar_c)] for i in range(len(kpi_labels))])
        apply_chart_style(ax1, xlabel='KPI Dimension', ylabel='Count',
                          title='Key Logistics KPI Summary')
        for i, v in enumerate(kpi_values):
            ax1.text(i, v + max(kpi_values) * 0.01, f'{v:,}',
                     ha='center', color='white', fontsize=9)
        fig1.tight_layout()
        full_chart(fig1, "Logistics KPI Overview",
                   [(NAVY, 'Primary KPIs'), (NAVY2, 'Secondary KPIs'), (RED, 'Operational')])

    # ── Cost vs Fuel pie (side by side) ──
    cost_total = pd.to_numeric(df[col_total_cost], errors="coerce").sum() if col_total_cost else 0
    labor_val  = pd.to_numeric(df[col_labor_cost], errors="coerce").sum() if col_labor_cost else 0
    fuel_val   = pd.to_numeric(df[col_fuel_cost],  errors="coerce").sum() if col_fuel_cost  else 0
    other_val  = max(0, cost_total - labor_val - fuel_val)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.set_facecolor(GREEN_BG); fig2.patch.set_facecolor(GREEN_BG)
    pie_vals = [fuel_val, labor_val, other_val]
    pie_lbls = ['Fuel', 'Labour', 'Other']
    pie_c    = ['#e67e22', '#2980b9', '#8e44ad']
    non_zero = [(v, l, c) for v, l, c in zip(pie_vals, pie_lbls, pie_c) if v > 0]
    if non_zero:
        v_list, l_list, c_list = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            v_list, labels=l_list, autopct='%1.1f%%',
            colors=c_list, startangle=90, wedgeprops=dict(width=0.6)
        )
        for t in texts + autotexts:
            t.set_color('white'); t.set_fontsize(9)
        ax2.set_title('Cost Category Split', color='white', fontsize=11, fontweight='bold')
        leg2 = list(zip(c_list, l_list))
    else:
        ax2.text(0.5, 0.5, 'Cost columns not found',
                 ha='center', va='center', color='white', transform=ax2.transAxes)
        leg2 = None
    fig2.tight_layout()

    # ── Breach vs no-breach summary pie ──
    breach_s     = pd.to_numeric(df[col_breach], errors="coerce").dropna() if col_breach else pd.Series(dtype=float)
    breach_count = int(breach_s.sum()) if not breach_s.empty else 0
    no_breach    = len(breach_s) - breach_count

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.set_facecolor(GREEN_BG); fig3.patch.set_facecolor(GREEN_BG)
    if not breach_s.empty:
        wedges, texts, autotexts = ax3.pie(
            [no_breach, breach_count],
            labels=['Compliant', 'Breach'],
            autopct='%1.1f%%', colors=['#1a6b3c', RED],
            startangle=90, wedgeprops=dict(width=0.6)
        )
        for t in texts + autotexts:
            t.set_color('white'); t.set_fontsize(9)
        ax3.set_title('Cold-Chain Compliance', color='white', fontsize=11, fontweight='bold')
        leg3 = [('#1a6b3c', 'Compliant'), (RED, 'Breach')]
    else:
        ax3.text(0.5, 0.5, 'Breach column not found',
                 ha='center', va='center', color='white', transform=ax3.transAxes)
        leg3 = None
    fig3.tight_layout()

    two_charts(fig2, "Cost Category Split", leg2,
               fig3, "Cold-Chain Compliance", leg3)

    # ── Delay distribution full width ──
    if col_delay and col_delay in df.columns:
        delay_s = pd.to_numeric(df[col_delay], errors="coerce").dropna()
        delayed_only = delay_s[delay_s > 0]
        if not delayed_only.empty:
            fig4, ax4 = plt.subplots(figsize=(10, 3.5))
            ax4.hist(delayed_only, bins=30, color=NAVY, edgecolor='white', linewidth=0.5)
            apply_chart_style(ax4, xlabel='Delay (minutes)', ylabel='Number of Deliveries',
                              title='Overall Delay Distribution (Delayed Trips Only)')
            fig4.tight_layout()
            full_chart(fig4, "Delay Distribution Summary", [(NAVY, 'Delayed trips')])

    section_footer()


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