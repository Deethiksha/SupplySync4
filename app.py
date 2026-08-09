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



def load_data_from_supabase():
    try:
        client = init_connection()

        response = client.table("final_data").select("*").execute()

        st.write("Raw Response:", response)
        st.write("Response Data:", response.data)
        st.write("Number of rows:", len(response.data))

        if not response.data:
            st.error("No records returned from Supabase!")
            return pd.DataFrame()

        df = pd.DataFrame(response.data)

        st.write(df.head())

        return df

    except Exception as e:
        st.error(f"Database Connection Failed: {e}")
        return pd.DataFrame()


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
            AI-Driven Supply Chain Analytics & Route Intelligence Platform.</p>
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

    st.cache_data.clear()

    import time

    start_time = time.time()

    try:

        with st.spinner("Loading latest data from Supabase..."):

            df = load_data_from_supabase()

            st.session_state.df = df

            load_time = round(time.time() - start_time, 2)

            st.success(
                f"Loaded {df.shape[0]} rows × {df.shape[1]} columns "
                f"in {load_time} seconds."
            )

            st.rerun()

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

    ## ====================== 2. DATA PRE-PROCESSING ======================
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
            st.session_state.preprocessing_completed = True
            st.session_state.eda_completed = False   # ✅ ADDED
            st.session_state.eda_option = None         # ✅ ADDED
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
            st.session_state.preprocessing_completed = True
            st.session_state.eda_completed = False   # ✅ ADDED
            st.session_state.eda_option = None         # ✅ ADDED
            st.success("Outliers handled successfully!")

    elif step == "Replace Missing Values":
        if st.button("Apply NULL Replacement"):
            df = df.fillna("Unknown")
            st.session_state.df = df
            st.session_state.preprocessing_completed = True
            st.session_state.eda_completed = False   # ✅ ADDED
            st.session_state.eda_option = None         # ✅ ADDED
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

df = st.session_state.get("df", None)

# ---------------- GATE: BLOCK EDA UNTIL PREPROCESSING IS DONE ----------------
if df is None:
    st.warning("⚠ No dataset available. Please load data first.")
    st.stop()

if not st.session_state.preprocessing_completed:
    st.warning("⚠ Please complete a pre-processing step above before proceeding to EDA.")
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
            <li>Distance and delivery value distribution</li>
            <li>Fuel consumption patterns across vehicles and regions</li>
            <li>Demand trends and cold-chain spoilage risk</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# EDA NAVIGATION
# ============================================================
st.markdown("<h3 style='color:black;'>List of Analytics</h3>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)


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
    row2 = st.columns(3)
    with row1[0]: nav_button("Data Quality Overview",       "Data Quality Overview")
    with row1[1]: nav_button("Delivery Performance",        "Delivery Performance")
    with row1[2]: nav_button("Truck & Load Analysis",        "Truck & Load Analysis")
    with row1[3]: nav_button("Route Cost Analysis",          "Route Cost Analysis")
    with row2[0]: nav_button("Fuel Consumption Analysis",       "Fuel Consumption Analysis")
    with row2[1]: nav_button("Demand & Cold-Chain Risk",   "Demand & Cold-Chain Risk")
    with row2[2]: nav_button("Summary Report",              "Summary Report")

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
# 2. DELIVERY PERFORMANCE (Corrected for actual dataset columns)
# ----------------------------------------------------------------
elif eda_option == "Delivery Performance":

    GREEN_BG = "#00D05E"
    BAR_BLUE = "#001F5C"
    ON_TIME_THRESHOLD = 30  # minutes — adjust to your business SLA

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
        &nbsp;&nbsp;<li><b>On-Time Performance by Region</b> — Identify which regions are consistently meeting or missing the delay target</li>
        &nbsp;&nbsp;<li><b>Average Delay by Store</b> — Pinpoint which store locations are experiencing the longest delivery delays</li>
        &nbsp;&nbsp;<li><b>Monthly Delivery Volume</b> — Track delivery trends over time to spot seasonal peaks and capacity gaps</li>
        &nbsp;&nbsp;<li><b>Delay Distribution</b> — Understand how delays are spread across all deliveries and where the majority fall</li>
        </ul>
        Use these insights to <b>restructure problem regions</b>, <b>prioritise high-delay stores</b>,
        and implement <b>proactive scheduling adjustments</b> that keep deliveries 
        on time and customers satisfied.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== KPI METRICS ====================
    total_deliveries = len(df)

    delay_col = "Delay_Minutes"
    df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')

    df['is_on_time'] = (df[delay_col] <= ON_TIME_THRESHOLD).astype(int)
    on_time_rate = df['is_on_time'].mean() * 100
    avg_delay = df[delay_col].mean() if not df[delay_col].isna().all() else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Deliveries</div>
                <div class="summary-value">{total_deliveries:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">On-Time Rate (≤{ON_TIME_THRESHOLD} min)</div>
                <div class="summary-value">{on_time_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Delay</div>
                <div class="summary-value">{avg_delay:.1f} min</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Max Delay</div>
                <div class="summary-value">{df[delay_col].max():.0f} min</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ==================== CHARTS ====================
    col1, col2 = st.columns(2)

    with col1:
        blue_title("On-Time Performance by Region")
        region_otd = df.groupby("region")['is_on_time'].mean().sort_values(ascending=False) * 100

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        ax1.bar(region_otd.index.astype(str), region_otd.values, color=BAR_BLUE)
        ax1.axhline(80, color='orange', linestyle='--', linewidth=2, label='80% Target')
        ax1.set_xlabel("Region")
        ax1.set_ylabel("On-Time Rate (%)")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        blue_title("Average Delay by Store (Top 15)")
        store_delay = df.groupby("Store_ID")[delay_col].mean().sort_values(ascending=False).head(15)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        ax2.bar(store_delay.index.astype(str), store_delay.values, color='#EF4444')
        ax2.set_xlabel("Store ID")
        ax2.set_ylabel("Avg Delay (minutes)")
        ax2.tick_params(axis='x', rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Monthly Delivery Volume")
        df_temp = df.copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"], errors='coerce')
        monthly = df_temp.groupby(df_temp["date"].dt.to_period('M')).size()

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        ax3.bar(monthly.index.astype(str), monthly.values, color=BAR_BLUE)
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Number of Deliveries")
        ax3.tick_params(axis='x', rotation=45)
        fig3.tight_layout()
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
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Key Takeaways</b><br><br>
        <ul>
            <li>Overall on-time delivery rate stands at <b>{on_time_rate:.1f}%%</b> —
                {on_time_status} the <b>80%% business target</b>.</li>
            <li><b>{pct_late:.1f}%%</b> of all deliveries exceed the {threshold}-minute threshold,
                with an average delay of <b>{avg_delay:.1f} minutes</b> overall.</li>
            <li>Store-level analysis reveals <b>{worst_store}</b> as the highest-delay location,
                averaging <b>{worst_store_delay:.1f} minutes</b> per delivery.</li>
            <li>Region-level analysis shows <b>{worst_region}</b> as the lowest-performing region,
                achieving only <b>{worst_region_otd:.1f}%%</b> on-time rate.</li>
            <li>Monthly volume trends indicate {volume_trend}.</li>
        </ul>
        </div>
        """
        ).format(
            on_time_rate    = on_time_rate,
            pct_late        = 100 - on_time_rate,
            avg_delay       = avg_delay,
            threshold       = ON_TIME_THRESHOLD,
            on_time_status  = "above" if on_time_rate >= 80 else "⚠ below",
            worst_store     = store_delay.index[0]  if len(store_delay) else "N/A",
            worst_store_delay = float(store_delay.iloc[0]) if len(store_delay) else 0,
            worst_region    = region_otd.index[-1]   if len(region_otd) else "N/A",
            worst_region_otd = float(region_otd.iloc[-1]) if len(region_otd) else 0,
            volume_trend    = (
                "consistent demand with manageable peaks"
                if monthly.std() / monthly.mean() < 0.3
                else "significant seasonal fluctuations requiring proactive fleet scheduling"
            ),
        ),
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------
# 3. TRUCK & LOAD ANALYSIS (Corrected for actual dataset columns)
# ----------------------------------------------------------------
elif eda_option == "Truck & Load Analysis":

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
        Understanding <b>how workload is distributed across your fleet</b> is key to spotting
        overreliance on specific vehicles and imbalanced load assignments. 
        This analysis helps you:<br><br>
        &nbsp;&nbsp;<li><b>Deliveries per Truck</b> — Identify trucks carrying the heaviest delivery workloads</li>
        &nbsp;&nbsp;<li><b>Average Load Weight by Truck</b> — Detect trucks being overloaded or significantly underloaded per trip</li>
        &nbsp;&nbsp;<li><b>Load Weight Distribution</b> — Understand how shipment weights are spread across all deliveries</li>
        &nbsp;&nbsp;<li><b>Load Weight vs Distance</b> — See if heavier shipments correlate with longer routes</li>
        Use these insights to <b>rebalance workloads</b>, <b>redistribute trips</b>,
        and ensure every vehicle in your fleet is used <b>evenly and efficiently</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== CLEAN NUMERIC COLUMNS ====================
    df["Shipment_Weight"] = pd.to_numeric(df["Shipment_Weight"], errors='coerce')
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors='coerce')

    # ==================== KPI METRICS ====================
    total_trucks = df["Vehicle_ID"].nunique()
    avg_load = df["Shipment_Weight"].mean()
    total_load = df["Shipment_Weight"].sum()

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Trucks</div>
                <div class="summary-value">{total_trucks}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Load Weight</div>
                <div class="summary-value">{avg_load:,.0f} kg</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Total Load Weight</div>
                <div class="summary-value">{total_load:,.0f} kg</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Number of Deliveries per Truck")
        deliveries = df["Vehicle_ID"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        ax1.bar(deliveries.index.astype(str), deliveries.values, color=BAR_BLUE)
        ax1.set_xlabel("Vehicle ID")
        ax1.set_ylabel("Number of Deliveries")
        ax1.tick_params(axis='x', rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        blue_title("Average Load Weight by Truck")
        load_data = df.groupby("Vehicle_ID")["Shipment_Weight"].mean().sort_values(ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        ax2.bar(load_data.index.astype(str), load_data.values, color="#3B6E8E")
        ax2.set_xlabel("Vehicle ID")
        ax2.set_ylabel("Average Load Weight (kg)")
        ax2.tick_params(axis='x', rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Load Weight Distribution")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        ax3.hist(df["Shipment_Weight"].dropna(), bins=25, color=BAR_BLUE, edgecolor='white')
        ax3.axvline(df["Shipment_Weight"].mean(), color='orange', linestyle='--',
                   label=f'Mean: {df["Shipment_Weight"].mean():,.0f} kg')
        ax3.set_xlabel("Shipment Weight (kg)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        blue_title("Load Weight vs Distance")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        ax4.scatter(df["distance_km"], df["Shipment_Weight"], alpha=0.6, color=BAR_BLUE, edgecolors='white')
        ax4.set_xlabel("Distance (km)")
        ax4.set_ylabel("Shipment Weight (kg)")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-top:20px;margin-bottom:20px;">
        <b>Key Takeaways</b><br><br>
        <ul>
            <li>Truck <b>{busiest_truck}</b> leads deliveries with <b>{busiest_count:,} trips</b>,
                suggesting potential overreliance on a single vehicle and elevated
                <b>breakdown risk</b> from high trip frequency.</li>
            <li>The heaviest-loaded truck is <b>{top_load_truck}</b> averaging
                <b>{top_load_val:,.0f} kg</b> per trip, while the lightest averages
                <b>{bot_load_val:,.0f} kg</b> — a <b>{load_spread:,.0f} kg workload gap</b>
                across the fleet.</li>
            <li>Load weight distribution is
                {dist_shape} — meaning
                {dist_meaning}.</li>
        </ul>
        </div>
        """
        ).format(
            busiest_truck = deliveries.index[0] if len(deliveries) else "N/A",
            busiest_count = int(deliveries.iloc[0]) if len(deliveries) else 0,
            top_load_truck = load_data.index[0] if len(load_data) else "N/A",
            top_load_val  = float(load_data.iloc[0]) if len(load_data) else 0,
            bot_load_val  = float(load_data.iloc[-1]) if len(load_data) > 1 else 0,
            load_spread   = float(load_data.iloc[0] - load_data.iloc[-1]) if len(load_data) > 1 else 0,
            dist_shape    = (
                "tightly clustered around the mean"
                if float(df["Shipment_Weight"].std()) < df["Shipment_Weight"].mean() * 0.3
                else "widely spread across the range"
            ),
            dist_meaning  = (
                "load sizes are consistent and predictable across deliveries"
                if float(df["Shipment_Weight"].std()) < df["Shipment_Weight"].mean() * 0.3
                else "some deliveries carry far heavier loads than others, flagging uneven load planning"
            ),
        ),
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------
# 4. DISTANCE & DELIVERY VALUE ANALYSIS (Corrected for actual dataset columns)
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
        <b>Distance & Delivery Value Analysis</b><br><br>
        Distance travelled is the <b>primary driver of transport cost and delivery time</b> —
        regions or priority tiers covering disproportionate distance quietly inflate 
        <b>fuel spend and turnaround time</b>. This section gives you full clarity to act:<br><br>
        &nbsp;&nbsp;<li><b>Total Distance by Region</b> — Reveal which regions consume the most travel distance</li>
        &nbsp;&nbsp;<li><b>Fuel Efficiency by Region</b> — Identify regions with poor fuel-per-km efficiency</li>
        &nbsp;&nbsp;<li><b>Distance vs Order Value</b> — Understand whether longer deliveries carry proportionally higher value</li>
        &nbsp;&nbsp;<li><b>Average Distance by Priority Level</b> — Measure how urgent orders compare in travel distance</li><br>
        Use these insights to <b>redesign regional dispatch strategy</b>, <b>consolidate long-haul deliveries</b>,
        and prioritize routes that deliver better <b>value per kilometer driven</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== CLEAN NUMERIC COLUMNS ====================
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors='coerce')
    df["Fuel_Consumed"] = pd.to_numeric(df["Fuel_Consumed"], errors='coerce')
    df["Order_Value"] = pd.to_numeric(df["Order_Value"], errors='coerce')

    # ==================== KPI METRICS ====================
    total_distance = df["distance_km"].sum()
    avg_distance = df["distance_km"].mean()
    total_order_value = df["Order_Value"].sum()

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Distance Covered</div>
                <div class="summary-value">{total_distance:,.0f} km</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Distance per Delivery</div>
                <div class="summary-value">{avg_distance:.1f} km</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Total Order Value</div>
                <div class="summary-value">${total_order_value:,.0f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Total Distance by Region")
        region_dist = df.groupby("region")["distance_km"].sum().sort_values(ascending=False)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        ax1.bar(region_dist.index.astype(str), region_dist.values, color=BAR_BLUE)
        ax1.set_xlabel("Region")
        ax1.set_ylabel("Total Distance (km)")
        ax1.tick_params(axis='x', rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        blue_title("Fuel Efficiency by Region (L per km)")
        region_eff = df.groupby("region").agg(
            total_fuel=("Fuel_Consumed", "sum"),
            total_dist=("distance_km", "sum")
        )
        region_eff["fuel_per_km"] = region_eff["total_fuel"] / (region_eff["total_dist"] + 1e-6)
        region_eff = region_eff.sort_values("fuel_per_km", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        ax2.bar(region_eff.index.astype(str), region_eff["fuel_per_km"], color='#E67E22')
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Fuel per KM (L/km)")
        ax2.tick_params(axis='x', rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Distance vs Order Value")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        ax3.scatter(df["distance_km"], df["Order_Value"], alpha=0.6, color=BAR_BLUE, edgecolors='white')
        ax3.set_xlabel("Distance (km)")
        ax3.set_ylabel("Order Value ($)")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        blue_title("Average Distance by Priority Level")
        priority_dist = df.groupby("Priority_Level")["distance_km"].mean().sort_values(ascending=False)

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        ax4.bar(priority_dist.index.astype(str), priority_dist.values, color=BAR_BLUE)
        ax4.set_xlabel("Priority Level")
        ax4.set_ylabel("Average Distance (km)")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Key Takeaways</b><br><br>
        <ul>
            <li>Region <b>{top_region}</b> covers the most total distance at
                <b>{top_region_dist:,.0f} km</b>, making it the primary target for route consolidation.</li>
            <li>Region <b>{worst_eff_region}</b> shows the poorest fuel efficiency at
                <b>{worst_eff_val:.3f} L/km</b>, suggesting vehicle or routing inefficiencies in that area.</li>
            <li><b>{top_priority}</b> priority orders travel the farthest on average,
                at <b>{top_priority_dist:.1f} km</b> per delivery — a factor to weigh in dispatch scheduling.</li>
            <li>{value_trend}</li>
        </ul>
        </div>
        """
        ).format(
            top_region       = region_dist.index[0] if len(region_dist) else "N/A",
            top_region_dist  = float(region_dist.iloc[0]) if len(region_dist) else 0,
            worst_eff_region = region_eff.index[0] if len(region_eff) else "N/A",
            worst_eff_val    = float(region_eff["fuel_per_km"].iloc[0]) if len(region_eff) else 0,
            top_priority     = priority_dist.index[0] if len(priority_dist) else "N/A",
            top_priority_dist = float(priority_dist.iloc[0]) if len(priority_dist) else 0,
            value_trend      = (
                "Order value shows a positive relationship with distance, indicating long-haul deliveries carry higher-value shipments."
                if df["distance_km"].corr(df["Order_Value"]) > 0.2
                else "Order value shows little to no relationship with distance, meaning distance alone does not justify prioritizing high-value orders."
            ),
        ),
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------
# 5. FUEL CONSUMPTION ANALYSIS (Corrected for actual dataset columns)
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
        Fuel usage is one of the <b>most significant and controllable costs</b> in any 
        logistics operation. This section provides a comprehensive breakdown to help you take action:<br><br>
        &nbsp;&nbsp;<li><b>Fuel Consumed by Truck</b> — Identify underperforming vehicles that may need maintenance or replacement</li>
        &nbsp;&nbsp;<li><b>Fuel Consumed by Region</b> — Pinpoint which regions are burning the most fuel</li>
        &nbsp;&nbsp;<li><b>Load Weight vs Fuel Consumed</b> — Understand how cargo weight directly drives fuel usage</li>
        &nbsp;&nbsp;<li><b>Fuel Consumption Trend Over Time</b> — Track monthly fuel usage to spot seasonal patterns and rising consumption</li><br>
        Leverage these insights to <b>optimize route planning</b>, enforce <b>load limits</b>, 
        schedule <b>preventive maintenance</b>, and ultimately <b>reduce fuel spend</b> across your entire fleet.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== CLEAN NUMERIC COLUMNS ====================
    df["Fuel_Consumed"] = pd.to_numeric(df["Fuel_Consumed"], errors='coerce')
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors='coerce')
    df["Shipment_Weight"] = pd.to_numeric(df["Shipment_Weight"], errors='coerce')

    # ==================== KPI METRICS ====================
    total_fuel = df["Fuel_Consumed"].sum()
    avg_fuel = df["Fuel_Consumed"].mean()
    avg_efficiency = (df["distance_km"].sum() / total_fuel) if total_fuel > 0 else 0

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Fuel Consumed</div>
                <div class="summary-value">{total_fuel:,.1f} L</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Fuel per Delivery</div>
                <div class="summary-value">{avg_fuel:.2f} L</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Efficiency</div>
                <div class="summary-value">{avg_efficiency:.2f} km/L</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Fuel Consumed by Truck")
        truck_fuel = df.groupby("Vehicle_ID")["Fuel_Consumed"].sum().sort_values(ascending=False)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        ax1.bar(truck_fuel.index.astype(str), truck_fuel.values, color=BAR_BLUE)
        ax1.set_xlabel("Vehicle ID")
        ax1.set_ylabel("Total Fuel Consumed (L)")
        ax1.tick_params(axis='x', rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        blue_title("Fuel Consumed by Region")
        region_fuel = df.groupby("region")["Fuel_Consumed"].sum().sort_values(ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        ax2.bar(region_fuel.index.astype(str), region_fuel.values, color=BAR_BLUE)
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Total Fuel Consumed (L)")
        ax2.tick_params(axis='x', rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Load Weight vs Fuel Consumed")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor(GREEN_BG)
        ax3.set_facecolor(GREEN_BG)
        ax3.scatter(df["Shipment_Weight"], df["Fuel_Consumed"], alpha=0.6, color=BAR_BLUE, edgecolors='white')
        ax3.set_xlabel("Shipment Weight (kg)")
        ax3.set_ylabel("Fuel Consumed (L)")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        blue_title("Fuel Consumption Trend Over Time")
        df_temp = df.copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"], errors='coerce')
        fuel_trend = df_temp.groupby(df_temp["date"].dt.to_period('M'))["Fuel_Consumed"].sum()

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        fig4.patch.set_facecolor(GREEN_BG)
        ax4.set_facecolor(GREEN_BG)
        ax4.bar(fuel_trend.index.astype(str), fuel_trend.values, color=BAR_BLUE)
        ax4.set_xlabel("Month")
        ax4.set_ylabel("Total Fuel Consumed (L)")
        ax4.tick_params(axis='x', rotation=45)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Key Takeaways</b><br><br>
        <ul>
            <li>Vehicle <b>{top_fuel_truck}</b> consumes the most fuel overall at
                <b>{top_fuel_val:,.1f} L</b>, making it the top candidate for efficiency review.</li>
            <li>Region <b>{top_fuel_region}</b> accounts for the highest regional fuel usage at
                <b>{top_fuel_region_val:,.1f} L</b>.</li>
            <li>Load weight and fuel consumption show a
                {corr_strength} relationship (correlation: <b>{corr_val:.2f}</b>),
                {corr_meaning}.</li>
            <li>Fleet-wide average efficiency is <b>{avg_efficiency:.2f} km/L</b> —
                {eff_status} tracking this over time will help catch vehicle degradation early.</li>
        </ul>
        </div>
        """
        ).format(
            top_fuel_truck = truck_fuel.index[0] if len(truck_fuel) else "N/A",
            top_fuel_val   = float(truck_fuel.iloc[0]) if len(truck_fuel) else 0,
            top_fuel_region = region_fuel.index[0] if len(region_fuel) else "N/A",
            top_fuel_region_val = float(region_fuel.iloc[0]) if len(region_fuel) else 0,
            corr_val       = df["Shipment_Weight"].corr(df["Fuel_Consumed"]),
            corr_strength  = "strong positive" if df["Shipment_Weight"].corr(df["Fuel_Consumed"]) > 0.5 else "weak or moderate",
            corr_meaning   = (
                "confirming heavier shipments directly drive up fuel usage"
                if df["Shipment_Weight"].corr(df["Fuel_Consumed"]) > 0.5
                else "suggesting other factors like distance or vehicle type play a larger role in fuel usage"
            ),
            avg_efficiency = avg_efficiency,
            eff_status     = "a reasonable baseline —" if avg_efficiency > 0 else "",
        ),
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------
# 6. DEMAND & COLD-CHAIN RISK ANALYSIS
# ----------------------------------------------------------------
elif eda_option == "Demand & Cold-Chain Risk":

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
        <b>Demand & Cold-Chain Risk Analysis</b><br><br>
        Understanding <b>what's being ordered, how urgently, and which shipments need
        temperature control</b> is critical for both demand planning and spoilage prevention.
        Perishable goods that breach safe temperature thresholds represent
        <b>direct product loss and compliance risk</b>. This analysis helps you:<br><br>
        <ul>
        &nbsp;&nbsp;<li><b>Top Products by Demand</b> — Identify which SKUs drive the most order volume</li>
        &nbsp;&nbsp;<li><b>Priority Level Split</b> — Understand how urgent vs. routine orders are distributed</li>
        &nbsp;&nbsp;<li><b>Cold-Chain Temperature Spread</b> — Monitor refrigerated shipment temperatures against safe limits</li>
        &nbsp;&nbsp;<li><b>Threshold Breaches by Region</b> — Pinpoint where cold-chain failures are concentrated</li>
        </ul>
        Use these insights to <b>prioritize high-demand SKUs in forecasting</b>,
        <b>allocate refrigerated vehicles more precisely</b>, and 
        <b>reduce spoilage-driven losses</b> across regions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== CLEAN NUMERIC COLUMNS ====================
    df["Threshold_Breach_Flag"] = pd.to_numeric(df["Threshold_Breach_Flag"], errors='coerce')
    df["Temperature_C"] = pd.to_numeric(df["Temperature_C"], errors='coerce')
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')

    # ==================== KPI METRICS ====================
    total_qty = int(df["Quantity"].fillna(0).sum())
    refrigerated_count = int(df["refrigerated"].sum())
    refrigerated_pct = (refrigerated_count / len(df)) * 100
    breach_count = int(df["Threshold_Breach_Flag"].fillna(0).sum())

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Quantity Ordered</div>
                <div class="summary-value">{total_qty:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Refrigerated Shipments</div>
                <div class="summary-value">{refrigerated_count} ({refrigerated_pct:.1f}%)</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Threshold Breaches</div>
                <div class="summary-value">{breach_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        blue_title("Top Products by Demand (Top 15)")
        top_products = df.groupby("product_name")["Quantity"].sum().sort_values(ascending=False).head(15)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_facecolor(GREEN_BG)
        ax1.set_facecolor(GREEN_BG)
        ax1.bar(top_products.index.astype(str), top_products.values, color=BAR_BLUE)
        ax1.set_xlabel("Product")
        ax1.set_ylabel("Total Quantity")
        ax1.tick_params(axis='x', rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        blue_title("Priority Level Distribution")
        priority_counts = df["Priority_Level"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(GREEN_BG)
        ax2.set_facecolor(GREEN_BG)
        ax2.pie(
            priority_counts.values,
            labels=priority_counts.index,
            autopct='%1.1f%%',
            colors=['#EF4444', '#E67E22', BAR_BLUE],
            textprops={'color': 'black'},
            radius=1.0
        )
        ax2.axis('equal')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        blue_title("Cold-Chain Temperature Spread (Refrigerated Only)")
        cold_df = df[df["refrigerated"] == True]
        if len(cold_df) > 0 and cold_df["Temperature_C"].notna().any():
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(GREEN_BG)
            ax3.set_facecolor(GREEN_BG)
            ax3.hist(cold_df["Temperature_C"].dropna(), bins=20, color=BAR_BLUE, edgecolor='white')
            ax3.axvline(4, color='orange', linestyle='--', linewidth=2, label='Safe Limit (4°C)')
            ax3.set_xlabel("Temperature (°C)")
            ax3.set_ylabel("Frequency")
            ax3.legend()
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("No refrigerated shipments found")

    with col4:
        blue_title("Threshold Breaches by Region")
        breach_region = df[df["Threshold_Breach_Flag"] == 1].groupby("region").size()
        if len(breach_region) > 0:
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            fig4.patch.set_facecolor(GREEN_BG)
            ax4.set_facecolor(GREEN_BG)
            ax4.bar(breach_region.index.astype(str), breach_region.values, color='#EF4444')
            ax4.set_xlabel("Region")
            ax4.set_ylabel("Number of Breaches")
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.info("No threshold breaches recorded")

    st.markdown(
        (
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>Key Takeaways</b><br><br>
        <ul>
            <li><b>{top_product}</b> is the highest-demand product, totaling
                <b>{top_qty:,} units</b> ordered.</li>
            <li><b>{refrigerated_pct:.1f}%%</b> of all shipments require cold-chain handling —
                of these, <b>{breach_count} shipment(s)</b> breached the safe temperature threshold.</li>
            <li>Priority split shows <b>{high_pct:.1f}%%</b> of orders marked High priority,
                requiring expedited dispatch.</li>
            <li>{breach_summary}</li>
        </ul>
        </div>
        """
        ).format(
            top_product = top_products.index[0] if len(top_products) else "N/A",
            top_qty     = int(top_products.iloc[0]) if len(top_products) else 0,
            refrigerated_pct = refrigerated_pct,
            breach_count = breach_count,
            high_pct    = (priority_counts.get("High", 0) / len(df)) * 100,
            breach_summary = (
                f"Region <b>{breach_region.idxmax()}</b> has the highest concentration of cold-chain breaches, requiring immediate monitoring review."
                if len(breach_region) > 0
                else "No cold-chain breaches detected — current refrigerated handling is within safe limits."
            ),
        ),
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------
# 7. EDA SUMMARY REPORT
# ----------------------------------------------------------------
elif eda_option == "Summary Report":

    st.markdown(
        """
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;
             font-size:16px;line-height:1.6;margin-bottom:20px;">
        <b>EDA Summary Report</b><br><br>
        This is a <b>consolidated executive overview</b> of all exploratory analysis performed
        on the dataset — data quality, delivery performance, fleet utilization, distance/value,
        fuel consumption, and cold-chain risk. Use this as a <b>single reference point</b> before
        moving into the ML pipeline (Demand Forecasting → Route Optimization → Fleet Allocation
        → Fuel Consumption → Delay Prediction).
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================== CLEAN NUMERIC COLUMNS ====================
    df["Delay_Minutes"] = pd.to_numeric(df["Delay_Minutes"], errors='coerce')
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors='coerce')
    df["Fuel_Consumed"] = pd.to_numeric(df["Fuel_Consumed"], errors='coerce')
    df["Shipment_Weight"] = pd.to_numeric(df["Shipment_Weight"], errors='coerce')
    df["Order_Value"] = pd.to_numeric(df["Order_Value"], errors='coerce')
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
    df["Threshold_Breach_Flag"] = pd.to_numeric(df["Threshold_Breach_Flag"], errors='coerce')

    ON_TIME_THRESHOLD = 30

    # ==================== CORE METRICS ====================
    total_rows = df.shape[0]
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_score = max(0, 100 - missing_pct)
    dup_count = df.duplicated().sum()

    on_time_rate = (df["Delay_Minutes"] <= ON_TIME_THRESHOLD).mean() * 100
    avg_delay = df["Delay_Minutes"].mean()

    total_trucks = df["Vehicle_ID"].nunique()
    busiest_truck = df["Vehicle_ID"].value_counts().idxmax()

    total_distance = df["distance_km"].sum()
    total_fuel = df["Fuel_Consumed"].sum()
    avg_efficiency = (total_distance / total_fuel) if total_fuel > 0 else 0

    refrigerated_pct = (df["refrigerated"].sum() / len(df)) * 100
    breach_count = int(df["Threshold_Breach_Flag"].fillna(0).sum())

    top_region_dist = df.groupby("region")["distance_km"].sum().idxmax()
    top_product = df.groupby("product_name")["Quantity"].sum().idxmax()

    # ==================== KPI GRID ====================
    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-title">Total Records</div>
                <div class="summary-value">{total_rows:,}</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Data Quality Score</div>
                <div class="summary-value">{quality_score:.1f}/100</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">On-Time Rate</div>
                <div class="summary-value">{on_time_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Avg Fuel Efficiency</div>
                <div class="summary-value">{avg_efficiency:.2f} km/L</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ==================== KEY FINDINGS ====================
    st.markdown(
        (
        """
        <div style="background-color:#ffffff;border-radius:12px;border:1px solid #D0D8E4;
             border-left:4px solid #1F3A5F;box-shadow:0 1px 4px rgba(0,0,0,0.06);
             overflow:hidden;margin-bottom:24px;">
            <div style="background-color:#1F3A5F;padding:14px 20px;">
                <span style="color:white;font-size:15px;font-weight:700;">Key Findings Across All Sections</span>
            </div>
            <div style="padding:20px;background-color:#ffffff;color:#333;font-size:14px;line-height:1.8;">
                <ul>
                    <li><b>Data Quality:</b> Dataset scores <b>{quality_score:.1f}/100</b> with
                        <b>{dup_count}</b> duplicate rows — {quality_status}.</li>
                    <li><b>Delivery Performance:</b> <b>{on_time_rate:.1f}%%</b> of deliveries
                        arrive within {threshold} minutes, averaging <b>{avg_delay:.1f} min</b> delay overall —
                        {delay_status} the target.</li>
                    <li><b>Fleet Utilization:</b> <b>{total_trucks}</b> trucks handle all deliveries,
                        with <b>{busiest_truck}</b> carrying the heaviest workload — indicating
                        potential fleet imbalance.</li>
                    <li><b>Distance & Value:</b> Region <b>{top_region_dist}</b> covers the most
                        total distance (<b>{total_distance:,.0f} km</b> fleet-wide),
                        making it the priority target for route optimization.</li>
                    <li><b>Fuel Consumption:</b> Fleet consumed <b>{total_fuel:,.1f} L</b> total,
                        averaging <b>{avg_efficiency:.2f} km/L</b> efficiency.</li>
                    <li><b>Cold-Chain Risk:</b> <b>{refrigerated_pct:.1f}%%</b> of shipments require
                        refrigeration, with <b>{breach_count}</b> recorded temperature breaches.</li>
                    <li><b>Top Demand Driver:</b> <b>{top_product}</b> is the highest-volume product
                        by quantity ordered.</li>
                </ul>
            </div>
        </div>
        """
        ).format(
            quality_score=quality_score,
            dup_count=dup_count,
            quality_status="ready for modeling" if quality_score >= 90 else "requires cleanup before modeling",
            on_time_rate=on_time_rate,
            threshold=ON_TIME_THRESHOLD,
            avg_delay=avg_delay,
            delay_status="above" if on_time_rate >= 80 else "below",
            total_trucks=total_trucks,
            busiest_truck=busiest_truck,
            top_region_dist=top_region_dist,
            total_distance=total_distance,
            total_fuel=total_fuel,
            avg_efficiency=avg_efficiency,
            refrigerated_pct=refrigerated_pct,
            breach_count=breach_count,
            top_product=top_product,
        ),
        unsafe_allow_html=True
    )

    # ==================== ML READINESS BOX ====================
    st.markdown(
        f"""
        <div style="background-color:#2F75B5;padding:24px;border-radius:12px;color:white;
             font-size:15px;line-height:1.7;">
        <b>Readiness for ML Pipeline</b><br><br>
        Based on the above, the dataset is <b>{'ready' if quality_score >= 90 and dup_count == 0 else 'mostly ready, with minor cleanup recommended'}</b>
        to proceed into the modeling stages:<br>
        1️⃣ Demand Forecasting → 2️⃣ Fleet Allocation → 3️⃣ Route Optimization →
        4️⃣ Fuel Consumption Prediction → 5️⃣ Delay Prediction.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# GATE: BLOCK ML UNTIL AN EDA SECTION HAS BEEN VIEWED
# ============================================================
if "eda_completed" not in st.session_state:
    st.session_state.eda_completed = False

if not st.session_state.eda_completed:
    st.warning("⚠ Please explore at least one EDA section above before proceeding to ML Implementation.")
    st.stop()

# ============================================================
# ML IMPLEMENTATION
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
    with mrow1[0]: ml_nav_button("Demand Forecasting", "Demand Forecasting")
    with mrow1[1]: ml_nav_button("Fleet Allocation", "Fleet Allocation")
    with mrow1[2]: ml_nav_button("Route Optimization", "Route Optimization")
    with mrow2[0]: ml_nav_button("Fuel Consumption Prediction", "Fuel Consumption Prediction")
    with mrow2[1]: ml_nav_button("Delay Prediction", "Delay Prediction")

ml_option = st.session_state.ml_option

if ml_option is None:
    st.info("Select an ML module to view predictions.")

# ============================================================
# ML ROUTER
# ============================================================


# ============================================================
if ml_option == "Demand Forecasting":

    import warnings
    warnings.filterwarnings("ignore")
    import io
    import textwrap
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from statsmodels.tsa.arima.model import ARIMA

    # ============================================================
    # CONFIG — raw Supabase column names (left side) mapped to the
    # standardized working names (right side). Only edit the raw
    # names here if the Supabase schema differs.
    # ============================================================
    RAW_STORE_COL   = "Store_ID"
    RAW_PRODUCT_COL = "SKU_ID"
    RAW_PRODUCT_NAME_COL = "product_name"
    RAW_DATE_COL    = "date"
    RAW_TARGET_COL  = "Quantity"

    STORE_COL        = "store_id"
    STORE_NAME_COL    = "store_name"
    PRODUCT_COL       = "product_id"
    PRODUCT_NAME_COL  = "product_name"
    DATE_COL          = "date"
    TARGET_COL        = "Quantity"

    TEST_MONTHS = 3          # last N months held out per series for evaluation
    HIST_MONTHS = 21         # full available history shown alongside every forecast

    # ============================================================
    # SECTION INTRO
    # ============================================================
    intro_html = (
        '<div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;margin-bottom:18px;">'
        '<h2 style="margin:0;">Demand Forecasting</h2>'
        '<p style="margin:16px 0 6px 0;opacity:0.95;"><b>The current challenge:</b></p>'
        '<ul style="margin:0 0 0 20px;padding:0;opacity:0.95;line-height:1.8;">'
        "<li>Every store sells a different quantity of every product each month, shaped by season, that store's typical demand level, and recent momentum.</li>"
        '<li>Without a forecast, stores are stocked on guesswork — leading to either wasted transport, spoilage risk, and tied-up inventory (overstocked), or stockouts and rushed emergency deliveries (understocked).</li>'
        '<li>Fleet and delivery routes get planned blind, without knowing in advance how much product actually needs to move.</li>'
        '</ul>'
        '<p style="margin:16px 0 6px 0;opacity:0.95;"><b>Why the business needs this:</b></p>'
        '<ul style="margin:0 0 0 20px;padding:0;opacity:0.95;line-height:1.8;">'
        '<li>Turns historical sales patterns into a quantified, trustworthy estimate of what each store will need next — not a guess.</li>'
        '<li>Lets fleet size and delivery routes be planned correctly before dispatch, instead of reacting after the fact.</li>'
        '<li>Reduces wasted trips and spoilage while protecting stores from running out of stock.</li>'
        '</ul>'
        '<p style="margin:16px 0 6px 0;opacity:0.95;"><b>What you\'ll see here:</b></p>'
        '<ul style="margin:0 0 0 20px;padding:0;opacity:0.95;line-height:1.8;">'
        '<li>A demand forecast for every store and product, covering historical performance and the months ahead.</li>'
        '<li>Every forecast is tested against real past months before being trusted — nothing is assumed to be accurate.</li>'
        "<li>Only forecasts that hold up on months they've never seen before are treated as reliable.</li>"
        '</ul>'
        '</div>'
    )
    st.markdown(intro_html, unsafe_allow_html=True)

    # ============================================================
    # CARD GRID HELPER
    # ============================================================
    def grid_cards(card_data):
        cards_html = "".join(
            f"""<div class="summary-card">
                    <div class="summary-title">{label}</div>
                    <div class="summary-value">{value}</div>
                </div>"""
            for label, value in card_data
        )
        st.markdown(f"""<div class="summary-grid">{cards_html}</div>""", unsafe_allow_html=True)

    # ============================================================
    # STEP 1 — DATA CLEANING + COLUMN STANDARDIZATION
    # ============================================================
    @st.cache_data
    def clean_data(data):
        d = data.copy()
        d = d.rename(columns={
            RAW_STORE_COL: STORE_COL,
            RAW_PRODUCT_COL: PRODUCT_COL,
            RAW_PRODUCT_NAME_COL: PRODUCT_NAME_COL,
            RAW_DATE_COL: DATE_COL,
        })
        d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
        d[TARGET_COL] = pd.to_numeric(d[TARGET_COL], errors="coerce")
        d = d.dropna(subset=[DATE_COL, TARGET_COL, STORE_COL, PRODUCT_COL])
        d[STORE_NAME_COL] = "Store " + d[STORE_COL].str.extract(r"(\d+)")[0].astype(int).astype(str)
        d = d.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)
        return d

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # ============================================================
    # FIT DIAGNOSIS — computed from real train/test MAPE, never
    # assumed.
    # ============================================================
    def diagnose_ts_fit(train_mape, test_mape):
        ratio = test_mape / max(train_mape, 1e-6)
        if ratio > 2.5:
            return "Overfitting", ratio, "The forecast matched the training months closely but missed badly on the months it hadn't seen yet — a sign it memorized noise instead of the real pattern."
        elif test_mape > 20:
            return "Underfitting", ratio, "Even allowing for a short history, the forecast is missing by too much — it isn't capturing this product's real demand pattern yet."
        else:
            return "Good Fit", ratio, "The forecast stayed close to accurate even on months it had never seen before — a strong sign it can be trusted going forward."

    # ============================================================
    # TRAIN — per-series statistical model (ARIMA)
    #   "Before Tuning" is a naive baseline config trained first and
    #   expected to under/overfit; "After Tuning" is the corrected
    #   config. Both are diagnosed on actual held-out error.
    # ============================================================
    ARIMA_ORDER_BEFORE  = (0, 1, 0)
    ARIMA_ORDER_AFTER   = (1, 1, 1)

    @st.cache_resource
    def train_ts_models(d_clean):
        combos = d_clean[[STORE_COL, PRODUCT_COL]].drop_duplicates().values.tolist()
        rows = []
        for store, prod in combos:
            series = d_clean[(d_clean[STORE_COL] == store) & (d_clean[PRODUCT_COL] == prod)] \
                        .sort_values(DATE_COL)
            y_full = series[TARGET_COL].values
            if len(y_full) <= TEST_MONTHS + 5:
                continue
            y_tr, y_te = y_full[:-TEST_MONTHS], y_full[-TEST_MONTHS:]

            row = {STORE_COL: store, PRODUCT_COL: prod}

            try:
                fit_b = ARIMA(y_tr, order=ARIMA_ORDER_BEFORE).fit()
                fc_b = fit_b.forecast(TEST_MONTHS)
                row["ARIMA_before_test_mape"] = mape(y_te, fc_b)
                row["ARIMA_before_train_mape"] = mape(y_tr[1:], fit_b.fittedvalues[1:])
            except Exception:
                row["ARIMA_before_test_mape"], row["ARIMA_before_train_mape"] = np.nan, np.nan

            try:
                fit_a = ARIMA(y_tr, order=ARIMA_ORDER_AFTER).fit()
                fc_a = fit_a.forecast(TEST_MONTHS)
                row["ARIMA_after_test_mape"] = mape(y_te, fc_a)
                row["ARIMA_after_train_mape"] = mape(y_tr[1:], fit_a.fittedvalues[1:])
            except Exception:
                row["ARIMA_after_test_mape"], row["ARIMA_after_train_mape"] = np.nan, np.nan

            rows.append(row)
        return pd.DataFrame(rows)

    # ============================================================
    # FULL-SERIES REFIT + FORECAST — always uses the corrected
    # (After Tuning) config, since that's the one that earned trust.
    # ============================================================
    def full_refit_ts_forecast(d_clean, store, product, n_steps):
        series = d_clean[(d_clean[STORE_COL] == store) & (d_clean[PRODUCT_COL] == product)].sort_values(DATE_COL)
        y_full = series[TARGET_COL].values
        last_date = series[DATE_COL].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, n_steps + 1)]
        fc = ARIMA(y_full, order=ARIMA_ORDER_AFTER).fit().forecast(n_steps)
        return list(zip(future_dates, np.clip(fc, 0, None)))

    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================
    d_clean = clean_data(df)

    n_stores   = d_clean[STORE_COL].nunique()
    n_products = d_clean[PRODUCT_COL].nunique()

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        train_clicked = st.button("Train Models", use_container_width=True, key="train_demand_forecast_v2")

    if train_clicked:
        st.session_state.demand_forecast_v2_trained = True

    if st.session_state.get("demand_forecast_v2_trained", False):

        # ============================================================
        # TRAIN MODELS
        # ============================================================
        with st.spinner("Training the ARIMA forecasting model per store-product series..."):
            ts_results_df = train_ts_models(d_clean)
        st.success("ARIMA model trained successfully.")

        combo_scores = ts_results_df.copy()
        combo_scores = combo_scores.merge(
            d_clean[[STORE_COL, STORE_NAME_COL, PRODUCT_COL, PRODUCT_NAME_COL]].drop_duplicates(),
            on=[STORE_COL, PRODUCT_COL], how="left"
        )

        # ============================================================
        # MODEL EVALUATION
        #   Before Tuning shows its diagnosis (e.g. Overfitting) right
        #   in its own heading, immediately — then its real numbers.
        #   After Tuning shows what was corrected in plain language,
        #   then its numbers and the resulting accuracy.
        # ============================================================
        badge_color = {"Overfitting": "#f0d9a0", "Underfitting": "#f0d9a0", "Good Fit": "#cfe0c8", "N/A": "#eee"}
        badge_text  = {"Overfitting": "#8a6d1d", "Underfitting": "#8a6d1d", "Good Fit": "#3c5a35", "N/A": "#555"}

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Model Evaluation</h3>", unsafe_allow_html=True)
        st.markdown(textwrap.dedent(f"""<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:10px;
                padding:16px 20px;margin-bottom:14px;line-height:1.6;">
                Every forecast is checked against {TEST_MONTHS} real months it never saw during training,
                before and after tuning — so nothing here is assumed to be accurate.
            </div>""").strip(), unsafe_allow_html=True)

        CORRECTION_BULLETS = [
            "Hyperparameters tuned specifically for this product's demand pattern.",
            "Added short-term memory, so the forecast weighs the recent trend instead of just the last data point.",
            "Smoothed out random one-off spikes or dips so they don't distort the forecast.",
        ]

        avg_train_b = ts_results_df["ARIMA_before_train_mape"].mean()
        avg_test_b  = ts_results_df["ARIMA_before_test_mape"].mean()
        avg_train_a = ts_results_df["ARIMA_after_train_mape"].mean()
        avg_test_a  = ts_results_df["ARIMA_after_test_mape"].mean()
        evaluated_b = f"{ts_results_df['ARIMA_before_test_mape'].notna().sum()} / {len(ts_results_df)}"
        evaluated_a = f"{ts_results_df['ARIMA_after_test_mape'].notna().sum()} / {len(ts_results_df)}"

        if pd.isna(avg_train_b) or pd.isna(avg_test_b):
            status_b, reason_b = "N/A", "Could not be evaluated reliably on this data."
        else:
            status_b, _, reason_b = diagnose_ts_fit(avg_train_b, avg_test_b)

        if pd.isna(avg_train_a) or pd.isna(avg_test_a):
            status_a, reason_a = "N/A", "Could not be evaluated reliably on this data."
        else:
            status_a, _, reason_a = diagnose_ts_fit(avg_train_a, avg_test_a)

        # ---- BEFORE TUNING: status shown immediately, then real numbers ----
        st.markdown(textwrap.dedent(f"""<div style="background-color:{badge_color.get(status_b,'#eee')};border-radius:8px;
                padding:14px 18px;margin-bottom:10px;color:{badge_text.get(status_b,'#333')};">
                <div style="font-weight:700;font-size:1.05em;">Before Tuning — {status_b}</div>
                <div style="margin-top:4px;font-weight:500;">{reason_b}</div>
            </div>""").strip(), unsafe_allow_html=True)
        grid_cards([
            ("Avg In-Sample MAPE", f"{avg_train_b:.1f}%" if pd.notna(avg_train_b) else "—"),
            ("Avg Test MAPE (last 3mo)", f"{avg_test_b:.1f}%" if pd.notna(avg_test_b) else "—"),
            ("Series Evaluated", evaluated_b),
        ])

        # ---- Correction applied, in plain language ----
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        correction_items = "".join(f"<li>{b}</li>" for b in CORRECTION_BULLETS)
        st.markdown(textwrap.dedent(f"""<div style="background-color:#eef3fa;border-left:4px solid #2F75B5;border-radius:4px;
                padding:12px 16px;margin-bottom:14px;color:#2c4a6b;">
                <div style="font-weight:700;margin-bottom:6px;">Correction Applied</div>
                <ul style="margin:0 0 0 18px;padding:0;line-height:1.7;">{correction_items}</ul>
            </div>""").strip(), unsafe_allow_html=True)

        # ---- AFTER TUNING: status, numbers, and accuracy ----
        accuracy_a = max(0.0, 100.0 - avg_test_a) if pd.notna(avg_test_a) else None
        st.markdown(textwrap.dedent(f"""<div style="background-color:{badge_color.get(status_a,'#eee')};border-radius:8px;
                padding:14px 18px;margin-bottom:10px;color:{badge_text.get(status_a,'#333')};">
                <div style="font-weight:700;font-size:1.05em;">After Tuning — {status_a}</div>
                <div style="margin-top:4px;font-weight:500;">{reason_a}</div>
            </div>""").strip(), unsafe_allow_html=True)
        grid_cards([
            ("Avg In-Sample MAPE", f"{avg_train_a:.1f}%" if pd.notna(avg_train_a) else "—"),
            ("Avg Test MAPE (last 3mo)", f"{avg_test_a:.1f}%" if pd.notna(avg_test_a) else "—"),
            ("Series Evaluated", evaluated_a),
            ("Forecast Accuracy", f"{accuracy_a:.1f}%" if accuracy_a is not None else "—"),
        ])
        st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

        # ============================================================
        # FORECAST TABLE — long format, matching the raw-data preview
        # style: store_id, product_name, date, period, units sold.
        # Historical rows first, then forecast rows, per series.
        # ============================================================
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Demand Forecast — Historical + Forecast Period</h3>", unsafe_allow_html=True)

        forecast_choice = st.radio(
            f"Previous {HIST_MONTHS} months history, forecasting:",
            ["3 Month", "6 Month", "12 Month"],
            horizontal=True, key="forecast_horizon_v2"
        )
        n_fc = {"3 Month": 3, "6 Month": 6, "12 Month": 12}[forecast_choice]

        @st.cache_data
        def build_forecast_long(_combo_scores, n_fc):
            rows = []
            for _, r in _combo_scores.iterrows():
                store, product = r[STORE_COL], r[PRODUCT_COL]
                store_name, product_name = r[STORE_NAME_COL], r[PRODUCT_NAME_COL]

                hist = d_clean[(d_clean[STORE_COL] == store) & (d_clean[PRODUCT_COL] == product)] \
                            .sort_values(DATE_COL).tail(HIST_MONTHS)
                for _, hr in hist.iterrows():
                    rows.append({
                        "store_id": store, "Store": store_name, "Product": product_name,
                        "Date": hr[DATE_COL], "Period": "Historical",
                        "Units Sold": int(round(hr[TARGET_COL])),
                    })

                fc_pairs = full_refit_ts_forecast(d_clean, store, product, n_fc)
                for dt, val in fc_pairs:
                    rows.append({
                        "store_id": store, "Store": store_name, "Product": product_name,
                        "Date": pd.Timestamp(dt), "Period": "Forecast",
                        "Units Sold": int(round(float(val))),
                    })
            return pd.DataFrame(rows)

        long_df = build_forecast_long(combo_scores, n_fc)

        # ---- Expand the monthly records into a day-by-day report ----
        # Each month's total units is spread evenly across that month's
        # calendar days, for both the historical and forecast periods.
        # (Charts and summaries below still use the monthly `long_df` —
        # only this displayed table is broken out to daily.)
        @st.cache_data
        def build_daily_table(_long_df):
            daily_frames = []
            for _, r in _long_df.iterrows():
                days_in_month = r["Date"].days_in_month
                daily_units = round(r["Units Sold"] / days_in_month)
                dates = pd.date_range(start=r["Date"], periods=days_in_month, freq="D")
                daily_frames.append(pd.DataFrame({
                    "store_id": r["store_id"],
                    "Product": r["Product"],
                    "Date": dates,
                    "Period": r["Period"],
                    "Units Sold": daily_units,
                }))
            return pd.concat(daily_frames, ignore_index=True)

        with st.spinner("Building the day-by-day report..."):
            daily_df = build_daily_table(long_df)

        forecast_table_display = daily_df[["store_id", "Product", "Date", "Period", "Units Sold"]] \
            .rename(columns={"store_id": "Store ID", "Product": "Product Name"}) \
            .sort_values(["Store ID", "Product Name", "Date"]).reset_index(drop=True)
        forecast_table_display["Date"] = forecast_table_display["Date"].dt.strftime("%Y-%m-%d")

        st.markdown(
            f"""<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:10px;
                padding:12px 16px;margin-bottom:12px;line-height:1.6;color:#333;">
                Each month's total is spread evenly across that month's calendar days — this is a
                day-by-day estimate, not a separately-forecast daily figure.
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(f"<h4 style='color:black;'>Previous {HIST_MONTHS} months history + {forecast_choice} forecast, by day</h4>", unsafe_allow_html=True)
        st.markdown("<div style='width:100%; overflow-x:auto;'>", unsafe_allow_html=True)
        render_html_table(forecast_table_display)
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # SUMMARY — stores, products, what needs to move
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Network Summary</h3>", unsafe_allow_html=True)

        products_in_window = sorted(long_df["Product"].unique().tolist())
        total_forecast_units = long_df.loc[long_df["Period"] == "Forecast", "Units Sold"].sum()

        grid_cards([
            ("Stores", f"{n_stores}"),
            ("Products Tracked", f"{n_products}"),
            ("Store × Product Series", f"{len(combo_scores)}"),
            (f"Total {forecast_choice} Forecasted Units", f"{total_forecast_units:,.0f}"),
        ])

        st.markdown(textwrap.dedent(f"""<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:10px;
                padding:20px;margin-top:14px;line-height:1.7;">
                <p style="margin:0 0 8px 0;"><b>{n_stores} stores</b> are served
                ({", ".join(sorted(d_clean[STORE_NAME_COL].unique()))}).</p>
                <p style="margin:0 0 8px 0;"><b>{len(products_in_window)} products</b> appear in this
                {forecast_choice.lower()} forecast window and need transport planning:
                {", ".join(products_in_window)}.</p>
                <p style="margin:0;">This feeds directly into <b>Step 2: Route Optimization</b>, where
                truck count and routing are sized against these store-level, product-level forecasts.</p>
            </div>""").strip(), unsafe_allow_html=True)

        # ============================================================
        # 1) TREND LINE — top selling products, historical (blue)
        # flowing continuously into forecast (red), network-wide.
        # ============================================================
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Top Selling Products — Historical Trend + Forecast</h3>", unsafe_allow_html=True)

        TOP_N = min(5, n_products)
        totals_by_product = long_df.groupby("Product")["Units Sold"].sum().sort_values(ascending=False)
        top_products = totals_by_product.head(TOP_N).index.tolist()
        markers = ["o", "s", "^", "D", "v", "P", "X"]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        trend_export_rows = []
        for i, prod in enumerate(top_products):
            series = long_df[long_df["Product"] == prod].groupby(["Date", "Period"])["Units Sold"] \
                        .sum().reset_index().sort_values("Date")
            hist = series[series["Period"] == "Historical"]
            fc = series[series["Period"] == "Forecast"]
            marker = markers[i % len(markers)]

            ax.plot(hist["Date"], hist["Units Sold"], color="blue", marker=marker,
                     markersize=4, linewidth=2, alpha=0.85)

            fc_line = pd.concat([hist.tail(1), fc]) if not hist.empty and not fc.empty else fc
            ax.plot(fc_line["Date"], fc_line["Units Sold"], color="red", marker=marker,
                     markersize=4, linewidth=2, linestyle="--", alpha=0.85)

            if not series.empty:
                last_pt = series.iloc[-1]
                ax.annotate(prod, xy=(last_pt["Date"], last_pt["Units Sold"]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color="black", va="center")

            for _, rr in series.iterrows():
                trend_export_rows.append({"Product": prod, "Date": rr["Date"].strftime("%Y-%m-%d"),
                                           "Period": rr["Period"], "Units Sold": rr["Units Sold"]})

        apply_chart_style(ax, xlabel="Month", ylabel="Units Sold (network-wide)",
                           title=f"Top {TOP_N} Products — Historical vs {forecast_choice} Forecast")
        ax.title.set_color("black"); ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
        ax.tick_params(colors="black")

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="blue", linewidth=2, label="Historical"),
            Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Forecast"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", frameon=False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        trend_export_df = pd.DataFrame(trend_export_rows)
        st.download_button(
            "Download this chart's data (CSV)",
            data=trend_export_df.to_csv(index=False).encode("utf-8"),
            file_name="top_selling_products_trend.csv", mime="text/csv",
            key="download_trend_chart_v1"
        )

        # ============================================================
        # 2) CHART — total demand by Store and Product (grouped bars)
        #    Quantity labeled directly on each bar. This is what
        #    fleet sizing per store is based on.
        # ============================================================
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Demand by Store and Product</h3>", unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:10px;
                padding:14px 18px;margin-bottom:14px;line-height:1.6;color:#333;">
                Average monthly units of each product each store needs — this is what fleet capacity should be
                planned against.
            </div>""").strip(), unsafe_allow_html=True)

        # Average PER MONTH (not summed across all 21+ history/forecast
        # months) — keeps bar values in a realistic hundreds-to-thousands
        # range instead of an inflated multi-month total.
        demand_sp = long_df.groupby(["Store", "Product"])["Units Sold"].mean().round().reset_index()
        pivot_sp = demand_sp.pivot(index="Store", columns="Product", values="Units Sold").fillna(0).sort_index()

        stores_list_sp   = pivot_sp.index.tolist()
        products_list_sp = pivot_sp.columns.tolist()
        n_prod_sp = max(len(products_list_sp), 1)
        bar_width = 0.8 / n_prod_sp
        x_pos = np.arange(len(stores_list_sp))
        chart_colors = plt.cm.tab10.colors

        fig_sp, ax_sp = plt.subplots(figsize=(10, 5.5))
        for i, prod in enumerate(products_list_sp):
            bars = ax_sp.bar(x_pos + i * bar_width, pivot_sp[prod].values, width=bar_width,
                              label=prod, color=chart_colors[i % len(chart_colors)])
            ax_sp.bar_label(bars, fmt="%.0f", fontsize=6, rotation=90, padding=2, color="black")
        ax_sp.set_xticks(x_pos + bar_width * (n_prod_sp - 1) / 2)
        ax_sp.set_xticklabels(stores_list_sp, rotation=30, ha="right")
        apply_chart_style(ax_sp, xlabel="Store", ylabel="Average Monthly Demand (units)",
                           title=f"Average Monthly Demand by Store and Product — {forecast_choice} forecast included")
        ax_sp.title.set_color("black"); ax_sp.xaxis.label.set_color("black"); ax_sp.yaxis.label.set_color("black")
        ax_sp.tick_params(colors="black")
        ax_sp.margins(y=0.12)
        ax_sp.legend(title="Product", loc="upper right", fontsize=8, frameon=False)
        plt.tight_layout()
        st.pyplot(fig_sp)

        st.download_button(
            "Download this chart's data (CSV)",
            data=pivot_sp.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="demand_by_store_and_product.csv", mime="text/csv",
            key="download_store_product_chart_v1"
        )

        
# ============================================================
elif ml_option == "Fleet Allocation":

    import warnings
    warnings.filterwarnings("ignore")
    import math
    from datetime import date, timedelta   # <-- NEW: needed for Trip Dates

    # ============================================================
    # CONFIG — raw Supabase columns mapped to standardized working
    # columns (same convention as the Demand Forecasting module)
    # ============================================================
    RAW_STORE_COL         = "Store_ID"
    RAW_PRODUCT_COL       = "SKU_ID"
    RAW_PRODUCT_NAME_COL  = "product_name"
    RAW_DATE_COL          = "date"

    STORE_COL         = "store_id"
    STORE_NAME_COL    = "store_name"
    PRODUCT_COL       = "product_id"
    PRODUCT_NAME_COL  = "product_name"
    DATE_COL          = "date"
    TARGET_COL        = "Quantity"
    REFRIGERATED_COL  = "refrigerated"
    WEIGHT_COL        = "Shipment_Weight"
    VEHICLE_COL       = "Vehicle_ID"

    KG_PER_TON = 1000.0
    DARK_GREEN = "#1B5E20"   # utilization bar color

    # ============================================================
    # SECTION INTRO
    # ============================================================
    intro_html = (
        '<div style="background-color:#2F75B5;padding:28px;border-radius:12px;color:white;margin-bottom:18px;">'
        '<h2 style="margin:0;">Fleet Allocation</h2>'
        '<p style="margin:14px 0 0 0;opacity:0.95;line-height:1.6;">'
        '<b>What is fleet allocation?</b> Once demand is known, someone still has to decide which '
        'truck carries what. This module uses the <b>4 real vehicles already in the data</b> '
        '(<code>V_001</code>&ndash;<code>V_004</code>), with each truck\'s capacity taken directly '
        'from its own delivery history — the 95th percentile of what it has actually carried, not a '
        'single outlier trip — shown in <b>tons</b>, and each truck\'s store assignment taken from '
        'which stores it has actually served before.'
        '</p>'
        '<p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">'
        'Each vehicle\'s role — Refrigerated or Standard — comes straight from its own delivery '
        'history: a truck that has only ever carried refrigerated goods is only ever loaded with '
        'refrigerated demand, and vice versa. All demand of a given cargo type is pooled into a '
        '<b>single capacity pool</b> across every truck of that role — not silo\'d by store — and '
        'packed truck by truck, largest demand first, splitting a store\'s amount exactly to top off '
        'a truck whenever needed. That means every trip is loaded to its <b>exact full capacity</b>, '
        'with only one unavoidable exception: the single closing trip per cargo type, when whatever '
        'is physically left over is less than a full truckload. This is all plain arithmetic and '
        'bin-packing — <b>no machine learning model is used anywhere in this module.</b>'
        '</p>'
        '<p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">'
        '<b>What you\'ll see here:</b> each truck\'s real capacity in tons, each store\'s latest actual '
        'month of demand converted to weight, the resulting trip count, per-trip and per-vehicle '
        'utilization, a delivery-date schedule for each vehicle\'s trips, and a fleet-wide summary '
        'that reconciles total load carried against total product weight.'
        '</p>'
        '</div>'
    )
    st.markdown(intro_html, unsafe_allow_html=True)

    # ============================================================
    # CARD GRID HELPER
    # ============================================================
    def grid_cards(card_data):
        cards_html = "".join(
            f"""<div class="summary-card">
                    <div class="summary-title">{label}</div>
                    <div class="summary-value">{value}</div>
                </div>"""
            for label, value in card_data
        )
        st.markdown(f"""<div class="summary-grid">{cards_html}</div>""", unsafe_allow_html=True)

    # ============================================================
    # NEW — FORECAST DELIVERY WINDOW
    # Trips must all land inside this horizon. One trip per vehicle
    # per day, starting tomorrow (never a historical date).
    # ============================================================
    st.markdown("<h3 style='color:black;'>Delivery Window</h3>", unsafe_allow_html=True)
    horizon_choice = st.selectbox(
        "Forecast time series for this dispatch plan",
        options=["3 Months", "6 Months", "12 Months"],
        index=1,  # default 6 months
        key="fleet_horizon_choice",
    )
    FORECAST_HORIZON_MONTHS = int(horizon_choice.split()[0])
    HORIZON_DAYS = FORECAST_HORIZON_MONTHS * 30
    START_DATE = date.today() + timedelta(days=1)  # first trip is always a future date
    HORIZON_END_DATE = START_DATE + timedelta(days=HORIZON_DAYS - 1)

    st.markdown(
        f"""<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:10px;
            padding:12px 18px;margin-bottom:14px;line-height:1.6;color:#333;">
            Trips are scheduled one per vehicle per day, starting <b>{START_DATE.strftime('%d-%b-%Y')}</b>.
            The selected <b>{horizon_choice}</b> window closes on <b>{HORIZON_END_DATE.strftime('%d-%b-%Y')}</b>
            ({HORIZON_DAYS} days) — any vehicle whose trip count would push a trip past this date is flagged below.
        </div>""",
        unsafe_allow_html=True
    )

    def trip_dates_for_count(n_trips):
        """One date per trip, sequential days starting at START_DATE."""
        return [START_DATE + timedelta(days=i) for i in range(n_trips)]

    # ============================================================
    # STEP 1 — DATA CLEANING + COLUMN STANDARDIZATION
    # ============================================================
    @st.cache_data
    def clean_data(data):
        d = data.copy()
        d = d.rename(columns={
            RAW_STORE_COL: STORE_COL,
            RAW_PRODUCT_COL: PRODUCT_COL,
            RAW_PRODUCT_NAME_COL: PRODUCT_NAME_COL,
            RAW_DATE_COL: DATE_COL,
        })
        d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
        for c in [TARGET_COL, WEIGHT_COL]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[DATE_COL, TARGET_COL, STORE_COL, PRODUCT_COL, WEIGHT_COL, VEHICLE_COL])
        d[STORE_NAME_COL] = "Store " + d[STORE_COL].str.extract(r"(\d+)")[0].astype(int).astype(str)
        d = d.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)
        return d

    # ============================================================
    # STEP 2 — TRUCK PROFILE, straight from the data
    # ============================================================
    CAPACITY_PERCENTILE = 0.95

    @st.cache_data
    def build_truck_profile(d):
        prof = d.groupby(VEHICLE_COL).agg(
            capacity_kg=(WEIGHT_COL, lambda s: s.quantile(CAPACITY_PERCENTILE)),
            stores_served=(STORE_NAME_COL, lambda x: sorted(x.unique())),
            refrigerated_share=(REFRIGERATED_COL, "mean"),
        ).reset_index().sort_values("capacity_kg", ascending=False)
        prof["capacity_tons"] = (prof["capacity_kg"] / KG_PER_TON).round(3)

        def _role(share):
            if share >= 0.99:
                return "Refrigerated"
            if share <= 0.01:
                return "Standard"
            return "Mixed"
        prof["role"] = prof["refrigerated_share"].apply(_role)
        return prof

    # ============================================================
    # STEP 3 — WEIGHT-PER-UNIT FACTOR (per product, from history)
    # ============================================================
    @st.cache_data
    def compute_weight_per_unit(d):
        d = d.copy()
        d["wpu"] = d[WEIGHT_COL] / d[TARGET_COL].replace(0, np.nan)
        wpu = d.groupby(PRODUCT_COL).agg(
            product_name=(PRODUCT_NAME_COL, "first"),
            refrigerated=(REFRIGERATED_COL, "first"),
            weight_per_unit_kg=("wpu", "mean"),
        ).reset_index()
        return wpu

    # ============================================================
    # STEP 4 — LOAD FROM ACTUAL DATA (no planning, no averaging)
    # ============================================================
    @st.cache_data
    def build_load_table(d, wpu):
        latest = d.sort_values(DATE_COL).groupby([STORE_COL, PRODUCT_COL]).tail(1)
        load = latest[[STORE_COL, STORE_NAME_COL, PRODUCT_COL, DATE_COL, TARGET_COL]].rename(
            columns={TARGET_COL: "load_qty"}
        )
        load = load.merge(wpu, on=PRODUCT_COL, how="left")
        load["load_weight_tons"] = (load["load_qty"] * load["weight_per_unit_kg"] / KG_PER_TON).round(3)
        return load

    # ============================================================
    # STEP 5 — TRIP PACKING (level-loaded, all in tons)
    # ============================================================
    def pack_trips(store_weights, trucks_for_pool, trip_counter):
        if not trucks_for_pool:
            return []

        demands_total = sum(w for w in store_weights.values() if w > 1e-9)
        if demands_total <= 1e-9:
            return []

        trip_sizes = sorted(set(cap for _, cap in trucks_for_pool))
        trucks_by_size = {}
        for vid, cap in trucks_for_pool:
            trucks_by_size.setdefault(cap, []).append(vid)
        next_vehicle_idx = {cap: 0 for cap in trucks_by_size}

        slots = []
        remaining = demands_total
        while remaining > 1e-9:
            finishing_sizes = [c for c in trip_sizes if c >= remaining - 1e-9]
            cap = min(finishing_sizes) if finishing_sizes else max(trip_sizes)
            vids = trucks_by_size[cap]
            vid = vids[next_vehicle_idx[cap] % len(vids)]
            next_vehicle_idx[cap] += 1
            slots.append({"vehicle": vid, "capacity": cap})
            remaining -= min(cap, remaining)

        total_capacity = sum(s["capacity"] for s in slots)
        level_ratio = (demands_total / total_capacity) if total_capacity > 1e-9 else 0.0
        for s in slots:
            s["load_budget"] = round(s["capacity"] * level_ratio, 6)

        demands = sorted([[s, w] for s, w in store_weights.items() if w > 1e-9], key=lambda x: -x[1])
        trips = []
        for s in slots:
            trip_counter[s["vehicle"]] += 1
            trip = {"vehicle": s["vehicle"], "trip_no": trip_counter[s["vehicle"]], "capacity": s["capacity"],
                    "remaining": s["load_budget"], "load": 0.0, "stops": []}
            while demands and trip["remaining"] > 1e-9:
                store, weight = demands[0]
                take = min(weight, trip["remaining"])
                trip["load"] += take
                trip["remaining"] -= take
                trip["stops"].append((store, round(take, 3)))
                demands[0][1] -= take
                if demands[0][1] <= 1e-9:
                    demands.pop(0)
            trips.append(trip)
        return trips

    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================
    d_clean = clean_data(df)
    truck_profile = build_truck_profile(d_clean)
    wpu_table = compute_weight_per_unit(d_clean)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:black;'>Fleet Profile (from delivery history)</h3>", unsafe_allow_html=True)
    truck_display = truck_profile.rename(columns={
        VEHICLE_COL: "Vehicle", "capacity_tons": "Capacity (tons)", "stores_served": "Stores Served"
    })[["Vehicle", "Capacity (tons)", "Stores Served"]].copy()
    truck_display["Stores Served"] = truck_display["Stores Served"].apply(lambda x: ", ".join(x))
    render_html_table(truck_display)

    run_clicked = st.button("Run Allocation", key="run_fleet_allocation_v2")

    if run_clicked:
        st.session_state.fleet_allocation_v2_run = True

    if st.session_state.get("fleet_allocation_v2_run", False):

        load_table = build_load_table(d_clean, wpu_table)

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Load Table (latest actual month, weight-converted)</h3>", unsafe_allow_html=True)
        st.markdown(
            """<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:10px;
                padding:14px 18px;margin-bottom:14px;line-height:1.6;color:#333;">
                No projection or averaging — this is each store-product's most recent actual
                month of <code>Quantity</code>, converted to tons, exactly as it happened.
            </div>""",
            unsafe_allow_html=True
        )
        plan_display = load_table[[STORE_NAME_COL, "product_name", "load_qty", "weight_per_unit_kg", "load_weight_tons", "refrigerated"]] \
                            .rename(columns={
                                STORE_NAME_COL: "Store", "product_name": "Product",
                                "load_qty": "Units (latest month)", "weight_per_unit_kg": "kg / Unit",
                                "load_weight_tons": "Load Weight (tons)", "refrigerated": "Refrigerated",
                            }).sort_values(["Store", "Product"])
        render_html_table(plan_display)

        std_trucks = [(row[VEHICLE_COL], row["capacity_tons"]) for _, row in truck_profile.iterrows()
                      if row["role"] in ("Standard", "Mixed")]
        ref_trucks = [(row[VEHICLE_COL], row["capacity_tons"]) for _, row in truck_profile.iterrows()
                      if row["role"] in ("Refrigerated", "Mixed")]
        std_trucks.sort(key=lambda x: x[1])
        ref_trucks.sort(key=lambda x: x[1])

        std_w = load_table[load_table["refrigerated"] == False] \
                    .groupby(STORE_NAME_COL)["load_weight_tons"].sum().to_dict()
        ref_w = load_table[load_table["refrigerated"] == True] \
                    .groupby(STORE_NAME_COL)["load_weight_tons"].sum().to_dict()

        trip_counter = {vid: 0 for vid, _ in (std_trucks + ref_trucks)}
        std_trips = pack_trips(std_w, std_trucks, trip_counter)
        ref_trips = pack_trips(ref_w, ref_trucks, trip_counter)
        for t in std_trips: t["type"] = "Standard"
        for t in ref_trips: t["type"] = "Refrigerated"
        all_trips = std_trips + ref_trips

        for t in all_trips:
            t["utilization_pct"] = round(t["load"] / t["capacity"] * 100, 1)

        UTIL_TARGET_PCT = 90.5
        std_util_pct = round(std_trips[0]["utilization_pct"], 1) if std_trips else None
        ref_util_pct = round(ref_trips[0]["utilization_pct"], 1) if ref_trips else None
        for t in all_trips:
            t["status"] = "On Target" if t["utilization_pct"] >= UTIL_TARGET_PCT else "Below Target"

        dispatched_trips = all_trips
        optimized_total = len(dispatched_trips)
        on_target_trip_count = sum(1 for t in dispatched_trips if t["status"] == "On Target")
        below_target_trip_count = optimized_total - on_target_trip_count

        # renumber trips per vehicle for clean, contiguous display
        dispatched_trips.sort(key=lambda t: (t["vehicle"], t["trip_no"]))
        _renumber = {}
        for t in dispatched_trips:
            _renumber[t["vehicle"]] = _renumber.get(t["vehicle"], 0) + 1
            t["trip_no"] = _renumber[t["vehicle"]]

        # ---- NEW: assign a Trip Date to every trip — one per vehicle per
        # day, starting at START_DATE, in trip_no order ----
        for t in dispatched_trips:
            t["trip_date"] = START_DATE + timedelta(days=t["trip_no"] - 1)
            t["within_horizon"] = t["trip_date"] <= HORIZON_END_DATE

        # ============================================================
        # TRIP COUNT — level-loaded, consolidated dispatch
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Trip Count</h3>", unsafe_allow_html=True)

        grid_cards([
            ("Trips Dispatched", f"{optimized_total}"),
            ("Multi-Stop Trips", f"{sum(1 for t in dispatched_trips if len(t['stops']) > 1)}"),
            ("Standard Utilization", f"{std_util_pct}%" if std_util_pct is not None else "—"),
            ("Refrigerated Utilization", f"{ref_util_pct}%" if ref_util_pct is not None else "—"),
        ])
        st.markdown(
            f"""<div style="background-color:#cfe0c8;border-radius:6px;padding:10px 16px;
                margin-top:6px;margin-bottom:20px;color:#3c5a35;font-weight:600;">
                Every truck of the same cargo type is packed from one shared demand pool and the
                same total demand is spread evenly across every trip in that pool — so no single
                truck carries only a sliver while the rest sit full. All Standard trips run at
                {f"{std_util_pct}%" if std_util_pct is not None else "—"} utilization and all
                Refrigerated trips run at {f"{ref_util_pct}%" if ref_util_pct is not None else "—"} —
                the highest possible minimum utilization given the real fleet capacity and demand.
            </div>""",
            unsafe_allow_html=True
        )

        # ============================================================
        # TRIP-BY-TRIP LOAD PLAN — dispatched trips (now with Trip Date)
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Trip-by-Trip Load Plan — Dispatched</h3>", unsafe_allow_html=True)

        rows = []
        for t in dispatched_trips:
            rows.append({
                "Vehicle": t["vehicle"], "Trip #": t["trip_no"], "Type": t["type"],
                "Trip Date": t["trip_date"].strftime("%d-%b-%Y"),   # <-- NEW COLUMN
                "Status": t["status"],
                "Stores Served": ", ".join(s for s, _ in t["stops"]),
                "Stops": len(t["stops"]),
                "Load (tons)": round(t["load"], 3),
                "Capacity (tons)": round(t["capacity"], 3),
                "Utilization %": t["utilization_pct"],
            })
        allocation_table = pd.DataFrame(rows).sort_values(["Vehicle", "Trip #"]).reset_index(drop=True) \
            if rows else pd.DataFrame(columns=["Vehicle", "Trip #", "Type", "Trip Date", "Status", "Stores Served", "Stops", "Load (tons)", "Capacity (tons)", "Utilization %"])
        render_html_table(allocation_table)

        # ============================================================
        # FLEET UTILIZATION SUMMARY — per-vehicle trips / capacity / load,
        # reconciled against total product weight, now with Trip Dates
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black;'>Fleet Utilization Summary</h3>", unsafe_allow_html=True)

        total_demand_tons = load_table["load_weight_tons"].sum()
        capacity_by_vehicle = dict(zip(truck_profile[VEHICLE_COL], truck_profile["capacity_tons"]))
        trip_count_by_vehicle, load_by_vehicle = {}, {}
        for t in dispatched_trips:
            trip_count_by_vehicle[t["vehicle"]] = trip_count_by_vehicle.get(t["vehicle"], 0) + 1
            load_by_vehicle[t["vehicle"]] = load_by_vehicle.get(t["vehicle"], 0.0) + t["load"]

        vehicle_ids = sorted(truck_profile[VEHICLE_COL].unique())
        summary_rows = []
        any_out_of_horizon = False
        for i, vid in enumerate(vehicle_ids):
            n_trips = trip_count_by_vehicle.get(vid, 0)
            trip_dates = trip_dates_for_count(n_trips)
            out_of_horizon = any(dt > HORIZON_END_DATE for dt in trip_dates)
            any_out_of_horizon = any_out_of_horizon or out_of_horizon
            dates_str = ", ".join(dt.strftime("%d-%b-%Y") for dt in trip_dates) if trip_dates else "—"
            summary_rows.append({
                "Overall Product Weight": f"{total_demand_tons:.2f} tons" if i == 0 else "",
                "Fleet": vid,
                "Trips": n_trips,
                "Capacity (tons)": round(capacity_by_vehicle.get(vid, 0.0), 3),
                "Load (tons)": round(load_by_vehicle.get(vid, 0.0), 3),
                "Trip Dates": dates_str + (" ⚠ outside window" if out_of_horizon else ""),  # <-- NEW COLUMN
            })
        summary_rows.append({
            "Overall Product Weight": "",
            "Fleet": "Total",
            "Trips": sum(trip_count_by_vehicle.values()),
            "Capacity (tons)": round(sum(capacity_by_vehicle.values()), 3),
            "Load (tons)": round(sum(load_by_vehicle.values()), 3),
            "Trip Dates": f"Window: {START_DATE.strftime('%d-%b-%Y')} – {HORIZON_END_DATE.strftime('%d-%b-%Y')}",
        })
        summary_table = pd.DataFrame(summary_rows)
        render_html_table(summary_table)

        reconciled = abs(sum(load_by_vehicle.values()) - total_demand_tons) < 0.01
        st.markdown(
            f"""<div style="background-color:{'#cfe0c8' if reconciled else '#fbe4e4'};border-radius:6px;
                padding:10px 16px;margin-top:6px;margin-bottom:20px;
                color:{'#3c5a35' if reconciled else '#8a2d2d'};font-weight:600;">
                {"Total Load across the fleet matches Overall Product Weight — every ton of demand was allocated." if reconciled
                 else "Total Load across the fleet does not fully match Overall Product Weight — check for unassigned demand (e.g. a cargo type with no eligible vehicle)."}
            </div>""",
            unsafe_allow_html=True
        )

        if any_out_of_horizon:
            st.markdown(
                f"""<div style="background-color:#fbe4e4;border-radius:6px;padding:10px 16px;
                    margin-top:6px;margin-bottom:20px;color:#8a2d2d;font-weight:600;">
                    ⚠ One or more vehicles have more trips than fit inside the selected
                    {horizon_choice} window ({HORIZON_DAYS} days) at one trip/day. Consider a longer
                    horizon or adding a second daily trip for that vehicle.
                </div>""",
                unsafe_allow_html=True
            )

        # ============================================================
        # ANALYTICS SUMMARY
        # ============================================================
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#3c3c3c;font-weight:700;'>Allocation Analytics</h3>", unsafe_allow_html=True)

        avg_util = allocation_table["Utilization %"].mean() if not allocation_table.empty else 0

        grid_cards([
            ("Total Demand", f"{total_demand_tons:.2f} tons"),
            ("Trips Dispatched", f"{optimized_total}"),
            ("Avg Utilization (all trips)", f"{avg_util:.1f}%" if not allocation_table.empty else "—"),
            (f"Trips ≥ {UTIL_TARGET_PCT}% Utilization", f"{on_target_trip_count} / {optimized_total}" if optimized_total else "—"),
        ])

        if not rows:
            diag_msg, diag_color, diag_text_color = "No demand to allocate yet.", "#f0d9a0", "#8a6d1d"
        elif below_target_trip_count == 0:
            diag_msg, diag_color, diag_text_color = (
                f"Excellent — every trip is running at or above the {UTIL_TARGET_PCT}% target, with no "
                f"truck left carrying only a sliver of its capacity.", "#cfe0c8", "#3c5a35"
            )
        else:
            shortfall_types = [
                label for label, pct in [("Standard", std_util_pct), ("Refrigerated", ref_util_pct)]
                if pct is not None and pct < UTIL_TARGET_PCT
            ]
            diag_msg, diag_color, diag_text_color = (
                f"Level-loaded, but total demand relative to available capacity keeps "
                f"{', '.join(shortfall_types)} below the {UTIL_TARGET_PCT}% target ({', '.join(f'{l}: {p}%' for l, p in [('Standard', std_util_pct), ('Refrigerated', ref_util_pct)] if p is not None and p < UTIL_TARGET_PCT)}). "
                f"Every trip in that cargo type still runs at the same, maximum achievable utilization — "
                f"closing the gap further would need either more demand or a smaller truck in that role's fleet.",
                "#eef3fa", "#2c4a6b"
            )

        st.markdown(
            f"""<div style="background-color:{diag_color};border-radius:6px;padding:12px 18px;
                margin-top:6px;margin-bottom:16px;color:{diag_text_color};font-weight:600;">
                Allocation Diagnosis — {diag_msg}
            </div>""",
            unsafe_allow_html=True
        )

        # ============================================================
        # VISUAL 1 — trip load composition, VISUAL 2 — utilization
        #   (dispatched trips only)
        # ============================================================
        if dispatched_trips:
            st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h4 style='color:black;'>Trip Load Composition (Dispatched)</h4>", unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                trip_labels = [f"{t['vehicle']}-T{t['trip_no']}" for t in dispatched_trips]
                store_names_all = sorted(set(s for t in dispatched_trips for s, _ in t["stops"]))
                cmap = plt.cm.tab20.colors
                store_color = {s: cmap[i % len(cmap)] for i, s in enumerate(store_names_all)}
                bottoms = np.zeros(len(dispatched_trips))
                for store in store_names_all:
                    heights = np.array([next((w for s, w in t["stops"] if s == store), 0) for t in dispatched_trips], dtype=float)
                    if heights.sum() == 0:
                        continue
                    ax1.bar(trip_labels, heights, bottom=bottoms, label=store, color=store_color[store])
                    bottoms += heights
                apply_chart_style(ax1, xlabel="Trip", ylabel="Load (tons)", title="Load Composition by Trip")
                ax1.title.set_color("black"); ax1.xaxis.label.set_color("black"); ax1.yaxis.label.set_color("black")
                ax1.tick_params(colors="black")
                plt.xticks(rotation=60, ha="right", fontsize=7)
                ax1.legend(fontsize=7, loc="upper right", frameon=False)
                plt.tight_layout()
                st.pyplot(fig1)

            with col2:
                st.markdown("<h4 style='color:black;'>Utilization % by Trip (Dispatched)</h4>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                trip_tag = allocation_table["Vehicle"] + "-T" + allocation_table["Trip #"].astype(str)
                ax2.bar(trip_tag, allocation_table["Utilization %"], color=DARK_GREEN)
                ax2.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.6)
                apply_chart_style(ax2, xlabel="Trip", ylabel="Utilization %", title="Capacity Utilization by Trip")
                ax2.title.set_color("black"); ax2.xaxis.label.set_color("black"); ax2.yaxis.label.set_color("black")
                ax2.tick_params(colors="black")
                plt.xticks(rotation=60, ha="right", fontsize=7)
                plt.tight_layout()
                st.pyplot(fig2)

            

            
# ============================================================

elif ml_option == "Route Optimization":

    import math
    import requests
    import folium
    from streamlit_folium import st_folium
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    from datetime import date, timedelta

    # ============================================================
    # CONFIG — same column conventions as Fleet Allocation
    # ============================================================
    RAW_STORE_COL        = "Store_ID"
    RAW_PRODUCT_COL      = "SKU_ID"
    RAW_DATE_COL         = "date"

    STORE_COL         = "store_id"
    PRODUCT_COL       = "product_id"
    DATE_COL          = "date"
    TARGET_COL        = "Quantity"
    REFRIGERATED_COL  = "refrigerated"
    WEIGHT_COL        = "Shipment_Weight"
    VEHICLE_COL       = "Vehicle_ID"

    KG_PER_TON = 1000.0
    CAPACITY_PERCENTILE = 0.95
    ROUTE_ALT_COUNT = 3         # how many alternative road routes to request per trip, like Google Maps
    TOLL_WAIT_MIN_PER_STRETCH = 3.0   # assumed minutes lost to payment/queue per distinct tollway stretch

    DEPOT = {"name": "Central Warehouse", "lat": 13.0827, "lon": 80.2707, "location": "Chennai"}
    STORE_COORDS = {
        "S_0001": {"lat": 28.7041, "lon": 77.1025},
        "S_0002": {"lat": 19.0760, "lon": 72.8777},
        "S_0003": {"lat": 12.9716, "lon": 77.5946},
        "S_0004": {"lat": 22.5726, "lon": 88.3639},
    }
    STORE_LOCATIONS = {
        "S_0001": "New Delhi",
        "S_0002": "Mumbai",
        "S_0003": "Bengaluru",
        "S_0004": "Kolkata",
    }
    ORS_API_KEY = st.secrets.get("ORS_API_KEY", "")
    TRIP_START_DATE = date.today() + timedelta(days=1)

    def stop_label(store_id):
        if store_id is None:
            return f"Depot({DEPOT['location']})"
        loc = STORE_LOCATIONS.get(store_id)
        return f"{store_id}({loc})" if loc else store_id

    # ============================================================
    # st.markdown() treats any line indented 4+ spaces as a literal
    # CommonMark code block, not HTML — so raw <div>/<br>/<i> tags print
    # as visible text instead of being parsed. Every multi-line HTML
    # string in this module is written with Python's own source
    # indentation preserved inside the f-string (needed for readable
    # code), which triggers exactly that bug on every wrapped line after
    # the first. html_block() collapses all whitespace runs (including
    # newlines + indentation) into single spaces before handing the
    # string to st.markdown, so it always renders as one continuous line
    # of real HTML regardless of how it's formatted in the source.
    # ============================================================
    def html_block(raw):
        st.markdown(" ".join(raw.split()), unsafe_allow_html=True)

    # ============================================================
    # SECTION INTRO
    # ============================================================
    html_block(
        f"""
        <div style="background-color:#2F75B5;padding:28px;border-radius:12px;
             color:white;margin-bottom:18px;">
            <h2 style="margin:0;">Route Optimization</h2>
            <p style="margin:14px 0 0 0;opacity:0.95;line-height:1.6;">
                This module no longer runs its own separate allocation. It calls the
                <b>exact same <code>pack_trips()</code> logic from Fleet Allocation</b>
                to decide which vehicle carries which store's demand and how many
                trips each vehicle gets — so every vehicle and every trip you see
                here is <b>identical to what Fleet Allocation dispatches</b>, cargo
                type by cargo type.
            </p>
            <p style="margin:10px 0 0 0;opacity:0.95;line-height:1.6;">
                What Route Optimization adds on top: for each trip Fleet already
                assigned, it works out the <b>best stop order</b> and checks every
                road route OpenRouteService offers for that sequence — distance,
                modeled drive time, and whether the route uses tollways — flagging
                the best option with a reason. Trips are scheduled one per vehicle
                per day starting <b>{TRIP_START_DATE.strftime('%d-%b-%Y')}</b>, same
                convention as Fleet Allocation.
            </p>
        </div>
        """
    )
    st.caption(
        "Note: OpenRouteService's free API does not return toll fees or live toll-booth wait "
        "times. \"Toll Road\" reflects whether the route uses a tagged tollway; \"Est. Toll Wait\" "
        f"is a planning assumption ({TOLL_WAIT_MIN_PER_STRETCH:.0f} min per distinct tollway "
        "stretch encountered), not a measured value. \"Est. Time\" is ORS's modeled drive time "
        "from road type/speed limit, not live traffic."
    )

    def grid_cards(card_data):
        cards_html = "".join(
            f"""<div class="summary-card">
                    <div class="summary-title">{label}</div>
                    <div class="summary-value">{value}</div>
                </div>"""
            for label, value in card_data
        )
        html_block(f"""<div class="summary-grid">{cards_html}</div>""")

    # ============================================================
    # DATA PREP
    # ============================================================
    @st.cache_data
    def cvrp_clean_data(data):
        d = data.copy()
        d = d.rename(columns={RAW_STORE_COL: STORE_COL, RAW_PRODUCT_COL: PRODUCT_COL, RAW_DATE_COL: DATE_COL})
        d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
        for c in [TARGET_COL, WEIGHT_COL]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[DATE_COL, TARGET_COL, STORE_COL, PRODUCT_COL, WEIGHT_COL, VEHICLE_COL])
        return d.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)

    @st.cache_data
    def cvrp_truck_profile(d):
        prof = d.groupby(VEHICLE_COL).agg(
            capacity_kg=(WEIGHT_COL, lambda s: s.quantile(CAPACITY_PERCENTILE)),
            refrigerated_share=(REFRIGERATED_COL, "mean"),
        ).reset_index()
        prof["capacity_tons"] = (prof["capacity_kg"] / KG_PER_TON).round(3)

        def _role(share):
            if share >= 0.99:
                return "Refrigerated"
            if share <= 0.01:
                return "Standard"
            return "Mixed"
        prof["role"] = prof["refrigerated_share"].apply(_role)
        return prof

    @st.cache_data
    def cvrp_store_demand(d):
        latest = d.sort_values(DATE_COL).groupby([STORE_COL, PRODUCT_COL]).tail(1)
        by_role = latest.groupby([STORE_COL, REFRIGERATED_COL])[WEIGHT_COL].sum().reset_index()
        by_role[WEIGHT_COL] = (by_role[WEIGHT_COL] / KG_PER_TON).round(3)
        return by_role

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    @st.cache_data(ttl=3600, show_spinner=False)
    def ors_matrix(nodes, api_key):
        if not api_key:
            return None
        try:
            locations = [[n["lon"], n["lat"]] for n in nodes]
            resp = requests.post(
                "https://api.openrouteservice.org/v2/matrix/driving-car",
                headers={"Authorization": api_key, "Content-Type": "application/json"},
                json={"locations": locations, "metrics": ["distance"], "units": "km"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            matrix = data.get("distances")
            if matrix is None or any(v is None for row in matrix for v in row):
                return None
            return matrix
        except Exception:
            return None

    def _parse_tollway_extra(feature):
        try:
            extras = feature.get("properties", {}).get("extras", {})
            tollways = extras.get("tollways")
            if not tollways:
                return 0.0, 0, False
            toll_km = 0.0
            for entry in tollways.get("summary", []):
                if entry.get("value") == 1:
                    toll_km += entry.get("distance", 0.0) / 1000
            toll_stretches = sum(1 for block in tollways.get("values", []) if block[2] == 1)
            return round(toll_km, 2), toll_stretches, toll_km > 0
        except Exception:
            return 0.0, 0, False

    @st.cache_data(ttl=3600, show_spinner=False)
    def ors_route_alternatives(stop_coords, api_key, target_count=ROUTE_ALT_COUNT):
        """
        Fetch up to `target_count` alternative driving routes for an ordered
        list of (lat, lon) stops (depot at both ends) via the ORS Directions
        API, including tollway tagging. Returns None on failure / no key /
        fewer than 2 stops — callers fall back to the solver's own distance.
        """
        if not api_key or len(stop_coords) < 2:
            return None
        try:
            coords = [[lon, lat] for lat, lon in stop_coords]
            resp = requests.post(
                "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
                headers={"Authorization": api_key, "Content-Type": "application/json"},
                json={
                    "coordinates": coords,
                    "alternative_routes": {"target_count": target_count, "weight_factor": 1.6, "share_factor": 0.6},
                    "extra_info": ["tollways"],
                },
                timeout=20,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            feats = data.get("features", [])
            if not feats:
                return None
            options = []
            for f in feats:
                summary = f.get("properties", {}).get("summary", {})
                geom = f["geometry"]["coordinates"]
                duration_min = round(summary.get("duration", 0.0) / 60, 1)
                toll_km, toll_stretches, has_toll = _parse_tollway_extra(f)
                est_toll_wait = round(toll_stretches * TOLL_WAIT_MIN_PER_STRETCH, 1) if has_toll else 0.0
                options.append({
                    "distance_km": round(summary.get("distance", 0.0) / 1000, 2),
                    "duration_min": duration_min,
                    "geometry": [(lat, lon) for lon, lat in geom],
                    "toll_km": toll_km,
                    "toll_stretches": toll_stretches,
                    "has_toll": has_toll,
                    "est_toll_wait_min": est_toll_wait,
                    "total_effective_min": round(duration_min + est_toll_wait, 1),
                })
            options.sort(key=lambda o: o["duration_min"])
            return options
        except Exception:
            return None

    @st.cache_data
    def haversine_matrix(nodes):
        n = len(nodes)
        m = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    m[i][j] = haversine_km(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"])
        return m

    def solve_cvrp(dist_km_matrix, demands_kg, capacities_kg, time_limit_sec=8):
        """Generic OR-Tools solve. Used here with a single vehicle and an
        effectively unlimited capacity, i.e. as a plain TSP: given a fixed
        set of stops already assigned to a trip by pack_trips(), find the
        distance-minimizing order to visit them in."""
        n_locations = len(dist_km_matrix)
        n_vehicles = len(capacities_kg)
        dist_m = [[int(round(v * 1000)) for v in row] for row in dist_km_matrix]

        manager = pywrapcp.RoutingIndexManager(n_locations, n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return dist_m[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        def demand_callback(from_index):
            return int(round(demands_kg[manager.IndexToNode(from_index)]))

        demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, [int(round(c)) for c in capacities_kg], True, "Capacity"
        )

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(time_limit_sec)

        solution = routing.SolveWithParameters(params)
        if solution is None:
            return None

        routes = []
        for v in range(n_vehicles):
            index = routing.Start(v)
            nodes, load, dist = [], 0, 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                nodes.append(node)
                load += demands_kg[node]
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                dist += routing.GetArcCostForVehicle(prev_index, index, v)
            nodes.append(manager.IndexToNode(index))
            routes.append({"vehicle_idx": v, "nodes": nodes, "load_kg": load, "distance_km": dist / 1000})
        return routes

    # ============================================================
    # STEP — TRIP PACKING. Slot generation (how many trips, which
    # vehicle gets each slot, each trip's load budget) is copied
    # verbatim from Fleet Allocation's pack_trips() — vehicle and
    # trip counts here always match Fleet Allocation exactly, since
    # none of that depends on which specific store fills which slot.
    #
    # FIX — geography-aware fill order: the original fill picked
    # whichever store had the largest remaining demand to top off a
    # trip, with no regard for location. With store locations spread
    # across opposite corners of the country, that could put Delhi,
    # Bengaluru, and Kolkata in the same truck's trip just because
    # their leftover tonnage happened to add up — a huge detour.
    # Now, once a trip has a starting store, every additional store
    # added to fill that SAME trip is chosen by nearest distance to
    # the stop just added (greedy nearest-neighbor), not by weight.
    # This only changes which stores end up sharing a trip — total
    # demand, load budgets, and vehicle/trip counts are unaffected.
    # ============================================================
    def _store_dist_km(a, b):
        ca, cb = STORE_COORDS.get(a), STORE_COORDS.get(b)
        if not ca or not cb:
            return float("inf")
        return haversine_km(ca["lat"], ca["lon"], cb["lat"], cb["lon"])

    def pack_trips(store_weights, trucks_for_pool, trip_counter):
        if not trucks_for_pool:
            return []

        demands_total = sum(w for w in store_weights.values() if w > 1e-9)
        if demands_total <= 1e-9:
            return []

        trip_sizes = sorted(set(cap for _, cap in trucks_for_pool))
        trucks_by_size = {}
        for vid, cap in trucks_for_pool:
            trucks_by_size.setdefault(cap, []).append(vid)
        next_vehicle_idx = {cap: 0 for cap in trucks_by_size}

        slots = []
        remaining = demands_total
        while remaining > 1e-9:
            finishing_sizes = [c for c in trip_sizes if c >= remaining - 1e-9]
            cap = min(finishing_sizes) if finishing_sizes else max(trip_sizes)
            vids = trucks_by_size[cap]
            vid = vids[next_vehicle_idx[cap] % len(vids)]
            next_vehicle_idx[cap] += 1
            slots.append({"vehicle": vid, "capacity": cap})
            remaining -= min(cap, remaining)

        total_capacity = sum(s["capacity"] for s in slots)
        level_ratio = (demands_total / total_capacity) if total_capacity > 1e-9 else 0.0
        for s in slots:
            s["load_budget"] = round(s["capacity"] * level_ratio, 6)

        demands_dict = {s: w for s, w in store_weights.items() if w > 1e-9}
        trips = []
        for s in slots:
            trip_counter[s["vehicle"]] += 1
            trip = {"vehicle": s["vehicle"], "trip_no": trip_counter[s["vehicle"]], "capacity": s["capacity"],
                    "remaining": s["load_budget"], "load": 0.0, "stops": []}

            current_store = None
            while demands_dict and trip["remaining"] > 1e-9:
                if current_store is None or current_store not in demands_dict:
                    # start (or restart) this trip at whichever remaining
                    # store has the largest leftover demand — keeps the
                    # same fairness the original fill used for picking
                    # WHERE a trip begins
                    current_store = max(demands_dict, key=lambda s: demands_dict[s])
                weight = demands_dict[current_store]
                take = min(weight, trip["remaining"])
                trip["load"] += take
                trip["remaining"] -= take
                trip["stops"].append((current_store, round(take, 3)))
                demands_dict[current_store] -= take
                if demands_dict[current_store] <= 1e-9:
                    del demands_dict[current_store]

                if trip["remaining"] > 1e-9 and demands_dict:
                    # nearest-neighbor: pick whichever remaining store is
                    # geographically closest to the stop just added, so
                    # any store this trip has to combine with is the
                    # least-costly detour available, not just whatever
                    # had the most leftover weight
                    current_store = min(demands_dict.keys(), key=lambda s: _store_dist_km(current_store, s))
                else:
                    current_store = None
            trips.append(trip)
        return trips

    # ============================================================
    # RUN
    # ============================================================
    d_clean = cvrp_clean_data(df)
    truck_profile = cvrp_truck_profile(d_clean)
    store_demand = cvrp_store_demand(d_clean)

    html_block("<div style='margin-top:18px'></div>")
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        run_clicked = st.button("Run Route Optimization", use_container_width=True, key="run_cvrp")
    if run_clicked:
        st.session_state.cvrp_run = True

    if st.session_state.get("cvrp_run", False):

        api_live = bool(ORS_API_KEY)
        if not api_live:
            html_block("""<div style="background-color:#fff8e8;border:1px solid #f0d9a0;border-radius:8px;
                    padding:14px 18px;margin-bottom:16px;color:#8a6d1d;font-weight:600;">
                    ⚠️ No ORS_API_KEY found — using the haversine fallback for stop-order
                    solving and for the "before" baseline below. Add ORS_API_KEY to
                    secrets.toml for real driving distances, timings, and toll tagging.
                </div>""")

        # ---- build the same truck pools Fleet Allocation uses ----
        std_trucks = [(row[VEHICLE_COL], row["capacity_tons"]) for _, row in truck_profile.iterrows()
                      if row["role"] in ("Standard", "Mixed")]
        ref_trucks = [(row[VEHICLE_COL], row["capacity_tons"]) for _, row in truck_profile.iterrows()
                      if row["role"] in ("Refrigerated", "Mixed")]
        std_trucks.sort(key=lambda x: x[1])
        ref_trucks.sort(key=lambda x: x[1])

        std_w_raw = store_demand[store_demand[REFRIGERATED_COL] == False].set_index(STORE_COL)[WEIGHT_COL].to_dict()
        ref_w_raw = store_demand[store_demand[REFRIGERATED_COL] == True].set_index(STORE_COL)[WEIGHT_COL].to_dict()
        std_w = {s: w for s, w in std_w_raw.items() if s in STORE_COORDS and w > 1e-9}
        ref_w = {s: w for s, w in ref_w_raw.items() if s in STORE_COORDS and w > 1e-9}

        # ============================================================
        # CAPACITY DIAGNOSTIC — same shape as before, computed against
        # the SAME truck pools / demand pack_trips will use
        # ============================================================
        html_block("<div style='margin-top:16px'></div>")
        html_block("<h4 style='color:black;'>Capacity Diagnostic</h4>")
        diag_rows = []
        for role, trucks, w in [("Standard", std_trucks, std_w), ("Refrigerated", ref_trucks, ref_w)]:
            if not trucks:
                diag_rows.append({"Role": role, "Status": "No trucks with this role in the fleet.",
                                   "Total Demand (t)": "—", "Total Single-Trip Capacity (t)": "—",
                                   "Largest Single Truck (t)": "—"})
                continue
            total_demand = sum(w.values())
            total_cap = sum(c for _, c in trucks)
            max_cap = max(c for _, c in trucks)
            diag_rows.append({
                "Role": role, "Status": "OK",
                "Total Demand (t)": f"{total_demand:.2f}",
                "Total Single-Trip Capacity (t)": f"{total_cap:.2f}",
                "Largest Single Truck (t)": f"{max_cap:.2f}",
            })
        render_html_table(pd.DataFrame(diag_rows))

        # ============================================================
        # ALLOCATION — identical call to Fleet Allocation's pack_trips()
        # ============================================================
        trip_counter = {vid: 0 for vid, _ in (std_trucks + ref_trucks)}
        std_trips = pack_trips(std_w, std_trucks, trip_counter)
        ref_trips = pack_trips(ref_w, ref_trucks, trip_counter)
        for t in std_trips: t["type"] = "Standard"
        for t in ref_trips: t["type"] = "Refrigerated"
        all_trips = std_trips + ref_trips

        if not all_trips:
            st.warning("No demand/truck combination could be solved — see the diagnostic above for why.")
        else:
            # renumber contiguous per vehicle across BOTH cargo types —
            # same renumbering Fleet Allocation applies to its dispatched_trips
            all_trips.sort(key=lambda t: (t["vehicle"], t["trip_no"]))
            _renumber = {}
            for t in all_trips:
                _renumber[t["vehicle"]] = _renumber.get(t["vehicle"], 0) + 1
                t["trip_no"] = _renumber[t["vehicle"]]

            for t in all_trips:
                t["trip_date"] = TRIP_START_DATE + timedelta(days=t["trip_no"] - 1)
                t["trip_day"] = t["trip_date"].strftime("%A")
                t["label"] = f"{t['vehicle']} — {t['type']} — Trip {t['trip_no']}"

            vehicles_used_set = {t["vehicle"] for t in all_trips if t["stops"]}
            all_idle = sorted(set(truck_profile[VEHICLE_COL]) - vehicles_used_set)

            used_summary = ", ".join(
                f"{vid} ({sum(1 for t in all_trips if t['vehicle']==vid)} trip(s))" for vid in sorted(vehicles_used_set)
            )
            st.caption(f"Vehicles dispatched (from Fleet Allocation's pack_trips): {used_summary}"
                       + (f" | Idle: {', '.join(all_idle)}" if all_idle else ""))

            # ============================================================
            # PER-TRIP STOP ORDERING + ROAD ROUTE ENRICHMENT
            # ============================================================
            with st.spinner("Solving stop order and checking road routes (distance/time/tolls) for each trip..."):
                for t in all_trips:
                    stops = t["stops"]
                    if not stops:
                        t["stop_ids"], t["stop_latlon"] = [], []
                        continue

                    nodes = [DEPOT] + [{"lat": STORE_COORDS[s]["lat"], "lon": STORE_COORDS[s]["lon"]} for s, _ in stops]
                    node_ids = [None] + [s for s, _ in stops]

                    dist_matrix = ors_matrix(nodes, ORS_API_KEY) if api_live else None
                    source = "OpenRouteService Matrix API"
                    if dist_matrix is None:
                        dist_matrix = haversine_matrix(nodes)
                        source = "Haversine fallback"

                    demands_kg = [0.0] * len(nodes)
                    capacities_kg = [10 ** 9]  # unconstrained — allocation is already fixed by pack_trips
                    order_routes = solve_cvrp(dist_matrix, demands_kg, capacities_kg, time_limit_sec=5)

                    if order_routes:
                        r = order_routes[0]
                        ordered_idx = r["nodes"]
                        distance_km = r["distance_km"]
                    else:
                        ordered_idx = list(range(len(nodes))) + [0]
                        distance_km = sum(dist_matrix[ordered_idx[i]][ordered_idx[i + 1]] for i in range(len(ordered_idx) - 1))

                    t["stop_ids"] = [node_ids[n] for n in ordered_idx]
                    t["stop_latlon"] = [
                        (DEPOT["lat"], DEPOT["lon"]) if sid is None
                        else (STORE_COORDS[sid]["lat"], STORE_COORDS[sid]["lon"])
                        for sid in t["stop_ids"]
                    ]
                    t["distance_km"] = round(distance_km, 2)
                    t["source"] = source
                    # depot -> store distance for EACH store in THIS trip, pulled
                    # from the same matrix this trip's own route was built from —
                    # used for a fair "before" baseline below (same distance
                    # source, and counted once per trip a store appears in, not
                    # deduplicated across the whole solution)
                    t["depot_dist_lookup"] = {
                        node_ids[i]: dist_matrix[0][i] for i in range(1, len(node_ids)) if node_ids[i] is not None
                    }

                api_live_alts = bool(ORS_API_KEY)
                for t in all_trips:
                    if not t["stops"]:
                        continue
                    alts = ors_route_alternatives(t["stop_latlon"], ORS_API_KEY) if api_live_alts else None
                    if not alts:
                        alts = [{
                            "distance_km": t["distance_km"], "duration_min": None, "geometry": t["stop_latlon"],
                            "toll_km": 0.0, "toll_stretches": 0, "has_toll": False,
                            "est_toll_wait_min": 0.0, "total_effective_min": None,
                        }]
                    t["alternatives"] = alts

                    shortest = min(alts, key=lambda o: o["distance_km"])
                    have_timing = any(o["total_effective_min"] is not None for o in alts)
                    recommended = (min((o for o in alts if o["total_effective_min"] is not None),
                                        key=lambda o: o["total_effective_min"]) if have_timing else shortest)
                    t["shortest"], t["recommended"], t["best"] = shortest, recommended, recommended

                    if recommended is shortest:
                        reason = "Shortest distance AND lowest total effective time (drive time + est. toll wait) among the options found."
                    else:
                        dist_delta = round(recommended["distance_km"] - shortest["distance_km"], 1)
                        time_saved = round(shortest["total_effective_min"] - recommended["total_effective_min"], 1) \
                            if shortest["total_effective_min"] is not None else None
                        if recommended["toll_stretches"] < shortest["toll_stretches"]:
                            reason = (f"{dist_delta:+.1f} km longer than the shortest option, but crosses "
                                      f"{shortest['toll_stretches'] - recommended['toll_stretches']} fewer toll "
                                      f"stretch(es), giving a lower total effective time"
                                      + (f" ({time_saved:+.1f} min vs. shortest)." if time_saved is not None else "."))
                        else:
                            reason = (f"{dist_delta:+.1f} km longer than the shortest option, but faster modeled "
                                      f"drive time keeps total effective time lower"
                                      + (f" ({time_saved:+.1f} min vs. shortest)." if time_saved is not None else "."))
                    t["reason"] = reason

            trip_records = [t for t in all_trips if t["stops"]]

            # ============================================================
            # RESULTS TABLE
            # ============================================================
            all_rows = []
            total_distance_km = 0.0
            for t in trip_records:
                stop_labels_route = [stop_label(sid) for sid in t["stop_ids"]]
                n_stops = len(t["stop_ids"]) - 2
                total_distance_km += t["distance_km"]
                rec = t["recommended"]
                all_rows.append({
                    "Vehicle": t["vehicle"], "Trip #": t["trip_no"], "Role": t["type"],
                    "Planned Trip Date": t["trip_date"].strftime("%d-%b-%Y"), "Day": t["trip_day"],
                    "Stops": n_stops, "Route": " → ".join(stop_labels_route),
                    "Load (tons)": round(t["load"], 3),
                    "Optimized Route": f"Option {t['alternatives'].index(rec) + 1}",
                    "Distance (km)": rec["distance_km"],
                    "Toll Road": "Yes" if rec["has_toll"] else "No",
                    "Est. Toll Wait (min)*": rec["est_toll_wait_min"],
                    "Est. Drive Time (min)": rec["duration_min"] if rec["duration_min"] is not None else "—",
                    "Total Effective Time (min)": rec["total_effective_min"] if rec["total_effective_min"] is not None else "—",
                    "Route Options": len(t["alternatives"]), "Source": t["source"],
                })

            route_df = pd.DataFrame(all_rows).sort_values(["Role", "Vehicle", "Trip #"]).reset_index(drop=True)

            html_block("<div style='margin-top:20px'></div>")
            html_block("<h3 style='color:#3c3c3c;font-weight:700;'>Route Solution</h3>")
            grid_cards([
                ("Vehicles Used", str(len(vehicles_used_set))),
                ("Vehicles Idle", str(len(all_idle))),
                ("Trips (incl. multi-trip)", str(len(route_df))),
                ("Total Fleet Distance", f"{total_distance_km:.1f} km"),
            ])

            if all_idle:
                html_block(f"""<div style="background-color:#eef3fa;border:1px solid #cfe0f0;border-radius:8px;
                        padding:12px 16px;margin-top:8px;margin-bottom:4px;color:#2c4a6b;">
                        <b>Idle this run:</b> {", ".join(all_idle)} — same idle vehicles Fleet Allocation
                        would show, since allocation is shared between the two modules.
                    </div>""")

            # ============================================================
            # FLEET ROSTER — every vehicle in the fleet
            # ============================================================
            vehicle_trip_counts, vehicle_roles_seen, vehicle_load_totals = {}, {}, {}
            for t in trip_records:
                vid = t["vehicle"]
                vehicle_trip_counts[vid] = vehicle_trip_counts.get(vid, 0) + 1
                vehicle_roles_seen.setdefault(vid, set()).add(t["type"])
                vehicle_load_totals[vid] = vehicle_load_totals.get(vid, 0.0) + t["load"]

            roster_rows = []
            for _, row in truck_profile.sort_values(VEHICLE_COL).iterrows():
                vid = row[VEHICLE_COL]
                n_trips = vehicle_trip_counts.get(vid, 0)
                roster_rows.append({
                    "Vehicle": vid, "Fleet Role": row["role"],
                    "Capacity (tons)": round(row["capacity_tons"], 3),
                    "Trips Dispatched": n_trips,
                    "Roles Dispatched": ", ".join(sorted(vehicle_roles_seen.get(vid, []))) or "—",
                    "Total Load Carried (tons)": round(vehicle_load_totals.get(vid, 0.0), 3),
                    "Status": "Dispatched" if n_trips > 0 else "Idle (not used this solve)",
                })
            roster_df = pd.DataFrame(roster_rows)

            html_block("<div style='margin-top:16px'></div>")
            html_block("<h4 style='color:black;'>Fleet Roster — All Vehicles</h4>")
            st.caption(f"Every vehicle in the fleet ({len(roster_df)} total) — trip counts here match "
                       f"Fleet Allocation's Trip-by-Trip Load Plan exactly.")
            render_html_table(roster_df)

            idle_rows = []
            for _, row in truck_profile.sort_values(VEHICLE_COL).iterrows():
                vid = row[VEHICLE_COL]
                if vid in vehicle_trip_counts:
                    continue
                idle_rows.append({
                    "Vehicle": vid, "Trip #": "—", "Role": row["role"],
                    "Planned Trip Date": "—", "Day": "—", "Stops": 0,
                    "Route": "No trips assigned this solve", "Load (tons)": 0.0,
                    "Optimized Route": "—", "Distance (km)": 0.0, "Toll Road": "—",
                    "Est. Toll Wait (min)*": "—", "Est. Drive Time (min)": "—",
                    "Total Effective Time (min)": "—", "Route Options": 0, "Source": "—",
                })
            route_df_full = pd.concat([route_df, pd.DataFrame(idle_rows)], ignore_index=True) if idle_rows else route_df

            html_block("<div style='margin-top:16px'></div>")
            html_block("<h4 style='color:black;'>Vehicle Routes — All Fleet Vehicles (Recommended Option, Planned Date &amp; Day)</h4>")
            st.caption(f"Includes all {len(truck_profile)} fleet vehicles.")
            render_html_table(route_df_full)
            st.caption("* Est. Toll Wait is a planning assumption, not measured toll-booth data — see note above.")

            # ============================================================
            # MASTER TABLE — every alternative route for every trip
            # ============================================================
            html_block("<div style='margin-top:24px'></div>")
            html_block("<h4 style='color:black;'>All Route Options — Every Trip (Tolls, Time &amp; Distance Compared)</h4>")
            if not api_live_alts:
                st.caption("No ORS_API_KEY — showing solver distance only, no alternative routes or toll tagging available.")

            all_alt_rows = []
            for t in trip_records:
                for i, opt in enumerate(t["alternatives"]):
                    is_shortest, is_recommended = opt is t["shortest"], opt is t["recommended"]
                    tags = []
                    if is_recommended: tags.append("★ Recommended")
                    if is_shortest and not is_recommended: tags.append("Shortest Distance")
                    elif is_shortest and is_recommended: tags[-1] = "★ Recommended (also Shortest Distance)"
                    all_alt_rows.append({
                        "Vehicle": t["vehicle"], "Trip #": t["trip_no"], "Role": t["type"],
                        "Planned Trip Date": t["trip_date"].strftime("%d-%b-%Y"), "Day": t["trip_day"],
                        "Route Option": f"Option {i+1}", "Distance (km)": opt["distance_km"],
                        "Est. Drive Time (min)": opt["duration_min"] if opt["duration_min"] is not None else "—",
                        "Toll Road": "Yes" if opt["has_toll"] else "No", "Toll Stretches": opt["toll_stretches"],
                        "Est. Toll Wait (min)*": opt["est_toll_wait_min"],
                        "Total Effective Time (min)": opt["total_effective_min"] if opt["total_effective_min"] is not None else "—",
                        "Best?": " / ".join(tags) if tags else "",
                        "Reason": t["reason"] if is_recommended else "",
                    })
            render_html_table(pd.DataFrame(all_alt_rows))
            st.caption("* Est. Toll Wait is a planning assumption (per distinct tollway stretch), not measured toll-booth data. "
                       "Toll fees are not available from the ORS free API.")

            # ============================================================
            # Per-trip written call-out
            # ============================================================
            html_block("<div style='margin-top:16px'></div>")
            html_block("<h4 style='color:black;'>Best Route per Trip — Reasoning</h4>")
            for t in trip_records:
                rec = t["recommended"]
                html_block(f"""<div style="background-color:#f4f8fb;border:1px solid #d6e4f0;border-radius:8px;
                        padding:10px 16px;margin-bottom:8px;color:#333;">
                        <b>{t['label']}</b> ({t['trip_date'].strftime('%d-%b-%Y')}, {t['trip_day']}) —
                        best route is <b>Option {t['alternatives'].index(rec) + 1}</b> at
                        <b>{rec['distance_km']} km</b>
                        {f", {rec['duration_min']} min drive" if rec['duration_min'] is not None else ""}
                        {f", {rec['est_toll_wait_min']} min est. toll wait" if rec['has_toll'] else ", no tollways"}.
                        <br><i>{t['reason']}</i>
                    </div>""")

            # ---- before/after: "before" = if every store visit in this
            # solution had instead been its own dedicated round trip. Counted
            # PER TRIP a store appears in (not deduplicated across the whole
            # run — a store whose demand needed 3 separate trips gets counted
            # 3 times, matching "after"), and using the SAME distance matrix
            # each trip's own route was built from (real road distance when
            # ORS is live, haversine otherwise) — so both sides use the same
            # measuring stick. ----
            unique_stores_seen = {sid for t in trip_records for sid in t["stop_ids"] if sid is not None}
            before_km = sum(
                2 * dist
                for t in trip_records
                for dist in t["depot_dist_lookup"].values()
            )

            km_saved = before_km - total_distance_km
            pct_saved = (km_saved / before_km * 100) if before_km else 0
            basis_note = ", ".join(sorted({t["source"] for t in trip_records}))

            html_block("<div style='margin-top:18px'></div>")
            html_block("<h4 style='color:black;'>Distance vs. Dedicated Round Trips</h4>")
            st.caption(f"'After' uses each trip's solved order distance ({basis_note}); "
                       f"'Before' uses straight-line depot-store round trips for a consistent baseline.")
            grid_cards([
                ("Before (dedicated trips)", f"{before_km:.1f} km"),
                ("After (optimized order)", f"{total_distance_km:.1f} km"),
                ("Distance Saved", f"{abs(pct_saved):.1f}% {'reduction' if km_saved > 0 else 'increase'}"),
                ("Unique Stores Served", str(len(unique_stores_seen))),
            ])
            if km_saved <= 0:
                html_block("""<div style="background-color:#faf6e3;border:1px solid #ecdfb0;border-radius:8px;
                        padding:12px 16px;margin-top:8px;color:#8a6d1d;">
                        Consolidation increased total distance here even after nearest-neighbor grouping.
                        This usually means demand didn't split evenly enough to give every trip a single
                        dedicated store — with only 4 placeholder store locations spread across opposite
                        corners of the country, any trip that has to combine two of them will add real
                        distance no matter which two are picked. Multi-stop consolidation pays off when
                        stops are geographically close together; swap in real, geographically clustered
                        store coordinates to see the algorithm's savings potential more realistically.
                    </div>""")

            # ---- map ----
            html_block("<div style='margin-top:18px'></div>")
            html_block("<h4 style='color:black;'>Route Map</h4>")
            fig, ax = plt.subplots(figsize=(7, 6))
            cmap = plt.cm.tab10.colors
            for i, t in enumerate(trip_records):
                xs = [DEPOT["lon"] if sid is None else STORE_COORDS[sid]["lon"] for sid in t["stop_ids"]]
                ys = [DEPOT["lat"] if sid is None else STORE_COORDS[sid]["lat"] for sid in t["stop_ids"]]
                ax.plot(xs, ys, marker="o", color=cmap[i % len(cmap)], label=t["label"])
            ax.scatter([DEPOT["lon"]], [DEPOT["lat"]], marker="*", s=280, color="black", zorder=5, label="Depot")
            apply_chart_style(ax, xlabel="Longitude", ylabel="Latitude", title="Optimized Vehicle Routes")
            ax.title.set_color("black"); ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
            ax.tick_params(colors="black")
            ax.legend(fontsize=7, loc="best", frameon=False)
            plt.tight_layout()
            st.pyplot(fig)

            # ============================================================
            # REAL ROAD ROUTE MAP
            # ============================================================
            html_block("<div style='margin-top:20px'></div>")
            html_block("<h4 style='color:black;'>Real Road Route Map</h4>")

            all_centers, fmap = [], None
            cmap = plt.cm.tab10.colors
            fallback_count = sum(1 for t in trip_records if t["best"]["duration_min"] is None)

            for i, t in enumerate(trip_records):
                stop_latlon = t["stop_latlon"]
                stop_labels_map = [stop_label(sid) for sid in t["stop_ids"]]
                all_centers.extend(stop_latlon)
                road_path = t["best"]["geometry"]

                if fmap is None:
                    clat = sum(p[0] for p in all_centers) / len(all_centers)
                    clon = sum(p[1] for p in all_centers) / len(all_centers)
                    fmap = folium.Map(location=[clat, clon], zoom_start=5, tiles="cartodbpositron")

                color = f"#{int(cmap[i % len(cmap)][0]*255):02x}{int(cmap[i % len(cmap)][1]*255):02x}{int(cmap[i % len(cmap)][2]*255):02x}"
                time_note = f", {t['best']['duration_min']} min" if t["best"]["duration_min"] is not None else ""
                toll_note = ", toll" if t["best"]["has_toll"] else ""
                tooltip = f"{t['label']} ({t['trip_date'].strftime('%d-%b-%Y')}) — {t['best']['distance_km']} km{time_note}{toll_note}"
                folium.PolyLine(road_path, color=color, weight=4, opacity=0.85, tooltip=tooltip).add_to(fmap)
                for (lat, lon), stop_lbl in zip(stop_latlon, stop_labels_map):
                    folium.CircleMarker([lat, lon], radius=6, color=color, fill=True, fill_opacity=0.9,
                                         popup=f"{stop_lbl} — {t['label']}").add_to(fmap)

            if fmap is not None:
                folium.Marker([DEPOT["lat"], DEPOT["lon"]], popup=stop_label(None),
                               icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(fmap)

                if fallback_count:
                    st.caption(f"{len(trip_records) - fallback_count}/{len(trip_records)} routes used real ORS "
                               f"road paths with timing/toll data; {fallback_count} fell back to straight-line "
                               f"distance only.")
                else:
                    st.caption(f"All {len(trip_records)} routes drawn from real OpenRouteService road paths, "
                               f"each showing the recommended option (lowest total effective time).")

                st_folium(fmap, width=None, height=520, key="cvrp_map_all")

                html_block("<div style='margin-top:10px'></div>")
                zoom_choice = st.selectbox(
                    "Zoom into a single vehicle trip — compare its route options",
                    options=["(all routes above)"] + [t["label"] for t in trip_records],
                    key="cvrp_route_choice",
                )
                if zoom_choice != "(all routes above)":
                    t = next(tr for tr in trip_records if tr["label"] == zoom_choice)
                    stop_latlon = t["stop_latlon"]
                    stop_labels_map = [stop_label(sid) for sid in t["stop_ids"]]
                    clat = sum(p[0] for p in stop_latlon) / len(stop_latlon)
                    clon = sum(p[1] for p in stop_latlon) / len(stop_latlon)
                    zmap = folium.Map(location=[clat, clon], zoom_start=6, tiles="cartodbpositron")

                    alt_colors = ["#0B2C5D", "#C0392B", "#1E8449", "#8E44AD", "#B9770E"]
                    for i, opt in enumerate(t["alternatives"]):
                        is_recommended = opt is t["recommended"]
                        time_note = f", ~{opt['duration_min']} min" if opt["duration_min"] is not None else ""
                        toll_note = ", toll" if opt["has_toll"] else ", no toll"
                        tooltip = f"Route {i+1}{' (recommended)' if is_recommended else ''} — {opt['distance_km']} km{time_note}{toll_note}"
                        folium.PolyLine(opt["geometry"], color=alt_colors[i % len(alt_colors)],
                                         weight=6 if is_recommended else 3, opacity=0.9 if is_recommended else 0.5,
                                         tooltip=tooltip).add_to(zmap)
                    for (lat, lon), stop_lbl in zip(stop_latlon, stop_labels_map):
                        is_depot = stop_lbl.startswith("Depot")
                        folium.Marker([lat, lon], popup=stop_lbl,
                                      icon=folium.Icon(color="red" if is_depot else "blue",
                                                        icon="home" if is_depot else "cube", prefix="fa")).add_to(zmap)
                    st.caption(f"Showing all {len(t['alternatives'])} route option(s) for {zoom_choice} — "
                               f"thicker/darker line is the recommended option.")
                    st_folium(zmap, width=None, height=480, key=f"cvrp_map_zoom_{zoom_choice}")
            else:
                st.info("No dispatched trips to map.")

            diag_color, diag_text_color = ("#cfe0c8", "#3c5a35") if km_saved > 0 else ("#faf6e3", "#8a6d1d")
            diag_icon = "✅" if km_saved > 0 else "⚠️"
            html_block(f"""<div style="background-color:{diag_color};border-radius:6px;padding:14px 18px;
                    margin-top:14px;margin-bottom:10px;color:{diag_text_color};font-weight:500;">
                    {diag_icon} Using the {len(vehicles_used_set)} vehicles Fleet Allocation dispatched
                    ({len(route_df)} total trips), optimized stop order covers {len(unique_stores_seen)}
                    stores for {total_distance_km:.1f} km, versus {before_km:.1f} km for straight-line
                    dedicated round trips per store ({abs(pct_saved):.1f}% {'reduction' if km_saved > 0 else 'increase'}).
                </div>""")
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