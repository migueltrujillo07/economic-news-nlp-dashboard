import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(page_title="Economic News AI Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    /* This targets the metric cards container */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e6e9ef;
    }
    /* This forces the label and value to be dark/readable */
    [data-testid="stMetricLabel"] p {
        color: #555555 !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricValue"] div {
        color: #111111 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading (Memoized for performance) ---
@st.cache_data
def load_data():
    file_path = "data/results/sentiment_results_v1.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['seendate'], format='ISO8601')
        df['date'] = df['datetime'].dt.date
        return df
    return None

df = load_data()

# --- HEADER ---
st.title("📊 Global Economic News Sentiment Dashboard")
st.markdown("Analyzing market narratives using **FinBERT** (Deep Learning for Finance).")

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("User Filters")
    
    # Date Filter
    min_date, max_date = df['date'].min(), df['date'].max()
    selected_dates = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    
    # Confidence Slider (Demonstrates ML understanding)
    min_conf = st.sidebar.slider("Minimum Model Confidence", 0.5, 1.0, 0.8)
    
    # Source Filter
    all_sources = sorted(df['source_name'].unique())
    selected_sources = st.sidebar.multiselect("News Sources", all_sources, default=all_sources[:5])

    # Filter Data
    mask = (df['date'] >= selected_dates[0]) & \
           (df['date'] <= selected_dates[1]) & \
           (df['sentiment_score'] >= min_conf) & \
           (df['source_name'].isin(selected_sources))
    
    filtered_df = df[mask]

    # --- TOP KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Articles", len(filtered_df))
    kpi2.metric("Positive 🟢", f"{len(filtered_df[filtered_df['sentiment_label']=='positive'])}")
    kpi3.metric("Negative 🔴", f"{len(filtered_df[filtered_df['sentiment_label']=='negative'])}")
    avg_conf = filtered_df['sentiment_score'].mean()
    kpi4.metric("Avg. ML Confidence", f"{avg_conf:.2%}")

    st.divider()

    # --- CHARTS SECTION ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Sentiment Evolution Over Time")
        daily_sent = filtered_df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        fig_time = px.line(daily_sent, x='date', y='count', color='sentiment_label',
                          color_discrete_map={'positive':'#00CC96', 'neutral':'#636EFA', 'negative':'#EF553B'},
                          markers=True, template="plotly_white")
        st.plotly_chart(fig_time, use_container_width=True)

    with col_right:
        st.subheader("🏢 News Source Bias Analysis")
        # Stacked bar chart for Source vs Sentiment
        fig_bias = px.histogram(filtered_df, x="source_name", color="sentiment_label", 
                                barmode="group", # Using group for better comparison
                                color_discrete_map={'positive':'#00CC96', 'neutral':'#636EFA', 'negative':'#EF553B'},
                                template="plotly_white")
        st.plotly_chart(fig_bias, use_container_width=True)

    # --- DATA EXPLORER ---
    st.subheader("🔍 Headline Explorer")
    st.markdown("Review the raw headlines and the AI's reasoning.")
    st.dataframe(filtered_df[['date', 'clean_title', 'source_name', 'sentiment_label', 'sentiment_score']]
                 .sort_values(by='date', ascending=False), 
                 use_container_width=True)

else:
    st.warning("No data found. Please ensure you have run the Ingestion and NLP scripts.")