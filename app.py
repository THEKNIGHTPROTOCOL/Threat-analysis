import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

# Auto-refresh every 10 seconds (10000 ms)
st_autorefresh(interval=10000, key="datarefresh")

# App title
st.set_page_config(page_title="📊 Threat Analysis Dashboard", layout="wide")
st.title("📱 Threat Analysis on Mobile Activity Dataset")

# Dataset URL
DATA_URL = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/refs/heads/main/mobile_activity_big.csv"

# Load dataset safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL, on_bad_lines="skip")
        df = df[~df.iloc[:, 0].astype(str).str.contains(
            "import|set|DataFrame|print|#", case=False, na=False
        )]
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ No valid data found.")
    st.stop()

# Show dataset preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(20))

# Basic dataset info
with st.expander("ℹ️ Dataset Info"):
    st.write(df.describe(include="all"))
    st.write(f"Dataset Shape: {df.shape}")

# ================== VISUALIZATIONS ==================
st.subheader("📊 Data Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Categorical Distributions", "Numeric Trends", "Correlation Heatmap", "Insights"]
)

with tab1:
    st.markdown("### 🔹 Distribution of Categorical Variables")
    for col in df.select_dtypes(include="object").columns[:3]:
        st.write(f"**Top categories in {col}:**")
        fig, ax = plt.subplots(figsize=(6,3))
        df[col].value_counts().head(10).plot(kind="barh", ax=ax, color="skyblue")
        ax.set_title(f"Top 10 {col} categories")
        st.pyplot(fig)

with tab2:
    st.markdown("### 📈 Numeric Trends")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        col = st.selectbox("Select numeric column", numeric_cols)
        fig, ax = plt.subplots(figsize=(6,4))
        df[col].hist(bins=30, color="coral", edgecolor="black", ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    else:
        st.info("No numeric columns available.")

with tab3:
    st.markdown("### 🔥 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation.")

with tab4:
    st.markdown("### 📌 Key Insights")
    st.write("- Categorical variables show top usage patterns.")  
    st.write("- Numeric variables distribution highlights anomalies or peaks.")  
    st.write("- Correlation heatmap reveals relationships between activity features.")  
    st.success("✅ Dashboard ready for further exploration & threat detection models!")
