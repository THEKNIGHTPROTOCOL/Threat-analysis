import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="📱 Mobile Usage Dashboard", layout="wide")

st.title("📊 Mobile Activity Data Dashboard")

# Load dataset
DATA_URL = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/refs/heads/main/mobile_activity_big.csv"

@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url, on_bad_lines="skip")
        # Normalize column names (lowercase, no spaces)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return None

df = load_data(DATA_URL)

if df is not None:
    st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Show actual column names to avoid confusion
    st.write("📑 Columns in dataset:", list(df.columns))

    # Show preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Numeric overview
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    # --- Visualizations ---
    st.subheader("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    if "call_duration_min" in df.columns:
        with col1:
            st.markdown("### ⏱ Call Duration Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["call_duration_min"], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

    if "internet_usage_mb" in df.columns:
        with col2:
            st.markdown("### 🌐 Internet Usage Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["internet_usage_mb"], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

    if "subscription_plan" in df.columns:
        st.markdown("### 📦 Subscription Plan Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="subscription_plan", data=df, palette="pastel", ax=ax)
        st.pyplot(fig)

    if "device_type" in df.columns:
        st.markdown("### 📱 Device Type Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="device_type", data=df, palette="muted", ax=ax)
        st.pyplot(fig)

    # Correlation Heatmap
    st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Safely handle recharge amount
    if "avg_recharge_amount" in df.columns:
        st.markdown("### 💰 Recharge Amount vs Subscription Plan")
        fig, ax = plt.subplots()
        sns.boxplot(x="subscription_plan", y="avg_recharge_amount", data=df, palette="Set2", ax=ax)
        st.pyplot(fig)

    if {"internet_usage_mb", "app_sessions"}.issubset(df.columns):
        st.markdown("### 🔗 Internet Usage vs App Sessions")
        fig, ax = plt.subplots()
        sns.scatterplot(x="internet_usage_mb", y="app_sessions", hue="device_type", data=df, alpha=0.5, ax=ax)
        st.pyplot(fig)

    if "churn_probability" in df.columns:
        st.markdown("### ⚠️ Churn Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["churn_probability"], bins=20, kde=True, color="red", ax=ax)
        st.pyplot(fig)

    if "satisfaction_score" in df.columns:
        st.markdown("### ⭐ Satisfaction Score Count")
        fig, ax = plt.subplots()
        sns.countplot(x="satisfaction_score", data=df, palette="Blues", ax=ax)
        st.pyplot(fig)

    if "customer_complaints" in df.columns:
        st.markdown("### 🛑 Customer Complaints Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["customer_complaints"], bins=10, kde=False, color="orange", ax=ax)
        st.pyplot(fig)

else:
    st.error("Dataset could not be loaded. Please check the URL.")
