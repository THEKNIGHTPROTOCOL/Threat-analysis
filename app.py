import os
import zipfile
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mobile Activity Dashboard", layout="wide")

# File names (update if your dataset has different names)
ZIP_FILE = "mobile-phone-activity.zip"   # The Kaggle zip you downloaded
DATA_FILE = "mobile_activity.csv"        # CSV inside the zip

st.title("📱 Mobile Phone Activity Dashboard")

# Step 1: Check if CSV already exists
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    st.success(f"✅ Loaded dataset from {DATA_FILE}")

# Step 2: If CSV not found, but zip exists → extract
elif os.path.exists(ZIP_FILE):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(".")
    st.info("📂 Extracted dataset from zip file.")
    
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.success("✅ Dataset loaded successfully after extraction!")
    else:
        st.error("⚠️ CSV not found inside zip. Please upload manually.")
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()

# Step 3: If neither CSV nor zip found → ask user to upload
else:
    st.error("⚠️ Default dataset not found. Please upload your own CSV file.")
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Step 4: Show basic info about dataset
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("📈 Basic Statistics")
st.write(df.describe())

st.subheader("🔍 Dataset Info")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

