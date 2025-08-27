import kagglehub
import os
import pandas as pd
import streamlit as st

st.title("📊 Mobile Phone Activity Dataset Downloader")

st.write("Downloading dataset to your D: drive... This may take a moment.")

# Download dataset
dataset_path = kagglehub.dataset_download("marcodena/mobile-phone-activity")

st.success("✅ Download complete!")
st.write("Dataset is located at:", dataset_path)

# List files
st.subheader("Downloaded Files")
files = os.listdir(dataset_path)
for file_name in files:
    st.write(f"📂 {file_name}")

# Try reading main CSV
data_file = os.path.join(dataset_path, "mobile_activity.csv")
if os.path.exists(data_file):
    st.subheader("Preview of Dataset")
    df = pd.read_csv(data_file)
    st.dataframe(df.head())
else:
    st.error("Main CSV file not found. Please check the downloaded files above.")
