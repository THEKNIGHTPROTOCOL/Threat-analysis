# ====== IMPORT LIBRARIES ======
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ====== STREAMLIT CONFIG ======
st.set_page_config(page_title="📊 Mobile Phone Activity Analysis", layout="wide")
sns.set_style("whitegrid")

# ====== APP TITLE ======
st.title("📱 Mobile Phone Activity - Data Analysis & Clustering")
st.caption("An interactive Streamlit dashboard to explore and cluster mobile activity data.")

# ====== LOAD DATASET ======
@st.cache_data
def load_data():
    dataset_path = kagglehub.dataset_download("marcodena/mobile-phone-activity")
    files = os.listdir(dataset_path)

    data_file = None
    for file in files:
        if file.endswith(".csv"):
            data_file = os.path.join(dataset_path, file)
            break

    if not data_file:
        st.error("❌ No CSV file found. Files available: " + str(files))
        st.stop()

    return pd.read_csv(data_file), os.path.basename(data_file)

st.info("Downloading dataset from Kaggle... please wait ⏳")
df, filename = load_data()
st.success(f"✅ Dataset Loaded: `{filename}`")

# ====== DATA PREVIEW ======
with st.expander("🔍 Preview Dataset"):
    st.write("**First 5 Rows:**")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# ====== BASIC INFO ======
with st.expander("📐 Dataset Info"):
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

# ====== STATISTICS ======
with st.expander("📊 Basic Statistics"):
    st.dataframe(df.describe(), use_container_width=True)

# ====== VISUALIZATIONS ======
st.header("📈 Data Visualizations")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Select column for histogram
if numerical_cols:
    col_choice = st.selectbox("Choose a column for histogram:", numerical_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col_choice], kde=True, ax=ax, color="skyblue")
    ax.set_title(f"Distribution of {col_choice}")
    st.pyplot(fig)

# Correlation Heatmap
if len(numerical_cols) > 1:
    st.subheader("🔹 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# Boxplots
if numerical_cols:
    st.subheader("🔹 Boxplots")
    fig, ax = plt.subplots(figsize=(15, 6))
    df[numerical_cols].boxplot(ax=ax)
    ax.set_title("Boxplots of Numerical Variables")
    st.pyplot(fig)

# Categorical Variables
if categorical_cols:
    st.subheader("🔹 Top Categories")
    col_choice = st.selectbox("Choose a categorical column:", categorical_cols)
    fig, ax = plt.subplots()
    value_counts = df[col_choice].value_counts().head(10)
    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette="viridis")
    ax.set_title(f"Top 10 Values in {col_choice}")
    st.pyplot(fig)

# ====== K-MEANS CLUSTERING ======
if len(numerical_cols) >= 2:
    st.header("🤖 K-Means Clustering")

    X = df[numerical_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method (optional, hidden behind expander)
    with st.expander("📉 Elbow Method (Find Optimal k)"):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker="o", linestyle="--", color="red")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

    # Cluster selection
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = X.copy()
    df_clustered["Cluster"] = clusters

    # Cluster Scatterplot
    st.subheader("🔹 Cluster Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    ax.set_xlabel(numerical_cols[0])
    ax.set_ylabel(numerical_cols[1])
    ax.set_title("K-means Clustering Results")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    st.write("### 📊 Cluster Sizes")
    st.dataframe(pd.Series(clusters).value_counts().sort_index(), use_container_width=True)
