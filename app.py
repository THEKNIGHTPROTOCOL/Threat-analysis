# ====== IMPORT ALL REQUIRED LIBRARIES ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ====== STREAMLIT CONFIGURATION ======
st.set_page_config(page_title="📊 Mobile Phone Activity Analysis", layout="wide")
sns.set_style("whitegrid")

# ====== TITLE ======
st.title("📱 Mobile Phone Activity Analysis & Clustering")
st.write("Analyze mobile user behavior with clustering and visualizations.")

# ====== LOAD DATASET FROM GITHUB ======
csv_url = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/a8903074af0b9bab8366e824467cbb6d9bdeccb8/mobile_activity_big.csv"

st.info("Loading dataset from GitHub... ⏳")
try:
    df = pd.read_csv(csv_url)
    st.success(f"✅ Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# ====== SHOW DATA ======
st.subheader("First 5 Rows of the Dataset")
st.dataframe(df.head())

# ====== BASIC INFO ======
st.write("### 📐 Dataset Info")
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
st.write("**Data Types:**")
st.write(df.dtypes)
st.write("**Missing Values:**")
st.write(df.isnull().sum())

# ====== STATISTICS ======
st.subheader("📊 Basic Statistics")
st.dataframe(df.describe())

# ====== VISUALIZATIONS ======
st.header("📈 Data Visualizations")

numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Histograms for numerical columns
if len(numerical_cols) > 0:
    st.subheader("🔹 Distribution of Numerical Variables")
    for col in numerical_cols[:4]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# Correlation Heatmap
if len(numerical_cols) > 1:
    st.subheader("🔹 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# Boxplot
if len(numerical_cols) > 0:
    st.subheader("🔹 Boxplots of Numerical Variables")
    fig, ax = plt.subplots(figsize=(15, 6))
    df[numerical_cols].boxplot(ax=ax)
    ax.set_title("Boxplot of Numerical Variables")
    st.pyplot(fig)

# Categorical Variables
if len(categorical_cols) > 0:
    st.subheader("🔹 Distribution of Categorical Variables")
    for col in categorical_cols:
        fig, ax = plt.subplots()
        value_counts = df[col].value_counts().head(10)
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
        ax.set_title(f"Distribution of {col} (Top 10)")
        st.pyplot(fig)

# ====== K-MEANS CLUSTERING ======
if len(numerical_cols) >= 2:
    st.header("🤖 K-Means Clustering")
    X = df[numerical_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker="o", linestyle="--")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method for Optimal Clusters")
    st.pyplot(fig)

    # Let user select cluster count
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = X.copy()
    df_clustered["Cluster"] = clusters

    # Cluster visualization (using first 2 numerical columns)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    ax.set_xlabel(numerical_cols[0])
    ax.set_ylabel(numerical_cols[1])
    ax.set_title("K-means Clustering Results")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    st.subheader("Cluster Sizes")
    st.write(pd.Series(clusters).value_counts().sort_index())

# ====== SAVE RESULTS ======
st.subheader("💾 Save Clustered Data")
output_path = "mobile_activity_clustered.csv"
df_clustered.to_csv(output_path, index=False)
st.success(f"Clustered data saved as `{output_path}`")
