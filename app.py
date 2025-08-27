# ====== IMPORT LIBRARIES ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ====== STREAMLIT CONFIG ======
st.set_page_config(page_title="📊 Mobile Phone Activity Analysis", layout="wide")
sns.set_style("whitegrid")
st.title("📱 Mobile Phone Activity Analysis & Clustering")
st.markdown("#### Upload your own dataset or use the default one.")

# ====== DATA UPLOAD ======
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
else:
    # Default dataset
    csv_url = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/main/mobile_activity_big.csv"
    df = pd.read_csv(csv_url, on_bad_lines='skip')
st.success(f"✅ Dataset Loaded ({df.shape[0]} rows × {df.shape[1]} columns)")

# ====== DATA CLEANING ======
# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ====== BASIC INFO ======
with st.expander("📋 Dataset Overview"):
    st.write("*Shape:*", df.shape)
    st.write("*Data Types:*")
    st.dataframe(df.dtypes)
    st.write("*Missing Values:*")
    st.dataframe(df.isnull().sum())
    st.write("*First 5 rows:*")
    st.dataframe(df.head())

# ====== INTERACTIVE VISUALIZATIONS ======
st.header("📈 Visualizations")

# Numerical distributions
if len(numeric_cols) > 0:
    st.subheader("🔹 Numerical Features Distribution")
    selected_num_cols = st.multiselect("Select numerical columns to visualize", numeric_cols, default=numeric_cols[:4])
    for col in selected_num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, color="#69b3a2", ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# Correlation heatmap
if len(numeric_cols) > 1:
    st.subheader("🔹 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)

# Pairplot (scatter matrix)
if st.checkbox("Show Pairplot (may be slow for large datasets)"):
    fig = sns.pairplot(df[numeric_cols].sample(min(500, len(df))))
    st.pyplot(fig)

# Categorical features
if len(categorical_cols) > 0:
    st.subheader("🔹 Categorical Features")
    selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)
    fig, ax = plt.subplots()
    value_counts = df[selected_cat_col].value_counts().head(15)
    sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
    ax.set_title(f"Distribution of {selected_cat_col} (Top 15)")
    st.pyplot(fig)

# ====== K-MEANS CLUSTERING ======
st.header("🤖 K-Means Clustering")

if len(numeric_cols) >= 2:
    X = df[numeric_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    st.subheader("Elbow Method")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker="o", linestyle="--", color="#FF6B6B")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # User selects number of clusters
    k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.write("### Cluster Sizes")
    st.dataframe(pd.Series(clusters).value_counts().sort_index())

    # PCA 2D visualization
    st.subheader("Cluster Visualization (PCA 2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.7)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-Means Clusters (2D PCA)")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

# ====== DOWNLOAD CLEANED & CLUSTERED DATA ======
st.header("💾 Download Cleaned & Clustered Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='mobile_activity_cleaned.csv',
    mime='text/csv',
)
