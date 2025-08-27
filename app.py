# =========================================================================
# === Mobile Phone Activity Analysis & Clustering Dashboard ===
# This script combines two Streamlit applications to create a unified
# dashboard for mobile phone activity data analysis.
#
# Key Features:
# 1. Flexible data loading: Upload your own CSV or use a default dataset.
# 2. Robust data cleaning: Handles missing values and removes bad lines.
# 3. Interactive visualizations: Uses tabs for an organized layout.
#    - Distributions of numerical and categorical features.
#    - Correlation heatmap and pairplot for feature relationships.
# 4. K-Means Clustering:
#    - Elbow Method to help determine the optimal number of clusters.
#    - Interactive slider to select the number of clusters.
#    - 2D PCA visualization to show the clusters.
# 5. Data Export: Download the cleaned and clustered dataset as a CSV.
# =========================================================================

# ====== IMPORT LIBRARIES ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ====== STREAMLIT CONFIGURATION ======
# Set the page title and layout for a better user experience.
st.set_page_config(page_title="📊 Mobile Phone Activity Dashboard", layout="wide")
sns.set_style("whitegrid")
st.title("📱 Mobile Phone Activity Analysis & Clustering")
st.markdown("#### Upload your own dataset or use the default one.")

# ====== DATA LOADING & CLEANING ======
# Use Streamlit's file uploader to allow users to provide their data.
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Define a function to load the data, either from upload or default URL.
# The st.cache_data decorator caches the data so it doesn't reload on every interaction.
@st.cache_data
def load_data(uploaded_file=None):
    """Loads the dataset from either a user upload or a default URL."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        else:
            csv_url = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/main/mobile_activity_big.csv"
            df = pd.read_csv(csv_url, on_bad_lines='skip')

        # Robust cleaning: Remove rows where the first column contains code-like garbage
        df = df[~df.iloc[:, 0].astype(str).str.contains("import|set|DataFrame|print|#", case=False, na=False)]
        return df

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()

df = load_data(uploaded_file)

# Stop the application if the dataset is empty.
if df.empty:
    st.warning("⚠ No valid data found. Please check your CSV file.")
    st.stop()

st.success(f"✅ Dataset Loaded ({df.shape[0]} rows × {df.shape[1]} columns)")

# Data cleaning and preprocessing steps.
with st.spinner("Cleaning data..."):
    # Fill numeric missing values with the median of their respective columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing values with the mode (most frequent value).
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
st.success("✅ Data cleaning complete!")

# =========================================================================
# === TABS FOR ORGANIZED VISUALIZATION AND ANALYSIS ===
# Use st.tabs to organize the dashboard into logical sections.
tab1, tab2, tab3 = st.tabs(["🔍 Dataset Overview & Visuals", "📈 Correlation Analysis", "🤖 K-Means Clustering"])

with tab1:
    # ====== DATA OVERVIEW & VISUALIZATIONS ======
    st.header("🔍 Dataset Overview & Visualizations")

    # Expandable section for basic dataset info.
    with st.expander("📋 Dataset Information"):
        st.write("*Shape:*", df.shape)
        st.write("*Data Types:*")
        st.dataframe(df.dtypes)
        st.write("*First 5 rows:*")
        st.dataframe(df.head())

    # Numerical distributions
    st.subheader("🔹 Numerical Features Distribution")
    if len(numeric_cols) > 0:
        selected_num_cols = st.multiselect("Select numerical columns to visualize", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        for col in selected_num_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[col], kde=True, color="#69b3a2", ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
    else:
        st.info("No numerical columns available.")

    # Categorical features
    st.subheader("🔹 Categorical Features Distribution")
    if len(categorical_cols) > 0:
        selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        # Get top 15 values to prevent visual clutter
        value_counts = df[selected_cat_col].value_counts().head(15)
        sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
        ax.set_title(f"Distribution of {selected_cat_col} (Top 15)")
        st.pyplot(fig)
    else:
        st.info("No categorical columns available.")

with tab2:
    # ====== CORRELATION ANALYSIS ======
    st.header("📈 Correlation Analysis")

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for correlation analysis.")

    # Pairplot (scatter matrix)
    if st.checkbox("Show Pairplot (may be slow for large datasets)"):
        if len(numeric_cols) > 1:
            # Sample a smaller dataset for performance
            sample_df = df[numeric_cols].sample(min(500, len(df)))
            fig = sns.pairplot(sample_df)
            st.pyplot(fig)
        else:
            st.info("Not enough numerical columns to create a pairplot.")

with tab3:
    # ====== K-MEANS CLUSTERING ======
    st.header("🤖 K-Means Clustering")
    if len(numeric_cols) >= 2:
        # Prepare data for clustering
        X = df[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow method to find the optimal number of clusters
        st.subheader("Elbow Method to find Optimal K")
        wcss = []
        # Run KMeans for 1 to 10 clusters
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Plot the elbow method graph
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 11), wcss, marker="o", linestyle="--", color="#FF6B6B")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # User selects the number of clusters
        k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add the cluster labels to the original dataframe
        df["Cluster"] = pd.Series(clusters, index=X.index)

        # Display the size of each cluster
        st.write("### Cluster Sizes")
        st.dataframe(df["Cluster"].value_counts().sort_index())

        # PCA for 2D visualization of clusters
        st.subheader("Cluster Visualization (PCA 2D)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create a scatter plot of the clusters in the 2D PCA space
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title(f"K-Means Clusters (K={k}, 2D PCA)")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)
    else:
        st.info("Please ensure your dataset has at least two numerical columns to perform clustering.")

# ====== DOWNLOAD CLEANED & CLUSTERED DATA ======
st.header("💾 Download Cleaned & Clustered Data")
if "Cluster" in df.columns:
    # Convert the DataFrame to a CSV string for download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name='mobile_activity_processed.csv',
        mime='text/csv',
    )
else:
    st.info("The clustered data is not ready to download yet. Please perform the clustering first.")
