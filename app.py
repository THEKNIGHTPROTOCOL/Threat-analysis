# =========================================================================
# === Mobile Phone Activity Analysis & Clustering Dashboard ===
# This script is a professional, unified Streamlit dashboard for
# mobile phone activity data analysis. It combines flexible data handling,
# interactive visualizations, and machine learning clustering.
#
# Key Improvements:
# - Bug Fix: Updated KMeans initialization parameter 'n_init' to 'auto'.
# - Enhanced Visuals: Switched from Matplotlib/Seaborn to Plotly for interactive plots.
# - Better UX: Added a Cluster Profile Analysis section for data interpretation.
# - Robustness: Added explicit plot closing to prevent memory leaks.
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
import plotly.express as px

# ====== STREAMLIT CONFIGURATION & APP HEADER ======
# Set the page title and layout for a wide, professional-looking dashboard.
st.set_page_config(page_title="📊 Mobile Phone Activity Dashboard", layout="wide")
sns.set_style("whitegrid")
st.title("📱 Professional Mobile Phone Activity Analysis")
st.markdown("#### A comprehensive dashboard for data exploration, visualization, and user segmentation.")

# ====== DATA LOADING & CLEANING ======
# Use Streamlit's file uploader for user-provided CSV.
uploaded_file = st.file_uploader("Upload your own dataset (CSV)", type=["csv"])

# Define a robust function to load and clean the data.
@st.cache_data
def load_data(uploaded_file=None):
    """
    Loads the dataset from either a user upload or a default URL.
    Performs initial data cleaning steps.
    """
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
        st.error(f"❌ Failed to load dataset. Please check the file format. Error: {e}")
        return pd.DataFrame()

df = load_data(uploaded_file)

# Check if the dataset is valid before proceeding.
if df.empty:
    st.warning("⚠ No valid data found. Please upload a valid CSV file.")
    st.stop()

st.success(f"✅ Dataset Loaded ({df.shape[0]} rows × {df.shape[1]} columns)")

# Data preprocessing and handling missing values.
with st.spinner("Cleaning data..."):
    # Identify and fill missing values in numeric columns with the median.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Identify and fill missing values in categorical columns with the mode.
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
st.success("✅ Data cleaning complete!")

# =========================================================================
# === TABS FOR ORGANIZED ANALYSIS ===
# Streamlit tabs for a clean, professional layout.
tab1, tab2, tab3 = st.tabs(["🔍 Data Overview", "📈 Visualizations", "🤖 K-Means Clustering"])

with tab1:
    # ====== DATA OVERVIEW ======
    st.header("🔍 Dataset Overview")

    # Expander to show basic dataset information.
    with st.expander("📋 Dataset Information"):
        st.markdown(f"**Dataset Shape:** `{df.shape[0]}` rows, `{df.shape[1]}` columns")
        st.markdown("**Data Types:**")
        st.dataframe(df.dtypes)
        st.markdown("**Missing Values per Column:**")
        st.dataframe(df.isnull().sum())
        st.markdown("**First 5 Rows:**")
        st.dataframe(df.head())

with tab2:
    # ====== INTERACTIVE VISUALIZATIONS ======
    st.header("📈 Interactive Visualizations")

    # Sub-tab for numerical and categorical features.
    num_tab, cat_tab, corr_tab = st.tabs(["Numerical Features", "Categorical Features", "Correlations"])

    with num_tab:
        st.subheader("🔹 Distribution of Numerical Features")
        if len(numeric_cols) > 0:
            selected_num_cols = st.multiselect("Select numerical columns to visualize", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
            for col in selected_num_cols:
                fig = px.histogram(df, x=col, marginal="box", nbins=30, color_discrete_sequence=['#1f77b4'])
                fig.update_layout(title_text=f"Distribution of {col}", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns available for visualization.")

    with cat_tab:
        st.subheader("🔹 Distribution of Categorical Features")
        if len(categorical_cols) > 0:
            selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)
            value_counts = df[selected_cat_col].value_counts().head(15).reset_index()
            value_counts.columns = [selected_cat_col, 'count']
            fig = px.bar(value_counts, x='count', y=selected_cat_col, orientation='h', color_discrete_sequence=['#2ca02c'])
            fig.update_layout(title_text=f"Top 15 Categories in {selected_cat_col}", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available for visualization.")

    with corr_tab:
        st.subheader("🔥 Interactive Correlation Heatmap")
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().round(2)
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', 
                            aspect="auto")
            fig.update_layout(title_text="Feature Correlation Heatmap", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical columns to display a correlation heatmap.")
        
        st.markdown("---")
        if st.checkbox("Show Interactive Pairplot (Warning: May be slow for large datasets)"):
            if len(numeric_cols) > 1 and len(numeric_cols) <= 10:
                sample_df = df[numeric_cols].sample(min(500, len(df)))
                fig = px.scatter_matrix(sample_df)
                fig.update_layout(title_text="Interactive Pairplot", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pairplot is not recommended for this dataset (too many columns or rows).")

with tab3:
    # ====== K-MEANS CLUSTERING ======
    st.header("🤖 K-Means Clustering for User Segmentation")
    if len(numeric_cols) >= 2:
        st.markdown("### 1. Find the Optimal Number of Clusters")
        st.markdown("Use the **Elbow Method** below to visually identify a good number of clusters (K). Look for the point where the curve starts to bend.")

        # Prepare data for clustering
        X = df[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow method to find the optimal number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init='auto')
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Plot the elbow method graph with Matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 11), wcss, marker="o", linestyle="--", color="#e84a5f")
        ax.set_xlabel("Number of Clusters (K)", fontsize=12)
        ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
        ax.set_title("Elbow Method for K-Means", fontsize=16)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 2. Run K-Means Clustering")
        # User selects the number of clusters
        k = st.slider("Select the number of clusters (K)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add the cluster labels to the original dataframe
        df["Cluster"] = pd.Series(clusters, index=X.index)

        st.markdown("---")
        st.write("### Cluster Sizes")
        st.dataframe(df["Cluster"].value_counts().sort_index())

        st.markdown("### 3. Visualize the Clusters")
        st.markdown("An interactive 2D PCA plot helps visualize the clusters in a reduced dimension.")
        
        # PCA for 2D visualization of clusters
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2'], index=X.index)
        pca_df['Cluster'] = df['Cluster']

        # Create a scatter plot of the clusters in the 2D PCA space
        fig = px.scatter(pca_df, x='PCA Component 1', y='PCA Component 2', color='Cluster', 
                         hover_data=numeric_cols, # Show original data on hover
                         color_continuous_scale='viridis')
        fig.update_layout(title_text=f"K-Means Clusters (K={k}, 2D PCA)", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 4. Cluster Profile Analysis")
        st.markdown("This table shows the **average values** for each feature per cluster. This is key to understanding the characteristics of each user segment.")
        cluster_profiles = df.groupby('Cluster')[numeric_cols].mean().transpose()
        st.dataframe(cluster_profiles.style.highlight_max(axis=0))

    else:
        st.info("Please ensure your dataset has at least two numerical columns to perform clustering.")

# ====== DOWNLOAD CLEANED & CLUSTERED DATA ======
st.header("💾 Download Processed Data")
if "Cluster" in df.columns:
    st.markdown("Your cleaned and clustered dataset is ready for download. This file includes a new 'Cluster' column.")
    # Convert the DataFrame to a CSV string for download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name='mobile_activity_processed.csv',
        mime='text/csv',
    )
else:
    st.info("The clustered data is not ready to download yet. Please run the clustering process above.")
