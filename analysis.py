import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="📊 Mobile Activity Data Analysis", layout="wide")
st.title("📱 Mobile Phone Activity Dataset Analysis")

# Download dataset
st.write("Downloading dataset to D: drive...")
dataset_path = kagglehub.dataset_download("marcodena/mobile-phone-activity")
st.success(f"✅ Download complete! Dataset located at: {dataset_path}")

# Find CSV file
files = os.listdir(dataset_path)
data_file = None
for file in files:
    if file.endswith('.csv'):
        data_file = os.path.join(dataset_path, file)
        break

if data_file:
    st.info(f"Found data file: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)

    # === BASIC INFO ===
    st.subheader("📌 Dataset Basic Information")
    st.write(f"**Shape:** {df.shape}")
    st.dataframe(df.head())
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.write("**Basic Statistics:**")
    st.dataframe(df.describe())

    # === VISUALIZATIONS ===
    st.subheader("📊 Data Visualizations")

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # 1. Distribution of numerical columns
    if len(numerical_cols) > 0:
        st.write("### Distribution of Numerical Variables")
        for col in numerical_cols[:4]:  # Show up to 4 distributions
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    # 3. Boxplots
    if len(numerical_cols) > 0:
        st.write("### Boxplot of Numerical Variables")
        fig, ax = plt.subplots(figsize=(12, 6))
        df[numerical_cols].boxplot(ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 4. Count plots for categorical variables
    if len(categorical_cols) > 0:
        st.write("### Distribution of Categorical Variables")
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts = df[col].value_counts().head(10)
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
            ax.set_title(f"Top Categories in {col}")
            st.pyplot(fig)

    # === ADVANCED ANALYSIS: K-MEANS ===
    if len(numerical_cols) >= 2:
        st.subheader("🤖 Advanced Analysis: K-means Clustering")

        X = df[numerical_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow Method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # Fit 3 clusters
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df_clustered = X.copy()
        df_clustered['Cluster'] = clusters

        # Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_xlabel(numerical_cols[0])
        ax.set_ylabel(numerical_cols[1])
        ax.set_title("K-means Clustering Results")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

        st.write("**Cluster Sizes:**")
        st.write(pd.Series(clusters).value_counts().sort_index())

    st.success("✅ Analysis Complete!")

else:
    st.error("No CSV file found in the dataset directory.")
    st.write("Available files:", files)
