# ====== IMPORT LIBRARIES ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io

# ====== STREAMLIT CONFIG ======
st.set_page_config(page_title="📊 Mobile Phone Activity Analysis", layout="wide")
sns.set_style("whitegrid")
st.title("📱 Mobile Phone Activity Analysis & Clustering")
st.markdown("#### Upload your own dataset or use the default one.")

# ====== DATA UPLOAD ======
df = None
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    else:
        # Default dataset
        csv_url = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/main/mobile_activity_big.csv"
        df = pd.read_csv(csv_url, on_bad_lines='skip')
        
    # Check if dataframe is empty
    if df is None or df.empty:
        st.error("❌ Failed to load dataset or dataset is empty")
        st.stop()
        
    st.success(f"✅ Dataset Loaded ({df.shape[0]} rows × {df.shape[1]} columns)")
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("💡 Using sample data for demonstration")
    # Create sample data
    np.random.seed(42)
    n_rows = 1000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='H'),
        'device_id': np.random.choice(['Device_A', 'Device_B', 'Device_C'], n_rows),
        'app_name': np.random.choice(['Browser', 'Messaging', 'Social Media', 'Email'], n_rows),
        'activity_type': np.random.choice(['Network', 'SMS', 'Call', 'Location'], n_rows),
        'data_volume': np.random.exponential(100, n_rows),
        'duration': np.random.normal(300, 100, n_rows),
        'threat_score': np.random.beta(1, 5, n_rows)
    })

# ====== DATA CLEANING ======
# Make a copy of the original data
df_clean = df.copy()

# Fill numeric missing values with median
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")

# ====== BASIC INFO ======
with st.expander("📋 Dataset Overview"):
    st.write("**Shape:**", df_clean.shape)
    st.write("**Data Types:**")
    st.dataframe(df_clean.dtypes.astype(str))
    st.write("**Missing Values:**")
    st.dataframe(df_clean.isnull().sum())
    st.write("**First 5 rows:**")
    st.dataframe(df_clean.head())

# ====== INTERACTIVE VISUALIZATIONS ======
st.header("📈 Visualizations")

# Numerical distributions
numeric_cols_clean = df_clean.select_dtypes(include=[np.number]).columns
if len(numeric_cols_clean) > 0:
    st.subheader("🔹 Numerical Features Distribution")
    selected_num_cols = st.multiselect(
        "Select numerical columns to visualize", 
        numeric_cols_clean, 
        default=list(numeric_cols_clean[:min(4, len(numeric_cols_clean))])
    )
    
    for col in selected_num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df_clean[col], kde=True, color="#69b3a2", ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# Correlation heatmap
if len(numeric_cols_clean) > 1:
    st.subheader("🔹 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df_clean[numeric_cols_clean].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0, ax=ax, mask=mask)
    st.pyplot(fig)

# Pairplot (scatter matrix)
if len(numeric_cols_clean) > 1 and st.checkbox("Show Pairplot (may be slow for large datasets)"):
    sample_size = min(500, len(df_clean))
    fig = sns.pairplot(df_clean[numeric_cols_clean].sample(sample_size))
    st.pyplot(fig)

# Categorical features
categorical_cols_clean = df_clean.select_dtypes(include=['object']).columns
if len(categorical_cols_clean) > 0:
    st.subheader("🔹 Categorical Features")
    selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols_clean)
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts = df_clean[selected_cat_col].value_counts().head(15)
    sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
    ax.set_title(f"Distribution of {selected_cat_col} (Top 15)")
    plt.tight_layout()
    st.pyplot(fig)

# ====== K-MEANS CLUSTERING ======
st.header("🤖 K-Means Clustering")

if len(numeric_cols_clean) >= 2:
    # Remove any remaining NaN values
    X = df_clean[numeric_cols_clean].dropna()
    
    if len(X) > 10:  # Ensure we have enough data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow method
        st.subheader("Elbow Method")
        wcss = []
        max_clusters = min(10, len(X) - 1)
        
        if max_clusters > 1:
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
                
            fig, ax = plt.subplots()
            ax.plot(range(1, max_clusters + 1), wcss, marker="o", linestyle="--", color="#FF6B6B")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("WCSS")
            ax.set_title("Elbow Method")
            st.pyplot(fig)

            # User selects number of clusters
            k = st.slider("Select number of clusters", min_value=2, max_value=max_clusters, value=min(3, max_clusters))
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            df_clustered = df_clean.copy()
            df_clustered = df_clustered.loc[X.index]  # Align with the filtered data
            df_clustered["Cluster"] = clusters

            st.write("### Cluster Sizes")
            st.dataframe(pd.Series(clusters).value_counts().sort_index())

            # PCA 2D visualization
            st.subheader("Cluster Visualization (PCA 2D)")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("K-Means Clusters (2D PCA)")
            fig.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
            
            # Show cluster characteristics
            st.subheader("Cluster Characteristics")
            cluster_summary = df_clustered.groupby("Cluster")[numeric_cols_clean].mean()
            st.dataframe(cluster_summary)
        else:
            st.warning("Not enough data points for clustering. Need at least 10 samples.")
    else:
        st.warning("Not enough valid data for clustering after cleaning.")
else:
    st.warning("Need at least 2 numeric columns for clustering.")

# ====== DOWNLOAD CLEANED & CLUSTERED DATA ======
st.header("💾 Download Cleaned & Clustered Data")

try:
    if 'df_clustered' in locals():
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        filename = 'mobile_activity_clustered.csv'
    else:
        csv = df_clean.to_csv(index=False).encode('utf-8')
        filename = 'mobile_activity_cleaned.csv'
        
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )
except Exception as e:
    st.error(f"Error preparing download: {str(e)}")

# ====== ADDITIONAL ANALYSIS ======
st.header("🔍 Additional Analysis")

# Time series analysis if timestamp is available
time_cols = [col for col in df_clean.columns if 'time' in col.lower() or 'date' in col.lower()]
if time_cols:
    time_col = st.selectbox("Select time column for analysis", time_cols)
    try:
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])
        df_clean['time_period'] = df_clean[time_col].dt.to_period('D')
        
        if len(numeric_cols_clean) > 0:
            metric_col = st.selectbox("Select metric to analyze over time", numeric_cols_clean)
            time_series = df_clean.groupby('time_period')[metric_col].mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            time_series.plot(ax=ax, marker='o')
            ax.set_title(f'{metric_col} over Time')
            ax.set_ylabel(metric_col)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    except:
        st.warning("Could not parse time column for analysis")

# Outlier detection
if len(numeric_cols_clean) > 0:
    st.subheader("📊 Outlier Detection")
    outlier_col = st.selectbox("Select column for outlier detection", numeric_cols_clean)
    
    Q1 = df_clean[outlier_col].quantile(0.25)
    Q3 = df_clean[outlier_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_clean[(df_clean[outlier_col] < lower_bound) | (df_clean[outlier_col] > upper_bound)]
    st.write(f"Number of outliers in {outlier_col}: {len(outliers)}")
    
    if len(outliers) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(y=df_clean[outlier_col], ax=ax[0])
        ax[0].set_title(f'Boxplot of {outlier_col}')
        sns.histplot(df_clean[outlier_col], kde=True, ax=ax[1])
        ax[1].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
        ax[1].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
        ax[1].set_title(f'Distribution of {outlier_col} with Outlier Boundaries')
        ax[1].legend()
        st.pyplot(fig)
