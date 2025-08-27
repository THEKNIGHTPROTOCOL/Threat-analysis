# =========================================================================
# === Enhanced Mobile Phone Activity Analysis & Clustering Dashboard ===
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ====== STREAMLIT CONFIGURATION ======
# Set the page title and layout for a better user experience.
st.set_page_config(
    page_title="📊 Mobile Phone Activity Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.2rem;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; margin-top: 1.5rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📱 Mobile Phone Activity Analysis & Clustering</h1>', unsafe_allow_html=True)
st.markdown("#### Upload your own dataset or use the default one.")

# ====== SIDEBAR FOR CONTROLS ======
with st.sidebar:
    st.header("Dashboard Controls")
    st.info("Configure analysis parameters and filters")
    
    # Data source selection
    data_source = st.radio("Data Source:", ["Use Default Dataset", "Upload CSV File"])
    
    # Clustering parameters
    st.subheader("Clustering Settings")
    max_clusters = st.slider("Maximum clusters to test:", 5, 15, 10)
    
    # Visualization settings
    st.subheader("Visualization Settings")
    use_plotly = st.checkbox("Use Interactive Plotly Charts", value=True)
    sample_size = st.slider("Sample size for large datasets:", 100, 2000, 500)

# ====== DATA LOADING & CLEANING ======
@st.cache_data(show_spinner="Loading data...")
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
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        return df

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        # Create sample data for demonstration
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
        st.info("Using sample data for demonstration")
        return df

# Load data based on user selection
if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
else:
    uploaded_file = None

df = load_data(uploaded_file)

# Stop the application if the dataset is empty.
if df.empty:
    st.warning("⚠ No valid data found. Please check your CSV file.")
    st.stop()

# Display dataset info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
col4.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))

# Data cleaning and preprocessing steps.
with st.spinner("Cleaning data..."):
    # Fill numeric missing values with the median of their respective columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing values with the mode (most frequent value).
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
st.success("✅ Data cleaning complete!")

# =========================================================================
# === TABS FOR ORGANIZED VISUALIZATION AND ANALYSIS ===
# Use st.tabs to organize the dashboard into logical sections.
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Dataset Overview", "📊 Data Visualizations", "🤖 Clustering Analysis", "📈 Advanced Analytics"])

with tab1:
    # ====== DATA OVERVIEW ======
    st.header("🔍 Dataset Overview")

    # Expandable section for basic dataset info.
    with st.expander("📋 Dataset Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.astype(str))
        
        with col2:
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            st.dataframe(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values")
    
    # Show first and last few rows
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 rows:**")
        st.dataframe(df.head())
    with col2:
        st.write("**Last 5 rows:**")
        st.dataframe(df.tail())
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

with tab2:
    # ====== DATA VISUALIZATIONS ======
    st.header("📊 Data Visualizations")
    
    # Numerical distributions
    st.subheader("🔹 Numerical Features Distribution")
    if len(numeric_cols) > 0:
        selected_num_cols = st.multiselect("Select numerical columns to visualize", numeric_cols, 
                                          default=list(numeric_cols[:min(4, len(numeric_cols))]))
        
        # Create subplots
        cols = st.columns(2)
        for i, col in enumerate(selected_num_cols):
            with cols[i % 2]:
                if use_plotly:
                    fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
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
        
        if use_plotly:
            value_counts = df[selected_cat_col].value_counts().head(15)
            fig = px.bar(x=value_counts.values, y=value_counts.index, orientation='h',
                         title=f"Distribution of {selected_cat_col} (Top 15)",
                         labels={'x': 'Count', 'y': selected_cat_col})
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Get top 15 values to prevent visual clutter
            value_counts = df[selected_cat_col].value_counts().head(15)
            sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
            ax.set_title(f"Distribution of {selected_cat_col} (Top 15)")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No categorical columns available.")

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    if len(numeric_cols) > 1:
        if use_plotly:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                           title="Correlation Matrix", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for correlation analysis.")

    # Pairplot (scatter matrix)
    if st.checkbox("Show Pairplot (may be slow for large datasets)"):
        if len(numeric_cols) > 1:
            # Sample a smaller dataset for performance
            sample_df = df[numeric_cols].sample(min(sample_size, len(df)))
            if use_plotly:
                fig = px.scatter_matrix(sample_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = sns.pairplot(sample_df)
                st.pyplot(fig)
        else:
            st.info("Not enough numerical columns to create a pairplot.")

with tab3:
    # ====== CLUSTERING ANALYSIS ======
    st.header("🤖 Clustering Analysis")
    
    if len(numeric_cols) >= 2:
        # Prepare data for clustering
        X = df[numeric_cols].dropna()
        
        if len(X) < 10:
            st.warning("Not enough data points for clustering after cleaning.")
            st.stop()
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow method to find the optimal number of clusters
        st.subheader("Elbow Method to find Optimal K")
        wcss = []
        silhouette_scores = []
        
        # Run KMeans for 2 to max_clusters
        k_range = range(2, max_clusters+1)
        for i in k_range:
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if len(X) > 1:  # Silhouette score requires at least 2 samples
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Create subplots for elbow method and silhouette scores
        if use_plotly:
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Scores'))
            
            # Elbow method plot
            fig.add_trace(go.Scatter(x=list(k_range), y=wcss, mode='lines+markers', name='WCSS'), row=1, col=1)
            fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
            fig.update_yaxes(title_text="WCSS", row=1, col=1)
            
            # Silhouette score plot
            fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'), row=1, col=2)
            fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
            fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(k_range, wcss, marker="o", linestyle="--", color="#FF6B6B")
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("WCSS")
            ax1.set_title("Elbow Method")
            
            ax2.plot(k_range, silhouette_scores, marker="o", linestyle="--", color="#4ECDC4")
            ax2.set_xlabel("Number of Clusters")
            ax2.set_ylabel("Silhouette Score")
            ax2.set_title("Silhouette Scores")
            
            st.pyplot(fig)

        # User selects the number of clusters
        k = st.slider("Select number of clusters (K)", min_value=2, max_value=max_clusters, value=3)
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add the cluster labels to the original dataframe
        df["Cluster"] = pd.Series(clusters, index=X.index)

        # Display the size of each cluster
        st.write("### Cluster Sizes")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        st.dataframe(cluster_counts)
        
        # Show cluster characteristics
        st.subheader("📊 Cluster Characteristics")
        cluster_summary = df.groupby("Cluster")[numeric_cols].mean()
        st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))

        # Dimensionality reduction for visualization
        st.subheader("Cluster Visualization")
        
        # Let user choose between PCA and t-SNE
        reduction_method = st.radio("Dimensionality Reduction Method:", ["PCA", "t-SNE"])
        
        if reduction_method == "PCA":
            reducer = PCA(n_components=2)
            reducer_name = "PCA"
        else:
            reducer = TSNE(n_components=2, random_state=42)
            reducer_name = "t-SNE"
            
        X_reduced = reducer.fit_transform(X_scaled)
        
        if use_plotly:
            # Create interactive scatter plot
            plot_df = pd.DataFrame({
                'x': X_reduced[:, 0],
                'y': X_reduced[:, 1],
                'cluster': clusters,
                'size': np.ones(len(clusters)) * 10  # Constant size for all points
            })
            
            fig = px.scatter(plot_df, x='x', y='y', color='cluster', 
                             title=f"K-Means Clusters (K={k}, {reducer_name})",
                             labels={'x': f'{reducer_name} Component 1', 'y': f'{reducer_name} Component 2'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create static scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="viridis", alpha=0.7)
            ax.set_xlabel(f"{reducer_name} Component 1")
            ax.set_ylabel(f"{reducer_name} Component 2")
            ax.set_title(f"K-Means Clusters (K={k}, {reducer_name})")
            fig.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
    else:
        st.info("Please ensure your dataset has at least two numerical columns to perform clustering.")

with tab4:
    # ====== ADVANCED ANALYTICS ======
    st.header("📈 Advanced Analytics")
    
    # Time series analysis if timestamp is available
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_cols:
        st.subheader("⏰ Time Series Analysis")
        time_col = st.selectbox("Select time column for analysis", time_cols)
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df['time_period'] = df[time_col].dt.to_period('D')
            
            if len(numeric_cols) > 0:
                metric_col = st.selectbox("Select metric to analyze over time", numeric_cols)
                time_series = df.groupby('time_period')[metric_col].mean()
                
                if use_plotly:
                    fig = px.line(x=time_series.index.astype(str), y=time_series.values, 
                                 title=f'{metric_col} over Time',
                                 labels={'x': 'Time Period', 'y': metric_col})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    time_series.plot(ax=ax, marker='o')
                    ax.set_title(f'{metric_col} over Time')
                    ax.set_ylabel(metric_col)
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
        except:
            st.warning("Could not parse time column for analysis")
    
    # Outlier detection
    if len(numeric_cols) > 0:
        st.subheader("📊 Outlier Detection")
        outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
        
        Q1 = df[outlier_col].quantile(0.25)
        Q3 = df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
        st.write(f"Number of outliers in {outlier_col}: {len(outliers)}")
        
        if len(outliers) > 0:
            if use_plotly:
                # Create box plot
                fig = px.box(df, y=outlier_col, title=f'Boxplot of {outlier_col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(y=df[outlier_col], ax=ax[0])
                ax[0].set_title(f'Boxplot of {outlier_col}')
                sns.histplot(df[outlier_col], kde=True, ax=ax[1])
                ax[1].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
                ax[1].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
                ax[1].set_title(f'Distribution of {outlier_col} with Outlier Boundaries')
                ax[1].legend()
                st.pyplot(fig)

# ====== DOWNLOAD CLEANED & CLUSTERED DATA ======
st.header("💾 Download Processed Data")
if "Cluster" in df.columns:
    # Convert the DataFrame to a CSV string for download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed CSV with Clusters",
        data=csv,
        file_name='mobile_activity_processed.csv',
        mime='text/csv',
    )
else:
    # Download without clusters
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name='mobile_activity_cleaned.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("### 🛡️ Advanced Mobile Activity Analysis Dashboard v2.0")
st.caption("Powered by Streamlit | For security research purposes only")
