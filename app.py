#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import io

# App configuration
st.set_page_config(
    page_title="📊 Advanced Threat Analysis Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.2rem;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; margin-top: 1.5rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .threat-high {color: #d62728; font-weight: bold;}
    .threat-medium {color: #ff7f0e; font-weight: bold;}
    .threat-low {color: #2ca02c; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📱 Advanced Threat Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for filters and info
with st.sidebar:
    st.header("Dashboard Controls")
    st.info("Filter data and configure analysis options")
    
    # Date filter (if applicable)
    st.subheader("Date Range")
    min_date = datetime(2024, 1, 1)
    max_date = datetime(2024, 12, 31)
    selected_range = st.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Threat level threshold
    st.subheader("Threat Detection Settings")
    threat_threshold = st.slider(
        "Threat score threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Adjust sensitivity for threat detection"
    )
    
    # Data refresh
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# Dataset URL
DATA_URL = "https://raw.githubusercontent.com/THEKNIGHTPROTOCOL/Threat-analysis/refs/heads/main/mobile_activity_big.csv"

# Load dataset with improved error handling
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    try:
        # Try to read the CSV
        df = pd.read_csv(DATA_URL, on_bad_lines="skip")
        
        # Check if dataframe is empty
        if df.empty:
            st.error("Loaded an empty dataset")
            return pd.DataFrame()
            
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove rows with code-like content in any column
        code_patterns = ["import", "set", "dataframe", "print", "#", "def ", "np.random", "pd.read_csv"]
        for col in df.columns:
            if df[col].dtype == 'object':
                mask = df[col].astype(str).str.contains('|'.join(code_patterns), case=False, na=False)
                df = df[~mask]
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        
        # Create sample data for demonstration if real data fails
        st.warning("Showing sample data for demonstration purposes")
        np.random.seed(42)
        n_rows = 5000
        
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='min'),
            'device_id': np.random.choice(['Device_A', 'Device_B', 'Device_C', 'Device_D'], n_rows),
            'app_name': np.random.choice(['Browser', 'Messaging', 'Social Media', 'Email', 'Games'], n_rows),
            'activity_type': np.random.choice(['Network', 'SMS', 'Call', 'Location', 'Media'], n_rows),
            'data_volume': np.random.exponential(100, n_rows),
            'threat_score': np.random.beta(1, 5, n_rows),
            'location': np.random.choice(['Home', 'Work', 'Public', 'Unknown'], n_rows)
        })
        
        return sample_data

# Load the data
df = load_data()

if df.empty:
    st.error("No data available. Please check the data source.")
    st.stop()

# Show dataset info
st.markdown('<h2 class="sub-header">🔍 Dataset Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Data Columns", len(df.columns))
col3.metric("Time Range", f"{df.get('timestamp', pd.Series([pd.Timestamp.now()])).min().date()} to {df.get('timestamp', pd.Series([pd.Timestamp.now()])).max().date()}")
col4.metric("Threat Detection Rate", f"{(df.get('threat_score', pd.Series([0])) > threat_threshold).mean():.2%}")

# Show dataset preview
with st.expander("📋 View Raw Data Sample", expanded=False):
    st.dataframe(df.head(20))

# ================== THREAT ANALYSIS ==================
st.markdown('<h2 class="sub-header">📊 Threat Analysis Dashboard</h2>', unsafe_allow_html=True)

# Create tabs for different analysis views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Threat Overview", 
    "Activity Patterns", 
    "Correlation Analysis", 
    "Anomaly Detection", 
    "Export Results"
])

with tab1:
    st.subheader("Threat Level Distribution")
    
    # Create threat level categories
    if 'threat_score' in df.columns:
        df['threat_level'] = pd.cut(
            df['threat_score'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            threat_counts = df['threat_level'].value_counts()
            colors = ['#2ca02c', '#ff7f0e', '#d62728']
            wedges, texts, autotexts = ax.pie(
                threat_counts.values, 
                labels=threat_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title('Distribution of Threat Levels')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Threat Summary")
            for level, count in threat_counts.items():
                if level == 'High':
                    st.markdown(f'<p class="threat-high">🔴 {level} Threat: {count} events</p>', unsafe_allow_html=True)
                elif level == 'Medium':
                    st.markdown(f'<p class="threat-medium">🟡 {level} Threat: {count} events</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="threat-low">🟢 {level} Threat: {count} events</p>', unsafe_allow_html=True)
            
            # Show high threat events
            high_threats = df[df['threat_level'] == 'High']
            if not high_threats.empty:
                st.warning(f"🚨 {len(high_threats)} high threat events detected!")
                st.dataframe(high_threats.head(10))
    else:
        st.info("No threat score column found in the dataset.")

with tab2:
    st.subheader("Activity Patterns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # App usage distribution
        if 'app_name' in df.columns:
            st.write("**Top Applications by Activity**")
            top_apps = df['app_name'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_apps.plot(kind='barh', color=sns.color_palette("viridis", len(top_apps)), ax=ax)
            ax.set_xlabel('Activity Count')
            ax.set_title('Most Used Applications')
            st.pyplot(fig)
    
    with col2:
        # Activity type distribution
        if 'activity_type' in df.columns:
            st.write("**Activity Type Distribution**")
            activity_counts = df['activity_type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%')
            ax.set_title('Activity Types')
            st.pyplot(fig)
    
    # Time-based analysis if timestamp available
    if 'timestamp' in df.columns:
        st.subheader("Temporal Activity Patterns")
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_activity = df.groupby('hour').size()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        hourly_activity.plot(kind='line', marker='o', ax=ax, color='purple')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Activity Count')
        ax.set_title('Activity by Hour of Day')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab3:
    st.subheader("Correlation Analysis")
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 1:
        # Let user select which columns to include
        selected_cols = st.multiselect(
            "Select columns for correlation analysis:",
            options=list(numeric_cols),
            default=list(numeric_cols)[:min(5, len(numeric_cols))]
        )
        
        if len(selected_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[selected_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="RdBu_r", 
                center=0,
                square=True,
                mask=mask,
                ax=ax
            )
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            
            # Interpretation of high correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr:
                st.subheader("Strong Correlations Detected")
                for col1, col2, value in high_corr:
                    st.write(f"**{col1}** and **{col2}**: {value:.3f}")
        else:
            st.info("Please select at least 2 numeric columns for correlation analysis.")
    else:
        st.info("Not enough numeric columns for correlation analysis.")

with tab4:
    st.subheader("Anomaly Detection")
    
    if 'threat_score' in df.columns:
        # Show anomalies based on threat score
        anomalies = df[df['threat_score'] > threat_threshold]
        
        if not anomalies.empty:
            st.success(f"Detected {len(anomalies)} anomalous events (threat score > {threat_threshold})")
            
            # Group by different dimensions to show patterns
            group_by = st.selectbox(
                "Group anomalies by:",
                options=[col for col in df.columns if df[col].dtype == 'object' and col != 'threat_level'],
                index=0
            )
            
            anomaly_summary = anomalies[group_by].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            anomaly_summary.plot(kind='bar', color='red', alpha=0.7, ax=ax)
            ax.set_title(f'Top Anomalies by {group_by}')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            
            # Show detailed anomaly table
            with st.expander("View Detailed Anomaly Data"):
                st.dataframe(anomalies.sort_values('threat_score', ascending=False))
        else:
            st.info(f"No anomalies detected with current threshold ({threat_threshold})")
    else:
        st.info("Threat score column not available for anomaly detection.")

with tab5:
    st.subheader("Export Analysis Results")
    
    # Create a downloadable report
    report_text = f"""
    THREAT ANALYSIS REPORT
    Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Dataset: {DATA_URL}
    Records analyzed: {len(df)}
    Time range: {df.get('timestamp', pd.Series([pd.Timestamp.now()])).min()} to {df.get('timestamp', pd.Series([pd.Timestamp.now()])).max()}
    Threat threshold: {threat_threshold}
    
    SUMMARY:
    - High threat events: {len(df[df.get('threat_score', 0) > 0.7]) if 'threat_score' in df.columns else 'N/A'}
    - Medium threat events: {len(df[(df.get('threat_score', 0) > 0.3) & (df.get('threat_score', 0) <= 0.7)]) if 'threat_score' in df.columns else 'N/A'}
    - Low threat events: {len(df[df.get('threat_score', 0) <= 0.3]) if 'threat_score' in df.columns else 'N/A'}
    
    RECOMMENDATIONS:
    - Review high threat events for potential security incidents
    - Monitor applications with unusual activity patterns
    - Consider adjusting threat threshold based on false positive rate
    """
    
    # Convert report to bytes for download
    report_bytes = report_text.encode()
    
    st.download_button(
        label="📥 Download Analysis Report",
        data=report_bytes,
        file_name=f"threat_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    # Option to download filtered data
    if st.button("📊 Download Filtered Data as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_threat_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("### 🛡️ Threat Analysis Dashboard v2.0")
st.caption("Powered by Streamlit | For security research purposes only")
