# ====== IMPORT ALL REQUIRED LIBRARIES ======
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
%matplotlib inline

# ====== SETUP & CONFIGURATION ======
print("🚀 STARTING COMPREHENSIVE MOBILE PHONE ACTIVITY ANALYSIS")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
np.random.seed(42)  # For reproducible results

# ====== DOWNLOAD DATASET ======
print("\n📥 DOWNLOADING DATASET TO D: DRIVE...")
dataset_path = kagglehub.dataset_download("marcodena/mobile-phone-activity")
print(f"✅ DOWNLOAD COMPLETE! Path: {dataset_path}")

# ====== FIND AND LOAD DATA FILE ======
data_file = None
for file in os.listdir(dataset_path):
    if file.endswith('.csv'):
        data_file = os.path.join(dataset_path, file)
        print(f"📊 FOUND DATA FILE: {file}")
        break

if data_file:
    # Load the data
    df = pd.read_csv(data_file)
    
    # ====== BASIC DATA EXPLORATION ======
    print(f"\n📋 DATASET SHAPE: {df.shape}")
    print("\n🔍 FIRST 5 ROWS:")
    display(df.head())
    
    print("\n📝 DATA TYPES:")
    print(df.dtypes)
    
    print("\n❓ MISSING VALUES:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])
    
    print("\n📊 BASIC STATISTICS:")
    display(df.describe())
    
    # ====== DATA CLEANING ======
    print("\n🧹 DATA CLEANING...")
    df_clean = df.copy()
    
    # Handle missing values
    if df_clean.isnull().sum().sum() > 0:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    final_count = len(df_clean)
    print(f"✅ Removed {initial_count - final_count} duplicate rows")
    
    # ====== ADVANCED VISUALIZATIONS ======
    print("\n📈 CREATING COMPREHENSIVE VISUALIZATIONS...")
    
    # 1. Distribution of all numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Numerical Variables', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(numerical_cols[:4]):
            row, col_idx = i // 2, i % 2
            if row < 2 and col_idx < 2:  # Ensure we don't exceed subplot bounds
                sns.histplot(data=df_clean, x=col, kde=True, ax=axes[row, col_idx], color='skyblue')
                axes[row, col_idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[row, col_idx].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df_clean[numerical_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, mask=mask, 
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Variables', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    # 3. Boxplots for outlier detection
    if len(numerical_cols) > 0:
        plt.figure(figsize=(14, 6))
        df_clean[numerical_cols].boxplot()
        plt.title('Boxplot of Numerical Variables (Outlier Detection)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 4. Categorical analysis (if categorical columns exist)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(1, min(3, len(categorical_cols)), figsize=(16, 5))
        fig.suptitle('Categorical Variable Analysis', fontsize=16, fontweight='bold')
        
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols[:3]):
            value_counts = df_clean[col].value_counts().head(8)  # Top 8 categories
            colors = sns.color_palette('viridis', len(value_counts))
            
            if i < len(axes):
                bars = axes[i].bar(range(len(value_counts)), value_counts.values, color=colors)
                axes[i].set_title(f'Top {len(value_counts)} {col} Categories', fontweight='bold')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[i].set_ylabel('Count')
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    # ====== MACHINE LEARNING: CLUSTERING ANALYSIS ======
    if len(numerical_cols) >= 2:
        print("\n🤖 PERFORMING K-MEANS CLUSTERING ANALYSIS...")
        
        # Select and scale numerical data
        X = df_clean[numerical_cols].select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow method to find optimal clusters
        wcss = []
        cluster_range = range(1, 11)
        
        for i in cluster_range:
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Plot elbow method
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, wcss, marker='o', linestyle='--', color='#FF6B6B', linewidth=2)
        plt.xlabel('Number of Clusters', fontweight='bold')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontweight='bold')
        plt.title('Elbow Method for Optimal Cluster Selection', fontweight='bold')
        plt.xticks(cluster_range)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Apply K-means with 3 clusters (typical choice from elbow method)
        optimal_clusters = 3
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        df_clean['Cluster'] = clusters
        
        # Visualize clusters using first two features
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', 
                            alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
        plt.xlabel(X.columns[0], fontweight='bold')
        plt.ylabel(X.columns[1], fontweight='bold')
        plt.title(f'K-means Clustering Results ({optimal_clusters} Clusters)', fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Cluster analysis
        print(f"\n📊 CLUSTER DISTRIBUTION:")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} users ({count/len(clusters)*100:.1f}%)")
        
        # Cluster characteristics
        print(f"\n📈 CLUSTER CHARACTERISTICS (MEAN VALUES):")
        cluster_means = df_clean.groupby('Cluster')[numerical_cols].mean()
        display(cluster_means.style.background_gradient(cmap='Blues'))
    
    # ====== PREDICTIVE MODELING (If sufficient data) ======
    if len(numerical_cols) >= 2 and len(df_clean) > 100:
        print("\n🔮 BUILDING PREDICTIVE MODEL...")
        
        # Use the first numerical column as target, others as features
        target_col = numerical_cols[0]
        feature_cols = [col for col in numerical_cols if col != target_col]
        
        if len(feature_cols) > 0:
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"✅ MODEL TRAINED SUCCESSFULLY!")
            print(f"📏 RMSE: {rmse:.4f}")
            print(f"📊 R² Score: {r2:.4f}")
            
            # Plot predictions vs actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.6, color='#4ECDC4')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values', fontweight='bold')
            plt.ylabel('Predicted Values', fontweight='bold')
            plt.title(f'Actual vs Predicted Values for {target_col}\nR² = {r2:.4f}', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    # ====== SAVE RESULTS ======
    print("\n💾 SAVING RESULTS...")
    output_path = "D:/mobile_activity_analysis_results.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"✅ ANALYSIS COMPLETE! Results saved to: {output_path}")
    
    # ====== FINAL SUMMARY ======
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📁 Original dataset: {df.shape}")
    print(f"🧹 Cleaned dataset: {df_clean.shape}")
    print(f"❌ Missing values handled: {df.isnull().sum().sum()} → 0")
    print(f"♻️  Duplicates removed: {len(df) - len(df_clean)}")
    
    if 'Cluster' in df_clean.columns:
        print(f"🤖 Clusters created: {df_clean['Cluster'].nunique()}")
    
    print("📈 Visualizations generated: Distribution, Correlation, Boxplots, Clustering")
    print("💾 Results saved to D: drive")
    print("="*60)

else:
    print("❌ ERROR: No CSV file found in the dataset directory.")
    print("Available files:", os.listdir(dataset_path))
