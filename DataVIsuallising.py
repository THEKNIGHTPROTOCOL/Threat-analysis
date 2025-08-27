# Import all required libraries
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
%matplotlib inline

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Download the dataset to D: drive
print("Downloading dataset to D: drive...")
dataset_path = kagglehub.dataset_download("marcodena/mobile-phone-activity")
print(f"Download complete! Dataset located at: {dataset_path}")

# Find the correct file name
files = os.listdir(dataset_path)
data_file = None
for file in files:
    if file.endswith('.csv'):
        data_file = os.path.join(dataset_path, file)
        break

if data_file:
    print(f"Found data file: {data_file}")
    
    # Load the data
    df = pd.read_csv(data_file)
    
    # Display basic information
    print("\n=== DATASET BASIC INFORMATION ===")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    display(df.head())
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    print("\n=== BASIC STATISTICS ===")
    display(df.describe())
    
    # Visualizations
    print("\n=== DATA VISUALIZATIONS ===")
    
    # 1. Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribution of Numerical Variables', fontsize=16)
        
        for i, col in enumerate(numerical_cols[:4]):  # Plot first 4 numerical columns
            row, col_idx = i // 2, i % 2
            sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.show()
    
    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Variables')
        plt.show()
    
    # 3. Boxplots for numerical columns
    if len(numerical_cols) > 0:
        plt.figure(figsize=(15, 6))
        df[numerical_cols].boxplot()
        plt.title('Boxplot of Numerical Variables')
        plt.xticks(rotation=45)
        plt.show()
    
    # 4. Count plots for categorical columns (if any)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 5))
        fig.suptitle('Distribution of Categorical Variables', fontsize=16)
        
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols):
            value_counts = df[col].value_counts()
            if len(value_counts) > 10:  # If too many categories, show only top 10
                value_counts = value_counts.head(10)
            
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    # Advanced analysis: Clustering (if we have enough numerical data)
    if len(numerical_cols) >= 2:
        print("\n=== ADVANCED ANALYSIS: K-MEANS CLUSTERING ===")
        
        # Select numerical data and standardize it
        X = df[numerical_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.show()
        
        # Apply K-means with selected number of clusters (let's use 3 as an example)
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_clustered = X.copy()
        df_clustered['Cluster'] = clusters
        
        # Visualize clusters (using first two numerical columns)
        if len(numerical_cols) >= 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.xlabel(numerical_cols[0])
            plt.ylabel(numerical_cols[1])
            plt.title('K-means Clustering Results')
            plt.colorbar(scatter, label='Cluster')
            plt.show()
            
            print("\nCluster sizes:")
            print(pd.Series(clusters).value_counts().sort_index())
    
    print("\n=== ANALYSIS COMPLETE ===")
    
else:
    print("No CSV file found in the dataset directory.")
    print("Available files:", files)
