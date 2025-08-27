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

# ====== SETUP & CONFIGURATION ======
print("🚀 STARTING COMPREHENSIVE MOBILE PHONE ACTIVITY ANALYSIS")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
np.random.seed(42)  # For reproducible results
