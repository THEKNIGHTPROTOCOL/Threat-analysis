import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Mobile Phone Activity Analysis")

df = pd.read_csv("mobile_activity_analysis_results.csv")

st.write("### First 5 Rows of Cleaned Data")
st.dataframe(df.head())

st.write("### Summary Statistics")
st.write(df.describe())

st.write("### Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
