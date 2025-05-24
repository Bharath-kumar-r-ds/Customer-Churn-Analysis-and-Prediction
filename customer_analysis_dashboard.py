import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (upload manually or from local)
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Customer Churn Dashboard")

uploaded_file = st.file_uploader("Upload Telco Churn CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset successfully loaded!")

    # Display raw data
    if st.checkbox("Show raw data"):
        st.write(df.head())

    # Convert TotalCharges to numeric if needed
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.countplot(x='Churn', data=df, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Monthly Charges by Churn")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, alpha=0.5, ax=ax2)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Contract Type vs Churn")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
        contract_churn.plot(kind='bar', stacked=True, ax=ax3)
        plt.ylabel('Proportion')
        st.pyplot(fig3)

    with col4:
        st.subheader("Tenure by Churn")
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        sns.boxplot(x='Churn', y='tenure', data=df, ax=ax5)
        st.pyplot(fig5)

    # Visual 4: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number')
    fig4, ax4 = plt.subplots(figsize=(7,2))
    sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

else:
    st.info("Please upload the Telco Customer Churn dataset to view the dashboard.")