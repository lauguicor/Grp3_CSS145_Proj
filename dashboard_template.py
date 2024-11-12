import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, year
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Initialize Spark session
spark = SparkSession.builder.appName("DashboardDataProcessing").getOrCreate()

st.title("Customer Product Sales Analysis Dashboard")

# Load Data
@st.cache
def load_data():
    query = """
    SELECT ID, Year_Birth, Education, Marital_Status, Dt_Customer, Income, Kidhome, Teenhome, 
           Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, 
           MntGoldProds, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, 
           NumStorePurchases, NumWebVisitsMonth
    FROM cln_mkt_campaign
    """
    data_df = spark.sql(query)
    return data_df.toPandas()

# Sales Analysis
def calculate_product_sales(data_pd):
    prodsales_df = data_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()
    prodsales_pivot = prodsales_df.reset_index().rename(columns={0: "TotalSales", "index": "Product"})
    return prodsales_pivot

def plot_sales_distribution(prodsales_pivot):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Product', y='TotalSales', data=prodsales_pivot, palette='viridis')
    plt.title('Total Sales per Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    st.pyplot(plt)

# Clustering
def perform_clustering(data_pd):
    columns_to_encode = ['Year_Birth', 'Marital_Status', 'Education', 'Dt_Customer']
    label_encoders = {col: LabelEncoder().fit(data_pd[col]) for col in columns_to_encode}
    for col, encoder in label_encoders.items():
        data_pd[col] = encoder.transform(data_pd[col])
    
    kmeans_df = data_pd[['Year_Birth', 'Marital_Status', 'Education', 'Income', 'MntWines', 'MntFruits', 
                         'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
    scaler = StandardScaler().fit(kmeans_df)
    scaled_data = scaler.transform(kmeans_df)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_pd['Cluster'] = kmeans.fit_predict(scaled_data)
    
    return data_pd[['Cluster'] + list(kmeans_df.columns)], kmeans

# Regression Model
def train_regression_model(data_pd):
    X = data_pd.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])
    y = data_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred

# Accuracy Evaluation
def calculate_accuracy(y_test, y_pred):
    target_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    mae_values = {col: mean_absolute_error(y_test[col], y_pred[:, i]) for i, col in enumerate(target_columns)}
    return mae_values

# Main function to run the dashboard functions
def main():
    data_pd = load_data()
    
    # Display raw data
    st.subheader("Raw Data")
    st.write(data_pd.head())
    
    # Sales Distribution
    st.subheader("Total Sales per Product")
    prodsales_pivot = calculate_product_sales(data_pd)
    plot_sales_distribution(prodsales_pivot)
    
    # Clustering
    st.subheader("Customer Clustering")
    clustered_data, kmeans = perform_clustering(data_pd)
    st.write("Clustered Data Summary:", clustered_data.groupby('Cluster').mean())
    
    # Regression Model and Accuracy
    st.subheader("Regression Model Accuracy")
    model, y_test, y_pred = train_regression_model(data_pd)
    accuracy = calculate_accuracy(y_test, y_pred)
    st.write("Model Accuracy (Mean Absolute Error):", accuracy)

if __name__ == "__main__":
    main()
