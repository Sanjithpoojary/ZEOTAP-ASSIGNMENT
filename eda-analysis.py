import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Function to load and prepare data
def load_data():
    """
    Load the three CSV files and perform initial data cleaning
    """
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    return customers_df, products_df, transactions_df

def perform_eda(customers_df, products_df, transactions_df):
    """
    Perform Exploratory Data Analysis
    """
    # Basic statistics and info
    print("\nCustomer Data Summary:")
    print(customers_df.info())
    print("\nProduct Data Summary:")
    print(products_df.info())
    print("\nTransaction Data Summary:")
    print(transactions_df.info())
    
    # Customer analysis
    customer_metrics = {
        'total_customers': len(customers_df),
        'customers_by_region': customers_df['Region'].value_counts(),
        'signup_trends': customers_df['SignupDate'].dt.year.value_counts().sort_index()
    }
    
    # Product analysis
    product_metrics = {
        'total_products': len(products_df),
        'products_by_category': products_df['Category'].value_counts(),
        'price_statistics': products_df['Price'].describe()
    }
    
    # Transaction analysis
    transactions_df['Year'] = transactions_df['TransactionDate'].dt.year
    transactions_df['Month'] = transactions_df['TransactionDate'].dt.month
    
    transaction_metrics = {
        'total_transactions': len(transactions_df),
        'total_revenue': transactions_df['TotalValue'].sum(),
        'avg_transaction_value': transactions_df['TotalValue'].mean(),
        'revenue_by_year': transactions_df.groupby('Year')['TotalValue'].sum()
    }
    
    return customer_metrics, product_metrics, transaction_metrics

def generate_visualizations(customers_df, products_df, transactions_df):
    """
    Create visualizations for the analysis
    """
    # Set style
    plt.style.use('seaborn')
    
    # 1. Customer distribution by region
    plt.figure(figsize=(10, 6))
    sns.countplot(data=customers_df, x='Region')
    plt.title('Customer Distribution by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('customer_distribution.png')
    
    # 2. Product price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=products_df, x='Price', bins=30)
    plt.title('Product Price Distribution')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    
    # 3. Transaction value over time
    transactions_df['YearMonth'] = transactions_df['TransactionDate'].dt.to_period('M')
    monthly_revenue = transactions_df.groupby('YearMonth')['TotalValue'].sum().reset_index()
    monthly_revenue['YearMonth'] = monthly_revenue['YearMonth'].astype(str)
    
    plt.figure(figsize=(15, 6))
    plt.plot(monthly_revenue['YearMonth'], monthly_revenue['TotalValue'])
    plt.title('Monthly Revenue Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('revenue_trend.png')

def analyze_customer_behavior(customers_df, transactions_df):
    """
    Analyze customer purchasing behavior
    """
    # Merge transactions with customer data
    customer_transactions = pd.merge(
        transactions_df,
        customers_df[['CustomerID', 'Region']],
        on='CustomerID'
    )
    
    # Calculate key metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': ['sum', 'mean'],
        'Quantity': 'sum'
    }).reset_index()
    
    customer_metrics.columns = ['CustomerID', 'total_transactions', 'total_spend', 'avg_transaction_value', 'total_items']
    
    return customer_metrics

def main():
    # Load data
    customers_df, products_df, transactions_df = load_data()
    
    # Perform EDA
    customer_metrics, product_metrics, transaction_metrics = perform_eda(
        customers_df, products_df, transactions_df
    )
    
    # Generate visualizations
    generate_visualizations(customers_df, products_df, transactions_df)
    
    # Analyze customer behavior
    customer_behavior = analyze_customer_behavior(customers_df, transactions_df)
    
    # Save results
    customer_behavior.to_csv('customer_behavior_analysis.csv', index=False)
    
    return {
        'customer_metrics': customer_metrics,
        'product_metrics': product_metrics,
        'transaction_metrics': transaction_metrics,
        'customer_behavior': customer_behavior
    }

if __name__ == "__main__":
    results = main()
