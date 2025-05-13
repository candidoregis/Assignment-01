#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_excel('SalesDB.xlsx')

# Display the first few rows of the dataset
print(data.head())

# Check for missing/null values
print(data.isnull().sum())

# Extract Year and Month from A_Date
data['Year'] = data['A_Date'].dt.year
data['Month'] = data['A_Date'].dt.month

# Add calculated fields 
data['Total_Sales'] = data['B_Product_Unit_Price'] * data['M_Quantity']
data['Total_Cost'] = (data['B_Product_Unit_Cost'] * data['M_Quantity']) + data['M_Shipping_Cost'] + (data['M_Discount_%'] / 100 * data['Total_Sales'])
data['Profit'] = data['Total_Sales'] - data['Total_Cost']
data['Profit_Margin'] = data['Profit'] / data['Total_Sales']

# Add High_Profit column (1 if Profit_Margin >= 0.2, 0 if Profit_Margin < 0.2)
data['High_Profit'] = (data['Profit_Margin'] >= 0.2).astype(int)

# Create a new column 'Discount_Bucket' based on M_Discount_%
data['Discount_Bucket'] = pd.cut(
    data['M_Discount_%'], 
    bins=[0, 5, 10, 15, 20, 25, 30], 
    labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%']
)

# Display the first few rows of the updated dataset
print(data.head())

# Create a directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
    
# Visualization 01 - Distribution of Profit_Margin
plt.figure(figsize=(10, 6))
sns.histplot(data['Profit_Margin'], kde=True, color='navy')
plt.title('Distribution of Profit Margin')
plt.xlabel('Profit Margin')
plt.savefig('visualizations/01-profit_margin_distribution.png')
plt.close()

# Visualization 02 - Distribution of High_Profit
plt.figure(figsize=(10, 6))
sns.countplot(x='High_Profit', data=data, color='orange')
plt.title('Distribution of High Profit Transactions')
plt.xlabel('High Profit (1 = Yes, 0 = No)')
plt.savefig('visualizations/02-high_profit_distribution.png')
plt.close()

# Select only numerical columns for correlation analysis
numerical_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_columns.corr(method='spearman')

# Display the first few rows of the updated dataset
print(numerical_columns.head())

# Visualization 03 - Correlation Matrix [Method: Pearson]
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features - Pearson')
plt.tight_layout()
plt.savefig('visualizations/03-correlation_matrix_pearson.png')
plt.close()

# Changing the method to 'spearman' for correlation analysis comparison
correlation_matrix = numerical_columns.corr(method='spearman')

# Visualization 04 - Correlation Matrix [Method: Pearson]
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features - Spearman')
plt.tight_layout()
plt.savefig('visualizations/04-correlation_matrix_spearman.png')
plt.close()