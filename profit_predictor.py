#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_excel('SalesDB.xlsx')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Data Preparation: Additional features
# Extract Year and Month from A_Date
data['Year'] = data['A_Date'].dt.year
data['Month'] = data['A_Date'].dt.month

# Add calculated fields 
data['Total_Sales'] = data['B_Product_Unit_Price'] * data['M_Quantity']
data['Total_Cost'] = (data['B_Product_Unit_Cost'] * data['M_Quantity']) + data['M_Shipping_Cost'] + (data['M_Discount_%'] / 100 * data['Total_Sales'])
data['Profit'] = data['Total_Sales'] - data['Total_Cost']
data['Profit_Margin'] = data['Profit'] / data['Total_Sales']

# Display the first few rows of the updated dataset
print(data.head())