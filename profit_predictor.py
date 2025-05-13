#### ASSIGNMENT 01 - AIDI2000 ####

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_excel('SalesDB.xlsx')

# Display the first few rows of the dataset
""" print(data.head())"""

# Check for missing/null values
# print(data.isnull().sum())

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
    labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%'],
    include_lowest=True # to include the leftmost edge since there are some values equal to 0 (3 in total)
)

# Check for null values in Discount_Bucket
# print(f'Null values in Discount_Bucket: {data["Discount_Bucket"].isnull().sum()}')

# Visualization 01 - Distribution of Discounts in Discount_Bucket
""" plt.figure(figsize=(10, 6))
sns.countplot(data['Discount_Bucket'], color='red')
plt.title('Distribution of Discounts in Discount_Bucket')
plt.xlabel('Quantity of Discounts')
plt.savefig('visualizations/01-Discount_Bucket_distribution.png')
plt.close() """

# Display the first few rows of the updated dataset
# print(data.head())

# Create a directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
    
# Visualization 02 - Distribution of Profit_Margin
""" plt.figure(figsize=(10, 6))
sns.histplot(data['Profit_Margin'], kde=True, color='navy')
plt.title('Distribution of Profit Margin')
plt.xlabel('Profit Margin')
plt.savefig('visualizations/02-profit_margin_distribution.png')
plt.close() """

# Visualization 03 - Distribution of High_Profit
""" plt.figure(figsize=(10, 6))
sns.countplot(x='High_Profit', data=data, color='orange')
plt.title('Distribution of High Profit Transactions')
plt.xlabel('High Profit (1 = Yes, 0 = No)')
plt.savefig('visualizations/03-high_profit_distribution.png')
plt.close() """

# Select only numerical columns for correlation analysis
numerical_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_columns.corr(method='spearman')

# Display the first few rows of the updated dataset
# print(numerical_columns.head())

# Visualization 04 - Correlation Matrix [Method: Pearson]
""" plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features - Pearson')
plt.tight_layout()
plt.savefig('visualizations/04-correlation_matrix_pearson.png')
plt.close() """

# Changing the method to 'spearman' for correlation analysis comparison
correlation_matrix = numerical_columns.corr(method='spearman')

# Visualization 05 - Correlation Matrix [Method: Pearson]
""" plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features - Spearman')
plt.tight_layout()
plt.savefig('visualizations/05-correlation_matrix_spearman.png')
plt.close() """

# OBS.: To start prediction analysis, we need to identify features and targets to use
# We should not use Total_Sales, Total_Cost, or Profit as features as they are 
# directly used to calculate Profit_Margin. Not all features are used in the model, 
# customers information, dates/day of the week, as I think they won't bring much impact right now.


# Numerical features to include (avoiding derived features that would cause data leakage)
numerical_features = [
    'B_Product_Unit_Cost', 
    'B_Product_Unit_Price', 
    'M_Quantity', 
    'M_Discount_%', 
    'M_Shipping_Cost',
    'Year',
    'Month'
]

# Categorical features that need encoding
categorical_features = [
    'B_Product_Category', 
    'D_Store_Location', 
    'Discount_Bucket', 
    'Order_Type', 
    'Payment_Method'
]

# Print feature information
#print(f'\nNumerical features: {len(numerical_features)}')
##print(numerical_features)

#print(f'\nCategorical features: {len(categorical_features)}')
#print(categorical_features)

# For regression (predicting Profit_Margin)
y_regres = data['Profit_Margin']

# For classification (predicting High_Profit)
y_classif = data['High_Profit']

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Split data for regression: 80% train / 20% test
X = data[numerical_features + categorical_features]
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regres, test_size=0.2, random_state=42
)
# Split data for classification: 80% train / 20% test
# The _ is used to ignore the first two outputs of train_test_split since they're already defined
# in the previous split. 
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_classif, test_size=0.2, random_state=42
)

# Print the sizes of the training and test sets
# print(f'\nTraining set size: {X_train.shape[0]} samples')
# print(f'Test set size: {X_test.shape[0]} samples')

# Apply preprocessing to get transformed feature matrix
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after preprocessing
feature_names = (
    numerical_features +
    list(preprocessor.named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_transformed, y_reg_train)
lr_pred = lr.predict(X_test_transformed)

r2 = r2_score(y_reg_test, lr_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, lr_pred))
mae = mean_absolute_error(y_reg_test, lr_pred)

print(f'\nLinear Regression Performance:')
print(f'R² Score: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')