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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_excel('SalesDB.xlsx')

# Create a directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

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
    labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%'],
    include_lowest=True # to include the leftmost edge since there are some values equal to 0 (3 in total)
)

#### ====================================================================================== ####

# Check for null values in Discount_Bucket
print(f'Null values in Discount_Bucket: {data["Discount_Bucket"].isnull().sum()}')

# Visualization 01 - Distribution of Discounts in Discount_Bucket
plt.figure(figsize=(10, 6))
sns.countplot(data['Discount_Bucket'], color='red')
plt.title('Distribution of Discounts in Discount_Bucket')
plt.xlabel('Quantity of Discounts')
plt.savefig('visualizations/01-Discount_Bucket_distribution.png')
plt.close()

# Display the first few rows of the updated dataset
print(data.head())
    
# Visualization 02 - Distribution of Profit_Margin
plt.figure(figsize=(10, 6))
sns.histplot(data['Profit_Margin'], kde=True, color='navy')
plt.title('Distribution of Profit Margin')
plt.xlabel('Profit Margin')
plt.savefig('visualizations/02-profit_margin_distribution.png')
plt.close()

# Visualization 03 - Distribution of High_Profit
plt.figure(figsize=(10, 6))
sns.countplot(x='High_Profit', data=data, color='orange')
plt.title('Distribution of High Profit Transactions')
plt.xlabel('High Profit (1 = Yes, 0 = No)')
plt.savefig('visualizations/03-high_profit_distribution.png')
plt.close()

# Select only numerical columns for correlation analysis
numerical_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_columns.corr(method='spearman')

# Display the first few rows of the updated dataset
print(numerical_columns.head())

# Visualization 04 - Correlation Matrix [Method: Pearson]
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features - Pearson')
plt.tight_layout()
plt.savefig('visualizations/04-correlation_matrix_pearson.png')
plt.close()

# Changing the method to 'spearman' for correlation analysis comparison
correlation_matrix = numerical_columns.corr(method='spearman')

# Visualization 05 - Correlation Matrix [Method: Pearson]
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features - Spearman')
plt.tight_layout()
plt.savefig('visualizations/05-correlation_matrix_spearman.png')
plt.close()

#### ====================================================================================== ####

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
print(f'\nNumerical features: {len(numerical_features)}')
print(numerical_features)

print(f'\nCategorical features: {len(categorical_features)}')
print(categorical_features)

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
print(f'\nTraining set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')

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
linRegression = LinearRegression()
linRegression.fit(X_train_transformed, y_reg_train)
lr_pred = linRegression.predict(X_test_transformed)

# Random Forest Regressor
randForest = RandomForestRegressor(n_estimators=100, random_state=42)
randForest.fit(X_train_transformed, y_reg_train)
rf_pred = randForest.predict(X_test_transformed)

# Decision Tree Regressor
decTree = DecisionTreeRegressor(random_state=42)
decTree.fit(X_train_transformed, y_reg_train)
dt_pred = decTree.predict(X_test_transformed)

# Function to calculate evaluation metrics
def evaluate_regression(y_true, y_pred, model_name):
    r2 = round(r2_score(y_true, y_pred),4)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)),4)
    mae = round(mean_absolute_error(y_true, y_pred),4)
    
    print(f'\n{model_name} Performance:')
    print(f'R² Score: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    
    return {'Model': model_name, 'R² Score': r2, 'RMSE': rmse, 'MAE': mae}

# Evaluate all models and store results in a list
results = []
results.append(evaluate_regression(y_reg_test, lr_pred, 'Linear Regression'))
results.append(evaluate_regression(y_reg_test, rf_pred, 'Random Forest'))
results.append(evaluate_regression(y_reg_test, dt_pred, 'Decision Tree'))

# Create a summary table to display the results
regression_results_df = pd.DataFrame(results)
print('\nRegression Models Summary:')
print(regression_results_df)

#### ====================================================================================== ####

# Random Forest feature importance
randForest_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': randForest.feature_importances_
}).sort_values('Importance', ascending=False)

# Display the top 10 features for Random Forest
print('\nRandom Forest Feature Importance:')
print(randForest_importance.head(10))

# Visualization 06 - Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
top_features = randForest_importance.head(10)
ax = sns.barplot(x='Importance', y='Feature', data=top_features)
for i in ax.patches: # Add the value at the end of each bar
    ax.text(
        i.get_width() + 0.01,             # x position: slightly to the right of the bar
        i.get_y() + i.get_height() / 2,   # y position: vertical center of the bar
        f'{i.get_width():.6f}',           # text to display (formatted float)
        va='center'                       # vertical alignment
    )
plt.title('Random Forest - Top 10 Feature Importance for Profit Margin Prediction')
plt.tight_layout()
plt.savefig('visualizations/06-random_forest_feature_importance.png')
plt.close()

# Decision Tree feature importance
decTree_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': decTree.feature_importances_
}).sort_values('Importance', ascending=False)

# Display the top 10 features for Decision Tree
print('\nDecision Tree Feature Importance:')
print(decTree_importance.head(10))

# Visualization 07 - Plot feature importance for Decision Tree
plt.figure(figsize=(10, 6))
top_features = decTree_importance.head(10)
ax = sns.barplot(x='Importance', y='Feature', data=top_features)
for i in ax.patches: # Add the value at the end of each bar
    ax.text(
        i.get_width() + 0.01,             # x position: slightly to the right of the bar
        i.get_y() + i.get_height() / 2,   # y position: vertical center of the bar
        f'{i.get_width():.6f}',           # text to display (formatted float)
        va='center'                       # vertical alignment
    )
plt.title('Decision Tree - Top 10 Feature Importance for Profit Margin Prediction')
plt.tight_layout()
plt.savefig('visualizations/07-decision_tree_feature_importance.png')
plt.close()

#### ====================================================================================== ####

# Random Forest Classifier for High_Profit
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_transformed, y_clf_train)
rf_clf_pred = rf_clf.predict(X_test_transformed)
rf_clf_prob = rf_clf.predict_proba(X_test_transformed)[:, 1]

# Calculate and display metrics
accuracy = accuracy_score(y_clf_test, rf_clf_pred)
precision = precision_score(y_clf_test, rf_clf_pred)
recall = recall_score(y_clf_test, rf_clf_pred)
f1 = f1_score(y_clf_test, rf_clf_pred)
cm = confusion_matrix(y_clf_test, rf_clf_pred)

print('\nRandom Forest Classifier Performance:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('\nConfusion Matrix:')
print(cm)

# Visualization 08 - Random Forest Confusion Matrix Classifier
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - High Profit Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0.5, 1.5], ['Low Profit', 'High Profit'])
plt.yticks([0.5, 1.5], ['Low Profit', 'High Profit'])
plt.tight_layout()
plt.savefig('visualizations/08-confusion_matrix_randForest_classifier.png')
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_clf_test, rf_clf_prob)
roc_auc = auc(fpr, tpr)

# Visualization 09 - ROC Curve for High Profit Prediction
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - High Profit Prediction')
plt.legend(loc='lower right')
plt.savefig('visualizations/09-roc_curve_high_profit.png')
plt.close()

# Feature importance for Random Forest Classification
randForest_clf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)
print('\nTop 10 features for predicting High Profit:')
print(randForest_clf_importance.head(10))

# Visualization 10 - Plot feature importance for Random Forest Classification
plt.figure(figsize=(12, 8))
top_features = randForest_clf_importance.head(10)
ax = sns.barplot(x='Importance', y='Feature', data=top_features)
for i in ax.patches: # Add the value at the end of each bar
    ax.text(
        i.get_width() + 0.01,             # x position: slightly to the right of the bar
        i.get_y() + i.get_height() / 2,   # y position: vertical center of the bar
        f'{i.get_width():.6f}',           # text to display (formatted float)
        va='center'                       # vertical alignment
    )
plt.title('Random Forest - Top 10 Feature Importance for High Profit Prediction')
plt.tight_layout()
plt.savefig('visualizations/10-RandForest_classific_high_profit_feature_importance.png')
plt.close()

#### ====================================================================================== ####

# Summary and QA Section
# Can we predict High Profit transactions reliably?
print('\nQuestion 1 - Can we predict High Profit transactions reliably?')
if accuracy > 0.8 and precision > 0.8:
    reliability = "Based on our Random Forest classifier, we can predict high profit transactions with good reliability.\n"
    reliability += f" The model achieved an accuracy of {accuracy:.2f} and precision of {precision:.2f}, meaning it correctly\n"
    reliability += "identifies high profit transactions most of the time."
    reliability += f" The F1 score of {f1:.2f} indicates a good balance between precision and recall."
else:
    reliability = "Our model shows moderate reliability in predicting high profit transactions.\n"
    reliability += f"With an accuracy of {accuracy:.2f} and precision of {precision:.2f}, there is room for improvement.\n"
    reliability += "in the model's predictive power."

print(reliability)

# Which features are most influential for predicting High Profit?
print('\nWhich features are most influential for predicting High Profit?')
top_features = randForest_clf_importance.head(5)['Feature'].tolist()
features_conclusion = "The most influential features for predicting high profit transactions are: \n"
features_conclusion += ", ".join([f"'{feature}'" for feature in top_features])
features_conclusion += ". \n"
features_conclusion += "These features have the strongest impact on determining whether a transaction will yield high profit.\n"
features_conclusion += "Business strategies should focus on optimizing these key factors to maximize profitability.\n"

print(features_conclusion)

# Save the summary to a text file
with open('visualizations/summary.txt', 'w') as f:
    f.write("# Profit Margin Prediction Analysis\n\n")
    f.write("## Regression Models Performance\n")
    f.write(regression_results_df.to_string(index=False))
    f.write("\n\n## High Profit Classification Performance\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("\n## Can we predict High Profit transactions reliably?\n")
    f.write(reliability)
    f.write("\n\n## Which features are most influential for predicting High Profit?\n")
    f.write(features_conclusion)

print("\nAnalysis complete! Results and visualizations saved to the 'visualizations' directory.")
