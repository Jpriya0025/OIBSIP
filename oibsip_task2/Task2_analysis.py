# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# 1. Loading the Datasets 
housing = pd.read_csv(r'C:\Users\chanv\OIBSIP\oibsip_task2\Housing.csv')

# 2: Data Exploration and Cleaning

# Display the first few rows of the dataset
print("\nDataset Overview:\n")
print(housing.head())

# Check for missing values

print("\nMissing Values:")
missing_values = housing.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values\n")
else:
    print("\nMissing values\n")

# Display the data info
housing.info()
print()

# Remove rows with missing values:

if missing_values.sum() != 0:
    data = housing.dropna()            # Drop rows with missing values
    print("Dropped rows with missing values.")
else:
    housing = housing
    print("\nNo missing values. Skipped dropna.\n")

# 4: Feature Selection

numerical_features = housing.select_dtypes(include=[np.number]).columns.tolist() # all numerical fields in the dataset
print("\nNumerical Features Selected for Model:\n", numerical_features)

categorical_features = categorical_features = housing.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nCategorical Features Selected for Model:\n", categorical_features)

# Visualization using histograms for all numerical features for better understanding

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

plt.figure(figsize=(16, 12))

for i, col in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)  # Adjust layout for up to 9 features (3x3 grid)
    
    # Define bin_edges for each feature based on its range and set intervals
    if col == 'price':
        bin_edges = np.arange(0, housing['price'].max() + 2000000, 2000000)  # Bins every 2 million for price
    else:
        # Create bin ranges dynamically for other columns using an interval of ~10 bins
        bin_width = (housing[col].max() - housing[col].min()) / 10
        bin_edges = np.arange(housing[col].min(), housing[col].max() + bin_width, bin_width)
    
    # Plot histograms for each feature using the calculated bins
    sns.histplot(housing[col], bins=bin_edges, kde=True, color = colors[i])
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)

plt.tight_layout()
plt.show()



# Visualization using histograms for all categorical features for better understanding

plt.figure(figsize=(12, 10))
for i, col in enumerate(categorical_features):
    plt.subplot(3, 3, i + 1)  # 3 rows, 3 columns
    sns.countplot(x=housing[col], color=colors[i % len(colors)])  # Cycle through colors
    plt.title(f"Count of {col}")

plt.tight_layout()
plt.show()

# Define the target variable (assumed to be 'price')

t_variable = 'price'  # Adjust according to your dataset column name
if t_variable not in housing.columns:
    t_variable = numerical_features[-1]  # Use last numerical column as target if unknown
print("\nTarget variable:", t_variable)

# Calculate and plot correlation matrix

plt.figure(figsize=(10, 8))
correlation_matrix = housing.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='BrBG', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Scatter plots to visualize relationships between numerical features and price

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_features[1:]):  # Skipping 'price'
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=housing[col], y=housing['price'])
    plt.title(f"{col} vs Price")

plt.tight_layout()
plt.show()

# Boxplots for categorical features vs price

plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=housing[col], y=housing['price'], color = colors[i])
    plt.title(f"{col} vs Price")

plt.tight_layout()
plt.show()

# Checking for Outliers

# Boxplots to identify outliers in numerical features

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=housing[col], color = colors [i])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# Outliner Treatment using Z-scores

# Select only numerical columns for outlier detection
numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']  # Add any other numerical columns if present

# Calculate Z-scores for all numerical columns
z_scores = np.abs(stats.zscore(housing[numerical_columns]))

threshold = 3   # 3 is common for outlier detection)

# removing rows where Z-scores are greater than the threshold
housing = housing[(z_scores < threshold).all(axis=1)]
print(f"Shape of the dataset after outlier removal: {housing.shape}")

# displaying first few rows on the dataset

print("\nFiltered Housing dataset outlier removal:\n")
housing.head()

# Encoding the categorial fields to numbers(no = 0, yes = 1):

housing_encoded = pd.get_dummies(housing, columns=categorical_features, drop_first=True)
print("\nEncoded Housing dataset:\n")
housing_encoded. head()

# Scaling

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
housing_encoded[numerical_columns] = scaler.fit_transform(housing_encoded[numerical_columns])
print(housing_encoded[numerical_columns].describe())

# 5: Model Training

# Prepare the features (X) and target (y)

X = housing_encoded.drop(columns=['price'])  # Dropping the target variable 'price' 
y = housing_encoded['price']  # The target variable is 'price'

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate evaluation metrics

model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficients:", model.coef_)
print()
print("Intercept:", model.intercept_)

# Prediction on the test data

y_pred = model.predict(X_test)
print("\nFirst 10 Predictions:\n", y_pred[:10])

# 6: Model Evaluation
# Evaluating the model's performance on a separate test dataset:

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}\n")
print(f"\nR-squared: {r2:.2f}\n")


# Step 8: Visualization

# Plot the actual vs predicted values

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()

# Display the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

# Predict on the training data to get the fitted values
y_train_pred = model.predict(X_train)

# Residuals: Difference between actual and predicted on the training set
residuals = y_train - y_train_pred

# Plot Residuals vs Fitted Values (Predicted Values) on the training set
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values (Predicted Prices)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True)
plt.show()

import seaborn as sns

# Plot the histogram of the residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='green')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

