# Import necessary libraries

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Load the dataset

nyc_abnb_data = pd.read_csv(r'C:\Users\chanv\OIBSIP\oibsip_task3\AB_NYC_2019.csv')

# Display the first few rows of the dataset

print("\nDataset Overview:\n")
print(nyc_abnb_data.head())

# 1. Data Integrity: Checking the Info and the data types of the dataset:

nyc_abnb_data.info()

# Check for inconsistencies, errors, and data corruption.

for column in ['neighbourhood_group', 'room_type']:
    print(f"Unique values in {column}: {nyc_abnb_data[column].unique()}")

    # Function to remove junk characters from a string:

def clean_text(text):
    if isinstance(text, str):
        text = text.lower() # Convert to lowercase for consistency
        text = text.strip() # Remove any extra spaces at the beginning or end
        text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove non-ASCII characters (optional)

        # Remove junk characters while retaining non-ASCII (foreign language) text
        text = re.sub(r'[^A-Za-z0-9\s\.\,\!\?\@\#\$\%\&\*\(\)\-\_\+\=\:\;\x80-\uFFFF]', '', text)

    return text

character_columns = ['host_name', 'room_type', 'neighbourhood_group', 'neighbourhood', 'name']

# Apply the clean_text function to all text-based columns

for column in character_columns:
    nyc_abnb_data[column] = nyc_abnb_data[column].apply(clean_text)
    print(nyc_abnb_data.head())


nyc_abnb_data.to_csv('cleaned_AB_NYC_2019.csv', index=False)

# 2. Missing Data Handling: Identify missing values and decide whether to fill them or remove rows/columns:

nyc_missing_data = nyc_abnb_data.isnull().sum()
print("\nMissing data in each column:\n")
nyc_missing_data

# Calculate percentage of missing data

missing_percentage = (nyc_abnb_data.isnull().sum() / len(nyc_abnb_data)) * 100
print(missing_percentage)

# Filling missing values

nyc_abnb_data['reviews_per_month'].fillna(0, inplace=True)        # replacing missing reviews per month to 0
nyc_abnb_data['last_review'].fillna('No Review', inplace=True)    # replacing missing reviews'No Review'
nyc_abnb_data['host_name'].fillna('Unknown', inplace=True)        # replacing missing host name to 'Unknown'
nyc_abnb_data['name'].fillna('Unnamed', inplace=True)             # replacing missing name to 'Unnamed'

# Check again for remaining missing values

missing_nyc_data2 = nyc_abnb_data.isnull().sum()
print("Missing data after filling:\n")
missing_nyc_data2

# Checking and removing duplicate records:

duplicates = nyc_abnb_data.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
data_cleaned = nyc_abnb_data.drop_duplicates()

# Ensuring consistent formatting on text (categorical) columns for accurate analysis:



# Function to remove junk characters from a string:

def clean_text(text):
    if isinstance(text, str):
        text = text.lower() # Convert to lowercase for consistency
        text = text.strip() # Remove any extra spaces at the beginning or end
        text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove non-ASCII characters (optional)

        # Remove junk characters while retaining non-ASCII (foreign language) text
        text = re.sub(r'[^A-Za-z0-9\s\.\,\!\?\@\#\$\%\&\*\(\)\-\_\+\=\:\;\x80-\uFFFF]', '', text)

    return text

character_columns = ['host_name', 'room_type', 'neighbourhood_group', 'neighbourhood', 'name']

# Apply the clean_text function to all text-based columns
for column in character_columns:
    nyc_abnb_data[column] = nyc_abnb_data[column].apply(clean_text)
    print(nyc_abnb_data.head())


nyc_abnb_data.to_csv('cleaned_AB_NYC_2019.csv', index=False)

# Outlier Detection: Analyze numerical data to find and handle any outliers

numerical_columns = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']

for col in numerical_columns:
    plt.figure(figsize = (10, 6))
    nyc_abnb_data.boxplot(column = col)
    plt.title(f'Boxplot for {col}')
    plt.show()

    # Remove outliers using quantiles

Q1 = nyc_abnb_data[numerical_columns].quantile(0.25)
Q3 = nyc_abnb_data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
outliers = ((nyc_abnb_data[numerical_columns] < (Q1 - 1.5 * IQR)) | (nyc_abnb_data[numerical_columns] > (Q3 + 1.5 * IQR)))
outliers.head()

# Drop rows with outliers

data_cleaned = nyc_abnb_data[~outliers.any(axis=1)]
data_cleaned = data_cleaned.reset_index(drop=True)
data_cleaned