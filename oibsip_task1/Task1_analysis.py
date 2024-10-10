import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Datasets 

df = pd.read_csv(r'C:\Users\chanv\OIBSIP\data\retail_sales_dataset.csv')

# Showing the First few rows of the data

print(df.head())

# Consistent Formatting in text fields

df['Gender'] = df['Gender'].str.lower()
df['Product Category'] = df['Product Category'].str.lower()
print(df['Gender'].unique())
print(df['Product Category'].unique())

# Converting 'Date' column to Datetime

df['Date'] = pd.to_datetime(df['Date'])  

# Checking for missing values

print()
print("\n",df.isnull().sum())

# Checking for Duplicates:

print("\n", df.duplicated().sum())

# Analysis $ Visualization

#1: Descriptive Statistics: Calculate basic statistics (mean, median, mode, standard deviation).

mean_values = df.select_dtypes(include =['number']).mean()
median_values = df.select_dtypes(include =['number']).median()
mode_values = df.select_dtypes(include =['number']).mode().iloc[0]  # Selecting the first value
std_values = df.select_dtypes(include =['number']).std()

print("\n Mean:\n", mean_values)
print("\n Median:\n", median_values)
print("\n Mode:\n", mode_values)
print("\n Stand_D:\n", std_values)

# 2: Time Series Analysis (Sales trends over time)

daily_sales = df.groupby('Date')['Total Amount'].sum()
monthly_sales = daily_sales.resample('M').sum().reset_index()
monthly_sales['Date'] = monthly_sales['Date'].dt.strftime('%Y-%m')
print("\nMonthly Sales Trends:\n", monthly_sales) 


# Visualization using Line plot to show how sales vary over time (Monthly).

plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Total Amount', data = monthly_sales, marker='o', color='green', linestyle='-')
plt.title('Sales Trends Over Time (Monthly)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
for i in range(len(monthly_sales)):             ## Adding text labels on the points             
    plt.text(i, monthly_sales['Total Amount'].iloc[i], f"${monthly_sales['Total Amount'].iloc[i]:.0f}", ha='center', 
             va='bottom', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3: Customer and Product Analysis: Analyzing Customer Age and Gender Influence on Purchasing Behavior

age_bins = [18, 24, 34, 44, 54, 64, 100]  # Age category and labels for ranges
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['Age Range'] = pd.cut(df['Age'], bins = age_bins, labels = age_labels, right = False) #New column (Age Range)
gender_age_sales = df.groupby(['Gender', 'Age Range'])['Total Amount'].sum().reset_index()
print("\nTotal Sales by Age Range and Gender:\n", gender_age_sales)


# Visualization using a Barchart to show how age and gender impact spending:
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Age Range', y='Total Amount', hue='Gender', data=gender_age_sales, palette = "rocket")
plt.title('Total Sales by Age Range and Gender')
plt.xlabel('Age Range')
plt.ylabel('Total Sales (Total Amount)')

for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9),
                        textcoords = 'offset points')

# 4: Top Product Categories

pcategory_sales = df.groupby('Product Category')['Total Amount'].sum().reset_index()
print("\nTop product categories:\n", pcategory_sales)


# Visualization using Bar chart showing Total Sales per product category.

plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Product Category', y='Total Amount', data=pcategory_sales, palette="icefire")
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales ($)')

for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'),  
                      (p.get_x() + p.get_width() / 2., p.get_height()),  
                      ha = 'center', va = 'center', 
                      xytext = (0, 9), textcoords='offset points') 
plt.show()

# 5: Age, Spending, and Product Preferences (Relationships)

age_product_sales = df.groupby(['Age Range', 'Product Category'])['Total Amount'].sum().unstack() #pivoting
print("\nSpending by age and product categories:\n", age_product_sales)


# Visualization using Heatmap to analyze which product categories appeal most to different age groups.

plt.figure(figsize = (14, 8))
sns.heatmap(age_product_sales, cmap = 'coolwarm', annot = True, fmt='.0f')
plt.title('Total Spending by Age and Product Category')
plt.show()

# 6: Seasonal Trends (Analyzing by month)

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
monthly_product_sales = df.groupby(['Month', 'Product Category'])['Total Amount'].sum().unstack()
print("\nSeasonal Trends:\n", monthly_product_sales) 

# Visualization monthly sales (seasonal trends)

plt.figure(figsize = (14, 8))
sns.heatmap(monthly_product_sales, cmap = 'coolwarm', annot = True, fmt='.0f')
plt.title('Total Spending by Month and Product Category')
plt.show()

# 7: Number of Items Bought Per Transaction

item_count_sales = df.groupby('Quantity')['Transaction ID'].count().reset_index(name='Transaction Count')
print("\nItems Per Transaction:\n", item_count_sales)


# Visualization of Average Sales by Quantity of Items per Transaction

bar_plot = item_count_sales.plot(kind='bar', x='Quantity', y='Transaction Count', color = sns.color_palette("rocket"), 
                                    figsize=(10, 6),  legend=False)
plt.title('Count of Transactions by Quantity')
plt.ylabel('Transaction Count')
plt.xlabel('Quantity')

for p in bar_plot.patches:
    tcount_value = p.get_height()  # Get the height (count) of the current bar
    total_tcount = item_count_sales['Transaction Count'].sum()
    percentage_value = (tcount_value / total_tcount) * 100  # Calculate percentage for the current bar
    bar_plot.annotate(f'{tcount_value} ({percentage_value:.1f}%)',  
                      (p.get_x() + p.get_width() / 2., tcount_value),  
                      ha='center', va='bottom', 
                      xytext=(0, 5), textcoords='offset points')  
plt.show()

# 8: Product Price Distribution by Category

df['Quantity'] = df['Total Amount'] / df['Price per Unit']
category_gender_group = df.groupby(['Product Category', 'Gender']).agg(
    {'Total Amount': 'sum','Quantity': 'sum'}).reset_index()
category_gender_group['Avg Price per Unit'] = category_gender_group['Total Amount'] / category_gender_group['Quantity']
print("\nProduct Price Distribution by Category:\n", category_gender_group)


# Visualizing using bar plot for Total Amount and Avg Price per Unit by Product Category and Gender

plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x='Product Category', y='Avg Price per Unit', hue='Gender', data=category_gender_group, palette='rocket')
plt.title('Average Price per Unit by Product Category and Gender')
plt.xlabel('Product Category')
plt.ylabel('Average Price per Unit')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'),  
                      (p.get_x() + p.get_width() / 2., p.get_height()),  
                      ha = 'center', va = 'center', 
                      xytext = (0, 9), textcoords = 'offset points')
plt.legend(title='Gender', loc='lower right')
plt.show()

