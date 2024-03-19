# CT7206_sourcecode
import pandas as pd

# Specify the CSV file path
csv_file_path = 'temu_product_sales_dataset (1).csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the DataFrame
print(df)

df_shape = df.shape

# Print the shape
print("Shape:", df_shape)

df_describe = df.describe

#describe
print("Shape:", df_describe)

import pandas as pd
import numpy as np

# Assuming df is your DataFrame
# Replace 'leve_1_category_id' with the actual column name if it's different
distinct_ID = df['leve_1_category_id'].unique()

# Print distinct values
(print(distinct_ID))

# Counting distinct ID
count_of_distinct_ID = len(set(distinct_ID))

# Printing the result
print(f"Count of the total categories are: {count_of_distinct_ID}")

df.info()

df.head(5) 

df.dtypes


# Assuming your DataFrame is named 'df' and the column is 'sales_info'
df['sales_info'] = pd.to_numeric(df['sales_info'].str.replace(' sold', ''), errors='coerce')

# Display the cleaned DataFrame
df.head(5)

df.count()

print(df.isnull().sum())

import seaborn as sns                  
sns.boxplot(x=df['price'])

rows_with_missing_values = df[df.isna().any(axis=1)]

# Display the rows with missing values
print(rows_with_missing_values)


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preview the Data
print(df.head())

# 2. Check Data Types
print(df.dtypes)

# 3. Summary Statistics
print(df.describe())

# 4. Missing Values
print(df.isnull().sum())

# 5. Data Visualization
# Example: Histogram of a numerical feature
plt.hist(df['Numerical Feature'])
plt.xlabel('Numerical Feature')
plt.ylabel('Frequency')
plt.title('Histogram of Numerical Feature')
plt.show()

# 6. Correlation Analysis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("temu_product_sales_dataset (1).csv")

# Display the first few rows of the dataset
print("Head of the dataset:")
print(df.head())

# Display the summary statistics of the dataset
print("\nDescription of the dataset:")
print(df.describe())

# Display the information about the dataset
print("\nInformation about the dataset:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Data manipulation for missing values
# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Check for duplicates
print("\nNumber of duplicates in the dataset:")
print(df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Sorting and ranking
# Sort the dataset by a specific column
sorted_df = df.sort_values(by='column_name', ascending=False)

# Rank the dataset based on a specific column
df['Rank'] = df['column_name'].rank(ascending=False)

# Correlation and Covariance
# Calculate correlation matrix
correlation_matrix = df.corr()

# Calculate covariance matrix
covariance_matrix = df.cov()

# Descriptive analysis
# Perform descriptive analysis on a specific column
print("\nDescriptive analysis of a specific column:")
print(df['column_name'].describe())


# Import necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv("temu_product_sales_dataset (1).csv")

# Display the first few rows of the dataset
print("Head of the dataset:")
print(df.head())

# Display the summary statistics of the dataset
print("\nDescription of the dataset:")
print(df.describe())

# Display the information about the dataset
print("\nInformation about the dataset:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Exclude non-numeric columns from the calculation of the mean
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Now, non-numeric columns are not included in the mean calculation, avoiding the TypeError.

#Check for duplicates
print("\nNumber of duplicates in the dataset:")
print(df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Rank the dataset based on a specific column
df['Rank'] = df['leve_1_category_id'].rank(ascending=False)

# Drop non-numeric columns before calculating correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculating correlation matrix
correlation_matrix = numeric_df.corr()

# Printing correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# Drop non-numeric columns before calculating covariance and correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculating covariance matrix
covariance_matrix = numeric_df.cov()

# Calculating correlation matrix
correlation_matrix = numeric_df.corr()

# Printing covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)

# Printing correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Descriptive analysis
# Perform descriptive analysis on a specific column
print("\nDescriptive analysis of a specific column:")
print(df['leve_1_category_id'].describe())

TAG CLOUD 

!pip install wordcloud

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('temu_product_sales_dataset (1).csv')

# Extract the categories from the dataset
categories = data['leve_2_category_name'].tolist()

# Combine all categories into a single string
text = ' '.join(categories)

# Generate the WordCloud
wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(text)

# Display the WordCloud image
plt.figure(figsize = (8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

LAYOUT
# Distribution Analysis
# Plot histograms to visualize the distribution of numerical variables
plt.figure(figsize=(12, 6))
df.hist()
plt.tight_layout()
plt.show()

# Box plots to visualize the distribution of numerical variables
plt.figure(figsize=(14, 8))  # Increase width to accommodate longer labels
sns.boxplot(data=df)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Density plots to visualize the distribution of numerical variables
plt.figure(figsize=(12, 6))
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    sns.kdeplot(data=df[column], fill=True, label=column)
plt.legend()
plt.show()

# Violin plots to compare the distribution of numerical variables across different categories
plt.figure(figsize=(12, 6))
sns.violinplot(x='leve_1_category_id', y='leve_2_category_id', data=df)
plt.show()

# Swarm plots to compare the distribution of numerical variables across different categories
plt.figure(figsize=(12, 6))
sns.swarmplot(x='leve_1_category_id', y='leve_2_category_id', data=df)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'leve_1_category_name' is the categorical variable in your dataset
categorical_variable = 'leve_1_category_name'

# Count the frequency of the categorical variable
categorical_counts = df[categorical_variable].value_counts()

# Create a bar plot to visualize the distribution of the categorical variable
plt.figure(figsize=(12, 6))
categorical_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency of Categories')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Create a pie chart to visualize the distribution of the categorical variable
plt.figure(figsize=(8, 8))
plt.pie(categorical_counts, labels=categorical_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Categories')
plt.show()

# perform Product Performance Analysis and identify top-selling products based on 
#the number of units sold or revenue generated, as well as analyze product categories and 
#subcategories to understand sales patterns

# Identify top-selling products based on the number of units sold
top_products_by_units_sold = df.groupby('title')['sales_volume'].sum().sort_values(ascending=False).head(10)

# Identify top-selling products based on revenue generated
df['revenue'] = df['sales_volume'] * df['price']
top_products_by_revenue = df.groupby('title')['revenue'].sum().sort_values(ascending=False).head(10)

# Analyze product categories and subcategories to understand sales patterns
category_sales = df.groupby(['leve_1_category_name', 'leve_2_category_name'])['sales_volume'].sum().unstack().fillna(0)

# Plotting the top-selling products and category sales
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
top_products_by_units_sold.plot(kind='bar', color='skyblue')
plt.title('Top Selling Products by Units Sold')
plt.xlabel('Product Title')
plt.ylabel('Total Units Sold')

plt.subplot(2, 2, 3)
top_products_by_revenue.plot(kind='bar', color='salmon')
plt.title('Top Selling Products by Revenue Generated')
plt.xlabel('Product Title')
plt.ylabel('Total Revenue Generated')

plt.subplot(2, 2, 4)
category_sales.plot(kind='bar', stacked=True)
plt.title('Sales Volume by Category and Subcategory')
plt.xlabel('Category')
plt.ylabel('Total Sales Volume')

plt.tight_layout()
plt.show()

#Customer Segmentation based on purchasing behavior and conduct clustering analysis 
#to group customers with similar characteristics, you can use Python with libraries such as 
#pandas, scikit-learn, and matplotlib. 
#Below is a sample code snippet that demonstrates how to achieve this analysis using K-means clustering: 

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Calculate total spending per customer
df['total_spending'] = df['sales_volume'] * df['price']

# Aggregate data at the customer level based on available columns
customer_data = df.groupby(['leve_1_category_name', 'leve_2_category_name']).agg({
    'sales_volume': 'sum',
    'total_spending': 'sum'
}).reset_index()

# Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['sales_volume', 'total_spending']])

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['sales_volume'], customer_data['total_spending'], c=customer_data['cluster'], cmap='viridis', s=50)
plt.xlabel('Total Sales Volume')
plt.ylabel('Total Spending')
plt.title('Customer Segmentation based on Purchasing Behavior')
plt.show()
