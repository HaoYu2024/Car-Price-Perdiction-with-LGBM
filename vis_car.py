import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import datetime

# Load sample datasets
sample_sub = pd.read_csv('sample_submission.csv', nrows=1000)
train = pd.read_csv('train.csv', nrows=1000)
test = pd.read_csv('test.csv', nrows=1000)
original = pd.read_csv('used_cars.csv', nrows=1000)





# Load the uploaded training data


# Let's check the first few rows of the dataset to understand its structure
train.head()

# Visualizing missing data

missing_data = train.isnull().sum()

# Display missing data counts

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_data.index, y=missing_data.values)
plt.xticks(rotation=90)
plt.title('Missing Data Count for Each Feature')
plt.ylabel('Count of Missing Values')
plt.show()


# Visualizing the distribution of categorical variables (brand, fuel_type, transmission)
plt.figure(figsize=(12, 8))

# Brand distribution
plt.subplot(3, 1, 1)
sns.countplot(y='brand', data=train, order=train['brand'].value_counts().index)
plt.title('Distribution of Car Brands')

# Fuel Type distribution
plt.subplot(3, 1, 2)
sns.countplot(y='fuel_type', data=train, order=train['fuel_type'].value_counts().index)
plt.title('Distribution of Fuel Types')

# Transmission distribution
plt.subplot(3, 1, 3)
sns.countplot(y='transmission', data=train, order=train['transmission'].value_counts().index)
plt.title('Distribution of Transmission Types')

plt.tight_layout()
plt.show()
# Visualizing the relationship between numeric features (milage, model_year) and price

plt.figure(figsize=(12, 6))

# Scatter plot for milage vs price
plt.subplot(1, 2, 1)
sns.scatterplot(x='milage', y='price', data=train)
plt.title('Milage vs Price')

# Scatter plot for model_year vs price
plt.subplot(1, 2, 2)
sns.scatterplot(x='model_year', y='price', data=train)
plt.title('Model Year vs Price')

plt.tight_layout()
plt.show()

# Correlation heatmap for numeric variables
numeric_columns = ['milage', 'model_year', 'price']
correlation_matrix = train[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Numeric Features')
plt.show()



# Update the function based on the provided feature engineering strategy
def update(df):
    t = 100
    
    # Map accident reports
    df['accident'] = df['accident'].map({
        'None reported': 'not_reported',
        'At least 1 accident or damage reported': 'reported'
    })
    
    # Clean transmission column
    df['transmission'] = df['transmission'].str.replace('/', '').str.replace('-', '')
    df['transmission'] = df['transmission'].str.replace(' ', '_')
    
    # Columns to be processed
    cat_c = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    re_ = ['model', 'engine', 'transmission', 'ext_col', 'int_col']
    
    # Replace rare categories with "noise"
    for col in re_:
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < t, col] = "noise"
        
    # Fill missing values and convert to category
    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')
        
    return df

# Apply the function to the train dataset
train = update(train)

# List of luxury brands
luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                 'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                 'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']

# Current year for calculating vehicle age
current_year = datetime.datetime.now().year

# Add new features
train['Vehicle_Age'] = current_year - train['model_year']
train['Mileage_per_Year'] = train['milage'] / train['Vehicle_Age']

# Functions to extract Horsepower and Engine Size
def extract_horsepower(engine):
    try:
        return float(engine.split('HP')[0])
    except:
        return None

def extract_engine_size(engine):
    try:
        return float(engine.split(' ')[1].replace('L', ''))
    except:
        return None

# Apply the functions to extract horsepower and engine size
train['Horsepower'] = train['engine'].apply(extract_horsepower)
train['Engine_Size'] = train['engine'].apply(extract_engine_size)

# Calculate Power to Weight Ratio
train['Power_to_Weight_Ratio'] = train['Horsepower'] / train['Engine_Size']

# Check the head of the updated dataframe
train.head()


# Visualizing the distribution and relationship of the new features

plt.figure(figsize=(14, 10))

# Boxplot for Vehicle Age vs Price
plt.subplot(2, 2, 1)
sns.boxplot(x='Vehicle_Age', y='price', data=train)
plt.title('Vehicle Age vs Price')

# Scatter plot for Mileage per Year vs Price
plt.subplot(2, 2, 2)
sns.scatterplot(x='Mileage_per_Year', y='price', data=train)
plt.title('Mileage per Year vs Price')

# Scatter plot for Horsepower vs Price
plt.subplot(2, 2, 3)
sns.scatterplot(x='Horsepower', y='price', data=train)
plt.title('Horsepower vs Price')

# Scatter plot for Power to Weight Ratio vs Price
plt.subplot(2, 2, 4)
sns.scatterplot(x='Power_to_Weight_Ratio', y='price', data=train)
plt.title('Power to Weight Ratio vs Price')

plt.tight_layout()
plt.show()
