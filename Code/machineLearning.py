import pandas

import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



dataset = pandas.read_csv("dataset/alb-rainfall-adm2-full.csv")

# print(dataset.head())

print(dataset.isnull().sum());

# print(dataset.info())

categorical_columns = [column for column in dataset.columns if dataset[column].dtype=='object']
# # print("Categorical columns: ",categorical_columns)

numerical_columns = [column for column in dataset.columns if dataset[column].dtype!=object]
# # print("Numerical columns: ",numerical_columns)

# print("Unique values: ",dataset[categorical_columns].nunique)

numerical_columns = ["n_pixels", "rfh", "rfh_avg", "r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q", "r3q"]
for column in numerical_columns:
    dataset[column] = pandas.to_numeric(dataset[column],errors='coerce')

# print(dataset.dtypes)
# for column in numerical_columns:
#     print(f"Unique values in {column}: ", dataset[column].unique())

dataset.replace({'?': None, 'missing': None, 'N/A': None}, inplace=True)

numerical_columns = [col for col in dataset.columns if dataset[col].dtype in ['int64', 'float64']]
# print("Numerical columns: ", numerical_columns)

# # print(dataset.isnull().sum())

dataset[['n_pixels', 'rfh', 'rfh_avg', 'rfq']] = dataset[['n_pixels', 'rfh', 'rfh_avg', 'rfq']].fillna(dataset[['n_pixels', 'rfh', 'rfh_avg', 'rfq']].median())
dataset[['r1h', 'r1h_avg', 'r1q']] = dataset[['r1h', 'r1h_avg', 'r1q']].fillna(dataset[['r1h', 'r1h_avg', 'r1q']].median())
dataset[['r3h', 'r3h_avg', 'r3q']] = dataset[['r3h', 'r3h_avg', 'r3q']].fillna(dataset[['r3h', 'r3h_avg', 'r3q']].median())

dataset['date'] = pandas.to_datetime(dataset['date'], errors='coerce', format='%Y-%m-%d')
print(dataset['date'].head())

print(dataset[dataset['date'].isna()])
dataset.drop(index=0, inplace=True)

# # print(dataset.isnull().sum())
dataset = pandas.get_dummies(dataset, columns=['version'], drop_first=True)
encoder = LabelEncoder()
dataset['ADM2_PCODE'] = encoder.fit_transform(dataset['ADM2_PCODE'])

# #outliers

z_score = dataset[numerical_columns].apply(zscore)
outliers = (z_score.abs()>3)
print("Number of outliers per column (Z-Score):\n", outliers.sum())

for column in numerical_columns:
    lower_limit = dataset[column].quantile(0.01)  
    upper_limit = dataset[column].quantile(0.99)  
    dataset[column] = dataset[column].clip(lower=lower_limit, upper=upper_limit)

# columns_to_transform = ["rfh", "rfh_avg", "r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q", "r3q"]
# dataset[columns_to_transform] = dataset[columns_to_transform].apply(lambda x: numpy.log1p(x))

scaler = StandardScaler()
dataset[numerical_columns] = scaler.fit_transform(numerical_columns)

dataset['year_month'] = dataset['date'].dt.to_period('M')  

aggregated_data = dataset.groupby(['adm2_id', 'ADM2_PCODE', 'year_month']).agg({
    'n_pixels': 'mean',
    'rfh': 'mean',
    'rfh_avg': 'mean',
    'r1h': 'mean',
    'r1h_avg': 'mean',
    'r3h': 'mean',
    'r3h_avg': 'mean',
    'rfq': 'mean',
    'r1q': 'mean',
    'r3q': 'mean'
}).reset_index()

#     Adding the second phase

X = aggregated_data.drop(columns=['rfq', 'adm2_id', 'ADM2_PCODE', 'year_month'])
y = aggregated_data['rfq']
# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])

