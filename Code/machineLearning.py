import pandas
import numpy

dataset = pandas.read_csv("dataset/alb-rainfall-adm2-full.csv")
print(dataset.head())
# print(dataset.info())

categorical_columns = [column for column in dataset.columns if dataset[column].dtype=='object']
# print("Categorical columns: ",categorical_columns)

numerical_columns = [column for column in dataset.columns if dataset[column].dtype!=object]
# print("Numerical columns: ",numerical_columns)

print("Unique values: ",dataset[categorical_columns].nunique)

numerical_columns = ["n_pixels", "rfh", "rfh_avg", "r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q", "r3q"]
for column in numerical_columns:
    dataset[column] = pandas.to_numeric(dataset[column],errors='coerce')

# print(dataset.dtypes)
for column in numerical_columns:
    print(f"Unique values in {column}: ", dataset[column].unique())

dataset.replace({'?': None, 'missing': None, 'N/A': None}, inplace=True)


numerical_columns = [col for col in dataset.columns if dataset[col].dtype in ['int64', 'float64']]
print("Numerical columns: ", numerical_columns)

# print(dataset.isnull().sum())

dataset[['n_pixels', 'rfh', 'rfh_avg', 'rfq']] = dataset[['n_pixels', 'rfh', 'rfh_avg', 'rfq']].fillna(dataset[['n_pixels', 'rfh', 'rfh_avg', 'rfq']].median())
dataset[['r1h', 'r1h_avg', 'r1q']] = dataset[['r1h', 'r1h_avg', 'r1q']].fillna(dataset[['r1h', 'r1h_avg', 'r1q']].median())
dataset[['r3h', 'r3h_avg', 'r3q']] = dataset[['r3h', 'r3h_avg', 'r3q']].fillna(dataset[['r3h', 'r3h_avg', 'r3q']].median())

