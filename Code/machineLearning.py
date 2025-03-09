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