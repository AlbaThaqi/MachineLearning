import pandas as pd
import numpy  as np

from sklearn.preprocessing import LabelEncoder,StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Create target (example): classify days with high rainfall
dataset['rain_label'] = (dataset['rfh'] > dataset['rfh'].median()).astype(int)

# Check class distribution
print("\nRain label distribution:")
print(dataset['rain_label'].value_counts())

# Features and labels (remove rfh because it's used to generate target)
X = dataset[numerical_columns].copy()
if 'rfh' in X.columns:
    X = X.drop(columns=['rfh'])
y = dataset['rain_label']

# Split for supervised
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---- Supervised Models ----
print("\n--- SUPERVISED MODELS ---")

# Decision Tree with limited depth to avoid overfitting
dt = DecisionTreeClassifier(max_depth=5, random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest with limited depth to avoid overfitting
rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Confusion matrix plot (one-time call only)
def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Metrics function
def print_metrics(name, y_true, y_pred, plot=False):
    print(f"\n{name} Results:")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    print("F1 Score :", round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    if plot:
        plot_confusion(name, y_true, y_pred)

print_metrics("Decision Tree", y_test, y_pred_dt)
print_metrics("Random Forest", y_test, y_pred_rf)

# Cross-validation scores
rf_cv_score = cross_val_score(rf, X, y, cv=5).mean()
dt_cv_score = cross_val_score(dt, X, y, cv=5).mean()
print("\nCross-validation Scores:")
print("Decision Tree CV Accuracy:", round(dt_cv_score, 4))

# ---- Unsupervised Models ----
print("UNSUPERVISED MODELS (with sampling) ---")

# Sample to speed up
X_sample = X.sample(n=3000, random_state=42)
y_sample = y.loc[X_sample.index]

# Agglomerative Clustering
start = time.time()
agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X_sample)
print_metrics("Agglomerative (sampled)", y_sample, agg_labels)
print("Agglomerative ARI:", round(adjusted_rand_score(y_sample, agg_labels), 4))
print("Agglomerative time:", round(time.time() - start, 2), "seconds")


# Spectral Clustering
start = time.time()
spectral = SpectralClustering(n_clusters=2, assign_labels='kmeans', random_state=0, affinity='nearest_neighbors')
spectral_labels = spectral.fit_predict(X_sample)
print_metrics("Spectral Clustering (sampled)", y_sample, spectral_labels)
print("Spectral Clustering ARI:", round(adjusted_rand_score(y_sample, spectral_labels), 4))
print("Spectral Clustering time:", round(time.time() - start, 2), "seconds")

# ---- Visualization ----

metrics = {
    "Modeli": ["Decision Tree", "Random Forest"],
    "Saktësia": [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)],
    "Precision": [precision_score(y_test, y_pred_dt, average='weighted'), precision_score(y_test, y_pred_rf, average='weighted')],
    "Recall": [recall_score(y_test, y_pred_dt, average='weighted'), recall_score(y_test, y_pred_rf, average='weighted')],
    "F1-Score": [f1_score(y_test, y_pred_dt, average='weighted'), f1_score(y_test, y_pred_rf, average='weighted')],
    "CV Accuracy": [dt_cv_score, rf_cv_score]
}

df_metrics = pd.DataFrame(metrics)

df_plot = df_metrics.set_index("Modeli").T

# Vizualizimi
df_plot.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Krahasimi i Performancës së Modeleve Mbikëqyrëse")
plt.ylabel("Vlera")
plt.xlabel("Metrika")
plt.legend(title="Modeli")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)


plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agg_labels, palette='Set2', s=40)
plt.title("Agglomerative Clustering me PCA")
plt.xlabel("Komponenti PCA 1")
plt.ylabel("Komponenti PCA 2")
plt.legend(title="Cluster")


plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=spectral_labels, palette='Set1', s=40)
plt.title("Spectral Clustering me PCA")
plt.xlabel("Komponenti PCA 1")
plt.ylabel("Komponenti PCA 2")
plt.legend(title="Cluster")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
labels = ['Pak reshje', 'Shumë reshje']
sizes = dataset['rain_label'].value_counts()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
plt.title("Shpërndarja e Etiketave për Reshje")
plt.axis('equal')
plt.tight_layout()
plt.show()

#Adding the third phase









