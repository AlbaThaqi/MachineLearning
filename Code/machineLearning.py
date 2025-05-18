import os
os.environ["WANDB_MODE"] = "disabled"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

if not hasattr(np, "VisibleDeprecationWarning"):
    class VisibleDeprecationWarning(Warning):
        pass
    np.VisibleDeprecationWarning = VisibleDeprecationWarning

import sweetviz as sv
import wandb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import adjusted_rand_score

import sweetviz as sv

# Optional imports
try:
    from ydata_profiling import ProfileReport
    ENABLE_PROFILING = True
except ImportError:
    ENABLE_PROFILING = False

try:
    import mlflow
    import mlflow.sklearn
    ENABLE_MLFLOW = True
except ImportError:
    ENABLE_MLFLOW = False

try:
    import wandb
    ENABLE_WANDB = True
except ImportError:
    ENABLE_WANDB = False

try:
    from joblib import dump
    ENABLE_SAVE_MODEL = True
except ImportError:
    ENABLE_SAVE_MODEL = False


# Load dataset
dataset = pd.read_csv("C:/Users/lenovo/Documents/GitHub/MachineLearning/dataset/alb-rainfall-adm2-full.csv")

# Remove header row if accidentally included as data
if dataset.iloc[0].astype(str).str.contains("#").any():
    dataset = dataset.drop(index=0).reset_index(drop=True)

# Replace known missing value indicators with NaN
dataset.replace({'?': np.nan, 'missing': np.nan, 'N/A': np.nan}, inplace=True)

# Convert numeric columns and fill missing with median
numerical_columns = ["n_pixels", "rfh", "rfh_avg", "r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q", "r3q"]
dataset[numerical_columns] = dataset[numerical_columns].apply(pd.to_numeric, errors='coerce')
dataset[numerical_columns] = dataset[numerical_columns].fillna(dataset[numerical_columns].median())

# Convert date and drop invalid dates
dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')
dataset.dropna(subset=['date'], inplace=True)

# One-hot encode 'version' column
dataset = pd.get_dummies(dataset, columns=['version'], drop_first=True)

# Label encode 'ADM2_PCODE'
dataset['ADM2_PCODE'] = LabelEncoder().fit_transform(dataset['ADM2_PCODE'])

# Clip numeric outliers between 1st and 99th percentile
for col in numerical_columns:
    low, high = dataset[col].quantile(0.01), dataset[col].quantile(0.99)
    dataset[col] = dataset[col].clip(lower=low, upper=high)

# Log transform skewed columns
log_columns = ["r1h", "r3h", "r1q", "r3q"]
dataset[log_columns] = dataset[log_columns].apply(lambda x: np.log1p(x))

# Scale numerical features
scaler = StandardScaler()
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

# Create binary rain label based on median of 'rfh'
dataset['rain_label'] = (dataset['rfh'] > dataset['rfh'].median()).astype(int)

# Prepare features and target
X = dataset[numerical_columns].copy().drop(columns=['rfh'])
y = dataset['rain_label']

# Split data with stratification to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# === Profiling Reports ===
if ENABLE_PROFILING:
    profile = ProfileReport(dataset, title="Rainfall Profiling Report", explorative=True)
    profile.to_file("rainfall_report.html")
    print("Saved ydata_profiling report as rainfall_report.html")

report = sv.compare([X_train, "Train"], [X_test, "Test"])
report.show_html(filepath="sweetviz_report.html")
print("Saved Sweetviz report as sweetviz_report.html")

# === Modeling ===
rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

dt = DecisionTreeClassifier(max_depth=5, random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

def print_metrics(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='weighted'), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='weighted'), 4))
    print("F1 Score :", round(f1_score(y_true, y_pred, average='weighted'), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

print_metrics("Decision Tree", y_test, y_pred_dt)
print_metrics("Random Forest", y_test, y_pred_rf)

# === MLflow logging ===
if ENABLE_MLFLOW:
     with mlflow.start_run():
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_rf))
        # Pass a small input example for model signature inference
        input_example = X_train.iloc[:5]  # pick first 5 rows as example input
        mlflow.sklearn.log_model(rf, "random_forest_model", input_example=input_example)
# === wandb logging ===
if ENABLE_WANDB:
    wandb.init(project="rainfall-ml", name="RandomForest", reinit=True)
    wandb.config.update({"max_depth": 5, "n_estimators": 100})
    wandb.log({"accuracy": accuracy_score(y_test, y_pred_rf)})
    wandb.finish()

# === Grid Search for Random Forest ===
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
    cv=5,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
best_rf_pred = best_rf.predict(X_test)
print_metrics("Optimized Random Forest", y_test, best_rf_pred)

# === Save best model ===
if ENABLE_SAVE_MODEL:
    dump(best_rf, "best_rf_model.joblib")
    print("Saved best_rf_model.joblib")


# === Clustering and Visualization ===
X_sample = X.sample(n=3000, random_state=42)
y_sample = y.loc[X_sample.index]

agg = AgglomerativeClustering(n_clusters=2).fit_predict(X_sample)
spectral = SpectralClustering(n_clusters=2, assign_labels='kmeans', affinity='nearest_neighbors').fit_predict(X_sample)

print("\nARI Agglomerative:", adjusted_rand_score(y_sample, agg))
print("ARI Spectral     :", adjusted_rand_score(y_sample, spectral))

X_pca = PCA(n_components=2).fit_transform(X_sample)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agg, palette='Set1')
plt.title("Agglomerative Clustering")
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=spectral, palette='Set2')
plt.title("Spectral Clustering")
plt.tight_layout()
plt.show()


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
# === Optional Tool: YData Profiling ===
if ENABLE_PROFILING:
    profile = ProfileReport(dataset, title="Rainfall Profiling Report", explorative=True)
    profile.to_file("/mnt/data/rainfall_report.html")

# === Optional Tool: Sweetviz ===
if ENABLE_SWEETVIZ:
    report = sv.compare([X_train, "Train"], [X_test, "Test"])
    report.show_html(filepath="/mnt/data/sweetviz_report.html")


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

# === Optional Tool: MLflow ===
if ENABLE_MLFLOW:
    with mlflow.start_run():
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_rf))
        mlflow.sklearn.log_model(rf, "random_forest_model")

# === Optional Tool: Weights & Biases ===
if ENABLE_WANDB:
    wandb.init(project="rainfall-ml", name="RandomForest", reinit=True)
    wandb.config.update({"max_depth": 5, "n_estimators": 100})
    wandb.log({"accuracy": accuracy_score(y_test, y_pred_rf)})

# === Grid Search for Optimization ===
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
    cv=5
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
best_rf_pred = best_rf.predict(X_test)
print_metrics("Optimized Random Forest", y_test, best_rf_pred)

# === Optional Tool: Save model ===
if ENABLE_SAVE_MODEL:
    dump(best_rf, "/mnt/data/best_rf_model.joblib")
    

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

# Third phase

#Analiza dhe evaluimi

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), rf_param_grid, cv=5)
rf_grid.fit(X_train, y_train)

print("\n[Random Forest] Best Params:", rf_grid.best_params_)
print("Best CV Score RF:", round(rf_grid.best_score_, 4))

best_rf = rf_grid.best_estimator_
best_rf_pred = best_rf.predict(X_test)
print_metrics("Random Forest (Optimized)", y_test, best_rf_pred, plot=True)


columns_to_log = ["r1h", "r3h", "r1q", "r3q"]
dataset[columns_to_log] = dataset[columns_to_log].apply(lambda x: np.log1p(x))

dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

dataset['rain_label'] = (dataset['rfh'] > dataset['rfh'].median()).astype(int)
X = dataset[numerical_columns].drop(columns=['rfh'])
y = dataset['rain_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

models = ["Original RF", "Optimized RF"]
accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, best_rf_pred)]
f1_scores = [f1_score(y_test, y_pred_rf, average='weighted'), f1_score(y_test, best_rf_pred, average='weighted')]

compare_df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies,
    "F1 Score": f1_scores
})

compare_df.set_index("Model").plot(kind='bar', figsize=(8, 5), colormap='Set3')
plt.title("Krahasim i Modeleve: RF Origjinal vs RF Optimized")
plt.ylabel("Rezultati")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()




