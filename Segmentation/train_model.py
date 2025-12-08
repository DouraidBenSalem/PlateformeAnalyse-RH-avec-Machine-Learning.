import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# 1. Load Data
print("Loading data...")
data = pd.read_csv("data_FINAL_with_segments.csv", delimiter=';')

# 2. Define Features
# Note: 'KPIs_met >80%' was renamed to 'KPIs_met_gt_80' in the CSV
numerical_features = [
    'no_of_trainings', 'age', 'previous_year_rating',
    'length_of_service', 'KPIs_met_gt_80', 'awards_won?',
    'avg_training_score', 'performance_gap'
]

categorical_features = [
    'department', 'region', 'education', 'gender', 'recruitment_channel'
]

# 3. Create Preprocessing Pipeline
print("Creating pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='drop'
)

# 4. Fit Preprocessor
print("Fitting preprocessor...")
X = preprocessor.fit_transform(data)

# 5. Fit PCA
print("Fitting PCA...")
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X)

# 6. Fit KMeans
print("Fitting KMeans...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_pca)
labels = kmeans.labels_

# 7. Train Classifier (for Feature Importance)
from sklearn.ensemble import RandomForestClassifier
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, labels)

# 8. Save Models
print("Saving models...")
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(pca, 'pca.joblib')
joblib.dump(kmeans, 'kmeans.joblib')
joblib.dump(clf, 'classifier.joblib')

# Save Feature Names
def get_feature_names(column_transformer):
    output_features = []
    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder': continue
        if name == 'num':
            output_features.extend(features)
        elif name == 'cat':
            ohe = pipe.named_steps['encoder']
            output_features.extend(ohe.get_feature_names_out(features))
    return output_features

try:
    feature_names = get_feature_names(preprocessor)
    joblib.dump(feature_names, 'feature_names.joblib')
except Exception as e:
    print(f"Warning: Could not extract feature names: {e}")
joblib.dump(numerical_features, 'numerical_features.joblib')
joblib.dump(categorical_features, 'categorical_features.joblib')

print("Done!")
