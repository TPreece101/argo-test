import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'boto3'])

import io
import os
import boto3
import pandas as pd
import pickle
from toolkit import find_numeric_cat_cols
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix

# Get bucket from environment
s3_bucket = os.getenv("S3BUCKET")

s3 = boto3.client('s3')
obj = s3.get_object(Bucket=s3_bucket, Key='bank-full.csv')
data = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=';')

# Print age statistics
print("Age statistics")
print(data['age'].describe())

# Check for NAs
print("Checking for NAs")
print(data.isnull().sum())

# Split into X and y
X = data.drop(['y'], axis = 1)
y = data['y'].values

# Find column types
num_cols, cat_cols = find_numeric_cat_cols(X)

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, num_cols),
    ('categrorical', categorical_transformer, cat_cols)
], sparse_threshold=0)

estimator = Pipeline([('preprocessor', preprocessor),
                    ('classifier', GradientBoostingClassifier(n_estimators=500,
                                                            validation_fraction=0.2,
                                                            n_iter_no_change=5, tol=0.01,
                                                            random_state=0,
                                                            verbose=1))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)

params = {
    'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'classifier__max_depth': [1,3,5,7,9]
}

grid_search = GridSearchCV(estimator,
                           param_grid = params,
                           scoring = 'roc_auc',
                           cv = 5)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_, grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict_proba(X_test)

auc = roc_auc_score(y_true=y_test, y_score=y_pred[:,1])
print(auc)

print("Test set AUC: ", roc_auc_score(y_true=y_test, y_score=y_pred[:,1]))

# Dump best model to S3
with open('model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

# Copy to S3
s3 = boto3.resource('s3')
s3.Object(s3_bucket, 'model.pkl').put(Body=open('model.pkl', 'rb'))

print("Done")