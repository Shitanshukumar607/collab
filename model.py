<<<<<<< HEAD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 1. Data Loading & Understanding
df = pd.read_csv("train.csv")

# 2. Data Cleaning
# convert age column datatype to integer
df["age"] = df["age"].astype(int)

# dropping ID & age_desc column
df = df.drop(columns=["ID", "age_desc"])

# define the mapping dictionary for country names
mapping = {
    "Viet Nam": "Vietnam",
    "AmericanSamoa": "United States",
    "Hong Kong": "China",
}
# replace value in the country column
df["contry_of_res"] = df["contry_of_res"].replace(mapping)

# handle missing values in ethnicity and relation column
df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
df["relation"] = df["relation"].replace(
    {
        "?": "Others",
        "Relative": "Others",
        "Parent": "Others",
        "Health care professional": "Others",
    }
)

# 3. Label Encoding
# identify columns with "object" data type
object_columns = df.select_dtypes(include=["object"]).columns
# initialize a dictionary to store the encoders
encoders = {}
# apply label encoding and store the encoders
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder  # saving the encoder for this column

# save the encoders as a pickle file
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)


# 4. Data Preprocessing - Handling outliers
# function to replace the outliers with median
def replace_outliers_with_median(df_local, column):
    Q1 = df_local[column].quantile(0.25)
    Q3 = df_local[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median = df_local[column].median()

    # replace outliers with median value
    df_local[column] = df_local[column].apply(
        lambda x: median if x < lower_bound or x > upper_bound else x
    )

    return df_local


# replace outliers in the "age" column
df = replace_outliers_with_median(df, "age")

# replace outliers in the "result" column
df = replace_outliers_with_median(df, "result")

# 5. Train Test Split
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. SMOTE (Synthetic Minority Oversampling technique)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 7. Model Training & Hyperparameter Tuning
# Initializing models
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
xgboost_classifier = XGBClassifier(random_state=42)

# Hyperparameter grids for RandomizedSearchCV
param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 50, 70],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


param_grid_rf = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}


param_grid_xgb = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
}

# perform RandomizedSearchCV for each model
random_search_dt = RandomizedSearchCV(
    estimator=decision_tree,
    param_distributions=param_grid_dt,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
)
random_search_rf = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=param_grid_rf,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
)
random_search_xgb = RandomizedSearchCV(
    estimator=xgboost_classifier,
    param_distributions=param_grid_xgb,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
)

# fit the models
print("Fitting Decision Tree...")
random_search_dt.fit(X_train_smote, y_train_smote)
print("Fitting Random Forest...")
random_search_rf.fit(X_train_smote, y_train_smote)
print("Fitting XGBoost...")
random_search_xgb.fit(X_train_smote, y_train_smote)

# Get the model with best score
best_model = None
best_score = 0

if random_search_dt.best_score_ > best_score:
    best_model = random_search_dt.best_estimator_
    best_score = random_search_dt.best_score_

if random_search_rf.best_score_ > best_score:
    best_model = random_search_rf.best_estimator_
    best_score = random_search_rf.best_score_

if random_search_xgb.best_score_ > best_score:
    best_model = random_search_xgb.best_estimator_
    best_score = random_search_xgb.best_score_

print(f"Best Model: {best_model}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}")

# save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model training complete and best model saved as 'best_model.pkl'.")
print("Encoders saved as 'encoders.pkl'.")

# evaluate on test data
y_test_pred = best_model.predict(X_test)
print("\n--- Test Set Evaluation ---")
print("Accuracy score:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
=======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 1. Data Loading & Understanding
df = pd.read_csv("train.csv")

# 2. Data Cleaning
# convert age column datatype to integer
df["age"] = df["age"].astype(int)

# dropping ID & age_desc column
df = df.drop(columns=["ID", "age_desc"])

# define the mapping dictionary for country names
mapping = {
    "Viet Nam": "Vietnam",
    "AmericanSamoa": "United States",
    "Hong Kong": "China",
}
# replace value in the country column
df["contry_of_res"] = df["contry_of_res"].replace(mapping)

# handle missing values in ethnicity and relation column
df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
df["relation"] = df["relation"].replace(
    {
        "?": "Others",
        "Relative": "Others",
        "Parent": "Others",
        "Health care professional": "Others",
    }
)

# 3. Label Encoding
# identify columns with "object" data type
object_columns = df.select_dtypes(include=["object"]).columns
# initialize a dictionary to store the encoders
encoders = {}
# apply label encoding and store the encoders
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder  # saving the encoder for this column

# save the encoders as a pickle file
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)


# 4. Data Preprocessing - Handling outliers
# function to replace the outliers with median
def replace_outliers_with_median(df_local, column):
    Q1 = df_local[column].quantile(0.25)
    Q3 = df_local[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median = df_local[column].median()

    # replace outliers with median value
    df_local[column] = df_local[column].apply(
        lambda x: median if x < lower_bound or x > upper_bound else x
    )

    return df_local


# replace outliers in the "age" column
df = replace_outliers_with_median(df, "age")

# replace outliers in the "result" column
df = replace_outliers_with_median(df, "result")

# 5. Train Test Split
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. SMOTE (Synthetic Minority Oversampling technique)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 7. Model Training & Hyperparameter Tuning
# Initializing models
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
xgboost_classifier = XGBClassifier(random_state=42)

# Hyperparameter grids for RandomizedSearchCV
param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 50, 70],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


param_grid_rf = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}


param_grid_xgb = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
}

# perform RandomizedSearchCV for each model
random_search_dt = RandomizedSearchCV(
    estimator=decision_tree,
    param_distributions=param_grid_dt,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
)
random_search_rf = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=param_grid_rf,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
)
random_search_xgb = RandomizedSearchCV(
    estimator=xgboost_classifier,
    param_distributions=param_grid_xgb,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
)

# fit the models
print("Fitting Decision Tree...")
random_search_dt.fit(X_train_smote, y_train_smote)
print("Fitting Random Forest...")
random_search_rf.fit(X_train_smote, y_train_smote)
print("Fitting XGBoost...")
random_search_xgb.fit(X_train_smote, y_train_smote)

# Get the model with best score
best_model = None
best_score = 0

if random_search_dt.best_score_ > best_score:
    best_model = random_search_dt.best_estimator_
    best_score = random_search_dt.best_score_

if random_search_rf.best_score_ > best_score:
    best_model = random_search_rf.best_estimator_
    best_score = random_search_rf.best_score_

if random_search_xgb.best_score_ > best_score:
    best_model = random_search_xgb.best_estimator_
    best_score = random_search_xgb.best_score_

print(f"Best Model: {best_model}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}")

# save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model training complete and best model saved as 'best_model.pkl'.")
print("Encoders saved as 'encoders.pkl'.")

# evaluate on test data
y_test_pred = best_model.predict(X_test)
print("\n--- Test Set Evaluation ---")
print("Accuracy score:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
>>>>>>> 09a790945a3326e21d38cf468a99216ce3b469fe
