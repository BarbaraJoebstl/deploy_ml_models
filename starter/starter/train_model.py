# Script to train machine learning model.

# necessary imports for the starter code.
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ml.data import process_data
import joblib

# Add code to load in the data.
file_path = "starter/data/census_cleaned.csv"
data = pd.read_csv(file_path)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# use only split and test
# train, test = train_test_split(data, test_size=0.20)
# X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# use K-fold cross validation instead of a train-test split.
# use k-fold and save best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_acc = 0
best_model = None
best_encoder = None
best_lb = None

fold = 1
for train_index, test_index in kf.split(data):
    print(f"Fold {fold}")
    train = data.iloc[train_index]
    test = data.iloc[test_index]

    # Preprocess
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Track best
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_encoder = encoder
        best_lb = lb

    fold += 1

# Save the best model & encoders
joblib.dump(best_model, "starter/model/model.joblib")
joblib.dump(best_encoder, "starter/model/encoder.joblib")
joblib.dump(best_lb, "starter/model/label_binarizer.joblib")
print(f"✅ Best model saved (Accuracy: {best_acc:.4f})")
