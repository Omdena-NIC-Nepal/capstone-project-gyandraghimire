# models.py 🧠 Core ML training & evaluation logic

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# --- Classification Model Trainer ---
def train_classifier(X, y, model_type='random_forest', model_name='model', save_path='data/processed'):
    model_map = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    model = model_map.get(model_type)
    if not model:
        raise ValueError(f"Unsupported classifier: {model_type}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_file = f"{save_path}/{model_name}_{model_type}.joblib"
    joblib.dump(model, model_file)

    return {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': np.mean(cv),
        'cv_std': np.std(cv),
        'classification_report': report,
        'confusion_matrix': confusion,
        'model_path': model_file
    }

# --- Regression Model Trainer ---
def train_regressor(X, y, model_type='random_forest', model_name='regressor', save_path='data/processed'):
    model_map = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1)
    }

    model = model_map.get(model_type)
    if not model:
        raise ValueError(f"Unsupported regressor: {model_type}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring='r2')

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_file = f"{save_path}/{model_name}_{model_type}.joblib"
    joblib.dump(model, model_file)

    return {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': np.mean(cv),
        'cv_std': np.std(cv),
        'model_path': model_file
    }
