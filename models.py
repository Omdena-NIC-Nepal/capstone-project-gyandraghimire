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


# --- Model Registry ---
CLASSIFIERS = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

REGRESSORS = {
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'linear_regression': LinearRegression(),
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.1)
}


# --- Classification Trainer ---
def train_classifier(X, y, model_type='random_forest', model_name='classifier', save_path='data/processed'):
    """
    Train and evaluate a classification model.
    """
    if model_type not in CLASSIFIERS:
        raise ValueError(f"Unsupported classifier: {model_type}. Choose from {list(CLASSIFIERS.keys())}")

    model = CLASSIFIERS[model_type]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    print(f"🔁 Training classifier: {model_type}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # Save model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = Path(save_path) / f"{model_name}_{model_type}.joblib"
    joblib.dump(model, model_path)

    print(f"✅ Saved model to: {model_path}")
    print(f"📈 Accuracy: {acc:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return {
        'model': model,
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': report,
        'confusion_matrix': matrix,
        'model_path': str(model_path)
    }


# --- Regression Trainer ---
def train_regressor(X, y, model_type='random_forest', model_name='regressor', save_path='data/processed'):
    """
    Train and evaluate a regression model.
    """
    if model_type not in REGRESSORS:
        raise ValueError(f"Unsupported regressor: {model_type}. Choose from {list(REGRESSORS.keys())}")

    model = REGRESSORS[model_type]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"🔁 Training regressor: {model_type}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    # Save model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = Path(save_path) / f"{model_name}_{model_type}.joblib"
    joblib.dump(model, model_path)

    print(f"✅ Saved model to: {model_path}")
    print(f"📉 RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f} | CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model_path': str(model_path)
    }
