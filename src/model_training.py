
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


FEATURES = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "practice_tests"
]

TARGET = "final_score"


def train_model(df):

    # Separate Features and Target
    X = df[FEATURES]
    y = df[TARGET]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create safe preprocessing and training pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    return model, metrics, y_test, predictions


def predict_student(model, study_hours,
                    attendance_pct, previous_score,
                    practice_tests):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_pct": attendance_pct,
        "previous_score": previous_score,
        "practice_tests": practice_tests
    }])

    prediction = model.predict(input_data)

    return prediction[0]
