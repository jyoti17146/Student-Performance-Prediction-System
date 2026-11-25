
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "practice_tests",
    "final_score"
]


def load_dataset(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty.")

    except Exception as e:
        raise ValueError(f"Unable to read CSV file: {e}")

    if df.empty:
        raise ValueError("The dataset contains no records.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def clean_dataset(df):
    df = df.copy()

    rows_before = len(df)

    # Keep only required columns
    df = df[REQUIRED_COLUMNS].copy()

    # Convert required columns to numeric
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Replace infinite values with missing values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Validate study hours
    df.loc[
        (df["study_hours"] < 0)
        | (df["study_hours"] > 24),
        "study_hours"
    ] = np.nan

    # Validate attendance percentage
    df.loc[
        (df["attendance_pct"] < 0)
        | (df["attendance_pct"] > 100),
        "attendance_pct"
    ] = np.nan

    # Validate previous score
    df.loc[
        (df["previous_score"] < 0)
        | (df["previous_score"] > 100),
        "previous_score"
    ] = np.nan

    # Validate practice tests
    df.loc[
        df["practice_tests"] < 0,
        "practice_tests"
    ] = np.nan

    # Practice tests should be whole numbers
    invalid_practice_tests = (
        df["practice_tests"].notna()
        & (df["practice_tests"] % 1 != 0)
    )

    df.loc[
        invalid_practice_tests,
        "practice_tests"
    ] = np.nan

    # Validate target score
    df.loc[
        (df["final_score"] < 0)
        | (df["final_score"] > 100),
        "final_score"
    ] = np.nan

    # Remove duplicate rows
    duplicates_before = len(df)

    df = df.drop_duplicates()

    duplicates_removed = (
        duplicates_before - len(df)
    )

    # Remove rows where target value is missing
    target_rows_before = len(df)

    df = df.dropna(
        subset=["final_score"]
    )

    target_rows_removed = (
        target_rows_before - len(df)
    )

    rows_after = len(df)

    if rows_after < 10:
        raise ValueError(
            "At least 10 valid rows are required "
            "for model training."
        )

    cleaning_info = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "target_rows_removed": target_rows_removed,
        "rows_removed_total": rows_before - rows_after
    }

    return df, cleaning_info


def get_dataset_info(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str),
        "description": df.describe(),
        "missing_values": df.isnull().sum(),
        "duplicates": df.duplicated().sum()
    }
