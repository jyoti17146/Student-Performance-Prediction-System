
import streamlit as st
import pandas as pd

from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    get_dataset_info
)

from src.model_training import (
    train_model,
    predict_student
)

from src.visualization import (
    score_distribution_chart,
    study_hours_chart,
    attendance_chart,
    correlation_heatmap
)


st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="📚",
    layout="wide"
)


st.title("Student Performance Prediction System")

st.write("""
This project analyzes student academic data
and predicts student final scores using
Machine Learning.
""")

st.header("User Instructions")

st.write("""
1. Upload a valid student CSV dataset.
2. Preview and inspect the dataset.
3. Clean the dataset.
4. Explore student performance using charts.
5. Train the Linear Regression model.
6. Enter student details to predict the final score.
""")


st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your student dataset (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        original_df = load_dataset(uploaded_file)

    except ValueError as error:
        st.error(str(error))
        st.stop()

    except Exception as error:
        st.error(f"Dataset error: {error}")
        st.stop()

    st.success("Dataset uploaded successfully.")

    st.subheader("Dataset Preview")

    st.dataframe(
        original_df.head(),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info = get_dataset_info(original_df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Rows",
        original_df.shape[0]
    )

    col2.metric(
        "Total Columns",
        original_df.shape[1]
    )

    col3.metric(
        "Duplicate Rows",
        int(info["duplicates"])
    )

    st.write("Column Data Types")

    dtype_df = pd.DataFrame({
        "Column": original_df.columns,
        "Data Type": original_df.dtypes.astype(str).values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.write("Missing Values")

    missing_df = pd.DataFrame({
        "Column": original_df.columns,
        "Missing Values": original_df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.write("Statistical Description")

    st.dataframe(
        original_df.describe().T,
        use_container_width=True
    )

    st.header("2. Data Cleaning")

    try:
        cleaned_df, cleaning_info = clean_dataset(
            original_df
        )

    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.success("Dataset cleaning completed.")

    st.write("Cleaning Summary")

    cleaning_summary = pd.DataFrame({
        "Description": [
            "Rows before cleaning",
            "Rows after cleaning",
            "Duplicate rows removed",
            "Rows removed due to missing target",
            "Total rows removed"
        ],
        "Count": [
            cleaning_info["rows_before"],
            cleaning_info["rows_after"],
            cleaning_info["duplicates_removed"],
            cleaning_info["target_rows_removed"],
            cleaning_info["rows_removed_total"]
        ]
    })

    st.dataframe(
        cleaning_summary,
        use_container_width=True
    )

    st.info(
        "The original uploaded dataset is not modified. "
        "Cleaning is performed on a copy."
    )

    st.subheader("Cleaned Dataset Preview")

    st.dataframe(
        cleaned_df.head(),
        use_container_width=True
    )

    st.write("Missing Values After Cleaning")

    cleaned_missing_df = pd.DataFrame({
        "Column": cleaned_df.columns,
        "Missing Values": cleaned_df.isnull().sum().values
    })

    st.dataframe(
        cleaned_missing_df,
        use_container_width=True
    )

    # Clear model if the cleaned dataset has changed
    if "trained_dataset" in st.session_state:

        previous_dataset = st.session_state[
            "trained_dataset"
        ]

        if not cleaned_df.equals(previous_dataset):

            st.session_state.pop(
                "trained_model",
                None
            )

            st.session_state.pop(
                "trained_metrics",
                None
            )

            st.session_state.pop(
                "trained_dataset",
                None
            )

            st.warning(
                "The dataset has changed. "
                "Please train the model again."
            )

    st.header("3. Exploratory Data Analysis")

    st.write(
        "The following charts help explore "
        "student academic performance."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader("Final Score Distribution")

        fig1 = score_distribution_chart(
            cleaned_df
        )

        st.pyplot(
            fig1,
            clear_figure=True
        )

        st.write(
            "This chart shows how final scores "
            "are distributed among students."
        )

    with chart_col2:

        st.subheader("Study Hours vs Final Score")

        fig2 = study_hours_chart(
            cleaned_df
        )

        st.pyplot(
            fig2,
            clear_figure=True
        )

        st.write(
            "This chart helps observe the relationship "
            "between study hours and final scores."
        )

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:

        st.subheader("Attendance Distribution")

        fig3 = attendance_chart(
            cleaned_df
        )

        st.pyplot(
            fig3,
            clear_figure=True
        )

        st.write(
            "This chart shows the attendance "
            "percentage distribution."
        )

    with chart_col4:

        st.subheader("Correlation Heatmap")

        fig4 = correlation_heatmap(
            cleaned_df
        )

        st.pyplot(
            fig4,
            clear_figure=True
        )

        st.write(
            "The heatmap shows correlations "
            "between the numerical columns."
        )

    st.header("4. Model Training")

    st.write("""
    The Linear Regression model uses these features:

    - Study Hours
    - Attendance Percentage
    - Previous Score
    - Practice Tests

    The target variable is Final Score.
    """)

    if st.button("Train Model"):

        try:
            (
                model,
                metrics,
                y_test,
                predictions
            ) = train_model(cleaned_df)

            st.session_state["trained_model"] = model

            st.session_state["trained_metrics"] = metrics

            st.session_state["trained_dataset"] = (
                cleaned_df.copy()
            )

            st.success(
                "Model training completed successfully."
            )

        except Exception as error:
            st.error(
                f"Model training failed: {error}"
            )

    if "trained_model" in st.session_state:

        st.subheader("Model Evaluation")

        metrics = st.session_state[
            "trained_metrics"
        ]

        metric_col1, metric_col2, metric_col3 = (
            st.columns(3)
        )

        metric_col1.metric(
            "MAE",
            f"{metrics['MAE']:.2f}"
        )

        metric_col2.metric(
            "RMSE",
            f"{metrics['RMSE']:.2f}"
        )

        metric_col3.metric(
            "R² Score",
            f"{metrics['R2 Score']:.2f}"
        )

        st.write("""
        MAE and RMSE measure prediction errors.
        A lower value generally indicates smaller errors.

        R² Score describes how well the model
        explains variation in the target score.
        """)

        st.header("5. Predict Student Performance")

        st.write(
            "Enter the student details below "
            "to estimate the final score."
        )

        with st.form("prediction_form"):

            study_hours = st.number_input(
                "Study Hours per Day",
                min_value=0.0,
                max_value=24.0,
                value=2.0,
                step=0.5
            )

            attendance_pct = st.number_input(
                "Attendance Percentage",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

            previous_score = st.number_input(
                "Previous Score",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            )

            practice_tests = st.number_input(
                "Practice Tests",
                min_value=0,
                max_value=1000,
                value=5,
                step=1
            )

            predict_button = st.form_submit_button(
                "Predict Final Score"
            )

        if predict_button:

            try:
                prediction = predict_student(
                    st.session_state["trained_model"],
                    study_hours,
                    attendance_pct,
                    previous_score,
                    practice_tests
                )

                # Keep predicted score within 0 to 100
                prediction = max(
                    0.0,
                    min(100.0, float(prediction))
                )

                st.success(
                    f"Estimated Final Score: "
                    f"{prediction:.2f} / 100"
                )

                st.warning(
                    "This is a machine learning estimate, "
                    "not a guaranteed actual exam score."
                )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )

    else:

        st.info(
            "Please click 'Train Model' "
            "before making predictions."
        )

else:

    st.info(
        "Please upload a CSV dataset to begin."
    )


st.header("6. Project Limitations")

st.write("""
- Predictions depend on the uploaded dataset.
- Linear Regression may not capture every
  factor affecting student performance.
- Predicted scores are estimates.
- Uploaded files and trained models in
  Streamlit session memory are not permanent.
""")
