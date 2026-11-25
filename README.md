# Student Performance Prediction System Using Machine Learning

<p align="center">
  <strong>An Interactive Machine Learning Application for Academic Performance Analysis and Prediction</strong>
</p>

<p align="center">
  <a href="https://student-performance-prediction-system-nzsgeelumme9x3ofrzxrys.streamlit.app/">
    <strong>🚀 VIEW LIVE DEMO</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/jyoti17146/Student-Performance-Prediction-System">
    <strong>📂 VIEW SOURCE CODE</strong>
  </a>
</p>

---

## 1. Project Overview

The **Student Performance Prediction System** is a machine learning-based web application designed to analyze student academic data and estimate their final academic scores using multiple performance-related parameters.

The project applies data preprocessing, exploratory data analysis, statistical visualization, and supervised machine learning techniques to examine the relationship between students' academic habits and their final performance.

Developed using Python and Streamlit, the application provides an interactive interface through which users can upload student datasets, inspect data quality, perform data cleaning, visualize academic trends, train a Linear Regression model, and generate individual student performance predictions.

The primary objective is to demonstrate how machine learning can be applied to educational data analysis to identify performance patterns and provide data-driven academic estimates.

## 2. Project Objectives

The major objectives of this project are:

- To analyze student academic performance using structured datasets.
- To examine the relationship between study hours, attendance, previous academic scores, and practice test participation.
- To identify missing values, duplicate records, and invalid data entries.
- To perform data cleaning and preprocessing before model training.
- To visualize academic performance patterns through statistical charts.
- To develop a supervised machine learning model for predicting final student scores.
- To evaluate prediction performance using standard regression metrics.
- To provide an interactive web-based platform for dataset analysis and individual score estimation.

## 3. Key Features

### 3.1 Dataset Upload and Inspection
- Upload student datasets in CSV format.
- Preview uploaded records through an interactive table.
- Display dataset dimensions, including total rows and columns.
- Identify duplicate records and missing values.
- Examine column data types and statistical summaries.

### 3.2 Data Cleaning and Preprocessing
- Retain the required dataset attributes.
- Convert relevant columns into numerical formats.
- Handle missing and invalid numerical values.
- Identify and remove duplicate records.
- Validate academic parameters against defined ranges.
- Generate a cleaning summary showing the records before and after preprocessing.

The original uploaded dataset remains unchanged during the cleaning process.

### 3.3 Exploratory Data Analysis (EDA)

The application provides four visualizations:

- **Final Score Distribution:** Examines the distribution of students' final scores.
- **Study Hours vs Final Score:** Visualizes the relationship between daily study hours and final academic scores.
- **Attendance Distribution:** Displays the distribution of student attendance percentages.
- **Correlation Heatmap:** Illustrates correlations among the numerical academic attributes.

### 3.4 Machine Learning Model

The project implements a **Linear Regression** algorithm to estimate student final scores.

The model uses the following independent variables:

- Study Hours
- Attendance Percentage
- Previous Score
- Practice Tests

The target variable is:

- Final Score

The dataset is divided into training and testing subsets using an 80:20 split. A machine learning pipeline incorporating median imputation and Linear Regression is used for model training.

### 3.5 Model Evaluation

The trained model is evaluated using three standard regression metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Root Mean Squared Error (RMSE):** Measures prediction error while assigning greater weight to larger errors.
- **R² Score:** Indicates how well the model explains the variation in the target variable.

### 3.6 Individual Student Performance Prediction

After successful model training, users can enter individual student details, including study hours, attendance percentage, previous score, and practice test participation.

The application uses the trained model to estimate the student's final academic score on a scale of 0–100.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Git & GitHub | Version control and source code management |
| Streamlit Community Cloud | Application deployment and hosting |

## 5. Dataset Description

The application accepts structured student academic datasets in CSV format.

The following columns are required:

| Attribute | Description |
|---|---|
| `study_hours` | Number of hours spent studying per day |
| `attendance_pct` | Student attendance percentage |
| `previous_score` | Previous academic examination score |
| `practice_tests` | Number of practice tests attempted |
| `final_score` | Final academic score (prediction target) |

**Dataset requirements:**
- The uploaded file must be in CSV format.
- All five required column names must match exactly.
- Numerical attributes should contain valid numerical values.
- At least 10 valid records must remain after data cleaning for model training.

## 6. Project Architecture

The project follows a modular structure, separating the main application interface from data preprocessing, model training, and visualization functionalities.

```text
Student-Performance-Prediction-System/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── sample_dataset.csv
│
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    └── visualization.py
```

**Module Description:**

- `app.py`: Main Streamlit application responsible for the user interface and overall workflow.
- `data_preprocessing.py`: Handles dataset loading, validation, cleaning, and dataset inspection.
- `model_training.py`: Implements Linear Regression model training, evaluation, and prediction.
- `visualization.py`: Generates statistical charts and correlation heatmaps.
- `requirements.txt`: Contains the Python libraries required to execute the application.
- `sample_dataset.csv`: Provides a sample dataset for testing the application.

## 7. Installation and Setup

Follow the steps below to run the project locally.

### Prerequisites

- Python 3.9 or later
- pip package manager
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/jyoti17146/Student-Performance-Prediction-System.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Student-Performance-Prediction-System
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The application will automatically open in the default web browser.

## 8. Application Workflow

The application follows a structured sequence:

1. **Upload Dataset:** Import a student academic dataset in CSV format.
2. **Dataset Inspection:** Review records, column types, missing values, and statistical summaries.
3. **Data Preprocessing:** Clean and validate the uploaded dataset.
4. **Exploratory Data Analysis:** Generate charts to understand student performance patterns.
5. **Model Training:** Train the Linear Regression model using the cleaned dataset.
6. **Model Evaluation:** Review MAE, RMSE, and R² evaluation metrics.
7. **Performance Prediction:** Enter individual student details and obtain an estimated final score.

## 9. Live Application

The project is deployed using Streamlit Community Cloud and can be accessed through the following link:

**[🚀 Launch Student Performance Prediction System](https://student-performance-prediction-system-nzsgeelumme9x3ofrzxrys.streamlit.app/)**

Users can interact with the application directly through a web browser without installing Python or configuring a local development environment.

## 10. Project Limitations

- Prediction accuracy depends on the quality, size, and representativeness of the uploaded dataset.
- Linear Regression assumes a linear relationship between input variables and the target variable.
- The model may not capture every factor influencing student academic performance.
- Predictions represent statistical estimates and should not be treated as guaranteed examination results.
- Uploaded datasets and trained models are maintained within the active Streamlit session and are not permanently stored by the application.

## 11. Future Enhancements

Potential improvements to the project include:

- Implementation and comparison of additional regression algorithms.
- Hyperparameter tuning and model optimization.
- Advanced feature engineering and feature importance analysis.
- Integration of additional academic and behavioural parameters.
- Enhanced interactive dashboards and downloadable analytical reports.
- Persistent model storage and prediction history.
- Improved model generalization through larger and more diverse datasets.

## 12. Project Contributors

This project was collaboratively developed by the following team members:

1. **Jyoti Pandey**
2. **Riya Kumari**
3. **Shruti Kumari**
4. **Sneha Manna**
5. **Tatim Adak**

**Department:** Computer Science and Engineering  
**Degree:** Bachelor of Technology (B.Tech)

All contributors participated in the collaborative development of the Student Performance Prediction System, focusing on the application of machine learning, data analysis, preprocessing, and interactive web application development.
## 13. Acknowledgements

We acknowledge the open-source Python ecosystem and the developers of the libraries used in this project, particularly Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

Their tools and documentation have supported the development of this interactive machine learning application.

---

<p align="center">
  <strong>Student Performance Prediction System</strong><br>
  Machine Learning | Data Analysis | Academic Performance Prediction
</p>

<p align="center">
  <a href="https://student-performance-prediction-system-nzsgeelumme9x3ofrzxrys.streamlit.app/">
    Explore the Live Application
  </a>
</p>
