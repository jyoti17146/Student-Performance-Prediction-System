


import matplotlib.pyplot as plt
import seaborn as sns


def score_distribution_chart(df):
    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="final_score",
        bins=10,
        kde=True,
        ax=ax
    )

    ax.set_title("Student Final Score Distribution")
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Number of Students")

    fig.tight_layout()

    return fig


def study_hours_chart(df):
    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="study_hours",
        y="final_score",
        ax=ax
    )

    ax.set_title("Study Hours vs Final Score")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Score")

    fig.tight_layout()

    return fig


def attendance_chart(df):
    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="attendance_pct",
        bins=10,
        ax=ax
    )

    ax.set_title("Attendance Percentage Distribution")
    ax.set_xlabel("Attendance Percentage")
    ax.set_ylabel("Number of Students")

    fig.tight_layout()

    return fig


def correlation_heatmap(df):
    columns = [
        "study_hours",
        "attendance_pct",
        "previous_score",
        "practice_tests",
        "final_score"
    ]

    correlation = df[columns].corr()

    fig, ax = plt.subplots()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Student Performance Correlation")

    fig.tight_layout()

    return fig
