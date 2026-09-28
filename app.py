import streamlit as st
import pandas as pd
import joblib
import os

from pathlib import Path

from src.feature_extractor import extract_features
from src.static_rules import analyze_code
from src.config import FEATURE_COLUMNS


MODEL_PATH = Path("models/knn.pkl")
FEATURES_PATH = Path("output/features.csv")
MODEL_RESULTS_PATH = Path("output/model_results.csv")

st.set_page_config(
    page_title="BugLens",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Code editor */
    textarea {
        font-family: "JetBrains Mono", "Fira Code", "Consolas",
                     "Cascadia Code", monospace !important;
        font-size: 14px !important;
        line-height: 1.55 !important;
        tab-size: 4 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .buglens-brand {
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .buglens-tagline {
        font-size: 0.78rem;
        opacity: 0.65;
        margin-bottom: 1.2rem;
    }

    .nav-section {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
        margin: 0.8rem 0 0.45rem 0.25rem;
    }

    /* Navigation buttons */
    [data-testid="stSidebar"] button {
        border-radius: 9px !important;
        border: 1px solid transparent !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: all 0.15s ease;
    }

    [data-testid="stSidebar"] button:hover {
        border-color: rgba(128, 128, 128, 0.25) !important;
        background: rgba(128, 128, 128, 0.08) !important;
    }

    /* Main analyzer button gets stronger visual weight */
    [data-testid="stSidebar"] button[kind="primary"] {
        font-weight: 650 !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
    }

    .sidebar-note {
        font-size: 0.72rem;
        line-height: 1.45;
        opacity: 0.55;
        margin-top: 1rem;
    }

    /* Keep the main analyzer visually prominent */
    .analyzer-badge {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.18);
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def load_model_results():
    return pd.read_csv(MODEL_RESULTS_PATH)


def get_ml_analysis(code, model):

    features = extract_features(code)

    X = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    buggy_index = list(classes).index(1)

    risk_score = probabilities[buggy_index]

    if risk_score >= 0.75:
        classification = "High General Risk"
    elif risk_score >= 0.50:
        classification = "Medium General Risk"
    else:
        classification = "Low General Risk"

    return {
        "prediction": prediction,
        "risk_score": risk_score,
        "classification": classification,
        "features": features
    }


def calculate_overall_risk(issues, ml_risk):

    if any(
        issue["severity"].lower() == "critical"
        for issue in issues
    ):
        return "Critical"

    high_count = sum(
        1
        for issue in issues
        if issue["severity"].lower() == "high"
    )

    if high_count >= 1:
        return "High"

    if ml_risk >= 0.75:
        return "High"

    if ml_risk >= 0.50:
        return "Medium"

    return "Low"


def analyzer_page():

    st.markdown(
        '<div class="analyzer-badge">PRIMARY WORKSPACE</div>',
        unsafe_allow_html=True
    )

    st.title("🐞 BugLens")
    st.subheader("C/C++ Bug Prediction and Static Analysis")

    st.write(
        "Analyze C/C++ source code using machine learning "
        "and explainable static-analysis rules."
    )

    st.divider()

    try:
        model = load_model()
    except Exception as e:
        st.error("Could not load the trained KNN model.")
        st.exception(e)
        return

    st.subheader("🔍 Code Analyzer")

    input_method = st.radio(
        "Choose input method",
        ["Paste Code", "Upload File"],
        horizontal=True
    )

    code = ""

    if input_method == "Paste Code":

        code = st.text_area(
            "Paste your C/C++ code",
            height=350,
            placeholder="""#include <iostream>

int main()
{
    int arr[10];

    for(int i = 0; i <= 10; i++)
    {
        arr[i] = i;
    }

    return 0;
}"""
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload a C/C++ source file",
            type=["c", "cpp", "cc", "cxx", "h", "hpp"]
        )

        if uploaded_file is not None:

            code = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

            st.success(
                f"Loaded: {uploaded_file.name}"
            )

    if st.button(
        "🔎 Analyze Code",
        type="primary",
        use_container_width=True
    ):

        if not code.strip():

            st.warning(
                "Please enter or upload some C/C++ code first."
            )

            return

        with st.spinner("Analyzing code..."):

            ml_result = get_ml_analysis(
                code,
                model
            )

            static_issues = analyze_code(code)

            overall_risk = calculate_overall_risk(
                static_issues,
                ml_result["risk_score"]
            )

        st.session_state["analysis"] = {
            "code": code,
            "ml": ml_result,
            "issues": static_issues,
            "overall_risk": overall_risk
        }

    if "analysis" not in st.session_state:
        return

    analysis = st.session_state["analysis"]

    ml_result = analysis["ml"]
    static_issues = analysis["issues"]
    overall_risk = analysis["overall_risk"]

    st.divider()

    st.subheader("📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Risk",
            overall_risk
        )

    with col2:
        st.metric(
            "ML Risk Score",
            f"{ml_result['risk_score'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Static Issues",
            len(static_issues)
        )

    with col4:
        st.metric(
            "Code Lines",
            ml_result["features"]["code_lines"]
        )

    st.divider()

    st.subheader("🤖 ML Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "General Defect Risk",
            f"{ml_result['risk_score'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Classification",
            ml_result["classification"]
        )

    st.caption(
        "The ML score is a similarity-based risk estimate "
        "generated by the KNN model. It does not guarantee "
        "that the source code is bug-free or defective."
    )

    st.divider()

    st.subheader(
        f"🐞 Static Analysis ({len(static_issues)} issues)"
    )

    if not static_issues:

        st.success(
            "No potential issues were detected."
        )

    else:

        for index, issue in enumerate(
            static_issues,
            start=1
        ):

            severity = issue["severity"]

            if severity == "Critical":
                icon = "🔴"
            elif severity == "High":
                icon = "🟠"
            elif severity == "Medium":
                icon = "🟡"
            else:
                icon = "🔵"

            with st.expander(
                f"{icon} Issue {index}: "
                f"{issue['bug_type']} "
                f"(Line {issue['line']})"
            ):

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Severity:** {severity}"
                    )

                with col2:
                    st.write(
                        f"**Confidence:** "
                        f"{issue['confidence'] * 100:.0f}%"
                    )

                st.write(
                    f"**Explanation:** "
                    f"{issue['explanation']}"
                )

                st.info(
                    f"**Suggested Fix:** "
                    f"{issue['suggested_fix']}"
                )

    st.divider()

    st.subheader("📈 Code Metrics")

    features = ml_result["features"]

    metric_data = {
        "Metric": [
            "Total Lines",
            "Code Lines",
            "Functions",
            "If Statements",
            "For Loops",
            "While Loops",
            "Array Declarations",
            "Array Accesses",
            "Pointer Declarations",
            "Pointer Dereferences",
            "New Operations",
            "Delete Operations",
            "Malloc Calls",
            "Free Calls",
            "Cyclomatic Complexity",
            "Maximum Nesting Depth"
        ],
        "Value": [
            features["total_lines"],
            features["code_lines"],
            features["functions"],
            features["if_statements"],
            features["for_loops"],
            features["while_loops"],
            features["array_declarations"],
            features["array_accesses"],
            features["pointer_declarations"],
            features["pointer_dereferences"],
            features["new_operations"],
            features["delete_operations"],
            features["malloc_calls"],
            features["free_calls"],
            features["cyclomatic_complexity"],
            features["max_nesting_depth"]
        ]
    }

    st.dataframe(
        pd.DataFrame(metric_data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("💻 Source Code")

    st.code(
        analysis["code"],
        language="cpp"
    )


def ml_performance_page():

    st.title("📊 ML Performance")

    st.write(
        "Performance comparison of the machine-learning models "
        "trained on the Juliet C/C++ dataset."
    )

    try:
        results = load_model_results()
    except Exception as e:
        st.error("Could not load model results.")
        st.exception(e)
        return

    st.subheader("Model Comparison")

    display_results = results.copy()

    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:
        if column in display_results.columns:
            display_results[column] = (
                display_results[column] * 100
            ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("F1 Score Comparison")

    if "f1" in results.columns:

        chart_data = results[
            ["model", "f1"]
        ].set_index("model")

        st.bar_chart(chart_data)

    st.subheader("Selected Model")

    st.success(
        "BugLens uses KNN with k=3 as the selected model."
    )

    st.write(
        """
        The model was selected using grouped cross-validation
        and KNN hyperparameter tuning.

        Related Juliet good/bad variants are kept within the
        same validation group to reduce data leakage.
        """
    )

    st.subheader("KNN Cross-Validation Result")

    cv_data = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            70.20,
            60.35,
            86.26,
            71.01
        ]
    })

    st.dataframe(
        cv_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These values are the 5-fold grouped cross-validation "
        "results for KNN with k=3."
    )


def dataset_page():

    st.title("📁 Dataset Explorer")

    try:
        df = load_features()
    except Exception as e:
        st.error("Could not load features.csv.")
        st.exception(e)
        return

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Samples",
            len(df)
        )

    with col2:
        st.metric(
            "Buggy Samples",
            int((df["is_buggy"] == 1).sum())
        )

    with col3:
        st.metric(
            "Clean Samples",
            int((df["is_buggy"] == 0).sum())
        )

    with col4:
        st.metric(
            "Features",
            len(FEATURE_COLUMNS)
        )

    st.divider()

    st.subheader("Bug Distribution")

    bug_distribution = pd.DataFrame({
        "Class": ["Buggy", "Clean"],
        "Count": [
            int((df["is_buggy"] == 1).sum()),
            int((df["is_buggy"] == 0).sum())
        ]
    })

    st.bar_chart(
        bug_distribution.set_index("Class")
    )

    st.subheader("CWE Distribution")

    cwe_distribution = (
        df["cwe"]
        .value_counts()
        .head(20)
    )

    st.bar_chart(cwe_distribution)

    st.subheader("Language Distribution")

    language_distribution = (
        df["language"]
        .value_counts()
    )

    st.bar_chart(language_distribution)

    st.divider()

    st.subheader("Feature Statistics")

    numeric_features = df[FEATURE_COLUMNS]

    statistics = numeric_features.describe().T

    st.dataframe(
        statistics,
        use_container_width=True
    )


def visual_analytics_page():

    st.title("📈 Visual Analytics")

    st.write(
        "Explore the dataset, machine-learning performance, "
        "feature importance, and statistical relationships "
        "used by BugLens."
    )

    output_dir = Path("output")

    # ---------------------------------------------------------
    # MODEL PERFORMANCE
    # ---------------------------------------------------------

    st.header("🤖 Model Performance")

    try:

        results = load_model_results()

        display_results = results.copy()

        metric_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]

        for column in metric_columns:

            if column in display_results.columns:

                display_results[column] = (
                    display_results[column] * 100
                ).round(2)

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Model Metric Comparison")

        chart_data = results[
            ["model", "accuracy", "precision", "recall", "f1"]
        ].copy()

        chart_data = chart_data.set_index("model")

        chart_data.columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        chart_data = chart_data * 100

        st.bar_chart(chart_data)

    except Exception as e:

        st.error(
            "Could not load model performance data."
        )

        st.exception(e)

    st.divider()

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    st.header("🎯 Feature Importance")

    feature_importance_path = (
        output_dir / "feature_importance.png"
    )

    if feature_importance_path.exists():

        st.image(
            str(feature_importance_path),
            use_container_width=True
        )

        st.caption(
            "Feature importance generated from the Random Forest "
            "model. Higher values indicate greater contribution "
            "to the model's predictions."
        )

    else:

        st.warning(
            "Feature importance chart was not found."
        )

    st.divider()

    # ---------------------------------------------------------
    # CORRELATION HEATMAP
    # ---------------------------------------------------------

    st.header("🔥 Feature Correlation")

    correlation_path = (
        output_dir / "correlation_heatmap.png"
    )

    if correlation_path.exists():

        st.image(
            str(correlation_path),
            use_container_width=True
        )

        st.caption(
            "Correlation between numerical code features."
        )

    else:

        st.warning(
            "Correlation heatmap was not found."
        )

    st.divider()

    # ---------------------------------------------------------
    # CWE DISTRIBUTION
    # ---------------------------------------------------------

    st.header("🛡️ CWE Distribution")

    cwe_path = (
        output_dir / "cwe_distribution.png"
    )

    if cwe_path.exists():

        st.image(
            str(cwe_path),
            use_container_width=True
        )

    else:

        st.warning(
            "CWE distribution chart was not found."
        )

    st.divider()

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    st.header("🎯 Confusion Matrix")

    confusion_path = (
        output_dir / "random_forest_confusion_matrix.png"
    )

    if confusion_path.exists():

        st.image(
            str(confusion_path),
            width=650
        )

        st.caption(
            "Confusion matrix for the Random Forest model "
            "from the grouped evaluation."
        )

    else:

        st.warning(
            "Confusion matrix was not found."
        )

    st.divider()

    # ---------------------------------------------------------
    # DATASET STATISTICS
    # ---------------------------------------------------------

    st.header("📊 Dataset Statistics")

    try:

        df = load_features()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Samples",
                len(df)
            )

        with col2:

            st.metric(
                "Buggy Samples",
                int(
                    (df["is_buggy"] == 1).sum()
                )
            )

        with col3:

            st.metric(
                "Clean Samples",
                int(
                    (df["is_buggy"] == 0).sum()
                )
            )

        st.subheader("Buggy vs Clean")

        bug_counts = pd.DataFrame({
            "Class": [
                "Buggy",
                "Clean"
            ],
            "Count": [
                int(
                    (df["is_buggy"] == 1).sum()
                ),
                int(
                    (df["is_buggy"] == 0).sum()
                )
            ]
        })

        st.bar_chart(
            bug_counts.set_index("Class")
        )

    except Exception as e:

        st.error(
            "Could not load dataset statistics."
        )

        st.exception(e)


def about_page():

    st.title("ℹ️ About BugLens")

    st.markdown(
        """
        ## What is BugLens?

        BugLens is a hybrid C/C++ code-analysis system that combines
        **machine learning** and **static analysis** to identify
        potential software defects.

        ### Machine Learning

        BugLens uses a **K-Nearest Neighbors (KNN)** classifier
        trained using features extracted from the
        **NIST Juliet C/C++ dataset**.

        The selected configuration is:

        - Algorithm: KNN
        - k: 3
        - Feature scaling: StandardScaler
        - Validation: 5-fold GroupKFold

        ### Static Analysis

        Rule-based checks identify specific potential problems,
        including:

        - Division by zero
        - Array out-of-bounds
        - Unsafe functions
        - Null pointer dereferences
        - Memory leaks
        - Loop boundary problems

        ### Hybrid Architecture

        The ML model provides a general defect-risk estimate,
        while static analysis provides specific and explainable
        findings.

        The two systems are intentionally kept separate because
        a machine-learning prediction does not prove the presence
        or absence of a specific programming bug.

        ### Dataset

        BugLens uses the NIST Juliet Test Suite for C/C++ as the
        primary training and evaluation dataset.

        ### Project Goal

        The goal of BugLens is to provide a simple interface through
        which developers and students can understand potential
        problems in C/C++ source code and investigate the metrics
        associated with software quality.
        """
    )


def main():

    if "page" not in st.session_state:
        st.session_state["page"] = "Code Analyzer"

    st.sidebar.markdown(
        '<div class="buglens-brand">🐞 BugLens</div>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        '<div class="buglens-tagline">'
        'C/C++ Bug Prediction & Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        '<div class="nav-section">Main</div>',
        unsafe_allow_html=True
    )

    if st.sidebar.button(
        "🔍  Code Analyzer",
        key="nav_analyzer",
        type="primary" if st.session_state["page"] == "Code Analyzer" else "secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "Code Analyzer"
        st.rerun()

    st.sidebar.markdown(
        '<div class="nav-section">Explore</div>',
        unsafe_allow_html=True
    )

    if st.sidebar.button(
        "📊  ML Performance",
        key="nav_ml",
        type="secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "ML Performance"
        st.rerun()

    if st.sidebar.button(
        "📈  Visual Analytics",
        key="nav_visual",
        type="secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "Visual Analytics"
        st.rerun()

    if st.sidebar.button(
        "📁  Dataset Explorer",
        key="nav_dataset",
        type="secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "Dataset Explorer"
        st.rerun()

    st.sidebar.divider()

    if st.sidebar.button(
        "ℹ️  About",
        key="nav_about",
        type="secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "About"
        st.rerun()

    st.sidebar.markdown(
        '<div class="sidebar-note">'
        'BugLens combines machine learning with '
        'explainable static analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    page = st.session_state["page"]

    if page == "Code Analyzer":
        analyzer_page()

    elif page == "ML Performance":
        ml_performance_page()

    elif page == "Visual Analytics":
        visual_analytics_page()

    elif page == "Dataset Explorer":
        dataset_page()

    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()
