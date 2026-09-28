import sys
from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_COLUMNS
from feature_extractor import extract_features
from static_rules import analyze_code


MODEL_PATH = Path("../models/knn.pkl")


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def get_ml_prediction(code, model):

    features = extract_features(code)

    feature_values = [
        features.get(
            feature,
            0
        )
        for feature in FEATURE_COLUMNS
    ]

    X = pd.DataFrame(
        [feature_values],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(X)[0]

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(
            X
        )[0][1]

    else:

        probability = float(
            prediction
        )

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }


def calculate_risk(
    ml_probability,
    static_issues
):

    if not static_issues:

        if ml_probability >= 0.75:
            return "High"

        if ml_probability >= 0.50:
            return "Medium"

        return "Low"

    critical_count = sum(
        issue["severity"] == "Critical"
        for issue in static_issues
    )

    high_count = sum(
        issue["severity"] == "High"
        for issue in static_issues
    )

    if critical_count > 0:
        return "Critical"

    if high_count >= 2:
        return "High"

    if high_count == 1:
        return "High"

    if ml_probability >= 0.75:
        return "High"

    if ml_probability >= 0.50:
        return "Medium"

    return "Low"


def analyze(code):

    model = load_model()

    ml_result = get_ml_prediction(
        code,
        model
    )

    static_issues = analyze_code(
        code
    )

    risk = calculate_risk(
        ml_result["probability"],
        static_issues
    )

    return {
        "ml_prediction": ml_result,
        "static_issues": static_issues,
        "overall_risk": risk
    }


def print_report(result):

    print("\n" + "=" * 60)
    print("BUGLENS ANALYSIS")
    print("=" * 60)

    ml_result = result[
        "ml_prediction"
    ]

    probability = (
        ml_result["probability"] * 100
    )

    print(
        f"\nML Defect Probability: "
        f"{probability:.2f}%"
    )

    print(
        f"ML Classification: "
        f"{'Potentially Buggy' if ml_result['prediction'] else 'Likely Clean'}"
    )

    print(
        f"\nOverall Risk: "
        f"{result['overall_risk']}"
    )

    issues = result[
        "static_issues"
    ]

    print(
        f"\nStatic Issues Found: "
        f"{len(issues)}"
    )

    for index, issue in enumerate(
        issues,
        start=1
    ):

        print(
            f"\n--- Issue {index} ---"
        )

        print(
            f"Bug Type: "
            f"{issue['bug_type']}"
        )

        print(
            f"Line: "
            f"{issue['line']}"
        )

        print(
            f"Severity: "
            f"{issue['severity']}"
        )

        print(
            f"Confidence: "
            f"{issue['confidence'] * 100:.1f}%"
        )

        print(
            f"Explanation: "
            f"{issue['explanation']}"
        )

        print(
            f"Suggested Fix: "
            f"{issue['suggested_fix']}"
        )


def main():

    if len(sys.argv) > 1:

        file_path = Path(
            sys.argv[1]
        )

        if not file_path.exists():

            print(
                f"File not found: {file_path}"
            )

            return

        code = file_path.read_text(
            encoding="utf-8"
        )

    else:

        code = """
#include <iostream>
#include <cstring>

int main()
{
    int arr[10];

    for(int i = 0; i <= 10; i++)
    {
        arr[i] = i;
    }

    int x = 10 / 0;

    char buffer[10];
    strcpy(buffer, "This string is too long");

    int *ptr = nullptr;
    int value = *ptr;

    int *data = new int[100];

    return 0;
}
"""

    result = analyze(
        code
    )

    print_report(
        result
    )


if __name__ == "__main__":
    main()