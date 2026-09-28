import sys
from pathlib import Path

import joblib
import pandas as pd

from feature_extractor import extract_features
from config import FEATURE_COLUMNS


MODEL_PATH = Path("../models/knn.pkl")
DATASET_PATH = Path("../output/features.csv")


def main():

    print("=" * 60)
    print("BUGLENS ML VALIDATION")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)

    bad_samples = df[df["is_buggy"] == 1].head(5)
    good_samples = df[df["is_buggy"] == 0].head(5)

    samples = pd.concat(
        [bad_samples, good_samples]
    )

    correct = 0

    for index, row in samples.iterrows():

        file_path = Path(row["file"])

        try:
            code = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception as e:
            print(f"\nCould not read: {file_path}")
            print(e)
            continue

        features = extract_features(code)

        X = pd.DataFrame(
            [[features[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        classes = model.classes_

        buggy_index = list(classes).index(1)

        buggy_probability = probabilities[buggy_index]

        actual = int(row["is_buggy"])

        predicted_text = (
            "BUGGY"
            if prediction == 1
            else "CLEAN"
        )

        actual_text = (
            "BUGGY"
            if actual == 1
            else "CLEAN"
        )

        if prediction == actual:
            correct += 1

        print("\n" + "-" * 60)
        print("File:", file_path.name)
        print("Actual:", actual_text)
        print("Predicted:", predicted_text)
        print(
            f"Defect Probability: "
            f"{buggy_probability * 100:.2f}%"
        )

    total = len(samples)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Correct: {correct} / {total}")

    if total > 0:
        print(
            f"Accuracy: "
            f"{(correct / total) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()