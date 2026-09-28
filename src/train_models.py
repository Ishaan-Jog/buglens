import pandas as pd
import numpy as np
import re
from config import FEATURE_COLUMNS

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib
from pathlib import Path


INPUT_PATH = Path("../output/features.csv")
MODEL_DIR = Path("../models")
RESULTS_PATH = Path("../output/model_results.csv")


def load_data():

    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["is_buggy"]
    groups = df["file"].apply(get_group)

    return X, y, groups


def create_models():

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=3
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    SVC(
                        random_state=42
                    ),
                    ensemble=False
                )
            )
        ])
    }

    return models


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }


def cross_validate_models(X, y, groups):

    print("\n" + "=" * 60)
    print("5-FOLD GROUPED CROSS-VALIDATION")
    print("=" * 60)

    group_kfold = GroupKFold(
        n_splits=5
    )

    models = create_models()

    cv_results = []

    for name, model in models.items():

        print("\n" + "-" * 60)
        print("Model:", name)

        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(
            group_kfold.split(X, y, groups),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            fold_model = clone(model)

            fold_model.fit(
                X_train,
                y_train
            )

            evaluation = evaluate_model(
                fold_model,
                X_test,
                y_test
            )

            fold_scores.append({
                "accuracy": evaluation["accuracy"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"]
            })

            print(
                f"Fold {fold}: "
                f"Accuracy={evaluation['accuracy']:.4f}, "
                f"Precision={evaluation['precision']:.4f}, "
                f"Recall={evaluation['recall']:.4f}, "
                f"F1={evaluation['f1']:.4f}"
            )

        scores_df = pd.DataFrame(fold_scores)

        cv_results.append({
            "model": name,
            "accuracy_mean": scores_df["accuracy"].mean(),
            "accuracy_std": scores_df["accuracy"].std(),
            "precision_mean": scores_df["precision"].mean(),
            "precision_std": scores_df["precision"].std(),
            "recall_mean": scores_df["recall"].mean(),
            "recall_std": scores_df["recall"].std(),
            "f1_mean": scores_df["f1"].mean(),
            "f1_std": scores_df["f1"].std()
        })

        print(
            f"Average F1: "
            f"{scores_df['f1'].mean():.4f} "
            f"+/- {scores_df['f1'].std():.4f}"
        )

    results_df = pd.DataFrame(cv_results)

    results_df = results_df.sort_values(
        by="f1_mean",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


def get_group(file_path):

    filename = str(file_path).split("\\")[-1]

    filename = re.sub(
        r"_(bad|goodG2B)\.cpp$",
        "",
        filename,
        flags=re.IGNORECASE
    )

    return filename


def main():

    print("Loading dataset...")

    X, y, groups = load_data()

    print("Dataset shape:", X.shape)

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nNumber of groups:", groups.nunique())

    cross_validate_models(
        X,
        y,
        groups
    )

    print("\nSplitting dataset using grouped split...")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("Training groups:", train_groups.nunique())
    print("Testing groups:", test_groups.nunique())

    overlap = set(train_groups) & set(test_groups)

    print("Overlapping groups:", len(overlap))

    models = create_models()

    results = []

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, model in models.items():

        print("\n" + "=" * 50)
        print("Training:", name)

        model.fit(
            X_train,
            y_train
        )

        evaluation = evaluate_model(
            model,
            X_test,
            y_test
        )

        print(
            f"Accuracy:  {evaluation['accuracy']:.4f}"
        )

        print(
            f"Precision: {evaluation['precision']:.4f}"
        )

        print(
            f"Recall:    {evaluation['recall']:.4f}"
        )

        print(
            f"F1 Score:  {evaluation['f1']:.4f}"
        )

        print("\nConfusion Matrix:")

        print(
            evaluation["confusion_matrix"]
        )

        results.append({
            "model": name,
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"]
        })

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        joblib.dump(
            model,
            MODEL_DIR / filename
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(
        results_df.to_string(
            index=False
        )
    )

    best_model_name = "KNN"

    print(
        f"\nSelected BugLens model: "
        f"{best_model_name} (k=3)"
    )


if __name__ == "__main__":
    main()