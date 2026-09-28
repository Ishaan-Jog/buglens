import pandas as pd
from pathlib import Path

from config import FEATURE_COLUMNS

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


INPUT_PATH = Path("../output/features.csv")


def get_group(file_path):

    filename = str(file_path).split("\\")[-1]

    import re

    filename = re.sub(
        r"_(bad|goodG2B)\.cpp$",
        "",
        filename,
        flags=re.IGNORECASE
    )

    return filename


def load_data():

    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["is_buggy"]
    groups = df["file"].apply(get_group)

    return X, y, groups


def evaluate_knn(X, y, groups, k):

    group_kfold = GroupKFold(
        n_splits=5
    )

    scores = []

    for train_idx, test_idx in group_kfold.split(
        X,
        y,
        groups
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=k
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        scores.append({
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        })

    scores_df = pd.DataFrame(scores)

    return {
        "accuracy_mean": scores_df["accuracy"].mean(),
        "accuracy_std": scores_df["accuracy"].std(),

        "precision_mean": scores_df["precision"].mean(),
        "precision_std": scores_df["precision"].std(),

        "recall_mean": scores_df["recall"].mean(),
        "recall_std": scores_df["recall"].std(),

        "f1_mean": scores_df["f1"].mean(),
        "f1_std": scores_df["f1"].std()
    }


def main():

    print("Loading dataset...")

    X, y, groups = load_data()

    print("Dataset shape:", X.shape)
    print("Number of groups:", groups.nunique())

    k_values = [
        3,
        5,
        7,
        9,
        11,
        15,
        21
    ]

    results = []

    print("\n" + "=" * 70)
    print("KNN HYPERPARAMETER TUNING")
    print("=" * 70)

    for k in k_values:

        print(
            f"\nTesting k = {k}"
        )

        evaluation = evaluate_knn(
            X,
            y,
            groups,
            k
        )

        print(
            f"Accuracy:  "
            f"{evaluation['accuracy_mean']:.4f} "
            f"+/- "
            f"{evaluation['accuracy_std']:.4f}"
        )

        print(
            f"Precision: "
            f"{evaluation['precision_mean']:.4f} "
            f"+/- "
            f"{evaluation['precision_std']:.4f}"
        )

        print(
            f"Recall:    "
            f"{evaluation['recall_mean']:.4f} "
            f"+/- "
            f"{evaluation['recall_std']:.4f}"
        )

        print(
            f"F1:        "
            f"{evaluation['f1_mean']:.4f} "
            f"+/- "
            f"{evaluation['f1_std']:.4f}"
        )

        results.append({
            "k": k,
            **evaluation
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="f1_mean",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("KNN TUNING RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()