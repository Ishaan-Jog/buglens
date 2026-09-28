import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


df = pd.read_csv("../output/features.csv")


FEATURE_COLUMNS = [
    "total_lines",
    "blank_lines",
    "comment_lines",
    "code_lines",
    "functions",
    "if_statements",
    "else_statements",
    "for_loops",
    "while_loops",
    "do_while_loops",
    "switch_statements",
    "case_statements",
    "return_statements",
    "array_declarations",
    "array_accesses",
    "pointer_declarations",
    "pointer_dereferences",
    "new_operations",
    "delete_operations",
    "malloc_calls",
    "calloc_calls",
    "realloc_calls",
    "free_calls",
    "unsafe_function_calls",
    "division_operations",
    "modulo_operations",
    "logical_and",
    "logical_or",
    "comparisons",
    "assignments",
    "max_nesting_depth",
    "cyclomatic_complexity",
    "average_line_length"
]


X = df[FEATURE_COLUMNS]
y = df["is_buggy"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


configurations = [
    {
        "max_depth": 3,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    {
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    {
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    {
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    {
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    {
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    }
]


results = []


for config in configurations:

    model = DecisionTreeClassifier(
        **config,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    results.append({
        "max_depth": config["max_depth"],
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


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "f1",
    ascending=False
)


print(
    results_df.to_string(
        index=False
    )
)