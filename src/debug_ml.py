import pandas as pd
import joblib
from pathlib import Path

from config import FEATURE_COLUMNS
from feature_extractor import extract_features


MODEL_PATH = Path("../models/knn.pkl")


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


model = joblib.load(
    MODEL_PATH
)

features = extract_features(
    code
)

print("=" * 60)
print("EXTRACTED FEATURES")
print("=" * 60)

for feature in FEATURE_COLUMNS:
    print(
        f"{feature:35} {features.get(feature, 0)}"
    )


X = pd.DataFrame(
    [[features.get(feature, 0) for feature in FEATURE_COLUMNS]],
    columns=FEATURE_COLUMNS
)

print("\n" + "=" * 60)
print("MODEL PREDICTION")
print("=" * 60)

print(
    "Prediction:",
    model.predict(X)[0]
)

print(
    "Probability:",
    model.predict_proba(X)[0]
)

print(
    "Classes:",
    model.classes_
)