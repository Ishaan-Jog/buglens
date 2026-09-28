import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_COLUMNS
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import joblib


df = pd.read_csv(
    "../output/features.csv"
)

X = df[FEATURE_COLUMNS]
y = df["is_buggy"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = joblib.load(
    "../models/random_forest.pkl"
)


predictions = model.predict(
    X_test
)


matrix = confusion_matrix(
    y_test,
    predictions
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Buggy"],
    yticklabels=["Safe", "Buggy"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "../output/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()