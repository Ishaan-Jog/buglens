import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_COLUMNS
import joblib


df = pd.read_csv(
    "../output/features.csv"
)

model = joblib.load(
    "../models/random_forest.pkl"
)


importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.to_string(
        index=False
    )
)


top_features = importance.head(15)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="importance",
    y="feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "../output/feature_importance.png",
    dpi=300
)

plt.show()