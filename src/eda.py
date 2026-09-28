import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../output/features.csv")

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)


print("\nCorrelation with is_buggy:\n")

correlation = (
    numeric_df
    .corr()["is_buggy"]
    .sort_values(ascending=False)
)

print(correlation)

plt.figure(figsize=(16, 12))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "../output/correlation_heatmap.png",
    dpi=300
)

important_features = [
    "cyclomatic_complexity",
    "max_nesting_depth",
    "pointer_declarations",
    "array_accesses",
    "unsafe_function_calls",
    "total_lines"
]

for feature in important_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="is_buggy",
        y=feature
    )

    plt.xlabel("Buggy Code (0 = Safe, 1 = Buggy)")
    plt.ylabel(feature)

    plt.title(
        f"{feature} vs Bug Status"
    )

    plt.tight_layout()

    plt.savefig(
        f"../output/{feature}_comparison.png",
        dpi=300
    )

    plt.show()

cwe_counts = (
    df["cwe"]
    .value_counts()
    .head(20)
)

plt.figure(figsize=(12, 7))

cwe_counts.sort_values().plot(
    kind="barh"
)

plt.xlabel("Number of Samples")
plt.ylabel("CWE")
plt.title("Top 20 CWE Categories")

plt.tight_layout()

plt.savefig(
    "../output/cwe_distribution.png",
    dpi=300
)

plt.show()

print("Buggy code distribution:", df["is_buggy"].value_counts(normalize=True))
