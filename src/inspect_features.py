import pandas as pd

df = pd.read_csv("../output/features.csv")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nBug distribution:")
print(df["is_buggy"].value_counts())

print("\nBug percentages:")
print(
    df["is_buggy"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nNumerical statistics:")
print(df.describe())