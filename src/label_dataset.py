from pathlib import Path
import pandas as pd

INPUT_PATH = Path("../output/raw_dataset.csv")
OUTPUT_PATH = Path("../output/labeled_dataset.csv")


def determine_label(file_path):
    """
    Determine whether the Juliet test case is classified
    as good or bad based on its filename.
    """

    filename = Path(file_path).stem.lower()

    if "_bad" in filename:
        return 1

    if "_good" in filename:
        return 0

    return -1


def main():

    df = pd.read_csv(INPUT_PATH)

    df["is_buggy"] = df["file"].apply(determine_label)

    print("Label distribution:")
    print(df["is_buggy"].value_counts())

    unknown = df[df["is_buggy"] == -1]

    print("\nUnknown labels:", len(unknown))

    df = df[df["is_buggy"] != -1]
    df = df[df["language"] == "C++"]

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:", OUTPUT_PATH)
    print("Final rows:", len(df))


if __name__ == "__main__":
    main()