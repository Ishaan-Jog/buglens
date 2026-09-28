from pathlib import Path
import pandas as pd
import re

DATASET_PATH = Path("../dataset/juliet")
OUTPUT_PATH = Path("../output/raw_dataset.csv")


def get_cwe(file_path):
    """
    Extract CWE identifier from the file's parent directories.
    Example:
    CWE121_Stack_Based_Buffer_Overflow
    -> CWE121
    """

    for part in file_path.parts:
        match = re.match(r"(CWE\d+)_", part)

        if match:
            return match.group(1)

    return "UNKNOWN"


def get_language(file_path):
    if file_path.suffix.lower() == ".cpp":
        return "C++"

    if file_path.suffix.lower() == ".c":
        return "C"

    return "Unknown"


def read_code(file_path):
    try:
        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return ""


def main():

    files = list(DATASET_PATH.rglob("*.cpp"))
    # files += list(DATASET_PATH.rglob("*.c"))

    records = []

    print(f"Found {len(files)} source files.")

    for index, file_path in enumerate(files):

        code = read_code(file_path)

        if not code.strip():
            continue

        records.append({
            "file": str(file_path),
            "cwe": get_cwe(file_path),
            "language": get_language(file_path),
            "code": code
        })

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1} files...")

    df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nDataset created successfully!")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nCWE distribution:")
    print(df["cwe"].value_counts())


if __name__ == "__main__":
    main()
