import re
from pathlib import Path
import pandas as pd


INPUT_PATH = Path("../output/labeled_dataset.csv")
OUTPUT_PATH = Path("../output/features.csv")


def count_pattern(code, pattern):
    return len(re.findall(pattern, code, re.MULTILINE))


def count_lines(code):
    lines = code.splitlines()

    total_lines = len(lines)

    blank_lines = sum(
        1 for line in lines
        if not line.strip()
    )

    comment_lines = sum(
        1 for line in lines
        if line.strip().startswith("//")
        or line.strip().startswith("/*")
        or line.strip().startswith("*")
    )

    code_lines = total_lines - blank_lines - comment_lines

    return total_lines, blank_lines, comment_lines, code_lines


def remove_comments(code):
    code = re.sub(
        r"//.*",
        "",
        code
    )

    code = re.sub(
        r"/\*[\s\S]*?\*/",
        "",
        code
    )

    return code


def calculate_max_nesting(code):
    max_depth = 0
    current_depth = 0

    for char in code:
        if char == "{":
            current_depth += 1
            max_depth = max(max_depth, current_depth)

        elif char == "}":
            current_depth = max(0, current_depth - 1)

    return max_depth


def calculate_cyclomatic_complexity(code):

    complexity = 1

    patterns = [
        r"\bif\s*\(",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\bcase\b",
        r"\bcatch\s*\(",
        r"\?\s*"
    ]

    for pattern in patterns:
        complexity += count_pattern(code, pattern)

    complexity += count_pattern(code, r"&&")
    complexity += count_pattern(code, r"\|\|")

    return complexity


def extract_features(code):

    cleaned_code = remove_comments(code)

    total_lines, blank_lines, comment_lines, code_lines = count_lines(code)

    features = {}

    features["total_lines"] = total_lines
    features["blank_lines"] = blank_lines
    features["comment_lines"] = comment_lines
    features["code_lines"] = code_lines

    features["functions"] = count_pattern(
        code,
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^;{}]*\)\s*\{"
    )

    features["if_statements"] = count_pattern(
        code,
        r"\bif\s*\("
    )

    features["else_statements"] = count_pattern(
        code,
        r"\belse\b"
    )

    features["for_loops"] = count_pattern(
        code,
        r"\bfor\s*\("
    )

    features["while_loops"] = count_pattern(
        code,
        r"\bwhile\s*\("
    )

    features["do_while_loops"] = count_pattern(
        code,
        r"\bdo\s*\{"
    )

    features["switch_statements"] = count_pattern(
        code,
        r"\bswitch\s*\("
    )

    features["case_statements"] = count_pattern(
        code,
        r"\bcase\b"
    )

    features["return_statements"] = count_pattern(
        code,
        r"\breturn\b"
    )

    features["array_declarations"] = len(
        re.findall(
            r"\b(?:int|char|float|double|long|short|bool)\s+[A-Za-z_]\w*\s*\[\s*\d+\s*\]",
            cleaned_code
        )
    )

    array_accesses = 0

    for line in cleaned_code.splitlines():

        if re.search(
            r"\b(?:int|char|float|double|long|short|bool)\s+[A-Za-z_]\w*\s*\[",
            line
        ):
            continue

        if re.search(
            r"\bnew\s+[A-Za-z_]\w*\s*\[",
            line
        ):
            continue

        array_accesses += len(
            re.findall(
                r"\b[A-Za-z_]\w*\s*\[[^\]]+\]",
                line
            )
        )

    features["array_accesses"] = array_accesses

    features["pointer_declarations"] = count_pattern(
        code,
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\*+\s*[A-Za-z_][A-Za-z0-9_]*"
    )

    pointer_dereferences = 0

    for line in cleaned_code.splitlines():

        if re.search(
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*\*+\s*[A-Za-z_][A-Za-z0-9_]*",
            line
        ):
            line_without_declaration = re.sub(
                r"\b[A-Za-z_][A-Za-z0-9_]*\s*\*+\s*[A-Za-z_][A-Za-z0-9_]*",
                "",
                line
            )
        else:
            line_without_declaration = line

        pointer_dereferences += len(
            re.findall(
                r"\*[A-Za-z_][A-Za-z0-9_]*",
                line_without_declaration
            )
        )

    features["pointer_dereferences"] = pointer_dereferences

    features["new_operations"] = count_pattern(
        code,
        r"\bnew\b"
    )

    features["delete_operations"] = count_pattern(
        code,
        r"\bdelete\b"
    )

    features["malloc_calls"] = count_pattern(
        code,
        r"\bmalloc\s*\("
    )

    features["calloc_calls"] = count_pattern(
        code,
        r"\bcalloc\s*\("
    )

    features["realloc_calls"] = count_pattern(
        code,
        r"\brealloc\s*\("
    )

    features["free_calls"] = count_pattern(
        code,
        r"\bfree\s*\("
    )

    unsafe_functions = [
        r"\bgets\s*\(",
        r"\bstrcpy\s*\(",
        r"\bstrcat\s*\(",
        r"\bsprintf\s*\(",
        r"\bscanf\s*\("
    ]

    features["unsafe_function_calls"] = sum(
        count_pattern(code, pattern)
        for pattern in unsafe_functions
    )

    features["division_operations"] = count_pattern(
        cleaned_code,
        r"/"
    )

    features["modulo_operations"] = count_pattern(
        cleaned_code,
        r"%"
    )

    features["logical_and"] = count_pattern(
        cleaned_code,
        r"&&"
    )

    features["logical_or"] = count_pattern(
        cleaned_code,
        r"\|\|"
    )

    features["comparisons"] = count_pattern(
        cleaned_code,
        r"==|!=|<=|>=|<|>"
    )

    features["assignments"] = count_pattern(
        cleaned_code,
        r"(?<![=!<>])=(?!=)"
    )

    features["max_nesting_depth"] = calculate_max_nesting(code)

    features["cyclomatic_complexity"] = calculate_cyclomatic_complexity(code)

    if total_lines > 0:
        features["average_line_length"] = (
            len(code) / total_lines
        )
    else:
        features["average_line_length"] = 0

    features["less_than_comparisons"] = count_pattern(
        code,
        r"<(?!=)"
    )

    features["less_equal_comparisons"] = count_pattern(
        code,
        r"<=" 
    )

    features["greater_than_comparisons"] = count_pattern(
        code,
        r">(?!=)"
    )

    features["greater_equal_comparisons"] = count_pattern(
        code,
        r">="
    )

    features["equality_comparisons"] = count_pattern(
        cleaned_code,
        r"=="
    )

    features["inequality_comparisons"] = count_pattern(
        cleaned_code,
        r"!="
    )

    features["gets_calls"] = count_pattern(
        code,
        r"\bgets\s*\("
    )

    features["strcpy_calls"] = count_pattern(
        code,
        r"\bstrcpy\s*\("
    )

    features["strcat_calls"] = count_pattern(
        code,
        r"\bstrcat\s*\("
    )

    features["sprintf_calls"] = count_pattern(
        code,
        r"\bsprintf\s*\("
    )

    features["scanf_calls"] = count_pattern(
        code,
        r"\bscanf\s*\("
    )

    features["memcpy_calls"] = count_pattern(
        code,
        r"\bmemcpy\s*\("
    )

    features["memmove_calls"] = count_pattern(
        code,
        r"\bmemmove\s*\("
    )

    features["malloc_free_difference"] = (
        features["malloc_calls"]
        + features["calloc_calls"]
        + features["realloc_calls"]
        - features["free_calls"]
    )

    features["new_delete_difference"] = (
        features["new_operations"]
        - features["delete_operations"]
    )

    features["null_checks"] = (
        count_pattern(
            cleaned_code,
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*==\s*(?:NULL|nullptr)"
        )
        +
        count_pattern(
            cleaned_code,
            r"\b(?:NULL|nullptr)\s*==\s*[A-Za-z_][A-Za-z0-9_]*"
        )
        +
        count_pattern(
            cleaned_code,
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*!=\s*(?:NULL|nullptr)"
        )
        +
        count_pattern(
            cleaned_code,
            r"\b(?:NULL|nullptr)\s*!=\s*[A-Za-z_][A-Za-z0-9_]*"
        )
    )

    features["address_of_operations"] = count_pattern(
        code,
        r"&[A-Za-z_][A-Za-z0-9_]*"
    )

    features["pointer_casts"] = count_pattern(
        code,
        r"\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\*\s*\)"
    )

    features["pointer_risk_ratio"] = (
        features["pointer_dereferences"]
        / max(features["pointer_declarations"], 1)
    )

    features["increment_operations"] = count_pattern(
        code,
        r"\+\+|--"
    )

    features["break_statements"] = count_pattern(
        code,
        r"\bbreak\b"
    )

    features["continue_statements"] = count_pattern(
        code,
        r"\bcontinue\b"
    )

    features["loop_condition_le"] = count_pattern(
        code,
        r"(for|while)\s*\([^)]*<="
    )

    features["loop_condition_ge"] = count_pattern(
        code,
        r"(for|while)\s*\([^)]*>="
    )

    features["division_by_literal_zero"] = count_pattern(
        code,
        r"/\s*0"
    )

    features["modulo_by_literal_zero"] = count_pattern(
        code,
        r"%\s*0"
    )

    features["division_by_variable"] = count_pattern(
        code,
        r"/\s*[A-Za-z_][A-Za-z0-9_]*"
    )

    return features


def main():

    df = pd.read_csv(INPUT_PATH)

    feature_rows = []

    print("Extracting features...")

    for index, row in df.iterrows():

        code = row["code"]

        features = extract_features(code)

        features["file"] = row["file"]
        features["cwe"] = row["cwe"]
        features["language"] = row["language"]
        features["is_buggy"] = row["is_buggy"]

        feature_rows.append(features)

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1} files")

    feature_df = pd.DataFrame(feature_rows)

    columns = (
        ["file"]
        + [
            column
            for column in feature_df.columns
            if column not in ["file", "cwe", "language", "is_buggy"]
        ]
        + ["cwe", "language", "is_buggy"]
    )

    feature_df = feature_df[columns]

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature extraction complete!")
    print("Rows:", len(feature_df))
    print("Columns:", len(feature_df.columns))

    print("\nFeatures:")
    print(feature_df.columns.tolist())

    print("\nDataset preview:")
    print(feature_df.head())


if __name__ == "__main__":
    main()