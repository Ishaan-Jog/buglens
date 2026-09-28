import re


def add_issue(
    issues,
    bug_type,
    line,
    severity,
    confidence,
    explanation,
    suggested_fix
):
    issues.append({
        "bug_type": bug_type,
        "line": line,
        "severity": severity,
        "confidence": confidence,
        "explanation": explanation,
        "suggested_fix": suggested_fix
    })


def check_unsafe_functions(lines, issues):

    unsafe_functions = {
        "gets": 0.99,
        "strcpy": 0.95,
        "strcat": 0.95,
        "sprintf": 0.90,
        "scanf": 0.70,
        "memcpy": 0.65,
        "memmove": 0.60
    }

    for line_number, line in enumerate(lines, start=1):

        for function, confidence in unsafe_functions.items():

            pattern = rf"\b{function}\s*\("

            if re.search(pattern, line):

                add_issue(
                    issues,
                    "Unsafe Function",
                    line_number,
                    "High",
                    confidence,
                    f"The function {function}() can cause security or memory-safety problems when used incorrectly.",
                    f"Review the use of {function}() and consider a bounds-checked or safer alternative."
                )


def check_division_by_zero(lines, issues):

    for line_number, line in enumerate(lines, start=1):

        if re.search(
            r"/\s*0(?:\D|$)",
            line
        ):

            add_issue(
                issues,
                "Division by Zero",
                line_number,
                "Critical",
                0.99,
                "The expression appears to divide by the literal value zero.",
                "Ensure the divisor cannot be zero before performing the division."
            )

        if re.search(
            r"%\s*0(?:\D|$)",
            line
        ):

            add_issue(
                issues,
                "Modulo by Zero",
                line_number,
                "Critical",
                0.99,
                "The expression appears to perform modulo operation with zero.",
                "Ensure the divisor cannot be zero before performing the modulo operation."
            )


def check_assignment_in_condition(lines, issues):

    for line_number, line in enumerate(lines, start=1):

        if re.search(
            r"\b(if|while)\s*\([^)]*[^=!<>]=[^=][^)]*\)",
            line
        ):

            add_issue(
                issues,
                "Assignment in Condition",
                line_number,
                "Medium",
                0.90,
                "An assignment operator appears inside a conditional expression.",
                "If assignment is intentional, make it explicit. Otherwise use == for comparison."
            )


def check_loop_conditions(lines, issues):

    arrays = set()

    for line in lines:

        declarations = re.findall(
            r"\b(?:int|char|float|double|long|short|bool)\s+([A-Za-z_]\w*)\s*\[\s*\d+\s*\]",
            line
        )

        for array_name in declarations:
            arrays.add(array_name)

    for line_number, line in enumerate(lines, start=1):

        loop_match = re.search(
            r"\bfor\s*\([^;]*;\s*([A-Za-z_]\w*)\s*(<=|>=)[^;]*;",
            line
        )

        if not loop_match:
            continue

        loop_variable = loop_match.group(1)

        end = min(
            line_number + 10,
            len(lines)
        )

        accesses_array = False

        for current_line in lines[
            line_number:end
        ]:

            if re.search(
                rf"\b(?:{'|'.join(map(re.escape, arrays))})\s*\[\s*{re.escape(loop_variable)}\s*\]",
                current_line
            ):
                accesses_array = True
                break

        if accesses_array:
            continue

        add_issue(
            issues,
            "Potential Off-by-One Error",
            line_number,
            "Medium",
            0.65,
            "The loop uses <= or >= in its condition, which can cause an extra iteration depending on the boundary.",
            "Verify the loop boundary and consider whether < or > is required."
        )


def check_array_access(lines, issues):

    array_declarations = set()

    for line in lines:

        declarations = re.findall(
            r"\b(?:int|char|float|double|long|short|bool)\s+([A-Za-z_]\w*)\s*\[\s*\d+\s*\]",
            line
        )

        for array_name in declarations:
            array_declarations.add(array_name)

    for line_number, line in enumerate(lines, start=1):

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

        array_accesses = re.findall(
            r"\b([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]",
            line
        )

        for array_name, index in array_accesses:

            if array_name not in array_declarations:
                continue

            if index.isdigit():

                add_issue(
                    issues,
                    "Array Bounds Risk",
                    line_number,
                    "Medium",
                    0.55,
                    f"Array '{array_name}' is accessed using a fixed index. The index may exceed the declared array bounds.",
                    "Verify that the index is within the declared array size."
                )


def check_pointer_dereference(lines, issues):

    pointer_declarations = set()

    for line in lines:

        declarations = re.findall(
            r"\b(?:int|char|float|double|long|short|bool)\s*\*\s*([A-Za-z_]\w*)",
            line
        )

        for pointer in declarations:
            pointer_declarations.add(pointer)

    for line_number, line in enumerate(lines, start=1):

        for pointer in pointer_declarations:

            if re.search(
                rf"\b(?:int|char|float|double|long|short|bool)\s*\*\s*{re.escape(pointer)}\b",
                line
            ):
                continue

            if re.search(
                rf"\b{re.escape(pointer)}\s*=\s*new\b",
                line
            ):
                continue

            dereference_pattern = rf"\*\s*{re.escape(pointer)}\b"

            if not re.search(
                dereference_pattern,
                line
            ):
                continue

            nearby_start = max(
                0,
                line_number - 4
            )

            nearby_lines = lines[
                nearby_start:line_number
            ]

            null_check = any(
                re.search(
                    rf"\b{re.escape(pointer)}\s*(==|!=)\s*(nullptr|NULL|0)",
                    previous_line
                )
                for previous_line in nearby_lines
            )

            if not null_check:

                add_issue(
                    issues,
                    "Potential Null Pointer Dereference",
                    line_number,
                    "High",
                    0.60,
                    f"Pointer '{pointer}' is dereferenced without an apparent nearby null check.",
                    f"Check whether {pointer} is null before dereferencing it."
                )


def check_memory_management(lines, issues):

    allocation_lines = {}

    for line_number, line in enumerate(lines, start=1):

        malloc_match = re.search(
            r"\bmalloc\s*\(",
            line
        )

        new_match = re.search(
            r"\bnew\s+",
            line
        )

        if malloc_match:

            allocation_lines[line_number] = "malloc"

        elif new_match:

            allocation_lines[line_number] = "new"

    free_count = sum(
        bool(re.search(r"\bfree\s*\(", line))
        for line in lines
    )

    delete_count = sum(
        bool(re.search(r"\bdelete\s+", line))
        for line in lines
    )

    malloc_count = sum(
        value == "malloc"
        for value in allocation_lines.values()
    )

    new_count = sum(
        value == "new"
        for value in allocation_lines.values()
    )

    if malloc_count > free_count:

        add_issue(
            issues,
            "Potential Memory Leak",
            next(
                line
                for line, value in allocation_lines.items()
                if value == "malloc"
            ),
            "High",
            0.70,
            "More malloc() calls than free() calls were detected.",
            "Ensure every dynamically allocated block is released exactly once."
        )

    if new_count > delete_count:

        add_issue(
            issues,
            "Potential Memory Leak",
            next(
                line
                for line, value in allocation_lines.items()
                if value == "new"
            ),
            "High",
            0.70,
            "More new operations than delete operations were detected.",
            "Ensure dynamically allocated objects are properly released."
        )


def check_array_loop_bounds(lines, issues):

    arrays = {}

    for line_number, line in enumerate(lines, start=1):

        declarations = re.findall(
            r"\b(?:int|char|float|double|long|short|bool)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]",
            line
        )

        for array_name, size in declarations:
            arrays[array_name] = int(size)

    for line_number, line in enumerate(lines, start=1):

        loop_match = re.search(
            r"\bfor\s*\(\s*(?:int\s+)?([A-Za-z_]\w*)\s*=\s*(\d+)\s*;\s*([A-Za-z_]\w*)\s*(<=|<|>=|>)\s*(\d+)\s*;",
            line
        )

        if not loop_match:
            continue

        variable = loop_match.group(1)
        condition_variable = loop_match.group(3)
        operator = loop_match.group(4)
        boundary = int(loop_match.group(5))

        if variable != condition_variable:
            continue

        if operator != "<=":
            continue

        loop_end = min(
            line_number + 10,
            len(lines)
        )

        for current_line_number in range(
            line_number,
            loop_end + 1
        ):

            current_line = lines[
                current_line_number - 1
            ]

            accesses = re.findall(
                rf"\b([A-Za-z_]\w*)\s*\[\s*{re.escape(variable)}\s*\]",
                current_line
            )

            for array_name in accesses:

                if array_name not in arrays:
                    continue

                array_size = arrays[array_name]

                if boundary >= array_size:

                    add_issue(
                        issues,
                        "Potential Array Out-of-Bounds",
                        current_line_number,
                        "High",
                        0.92,
                        f"Array '{array_name}' has {array_size} elements, but loop variable '{variable}' can reach {boundary}. Valid indices are 0 to {array_size - 1}.",
                        f"Change the loop boundary so '{variable}' cannot reach {array_size}, for example use {variable} < {array_size}."
                    )


def analyze_code(code):

    issues = []

    lines = code.splitlines()

    check_unsafe_functions(
        lines,
        issues
    )

    check_division_by_zero(
        lines,
        issues
    )

    check_assignment_in_condition(
        lines,
        issues
    )

    check_loop_conditions(
        lines,
        issues
    )

    check_array_loop_bounds(
        lines,
        issues
    )

    check_array_access(
        lines,
        issues
    )

    check_pointer_dereference(
        lines,
        issues
    )

    check_memory_management(
        lines,
        issues
    )

    return issues