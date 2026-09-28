# 🐞 BugLens

**BugLens** is a hybrid C/C++ bug prediction and static-analysis system built with Python, machine learning, and rule-based code analysis.

The project combines:

- **Machine Learning** to estimate general defect risk from code metrics.
- **Static Analysis** to detect specific, explainable programming issues.
- **Streamlit** to provide an interactive interface for analyzing C/C++ code.
- **Data exploration and visualization** to study the dataset and model performance.

The main goal is not to claim that code is definitely buggy or safe, but to provide useful indicators that help identify code requiring further inspection.

---

## ✨ Features

### 🔍 C/C++ Code Analyzer
BugLens accepts C/C++ source code in two ways:

- Paste code directly into the analyzer.
- Upload a `.c`, `.cpp`, `.cc`, `.cxx`, `.h`, or `.hpp` file.

The analyzer extracts code features, runs the trained ML model, and applies static-analysis rules.

### 🤖 Machine Learning
BugLens uses a **K-Nearest Neighbors (KNN)** classifier.

Current configuration:

| Setting | Value |
|---|---|
| Algorithm | KNN |
| k | 3 |
| Feature scaling | StandardScaler |
| Validation | 5-fold GroupKFold |
| Dataset | NIST Juliet C/C++ Test Suite |

The selected KNN model achieved the following grouped cross-validation results:

| Metric | Score |
|---|---:|
| Accuracy | 70.20% |
| Precision | 60.35% |
| Recall | 86.26% |
| F1 Score | 71.01% |

The ML output is presented as a **risk score**, not as proof that a bug exists.

### 🛡️ Static Analysis
The rule-based analyzer checks for potential issues such as:

- Division by zero
- Modulo by zero
- Array out-of-bounds access
- Unsafe functions such as `gets`, `strcpy`, `strcat`, and `sprintf`
- Potential null-pointer dereferences
- Memory leaks
- Suspicious loop boundaries
- Assignment inside conditions
- Other pointer and memory-related risks

Each detected issue can include:

- Bug type
- Source-code line
- Severity
- Confidence
- Explanation
- Suggested fix

### 📊 Data Exploration & Visualization

The Streamlit application includes pages for:

- ML model performance
- Dataset statistics
- CWE distribution
- Language distribution
- Feature statistics
- Feature importance
- Feature correlation
- Confusion matrix
- Model comparison

---

# 🧠 How BugLens Works

BugLens uses two complementary analysis approaches.

```text
                   C/C++ Source Code
                           │
                           ▼
                 ┌───────────────────┐
                 │ Feature Extraction│
                 └─────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      ┌───────────────┐         ┌────────────────┐
      │  KNN Model    │         │ Static Rules   │
      │               │         │                │
      │ General risk  │         │ Specific bugs  │
      │ estimation    │         │ & explanations │
      └───────┬───────┘         └───────┬────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Hybrid Analysis │
                  │     Report      │
                  └─────────────────┘
                           │
                           ▼
                    Streamlit UI
```

## 1. Feature Extraction

The source code is processed by `feature_extractor.py`.

BugLens extracts measurable characteristics such as:

- Number of lines
- Number of functions
- `if`, `for`, and `while` statements
- Array declarations and accesses
- Pointer declarations and dereferences
- `new` / `delete`
- `malloc` / `free`
- Unsafe function calls
- Comparisons and assignments
- Division and modulo operations
- Nesting depth
- Cyclomatic complexity
- Loop-related patterns
- Pointer-risk indicators

These features form the input to the ML model.

## 2. Machine Learning Prediction

The trained KNN model receives the extracted feature vector.

Before prediction, the features are scaled using `StandardScaler`.

The model returns a general **defect-risk score** based on its learned similarity to the training data.

For example:

```text
ML Risk Score: 66.67%
Classification: Medium General Risk
```

Because KNN uses `k=3`, the probability-like score can occur in coarse steps such as approximately 0%, 33.33%, 66.67%, or 100%. Therefore, BugLens describes this value as a **risk score** rather than a precise probability.

## 3. Static Analysis

The static analyzer in `src/static_rules.py` scans the source code using deterministic rules.

For example:

```cpp
int arr[10];

for(int i = 0; i <= 10; i++)
{
    arr[i] = i;
}
```

The analyzer can identify the potential array-boundary problem:

```text
Potential Array Out-of-Bounds
Line: 5
Severity: High
```

Static analysis is useful because the result is explainable. The application can tell the user what pattern triggered the finding instead of simply producing a mysterious ML prediction. Humanity occasionally appreciates explanations.

## 4. Hybrid Risk

The application combines the ML risk score and static-analysis findings into an overall risk level.

A critical static finding can make the overall result **Critical**, while high-severity findings can produce **High** overall risk.

The ML model and static analyzer remain separate because an ML prediction does not prove that a particular bug exists.

---

# 📁 Project Structure

```text
BugLens/
│
├── dataset/
│   └── juliet/
│       └── NIST Juliet C/C++ dataset
│
├── output/
│   ├── raw_dataset.csv
│   ├── labeled_dataset.csv
│   ├── features.csv
│   ├── model_results.csv
│   ├── correlation_heatmap.png
│   ├── cwe_distribution.png
│   ├── feature_importance.png
│   └── random_forest_confusion_matrix.png
│
├── models/
│   └── knn.pkl
│
├── src/
│   ├── explore_dataset.py
│   ├── create_dataset.py
│   ├── label_dataset.py
│   ├── feature_extractor.py
│   ├── inspect_features.py
│   ├── eda.py
│   ├── train_models.py
│   ├── model_visualization.py
│   ├── feature_importance.py
│   ├── tune_decision_tree.py
│   ├── tune_random_forest.py
│   ├── tune_knn.py
│   ├── config.py
│   ├── static_rules.py
│   ├── test_static_rules.py
│   ├── hybrid_analyzer.py
│   └── debug_ml.py
│
├── app.py
└── README.md
```

---

# ⚙️ Requirements

- Python 3.10 or newer recommended
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

The project also requires the **NIST Juliet C/C++ Test Suite 1.3** for the dataset-generation and model-training pipeline.

---

# 🚀 Installation

## 1. Clone or download the project

```bash
git clone https://github.com/Ishaan-Jog/buglens.git
cd buglens
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset Setup

Download the (NIST Juliet C/C++ Test Suite 1.3)[https://samate.nist.gov/SARD/test-suites/112] and place the extracted dataset inside:

```text
dataset/juliet/
```

The resulting structure should look approximately like:

```text
BugLens/
└── dataset/
    └── juliet/
        ├── testcases/
        └── ...
```

The exact internal structure depends on how the Juliet archive is extracted.

---

# 🧪 Building the Dataset and Model

If the generated CSV files and trained model are already included, you can skip this section and run the application directly.

For a fresh setup, the general pipeline is:

```text
Juliet Dataset
      │
      ▼
explore_dataset.py
      │
      ▼
create_dataset.py
      │
      ▼
label_dataset.py
      │
      ▼
feature_extractor.py
      │
      ▼
features.csv
      │
      ▼
train_models.py
      │
      ▼
models / model results
```

Run the scripts from the project root.

### 1. Explore the dataset

```bash
python src/explore_dataset.py
```

This checks the Juliet dataset and provides information about the available C/C++ source files.

### 2. Create the raw dataset

```bash
python src/create_dataset.py
```

This recursively collects C/C++ files and extracts information such as:

- File name
- CWE
- Programming language
- Source code

Output:

```text
output/raw_dataset.csv
```

### 3. Label the dataset

```bash
python src/label_dataset.py
```

Juliet filenames contain `_bad` and `_good` variants.

These are converted into the binary target:

```text
0 = Clean
1 = Buggy
```

Output:

```text
output/labeled_dataset.csv
```

### 4. Extract features

```bash
python src/feature_extractor.py
```

This converts source code into numerical code metrics used by the ML models.

Output:

```text
output/features.csv
```

### 5. Inspect the features

```bash
python src/inspect_features.py
```

This checks:

- Dataset shape
- Data types
- Missing values
- Duplicate rows
- Class distribution
- Feature statistics

### 6. Run exploratory data analysis

```bash
python src/eda.py
```

This generates visualizations such as:

- Correlation heatmap
- CWE distribution
- Feature distributions

### 7. Train and evaluate models

```bash
python src/train_models.py
```

The project evaluates several machine-learning algorithms, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN
- SVM

The final BugLens model uses **KNN with k=3**, selected through grouped cross-validation and KNN tuning.

### 8. Tune KNN

```bash
python src/tune_knn.py
```

This evaluates different values of `k` using grouped cross-validation.

The selected configuration is:

```text
KNN
k = 3
```

---

# 🖥️ Running BugLens

Once the model and required output files are available, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually similar to:

```text
http://localhost:8501
```

Open that address in a browser.

---

# 🔍 Using the Code Analyzer

The **Code Analyzer** is the primary functionality of BugLens.

### Option 1: Paste Code

Paste C/C++ code into the editor and click:

```text
🔎 Analyze Code
```

### Option 2: Upload a File

Upload a supported source file:

```text
.c
.cpp
.cc
.cxx
.h
.hpp
```

BugLens then performs:

1. Feature extraction
2. KNN prediction
3. Static-rule analysis
4. Overall risk calculation
5. Display of findings and code metrics

---

# 🧪 Example

Input:

```cpp
#include <iostream>

int main()
{
    int arr[10];

    for(int i = 0; i <= 10; i++)
    {
        arr[i] = i;
    }

    return 0;
}
```

BugLens can identify the loop boundary and array-access pattern as a potential out-of-bounds issue.

The analyzer also displays:

- Overall risk
- ML risk score
- Number of static issues
- Code-line count
- Detailed findings
- Code metrics
- Source code

---

# 📊 Streamlit Pages

## 🔍 Code Analyzer

The main workspace.

Used for analyzing individual C/C++ programs.

## 📊 ML Performance

Displays:

- Model comparison
- Accuracy
- Precision
- Recall
- F1 score
- Selected KNN configuration
- Grouped cross-validation results

## 📈 Visual Analytics

Displays:

- Model performance charts
- Feature importance
- Feature correlation heatmap
- CWE distribution
- Confusion matrix
- Dataset statistics

## 📁 Dataset Explorer

Provides interactive information about:

- Total samples
- Buggy vs clean samples
- CWE distribution
- Programming-language distribution
- Feature statistics

## ℹ️ About

Provides a short explanation of the BugLens architecture, dataset, ML model, and static-analysis approach.

---

# 🔬 Validation Methodology

A major concern with the Juliet dataset is that related `_good` and `_bad` examples can be highly similar.

A normal random train/test split can therefore produce overly optimistic results if related examples appear in both sets.

BugLens uses **GroupKFold** validation to reduce this problem.

Related Juliet variants are assigned to the same group so that the model is evaluated on groups it did not directly train on.

The final KNN evaluation uses **5-fold grouped cross-validation**.

---

# ⚠️ Limitations

BugLens is an educational/research-oriented static-analysis and ML project, not a replacement for a production-grade compiler, sanitizer, or security scanner.

Important limitations include:

- Regex-based static analysis cannot fully understand C/C++ semantics.
- The ML model can produce false positives and false negatives.
- The ML risk score is not proof that code contains a defect.
- KNN risk scores are coarse because the selected model uses `k=3`.
- Static rules detect only the patterns implemented in `static_rules.py`.
- The analyzer does not guarantee that code will compile successfully.
- Complex control flow and inter-procedural behavior may not be detected.
- The model is trained on Juliet and may not generalize perfectly to arbitrary real-world code.

For example, an undeclared identifier such as:

```cpp
cout << hi;
```

is a C++ compilation error, but it is not automatically detected by the current static-rule engine. A future compiler-diagnostics layer could address this.

---

# 🛠️ Future Improvements

Possible extensions include:

- Compiler-based diagnostics using GCC/Clang
- More sophisticated AST-based analysis
- Infinite-loop detection for constant conditions such as `while(1)`
- More CWE-specific static-analysis rules
- Data-flow analysis
- Better pointer and memory analysis
- Support for larger real-world datasets
- Additional ML models
- Explainable ML visualizations
- Code-quality scoring
- Exportable analysis reports

---

# 📚 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main development language |
| Streamlit | Interactive web interface |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| C/C++ | Source-code analysis target |
| NIST Juliet | Training/evaluation dataset |

---

# 📌 Project Summary

**BugLens** combines machine learning and static analysis into a single C/C++ code-analysis platform.

The ML component learns patterns from code metrics and provides a general defect-risk estimate, while the static-analysis component detects specific patterns and explains why they may be problematic.

The combination provides both:

> **Prediction + Explanation**

which is the core idea behind the project.

---

## 👨‍💻 Project

**BugLens — C/C++ Bug Prediction and Static Analysis**

Built as a Data Exploration & Visualization / Machine Learning project.
