from static_rules import analyze_code


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


issues = analyze_code(code)


print("=" * 60)
print("STATIC ANALYSIS RESULTS")
print("=" * 60)

for issue in issues:

    print("\nBug Type:", issue["bug_type"])
    print("Line:", issue["line"])
    print("Severity:", issue["severity"])
    print("Confidence:", issue["confidence"])
    print("Explanation:", issue["explanation"])
    print("Suggested Fix:", issue["suggested_fix"])