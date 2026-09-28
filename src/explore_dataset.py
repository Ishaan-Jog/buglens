from pathlib import Path

dataset = Path("../dataset/juliet")

cpp_files = list(dataset.rglob("*.cpp"))
c_files = list(dataset.rglob("*.c"))
all_files = c_files + cpp_files

print("C files:", len(c_files))
print("C++ files:", len(cpp_files))
print("Total files:", len(all_files))

print("\nSample files: \n")

for file in all_files[:20]:
    print(file)
