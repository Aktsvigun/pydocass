import os

EXCLUDE_DIRS = {
    "node_modules",
    "build",
    "dist",
    ".git",
    "__pycache__",
    ".idea",
    ".next",
    ".venv",
}
EXCLUDE_EXTS = {".log", ".lock", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".py"}
EXCLUDE_FILES = {
    ".DS_Store",
    "package-lock.json",
    "repo_contents.txt",
    "st_run_python_documentation.py",
}


def should_exclude(path):
    for excl in EXCLUDE_DIRS:
        if f"/{excl}/" in path or path.startswith(f"./{excl}/"):
            return True
    return os.path.splitext(path)[1] in EXCLUDE_EXTS


def collect_files():
    all_files = []
    for root, _, files in os.walk(".", topdown=True):
        if any(excl in root for excl in EXCLUDE_DIRS):
            continue
        for file in files:
            if any(file in excl_file for excl_file in EXCLUDE_FILES):
                continue
            file_path = os.path.join(root, file)
            if not should_exclude(file_path):
                all_files.append(file_path)
    return all_files


def read_files(files):
    file_data = {}
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                file_data[file] = f.read()
        except Exception as e:
            file_data[file] = f"Error reading file: {e}"
    return file_data


if __name__ == "__main__":
    files = collect_files()
    contents = read_files(files)

    with open("repo_contents.txt", "w", encoding="utf-8") as out:
        for file, content in contents.items():
            out.write(f"### FILE: {file} ###\n{content}\n\n")
