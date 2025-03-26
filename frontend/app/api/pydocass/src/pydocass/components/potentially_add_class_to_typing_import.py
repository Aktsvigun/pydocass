import ast


def potentially_add_class_to_typing_import(
    code: str, tree: ast.Module | None = None, class_name: str = "Union"
) -> str:
    if tree is None:
        tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            # Get original import position
            start = node.lineno - 1
            end = node.end_lineno

            # Extract existing names
            names = [n.name for n in node.names]
            if class_name in names:
                return code

            # Split code into lines
            lines = code.splitlines()

            # Handle single-line imports
            if start == end - 1:
                old_line = lines[start]
                if "(" not in old_line:
                    new_line = old_line.replace("import ", f"import {class_name}, ")
                    lines[start] = new_line

            # Handle multi-line imports
            else:
                for i in range(start, end):
                    if "import" in lines[i]:
                        lines[i] = lines[i].replace("import ", f"import {class_name}, ")

            return "\n".join(lines)

    return f"from typing import {class_name}\n" + code
