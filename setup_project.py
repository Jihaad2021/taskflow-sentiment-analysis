#!/usr/bin/env python3
"""
Setup project structure for TaskFlow Sentiment Analysis
Cross-platform compatible
"""

from pathlib import Path


def create_structure():
    """Create complete project folder structure."""

    # Define structure
    structure = {
        ".github/workflows": [],
        "docs": [],
        "src/agents": ["__init__.py"],
        "src/tools": ["__init__.py"],
        "src/models": ["__init__.py"],
        "src/utils": ["__init__.py"],
        "src/llm": ["__init__.py"],
        "src/api": ["__init__.py"],
        "tests/unit": ["__init__.py"],
        "tests/integration": ["__init__.py"],
        "tests/e2e": ["__init__.py"],
        "tests/fixtures": ["__init__.py"],
    }

    # Create root __init__.py
    root_inits = ["src/__init__.py", "tests/__init__.py"]

    print("Creating project structure...")

    # Create directories and files
    for folder, files in structure.items():
        # Create directory
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}/")

        # Create files in directory
        for file in files:
            file_path = Path(folder) / file
            file_path.touch(exist_ok=True)
            print(f"  ✓ Created: {file_path}")

    # Create root __init__.py files
    for init_file in root_inits:
        Path(init_file).touch(exist_ok=True)
        print(f"✓ Created: {init_file}")

    print("\n✓ Project structure created successfully!")
    print("\nStructure overview:")
    print_tree(".", max_depth=3)


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure."""
    if current_depth >= max_depth:
        return

    # Skip these directories
    skip_dirs = {".git", "__pycache__", ".taskflow-venv", "venv", ".pytest_cache", ".mypy_cache"}

    try:
        entries = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        if entry.name in skip_dirs or entry.name.startswith(".") and entry.name != ".github":
            continue

        is_last = i == len(entries) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{entry.name}{'/' if entry.is_dir() else ''}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension, max_depth, current_depth + 1)


if __name__ == "__main__":
    create_structure()
    print("\nNext steps:")
    print("1. Create virtual environment: python -m venv .taskflow-venv")
    print("2. Activate it:")
    print("   - Linux/Mac: source .taskflow-venv/bin/activate")
    print("   - Windows: .taskflow-venv\\Scripts\\activate")
    print("3. Install dependencies: pip install -r requirements.txt")
