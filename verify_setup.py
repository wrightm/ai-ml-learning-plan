#!/usr/bin/env python3
"""
Quick verification script to check if the Python environment is set up correctly.
Run this after installing requirements.txt to verify your setup.
"""

import sys
from importlib import import_module

CORE_DEPENDENCIES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",  # scikit-learn
    "jupyter",
    "torch",
]

OPTIONAL_DEPENDENCIES = [
    "fastapi",
    "wandb",
    "langchain",
    "implicit",
    "lightfm",
]


def check_python_version():
    """Verify Python 3.11+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.11+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, optional=False):
    """Try to import a package and report status"""
    try:
        mod = import_module(package_name)
        version = getattr(mod, "__version__", "unknown")
        symbol = "✅" if not optional else "✓"
        print(f"{symbol} {package_name:20s} (v{version})")
        return True
    except ImportError:
        symbol = "❌" if not optional else "⚠️"
        status = "MISSING" if not optional else "optional"
        print(f"{symbol} {package_name:20s} [{status}]")
        return False


def main():
    print("=" * 60)
    print("Python Environment Verification")
    print("=" * 60)
    print()

    # Check Python version
    print("Python Version:")
    py_ok = check_python_version()
    print()

    # Check core dependencies
    print("Core Dependencies:")
    core_ok = all(check_package(pkg) for pkg in CORE_DEPENDENCIES)
    print()

    # Check optional dependencies
    print("Optional Dependencies:")
    for pkg in OPTIONAL_DEPENDENCIES:
        check_package(pkg, optional=True)
    print()

    # Summary
    print("=" * 60)
    if py_ok and core_ok:
        print("✅ Setup verified! You're ready to start learning.")
        print("\nNext steps:")
        print("  1. Run: jupyter lab")
        print("  2. Open: journal/weeks/WEEK_01/W01_day_by_day.ipynb")
        return 0
    else:
        print("❌ Setup incomplete. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

