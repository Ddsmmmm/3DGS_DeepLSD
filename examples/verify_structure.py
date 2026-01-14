#!/usr/bin/env python3
"""
Verification script to check code structure and imports.
This doesn't require dependencies to be installed.
"""

import ast
import sys
from pathlib import Path

def check_python_file(filepath):
    """Check if a Python file can be parsed."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_module_structure():
    """Verify the module structure is correct."""
    src_dir = Path(__file__).parent.parent / 'src'
    
    required_modules = [
        'line_detection/__init__.py',
        'line_detection/line_detector.py',
        'line_detection/video_loader.py',
        'line_reconstruction/__init__.py',
        'line_reconstruction/line_reconstructor.py',
        'line_reconstruction/camera_utils.py',
        'gaussian_constraint/__init__.py',
        'gaussian_constraint/gaussian_constraint.py',
        'utils/__init__.py',
        'utils/visualization.py',
        '__init__.py',
        'pipeline.py',
        'main.py'
    ]
    
    all_ok = True
    for module in required_modules:
        filepath = src_dir / module
        if not filepath.exists():
            print(f"❌ Missing: {module}")
            all_ok = False
        else:
            ok, error = check_python_file(filepath)
            if ok:
                print(f"✓ {module}")
            else:
                print(f"❌ {module}: {error}")
                all_ok = False
    
    return all_ok

def check_configs():
    """Verify configuration files exist."""
    configs_dir = Path(__file__).parent.parent / 'configs'
    
    required_configs = [
        'default_config.yaml',
        'high_quality_config.yaml',
        'fast_config.yaml'
    ]
    
    all_ok = True
    for config in required_configs:
        filepath = configs_dir / config
        if filepath.exists():
            print(f"✓ configs/{config}")
        else:
            print(f"❌ Missing: configs/{config}")
            all_ok = False
    
    return all_ok

def check_documentation():
    """Verify documentation files exist."""
    root_dir = Path(__file__).parent.parent
    
    required_docs = [
        'README.md',
        'requirements.txt',
        'setup.py'
    ]
    
    all_ok = True
    for doc in required_docs:
        filepath = root_dir / doc
        if filepath.exists():
            print(f"✓ {doc}")
        else:
            print(f"❌ Missing: {doc}")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("3DGS_DeepLSD Code Verification")
    print("=" * 60)
    
    print("\n1. Checking module structure...")
    print("-" * 60)
    modules_ok = check_module_structure()
    
    print("\n2. Checking configuration files...")
    print("-" * 60)
    configs_ok = check_configs()
    
    print("\n3. Checking documentation...")
    print("-" * 60)
    docs_ok = check_documentation()
    
    print("\n" + "=" * 60)
    if modules_ok and configs_ok and docs_ok:
        print("✓ All checks passed!")
        print("=" * 60)
        print("\nThe code structure is correct.")
        print("To run the pipeline, install dependencies first:")
        print("  pip install -r requirements.txt")
        return 0
    else:
        print("❌ Some checks failed!")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
