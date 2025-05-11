#!/usr/bin/env python3
"""
Path handling test script for AudioSegmenter.

This script helps diagnose path issues when using Windows-style paths in WSL.
"""

import os
import sys
import platform
import argparse
from pathlib import Path

def print_header(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

def detect_wsl():
    """Check if we're running in WSL."""
    if platform.system() == 'Linux' and 'microsoft' in platform.release().lower():
        return True
    return False

def print_env_info():
    """Print environment information."""
    print_header("ENVIRONMENT INFO")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Running in WSL: {detect_wsl()}")

def fix_windows_path(path_str):
    """Convert Windows-style paths to Unix paths."""
    # Convert backslashes to forward slashes
    if '\\' in path_str:
        path_str = path_str.replace('\\', '/')
    
    # Normalize relative paths (../foo → absolute path)
    if path_str.startswith('../') or path_str.startswith('./'):
        base_dir = os.getcwd()
        path_str = os.path.normpath(os.path.join(base_dir, path_str))
    
    return path_str

def test_path(path_str):
    """Test path handling for a given path string."""
    print_header(f"TESTING PATH: {path_str}")
    
    # Original path
    print(f"Original path: {path_str}")
    
    # Convert Windows-style paths
    fixed_path = fix_windows_path(path_str)
    print(f"Fixed path: {fixed_path}")
    
    # Normalize path
    path = Path(fixed_path).expanduser().resolve()
    norm_path = str(path)
    print(f"Normalized path: {norm_path}")
    
    # Check if exists
    exists = os.path.exists(norm_path)
    print(f"Path exists: {exists}")
    
    if exists:
        if os.path.isfile(norm_path):
            print(f"Path is a FILE")
        elif os.path.isdir(norm_path):
            print(f"Path is a DIRECTORY")
    else:
        print(f"WARNING: Path does not exist!")
    
    return norm_path, exists

def main():
    parser = argparse.ArgumentParser(description="Test path handling for AudioSegmenter in WSL.")
    parser.add_argument("paths", nargs="+", help="Paths to test")
    
    args = parser.parse_args()
    
    print_env_info()
    
    for path_str in args.paths:
        norm_path, exists = test_path(path_str)

if __name__ == "__main__":
    main() 