#!/usr/bin/env python3
"""
AudioSegmenter simplified wrapper script
Handles path issues in WSL environment
"""

import os
import sys
import re
import subprocess
from pathlib import Path

def print_header(msg):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}")

def fix_path(path_str):
    """Fix path issues, specifically for Windows paths in WSL."""
    # Look for patterns like 'test_audio2014-06-19' which should be 'test_audio/2014-06-19'
    if re.search(r'[a-zA-Z0-9_-]+\d{4}-\d{2}-\d{2}', path_str):
        # This looks like a filename with a date that's missing a separator
        match = re.search(r'([a-zA-Z0-9_-]+)(\d{4}-\d{2}-\d{2})', path_str)
        if match:
            dir_part = match.group(1)
            date_part = match.group(2)
            path_str = f"{dir_part}/{date_part}{path_str[len(dir_part+date_part):]}"
            print(f"Fixed missing separator: {path_str}")
    
    # Convert Windows backslashes to forward slashes
    path_str = path_str.replace('\\', '/')
    
    return path_str

def main():
    print_header("AudioSegmenter Wrapper")
    print("Fixing path issues and running AudioSegmenter...")
    
    # Get command line arguments
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python run.py [AudioSegmenter arguments]")
        print("Example: python run.py --normalize \"path/to/audio.mp3\" output_dir")
        return
    
    # Process each argument to fix paths
    fixed_args = []
    for arg in args:
        if arg.startswith('-'):
            # This is an option (like --normalize), don't modify
            fixed_args.append(arg)
        else:
            # This might be a path, fix it
            fixed_arg = fix_path(arg)
            fixed_args.append(fixed_arg)
    
    # Build the command to run audiosegmenter_cli.py
    cmd = ["python", "audiosegmenter_cli.py"] + fixed_args
    
    # Print the command
    print("Running command:", " ".join(cmd))
    
    # Run the command
    result = subprocess.run(cmd)
    
    # Return the same exit code
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 