import os
import csv
from pathlib import Path

def inspect_files(directory):
    path = Path(directory)
    for file in path.rglob('*'):
        if file.suffix.lower() == '.csv':
            print(f"\n--- File: {file} ---")
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    # Try to detect separator
                    line = f.readline()
                    sep = ',' if ',' in line else ';' if ';' in line else '\t'
                    f.seek(0)
                    reader = csv.reader(f, delimiter=sep)
                    header = next(reader)
                    print(f"Columns: {header}")
                    try:
                        sample = next(reader)
                        print(f"Sample data: {sample}")
                    except StopIteration:
                        print("File is empty or has only header")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        elif file.suffix.lower() == '.xlsx':
            print(f"\n--- File: {file} --- (Excel file, skipping for now)")

if __name__ == "__main__":
    inspect_files('/home/said/projects/translator/data/extracted')
