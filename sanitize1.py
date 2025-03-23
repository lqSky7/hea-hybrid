#!/usr/bin/env python3
# /Users/ca5/Desktop/qnn_fnl/remove_empty_columns.py

import csv
import argparse
import os
import chardet

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read a chunk to detect encoding
        result = chardet.detect(raw_data)
        return result['encoding']

def remove_columns_with_empty_cells(input_file, output_file, encoding=None):
    # Try to detect encoding if not specified
    if not encoding:
        try:
            encoding = detect_encoding(input_file)
            print(f"Detected encoding: {encoding}")
        except Exception as e:
            print(f"Error detecting encoding: {e}")
            encoding = 'latin-1'  # Fallback encoding that accepts most bytes
            print(f"Using fallback encoding: {encoding}")

    # Read the input CSV file with the detected or specified encoding
    try:
        with open(input_file, 'r', newline='', encoding=encoding, errors='replace') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
            
            if not data:
                print("The CSV file is empty.")
                return
            
            # Get the number of rows and columns
            num_rows = len(data)
            num_cols = max(len(row) for row in data) if num_rows > 0 else 0
            
            # Find columns that have at least one empty cell
            columns_to_remove = set()
            for col_idx in range(num_cols):
                for row_idx in range(num_rows):
                    # Check if column index exists for current row and if it's empty
                    if col_idx >= len(data[row_idx]) or data[row_idx][col_idx].strip() == '':
                        columns_to_remove.add(col_idx)
                        break
            
            # Create a new dataset without the columns to be removed
            new_data = []
            for row in data:
                new_row = [row[i] for i in range(len(row)) if i not in columns_to_remove]
                new_data.append(new_row)
            
            # Write the new data to the output CSV file
            with open(output_file, 'w', newline='', encoding=encoding) as outfile:
                writer = csv.writer(outfile)
                writer.writerows(new_data)
            
            removed_count = len(columns_to_remove)
            print(f"Removed {removed_count} column{'s' if removed_count != 1 else ''} with empty cells.")
            print(f"Output saved to {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Trying with latin-1 encoding...")
        # Retry with latin-1 encoding which can handle most byte sequences
        with open(input_file, 'r', newline='', encoding='latin-1') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
            # Continue with the same processing as above
            num_rows = len(data)
            num_cols = max(len(row) for row in data) if num_rows > 0 else 0
            columns_to_remove = set()
            for col_idx in range(num_cols):
                for row_idx in range(num_rows):
                    if col_idx >= len(data[row_idx]) or data[row_idx][col_idx].strip() == '':
                        columns_to_remove.add(col_idx)
                        break
            new_data = []
            for row in data:
                new_row = [row[i] for i in range(len(row)) if i not in columns_to_remove]
                new_data.append(new_row)
            with open(output_file, 'w', newline='', encoding='latin-1') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(new_data)
            removed_count = len(columns_to_remove)
            print(f"Removed {removed_count} column{'s' if removed_count != 1 else ''} with empty cells.")
            print(f"Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Remove columns with empty cells from a CSV file.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output CSV file (default: input_file_filtered.csv)')
    parser.add_argument('-e', '--encoding', help='Specify file encoding (e.g., utf-8, latin-1, cp1252)')
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}_filtered{ext}"
    
    remove_columns_with_empty_cells(args.input_file, output_file, args.encoding)

if __name__ == '__main__':
    main()