"""
CSV File Concatenator

This script loads and concatenates multiple CSV files from a folder into a single pandas DataFrame.
The script supports glob patterns for flexible file matching and preserves the index from the original files.

Example usage:
    python concat_csv.py -f ./data/ -p "db*convex*.csv" -o combined_output.csv
    python concat_csv.py -f ./data/ -p "*.csv" -o output.csv --reset-index

Author: [Author Name]
Date: October 31, 2025
"""

import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import pandas as pd


def log_info(message: str, prefix: str = "INFO") -> None:
    """Simple logging function for consistent output formatting."""
    print(f"[{prefix}] {message}")


def extract_size_from_filename(filename: str) -> Optional[int]:
    """
    Extract the size parameter from a filename with format XXX*_{size}_rank.csv
    
    Args:
        filename: Name of the file to parse
        
    Returns:
        The size value if found, None otherwise
    """
    match = re.search(r'_(\d+)_\d+\.csv$', filename)
    if match:
        return int(match.group(1))
    return None


def check_missing_files(folder_path: str, file_pattern: str, found_files: List[str]) -> List[str]:
    """
    Check if all expected rank files are present based on the size parameter.
    
    Args:
        folder_path: Path to the folder containing CSV files
        file_pattern: Original glob pattern used
        found_files: List of file paths that were found
        
    Returns:
        List of missing file names
    """
    if not found_files:
        return []
    
    # Extract size from the first file
    first_file = os.path.basename(found_files[0])
    size = extract_size_from_filename(first_file)
    
    if size is None:
        # Cannot determine expected size, skip validation
        return []
    
    # Extract the base pattern (everything before _{size}_rank.csv)
    match = re.match(r'(.+)_\d+_\d+\.csv$', first_file)
    if not match:
        return []
    
    base_pattern = match.group(1)
    
    # Generate expected filenames
    expected_files = [f"{base_pattern}_{size}_{rank}.csv" for rank in range(size)]
    found_basenames = {os.path.basename(f) for f in found_files}
    
    # Find missing files
    missing = [f for f in expected_files if f not in found_basenames]
    
    return missing


def load_and_concat_csv_files(folder_path: str, file_pattern: str, 
                               reset_index: bool = False,
                               check_completeness: bool = False) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files containing data.
    
    Args:
        folder_path: Path to the folder containing CSV files
        file_pattern: Glob pattern to match CSV files (e.g., "db*convex*.csv")
        reset_index: If True, reset the index in the combined DataFrame
        check_completeness: If True, check for missing rank files based on size parameter
        
    Returns:
        Concatenated DataFrame from all matching CSV files
        
    Raises:
        FileNotFoundError: If no files match the pattern
        ValueError: If files cannot be read or concatenated
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    csv_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern '{file_pattern}' in {folder_path}")
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    log_info(f"Found {len(csv_files)} CSV files to load")
    
    # Check for missing files if requested
    if check_completeness:
        missing_files = check_missing_files(folder_path, file_pattern, csv_files)
        if missing_files:
            log_info(f"WARNING: {len(missing_files)} expected files are missing:", prefix="WARNING")
            for missing_file in missing_files:
                print(f"  - {missing_file}")
        else:
            log_info("All expected rank files are present")
    
    try:
        dataframes = []
        for file_path in csv_files:
            df = pd.read_csv(file_path, index_col=0)
            dataframes.append(df)
            log_info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
        
        combined_df = pd.concat(dataframes, ignore_index=reset_index)
        log_info(f"Combined database contains {len(combined_df)} total rows")
        return combined_df
        
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {str(e)}")


def main():
    """Main function to orchestrate the CSV concatenation process."""
    parser = argparse.ArgumentParser(
        description="Load and concatenate multiple CSV files into a single DataFrame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-f", "--folder", type=str, required=True,
                       help="Path to the folder containing CSV files")
    parser.add_argument("-p", "--pattern", type=str, required=True,
                       help="Glob pattern for CSV files (e.g., 'db*test*.csv', '*.csv')")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output CSV file path")
    
    # Optional arguments
    parser.add_argument("--reset-index", action="store_true",
                       help="Reset index in the combined DataFrame (default: preserve original indices)")
    parser.add_argument("--check-completeness", action="store_true", default=True,
                       help="Check that all rank files are present for files with XXX*_{size}_rank.csv format")

    args = parser.parse_args()
    
    try:
        log_info("Starting CSV concatenation...")
        
        combined_df = load_and_concat_csv_files(
            folder_path=args.folder,
            file_pattern=args.pattern,
            reset_index=args.reset_index,
            check_completeness=args.check_completeness
        )
        
        log_info("Sorting and saving results...")
        if not args.reset_index:
            combined_df = combined_df.sort_index()
        combined_df.to_csv(args.output)
        
        log_info("CSV concatenation completed successfully!")
        log_info(f"Results saved to: {args.output}")
        log_info(f"Total rows: {len(combined_df)}")
        log_info(f"Total columns: {len(combined_df.columns)}")
        
        # Display column names
        print(f"\nColumns in combined DataFrame:")
        for col in combined_df.columns:
            print(f"  - {col}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
