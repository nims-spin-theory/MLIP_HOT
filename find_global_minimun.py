

"""
Global Minimum Finder for CSV Databases

This script identifies the global minimum energy structures from multiple CSV files.
It loads multiple datasets, groups them by a common identifier (e.g., composition or structure ID),
and finds the entry with the minimum energy for each group.

Example usage:
    # Using file pattern
    python find_global_minimun.py -f ./results/ -p "formE_*_*.csv" -o global_min.csv

    # Using explicit file list
    python find_global_minimun.py -i file1.csv file2.csv file3.csv -o global_min.csv
    
    # Specify custom energy column
    python find_global_minimun.py -f ./data/ -p "*.csv" -o output.csv --energy-column "ML_formE"

Author: [Author Name]
Date: October 31, 2025
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm


def log_info(message: str, prefix: str = "INFO") -> None:
    """Simple logging function for consistent output formatting."""
    print(f"[{prefix}] {message}")


def load_csv_files(file_paths: List[str], labels: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files into a dictionary of DataFrames.
    
    Args:
        file_paths: List of paths to CSV files
        labels: Optional list of labels for each file. If None, uses basenames without extension.
        
    Returns:
        Dictionary mapping labels to DataFrames
        
    Raises:
        FileNotFoundError: If any file doesn't exist
        ValueError: If files cannot be read
    """
    if labels is None:
        labels = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]
    
    if len(labels) != len(file_paths):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of files ({len(file_paths)})")
    
    dbs = {}
    log_info(f"Loading {len(file_paths)} CSV files...")
    
    for label, file_path in tqdm(zip(labels, file_paths), total=len(file_paths), desc="Loading files"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            dbs[label] = df
            log_info(f"Loaded '{label}': {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {str(e)}")
    
    return dbs


def get_global_minimum(dbs: Dict[str, pd.DataFrame], 
                       energy_column: str = 'Energy (eV/atom)',
                       group_by_index: bool = True,
                       group_by_column: Optional[str] = None) -> pd.DataFrame:
    """
    Find the global minimum energy structures from multiple databases.
    
    Args:
        dbs: Dictionary mapping database names to DataFrames
        energy_column: Name of the column containing energy values
        group_by_index: If True, group by DataFrame index (level 1 after concatenation)
        group_by_column: If specified, group by this column instead of index
        
    Returns:
        DataFrame containing only the minimum energy entries for each group
        
    Raises:
        KeyError: If energy_column doesn't exist in the DataFrames
        ValueError: If DataFrames cannot be concatenated
    """
    log_info("Combining databases and finding global minima...")
    
    # Validate that energy column exists in all dataframes
    for db_name, df in dbs.items():
        if energy_column not in df.columns:
            raise KeyError(f"Column '{energy_column}' not found in database '{db_name}'")
    
    # Concatenate all dataframes with hierarchical index
    # Level 0: database name, Level 1: original index
    combined = pd.concat(list(dbs.values()), keys=list(dbs.keys()))
    
    log_info(f"Combined database contains {len(combined)} total entries")
    
    # Find minimum energy for each group
    if group_by_column:
        log_info(f"Grouping by column: '{group_by_column}'")
        result = combined.groupby(group_by_column).apply(
            lambda group: group.loc[group[energy_column].idxmin()]
        )
    else:
        log_info("Grouping by index (level 1)")
        result = combined.groupby(level=1).apply(
            lambda group: group.loc[group[energy_column].idxmin()]
        )
    
    log_info(f"Found {len(result)} unique global minimum structures")
    
    # Add a column indicating which database each minimum came from
    if isinstance(result.index, pd.MultiIndex):
        result['source_database'] = result.index.get_level_values(0)
    
    return result


def print_summary_statistics(result_df: pd.DataFrame, energy_column: str) -> None:
    """Print summary statistics about the results."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if energy_column in result_df.columns:
        energies = result_df[energy_column].dropna()
        print(f"\nEnergy Statistics ({energy_column}):")
        print(f"  Count:  {len(energies)}")
        print(f"  Mean:   {energies.mean():.6f}")
        print(f"  Std:    {energies.std():.6f}")
        print(f"  Min:    {energies.min():.6f}")
        print(f"  Max:    {energies.max():.6f}")
        print(f"  Median: {energies.median():.6f}")
    
    if 'source_database' in result_df.columns:
        print(f"\nSource Database Distribution:")
        source_counts = result_df['source_database'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(result_df)) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
    
    print("="*60 + "\n")


def main():
    """Main function to orchestrate the global minimum finding process."""
    parser = argparse.ArgumentParser(
        description="Find global minimum energy structures from multiple CSV databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options - either folder pattern or explicit files
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--folder", type=str,
                           help="Path to folder containing CSV files (use with -p/--pattern)")
    input_group.add_argument("-i", "--input-files", nargs='+', type=str,
                           help="Explicit list of CSV file paths")
    
    parser.add_argument("-p", "--pattern", type=str,
                       help="Glob pattern for CSV files (e.g., 'formE_*.csv'). Required with -f.")
    
    # Output
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output CSV file path for global minima")
    
    # Column options
    parser.add_argument("--energy-column", type=str, default="Energy (eV/atom)",
                       help="Name of the column containing energy values")
    parser.add_argument("--group-by-column", type=str, default=None,
                       help="Column name to group by (default: use index)")
    
    # Optional features
    parser.add_argument("--labels", nargs='+', type=str,
                       help="Custom labels for input files (same order as files)")
    parser.add_argument("--no-summary", action="store_true",
                       help="Skip printing summary statistics")
    
    args = parser.parse_args()
    
    try:
        # Determine input files
        if args.folder:
            if not args.pattern:
                raise ValueError("--pattern is required when using --folder")
            
            if not os.path.exists(args.folder):
                raise FileNotFoundError(f"Folder not found: {args.folder}")
            
            file_pattern = os.path.join(args.folder, args.pattern)
            file_paths = glob.glob(file_pattern)
            
            if not file_paths:
                raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
            
            file_paths.sort()
            log_info(f"Found {len(file_paths)} files matching pattern")
            
        else:
            file_paths = args.input_files
            # Validate files exist
            for fp in file_paths:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"File not found: {fp}")
        
        # Load CSV files
        dbs = load_csv_files(file_paths, labels=args.labels)
        
        # Find global minima
        result = get_global_minimum(
            dbs,
            energy_column=args.energy_column,
            group_by_index=(args.group_by_column is None),
            group_by_column=args.group_by_column
        )
        
        # Save results
        log_info(f"Saving results to: {args.output}")
        result.to_csv(args.output)
        
        # Print summary
        if not args.no_summary:
            print_summary_statistics(result, args.energy_column)
        
        log_info("Global minimum search completed successfully!")
        log_info(f"Total unique minima: {len(result)}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())



