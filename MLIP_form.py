"""
Formation Energy Calculator for ML-predicted Energies

This script calculates formation energies from ML-predicted energies of compounds
and terminal elements. Formation energy is defined as:

    E_formation = E_compound - Σ(n_i * E_i) / Σ(n_i)

where:
    - E_compound is the energy per atom of the compound
    - n_i is the number of atoms of element i in the compound
    - E_i is the energy per atom of the terminal element i

All energies are in eV per atom units.

Example usage:
    Basic usage:
        python ML_formE.py -i compounds.csv -t terminal.csv -o results.csv
    
    Custom column names:
        python ML_formE.py -i compounds.csv -t terminal.csv -o results.csv \\
            --formula_column_compound "formula" \\
            --energy_column "energy_per_atom" \\
            --out_column "formation_energy"
    
    With specific terminal element column:
        python ML_formE.py -i compounds.csv -t elements.csv -o out.csv \\
            --formula_column_terminal "symbol"

Author: [Author Name]
Date: [Date]
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from tqdm import tqdm


# Configuration constants
DEFAULT_COLUMNS = {
    'compound_formula': 'composition',
    'terminal_element': 'element', 
    'energy': 'ML_e',
    'formation_energy': 'ML_formE'
}

SUPPORTED_FILE_EXTENSIONS = ['.csv']


def log_info(message: str, prefix: str = "INFO") -> None:
    """Simple logging function for consistent output formatting."""
    print(f"[{prefix}] {message}")

def validate_dataframe(dataframe: pd.DataFrame, required_columns: List[str], 
                      dataframe_name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame contains required columns and has valid data.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of column names that must be present
        dataframe_name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if dataframe.empty:
        raise ValueError(f"{dataframe_name} is empty")
    
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name} missing required columns: {missing_columns}")
    
    # Check for null values in required columns
    for col in required_columns:
        null_count = dataframe[col].isnull().sum()
        if null_count > 0:
            print(f"Warning: {dataframe_name} has {null_count} null values in column '{col}'")


def calculate_formation_energy(formula: str, total_energy: float, 
                             terminal_energies: Dict[str, float]) -> float:
    """
    Calculate formation energy from compound energy and terminal element energies.
    
    Formation energy = E_compound - sum(n_i * E_i) / sum(n_i)
    where n_i is the number of atoms of element i, and E_i is the energy per atom
    of the terminal element i.
    
    Args:
        formula: Chemical formula string (e.g., 'Li2O', 'NaCl')
        total_energy: Total energy per atom of the compound
        terminal_energies: Dictionary mapping element symbols to their energies
        
    Returns:
        Formation energy per atom
        
    Raises:
        KeyError: If an element in the formula is not found in terminal_energies
        ValueError: If the formula cannot be parsed
    """
    try:
        composition = Composition(formula)
        element_energies = []
        
        for element, amount in composition.items():
            element_str = str(element)
            if element_str not in terminal_energies:
                raise KeyError(f"Terminal energy not found for element: {element_str}")
            
            # Add energy for each atom of this element
            element_energies.extend([terminal_energies[element_str]] * int(amount))
        
        formation_energy = total_energy - np.mean(element_energies)
        return formation_energy
        
    except Exception as e:
        raise ValueError(f"Error calculating formation energy for {formula}: {str(e)}")


def create_energy_dictionary(dataframe: pd.DataFrame, key_column: str = 'element', 
                           value_column: str = 'ML_e') -> Dict[str, float]:
    """
    Create a dictionary mapping elements to their energies from a DataFrame.
    
    Args:
        dataframe: DataFrame containing element data
        key_column: Column name containing element symbols
        value_column: Column name containing energy values
        
    Returns:
        Dictionary mapping element symbols to energies
        
    Raises:
        KeyError: If specified columns don't exist in the DataFrame
        ValueError: If there are duplicate elements or invalid data
    """
    if key_column not in dataframe.columns:
        raise KeyError(f"Column '{key_column}' not found in DataFrame")
    if value_column not in dataframe.columns:
        raise KeyError(f"Column '{value_column}' not found in DataFrame")
    
    # Check for duplicates
    if dataframe[key_column].duplicated().any():
        duplicates = dataframe[dataframe[key_column].duplicated()][key_column].tolist()
        raise ValueError(f"Duplicate elements found: {duplicates}")
    
    return dict(zip(dataframe[key_column], dataframe[value_column]))


def update_formation_energies(dataframe: pd.DataFrame, terminal_energies: Dict[str, float],
                            formula_column: str = 'composition', 
                            energy_column: str = 'ML_e', 
                            output_column: str = 'ML_formE') -> pd.DataFrame:
    """
    Calculate formation energies for all compounds in a DataFrame.
    
    Args:
        dataframe: DataFrame containing compound data
        terminal_energies: Dictionary mapping element symbols to energies
        formula_column: Column name containing chemical formulas
        energy_column: Column name containing compound energies
        output_column: Column name for storing formation energies
        
    Returns:
        DataFrame with formation energies added
        
    Raises:
        KeyError: If required columns are missing
        ValueError: If formula parsing fails
    """
    # Validate required columns exist
    required_columns = [formula_column, energy_column]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = dataframe.copy()
    
    # Calculate formation energies with progress bar
    formation_energies = []
    failed_formulas = []
    
    for idx, row in tqdm(result_df.iterrows(), total=len(result_df), 
                        desc="Calculating formation energies"):
        try:
            formula = row[formula_column]
            energy = row[energy_column]
            formation_energy = calculate_formation_energy(formula, energy, terminal_energies)
            formation_energies.append(formation_energy)
        except (KeyError, ValueError) as e:
            print(f"Warning: Failed to calculate formation energy for row {idx}: {e}")
            formation_energies.append(np.nan)
            failed_formulas.append((idx, formula))
    
    result_df[output_column] = formation_energies
    
    if failed_formulas:
        print(f"Failed to process {len(failed_formulas)} formulas")
    
    return result_df


def load_compound_database(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file containing compound data.
    
    Args:
        file_path: Path to the CSV file containing compound data
        
    Returns:
        DataFrame from the CSV file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        log_info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


def main():
    """Main function to orchestrate the formation energy calculation process."""
    parser = argparse.ArgumentParser(
        description="Calculate formation energies from ML-predicted energies of compounds "
                   "and terminal elements. All energies are in eV per atom.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Path to the compound CSV file")
    parser.add_argument("-t", "--database_terminal", type=str, required=True,
                       help="Path to terminal elements energy CSV file")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output CSV file path")
    
    # Optional arguments
    parser.add_argument("--formula_column_compound", type=str, default="optimized_formula",
                       help="Column name containing chemical formulas in compound database")
    parser.add_argument("--formula_column_terminal", type=str, default="optimized_formula",
                       help="Column name containing element symbols in terminal database")
    parser.add_argument("--energy_column", type=str, default="Energy (eV/atom)",
                       help="Column name containing ML energies in both databases")
    parser.add_argument("--out_column", type=str, default="Formation Energy (eV/atom)",
                       help="Column name for storing calculated formation energies")

    args = parser.parse_args()
    
    try:
        log_info("Loading terminal elements database...")
        if not os.path.exists(args.database_terminal):
            raise FileNotFoundError(f"Terminal database file not found: {args.database_terminal}")
        
        terminal_db = pd.read_csv(args.database_terminal, index_col=0)
        log_info(f"Loaded {len(terminal_db)} terminal elements")
        
        # Validate terminal database
        validate_dataframe(
            terminal_db, 
            [args.formula_column_terminal, args.energy_column],
            "Terminal elements database"
        )
        
        log_info("Creating terminal energy dictionary...")
        terminal_energies = create_energy_dictionary(
            terminal_db, 
            key_column=args.formula_column_terminal, 
            value_column=args.energy_column
        )
        log_info(f"Terminal elements: {list(terminal_energies.keys())}")
        
        log_info("Loading compound database...")
        compound_db = load_compound_database(file_path=args.input)
        
        # Validate compound database
        validate_dataframe(
            compound_db,
            [args.formula_column_compound, args.energy_column],
            "Compound database"
        )
        
        log_info("Calculating formation energies...")
        compound_db = update_formation_energies(
            compound_db, 
            terminal_energies,
            formula_column=args.formula_column_compound,
            energy_column=args.energy_column,
            output_column=args.out_column
        )
        
        log_info("Sorting and saving results...")
        compound_db = compound_db.sort_index()
        compound_db.to_csv(args.output)
        
        log_info("Formation energies calculated successfully!")
        log_info(f"Results saved to: {args.output}")
        log_info(f"Total compounds processed: {len(compound_db)}")
        
        # Summary statistics
        if args.out_column in compound_db.columns:
            valid_energies = compound_db[args.out_column].dropna()
            if len(valid_energies) > 0:
                print(f"Formation energy statistics:")
                print(f"  Mean: {valid_energies.mean():.4f}")
                print(f"  Std:  {valid_energies.std():.4f}")
                print(f"  Min:  {valid_energies.min():.4f}")
                print(f"  Max:  {valid_energies.max():.4f}")
            
            failed_count = compound_db[args.out_column].isna().sum()
            if failed_count > 0:
                print(f"Warning: {failed_count} compounds failed to calculate formation energy")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
