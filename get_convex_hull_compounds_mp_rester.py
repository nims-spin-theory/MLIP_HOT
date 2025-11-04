"""
Convex Hull Phase Preparation for ML-predicted Formation Energies

This script prepares competing phase data for convex hull analysis by extracting
stable phases from the Materials Project database for each unique chemical system
found in the input compound database. 

This script uses the Materials Project API (MPRester) to retrieve phase data.
An API key is required - get yours at: https://materialsproject.org/api

The script uses MPI for parallel processing to efficiently handle large datasets
and multiple chemical systems.

Example usage:
    mpirun -n 4 python get_convex_hull_compounds_mp_rester.py -d compounds.csv -o convex_phases.csv --api_key YOUR_API_KEY
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ase.formula import Formula
from mpi4py import MPI
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from tqdm import tqdm

def log_info(message: str, rank: int = 0, current_rank: int = 0) -> None:
    """Log information message only from specified rank."""
    if current_rank == rank:
        print(f"[INFO] {message}")


def log_warning(message: str, rank: int = 0, current_rank: int = 0) -> None:
    """Log warning message only from specified rank."""
    if current_rank == rank:
        print(f"[WARNING] {message}")


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def print_mpi_info() -> None:
    """Print MPI configuration information."""
    if rank == 0:
        print(f"MPI Configuration:")
        print(f"  Total processes: {size}")
        print(f"  Master rank: 0")
        if size > 1:
            print(f"  Worker ranks: 1-{size-1}")
        else:
            print("  Running in single-process mode")
        print()


def extract_unique_chemical_systems(dataframe: pd.DataFrame, 
                                   composition_column: str = 'composition') -> List[List[str]]:
    """
    Extract unique chemical systems (element combinations) from compound database.
    
    Args:
        dataframe: DataFrame containing compound compositions
        composition_column: Column name containing chemical formulas
        
    Returns:
        List of unique element combinations, each as a sorted list of element symbols
        
    Raises:
        KeyError: If composition column is missing
        ValueError: If formulas cannot be parsed
    """
    if composition_column not in dataframe.columns:
        raise KeyError(f"Column '{composition_column}' not found in DataFrame")
    
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty")
    
    log_info(f"Processing {len(dataframe)} compounds...", current_rank=rank)
    
    element_systems = []
    failed_formulas = []
    
    for idx, row in dataframe.iterrows():
        try:
            composition = row[composition_column]
            if pd.isna(composition):
                continue
                
            # Parse chemical formula to get element counts
            element_counts = Formula(composition).count()
            elements = sorted(list(element_counts.keys()))
            element_systems.append(elements)
            
        except Exception as e:
            failed_formulas.append((idx, composition))
            log_warning(f"Failed to parse formula '{composition}' at row {idx}: {e}", 
                       current_rank=rank)
    
    log_info(f"Successfully parsed {len(element_systems)} compounds", current_rank=rank)
    
    if failed_formulas:
        log_warning(f"Failed to parse {len(failed_formulas)} formulas", current_rank=rank)
    
    # Get unique chemical systems
    unique_systems = list(set(tuple(system) for system in element_systems))
    unique_systems = [list(system) for system in unique_systems]
    
    log_info(f"Found {len(unique_systems)} unique chemical systems", current_rank=rank)
    
    return unique_systems

def extract_competing_phases(chemical_systems: List[List[str]], api_key: str) -> Optional[pd.DataFrame]:
    """
    Extract competing phases from Materials Project database for given chemical systems.
    
    Uses MPI to distribute workload across multiple processes for efficiency.
    
    Args:
        chemical_systems: List of chemical systems, each as a list of element symbols
        api_key: Materials Project API key
        
    Returns:
        DataFrame containing competing phase data (only on rank 0), None on other ranks
        
    Raises:
        Exception: If database connection or phase extraction fails
    """
    log_info(f"Processing {len(chemical_systems)} chemical systems across {size} MPI ranks", 
             current_rank=rank)
    
    # Distribute workload among MPI ranks
    local_systems = [system for i, system in enumerate(chemical_systems) if i % size == rank]
    
    log_info(f"Rank {rank} processing {len(local_systems)} chemical systems", 
             current_rank=rank)
    
    local_results = []
    failed_systems = []
    
    # Process local chemical systems with progress bar (only show on rank 0)
    for elements in tqdm(local_systems, disable=(rank != 0), 
                        desc=f"Rank {rank} extracting phases"):
        try:
            with MPRester(api_key) as mpr:
                # 1) Fetch entries with formation energies for the given chemical system
                entries = mpr.get_entries_in_chemsys(elements)

                # 2) Build phase diagram and get stable entries
                phase_diagram = PhaseDiagram(entries)
                stable_entries = phase_diagram.stable_entries

                # 3) Extract relevant data from each stable entry
                for entry in stable_entries:
                    mp_id = entry.data.get('material_id')
                    
                    E_Form = phase_diagram.get_form_energy_per_atom(entry)          
                    E_Hull = phase_diagram.get_e_above_hull(entry)
                    
                    # 4) Fetch structure from MP
                    if mp_id:
                        try:
                            structure = mpr.get_structure_by_material_id(mp_id)
                            
                            phase_data = {
                                'composition': str(structure.composition.reduced_formula),
                                'cell': str(structure.lattice.matrix.tolist()),
                                'positions': str(structure.frac_coords.tolist()),
                                'numbers': str(list(structure.atomic_numbers)),
                                'material_id': mp_id,
                                'MP Formation energy (eV/atom)': E_Form,
                                'MP Hull distance (eV/atom)': E_Hull,
                                'natoms': structure.num_sites,
                                'spacegroup': structure.get_space_group_info()[0]
                            }
                            
                            local_results.append(phase_data)
                            
                        except Exception as struct_error:
                            log_warning(f"Failed to fetch structure for {mp_id}: {struct_error}", 
                                      current_rank=rank)

        except Exception as e:
            failed_systems.append((elements, str(e)))
            log_warning(f"Failed to process system {elements}: {e}", current_rank=rank)
    
    log_info(f"Rank {rank} extracted {len(local_results)} phases", current_rank=rank)
    
    if failed_systems:
        log_warning(f"Rank {rank} failed to process {len(failed_systems)} systems", 
                   current_rank=rank)
    
    # Gather results from all MPI ranks
    all_results = comm.gather(local_results, root=0)
    all_failures = comm.gather(failed_systems, root=0)
    
    # Combine and process results on rank 0
    if rank == 0:
        return combine_phase_results(all_results, all_failures)
    else:
        return None




def combine_phase_results(all_results: List[List[Dict]], 
                         all_failures: List[List[Tuple]]) -> pd.DataFrame:
    """
    Combine phase results from all MPI ranks and create final DataFrame.
    
    Args:
        all_results: List of result lists from each MPI rank
        all_failures: List of failure lists from each MPI rank
        
    Returns:
        Combined DataFrame with competing phase data
    """
    # Flatten results from all ranks
    flat_results = [item for rank_results in all_results for item in rank_results]
    flat_failures = [item for rank_failures in all_failures for item in rank_failures]
    
    log_info(f"Combined {len(flat_results)} phases from all ranks")
    
    if flat_failures:
        log_warning(f"Total of {len(flat_failures)} chemical systems failed processing")
    
    # Create DataFrame
    if not flat_results:
        log_warning("No phases extracted - returning empty DataFrame")
        return pd.DataFrame()
    
    phases_df = pd.DataFrame(flat_results)
    log_info(f"Created DataFrame with {len(phases_df)} phases")
    
    # Remove duplicates based on material ID
    if 'material_id' in phases_df.columns:
        initial_count = len(phases_df)
        phases_df = phases_df.drop_duplicates(subset=['material_id'])
        removed_count = initial_count - len(phases_df)
        
        if removed_count > 0:
            log_info(f"Removed {removed_count} duplicate phases")
    
    log_info(f"Final competing phases database contains {len(phases_df)} unique phases")
    
    return phases_df

def validate_input_database(dataframe: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate input database has required columns and data.
    
    Args:
        dataframe: Input DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If validation fails
    """
    if dataframe.empty:
        raise ValueError("Input database is empty")
    
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values in required columns
    for col in required_columns:
        null_count = dataframe[col].isnull().sum()
        if null_count > 0:
            log_warning(f"Found {null_count} null values in column '{col}'", current_rank=rank)


def main() -> int:
    """
    Main function to orchestrate the competing phases extraction process.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    examples = """
Examples:
    Basic usage with MPI:
        mpirun -n 4 python get_convex_hull_compounds_mp_rester.py -d compounds.csv -o competing_phases.csv --api_key YOUR_API_KEY
    
    Single process:
        python get_convex_hull_compounds_mp_rester.py -d compounds.csv -o competing_phases.csv --api_key YOUR_API_KEY
    
    Custom composition column:
        mpirun -n 8 python get_convex_hull_compounds_mp_rester.py -d data.csv -o phases.csv \\
            --composition_column "formula" --api_key YOUR_API_KEY
    """
    
    parser = argparse.ArgumentParser(
        description="Extract competing phases from Materials Project database for convex hull analysis. "
                   "Processes compounds database to identify unique chemical phase space and "
                   "extracts stable phases for each system using MPI parallelization.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-d", "--database_candidate", type=str, required=True,
                       help="Path to compounds CSV file containing compositions for hull evaluation")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output CSV file path for competing phases database")
    parser.add_argument("--api_key", type=str, required=True,
                       help="API key for accessing the Materials Project database")
    
    # Optional arguments
    parser.add_argument("--composition_column", type=str, default="composition",
                       help="Column name containing chemical formulas")
    
    args = parser.parse_args()
    
    try:
        # Print MPI configuration
        print_mpi_info()
        
        # Load and validate input database
        log_info(f"Loading compound database from {args.database_candidate}...", current_rank=rank)
        
        if not os.path.exists(args.database_candidate):
            raise FileNotFoundError(f"Database file not found: {args.database_candidate}")
        
        compound_db = pd.read_csv(args.database_candidate, index_col=0)
        log_info(f"Loaded {len(compound_db)} compounds", current_rank=rank)
        
        # Validate input data
        validate_input_database(compound_db, [args.composition_column])
        
        # Extract unique chemical systems
        log_info("Extracting unique chemical systems...", current_rank=rank)
        chemical_systems = extract_unique_chemical_systems(
            compound_db, 
            composition_column=args.composition_column
        )
        
        # Extract competing phases using MPI
        log_info("Extracting competing phases from Materials Project database...", current_rank=rank)
        competing_phases_db = extract_competing_phases(chemical_systems, api_key=args.api_key)
        
        # Save results (only on rank 0)
        if rank == 0:
            if competing_phases_db is not None and not competing_phases_db.empty:
                log_info(f"Saving results to {args.output}...")
                competing_phases_db.to_csv(args.output, index=False)
                log_info("Competing phases extraction completed successfully!")
                
                # Print summary statistics
                print("\n=== Summary Statistics ===")
                print(f"Input compounds: {len(compound_db)}")
                print(f"Unique chemical systems: {len(chemical_systems)}")
                print(f"Competing phases found: {len(competing_phases_db)}")
                
            else:
                log_warning("No competing phases found - output file will be empty")
                pd.DataFrame().to_csv(args.output, index=False)
        
        return 0
        
    except Exception as e:
        log_warning(f"Error: {str(e)}", current_rank=rank)
        return 1


if __name__ == "__main__":
    exit(main())

  



