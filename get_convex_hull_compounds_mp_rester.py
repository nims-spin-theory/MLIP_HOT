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
import time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from ase.formula import Formula
from mpi4py import MPI
# from pymatgen.ext.matproj import MPRester
from mp_api.client import MPRester
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
                                   composition_column: str = 'composition') -> Tuple[List[List[str]], Dict[str, pd.DataFrame]]:
    """
    Extract unique chemical systems (element combinations) from compound database.
    
    This function should only be called on rank 0, then broadcast to other ranks.
    
    Args:
        dataframe: DataFrame containing compound compositions
        composition_column: Column name containing chemical formulas
        
    Returns:
        Tuple of (unique_systems, system_to_compounds):
            - unique_systems: List of unique element combinations, each as a sorted list of element symbols
            - system_to_compounds: Dictionary mapping chemical system string to DataFrame of compounds
        
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
    system_to_indices = {}  # Map from chemical system string to list of DataFrame indices
    
    for idx, row in dataframe.iterrows():
        try:
            composition = row[composition_column]
            if pd.isna(composition):
                continue
                
            # Parse chemical formula to get element counts
            element_counts = Formula(composition).count()
            elements = sorted(list(element_counts.keys()))
            element_systems.append(elements)
            
            # Track which compounds belong to which chemical system
            system_key = '-'.join(elements)
            if system_key not in system_to_indices:
                system_to_indices[system_key] = []
            system_to_indices[system_key].append(idx)
            
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
    
    # Create mapping from chemical system to compounds DataFrame
    system_to_compounds = {}
    for system_key, indices in system_to_indices.items():
        system_to_compounds[system_key] = dataframe.loc[indices].copy()
    
    log_info(f"Found {len(unique_systems)} unique chemical systems", current_rank=rank)
    
    return unique_systems, system_to_compounds


def extract_phase_data(doc: Any, entry: Any, phase_diagram: PhaseDiagram) -> Optional[Dict[str, Any]]:
    """
    Extract relevant data from a Materials Project document and entry.
    
    Args:
        doc: Materials Project document containing structure and material_id
        entry: ComputedStructureEntry from Materials Project
        phase_diagram: PhaseDiagram object for calculating formation energy and hull distance
        
    Returns:
        Dictionary containing extracted phase data, or None if extraction fails
    """
    try:
        structure = doc.structure
        mp_id = doc.material_id.string
        run_type = entry.data['run_type']
        E_Form = phase_diagram.get_form_energy_per_atom(entry)          
        E_Hull = phase_diagram.get_e_above_hull(entry)
        
        phase_data = {
            'material_id': mp_id,
            'run_type': run_type,
            'MP Formation energy (eV/atom)': E_Form,
            'MP Hull distance (eV/atom)': E_Hull,
            'composition': str(structure.composition.reduced_formula),
            'cell': str(structure.lattice.matrix.tolist()),
            'positions': str(structure.frac_coords.tolist()),
            'numbers': str(list(structure.atomic_numbers)),
            'natoms': structure.num_sites,
        }
        
        return phase_data
        
    except Exception as e:
        log_warning(f"Failed to extract data from entry {entry.data.get('material_id', 'unknown')}: {e}")
        return None


def query_mp_with_retry(elements: List[str], api_key: str, max_retries: int = 4,
                        initial_delay: float = 3.0) -> Tuple[List[Any], List[Any], PhaseDiagram]:
    """
    Query Materials Project database with exponential backoff retry logic.
    
    Args:
        elements: List of element symbols for the chemical system
        api_key: Materials Project API key
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        
    Returns:
        Tuple of (docs, stable_entries, phase_diagram):
            - docs: List of Materials Project documents with structure data
            - stable_entries: List of stable ComputedStructureEntry objects
            - phase_diagram: PhaseDiagram object for the chemical system
        
    Raises:
        Exception: If all retries fail
    """
    elements_str = '-'.join(elements)
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            with MPRester(api_key) as mpr:
                # 1) Fetch entries with formation energies for the given chemical system
                entries = mpr.get_entries_in_chemsys(elements)

                # 2) Build phase diagram and get stable entries
                phase_diagram = PhaseDiagram(entries)
                stable_entries = phase_diagram.stable_entries

                # 3) Get material IDs for structure fetching
                mp_id_list = [entry.data.get('material_id') for entry in stable_entries]

            with MPRester(api_key) as mpr:
                # 4) Fetch representative structures from MP
                docs = mpr.materials.search(
                        material_ids=mp_id_list,
                        fields=["material_id", "structure",],
                    )

            return docs, stable_entries, phase_diagram
                
        except Exception as e:
            last_exception = e
            if attempt < max_retries :
                delay = initial_delay * (2 ** attempt)
                log_warning(
                    f"Rank {rank}: Query failed for {elements_str} (attempt {attempt + 1}/{max_retries}). Error: {e}. "
                    f"Retrying in {delay:.1f}s...",
                    current_rank=rank, rank=rank
                )
                time.sleep(delay)
            else:
                log_warning(
                    f"Rank {rank}: All {max_retries} attempts failed for {elements_str}.",
                    current_rank=rank, rank=rank
                )
    
    # If all retries failed, raise the last exception
    raise last_exception


def extract_competing_phases(chemical_systems: List[List[str]], api_key: str, 
                            system_to_compounds: Dict[str, pd.DataFrame] = None) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract competing phases from Materials Project database for given chemical systems.
    
    Uses MPI to distribute workload across multiple processes for efficiency.
    
    Args:
        chemical_systems: List of chemical systems, each as a list of element symbols
        api_key: Materials Project API key
        system_to_compounds: Dictionary mapping chemical system string to DataFrame of compounds
        
    Returns:
        Tuple of (phases_df, failed_compounds_df) on rank 0, None on other ranks:
            - phases_df: DataFrame containing competing phase data
            - failed_compounds_df: DataFrame containing failed compounds (original input format)
        
    Raises:
        Exception: If database connection or phase extraction fails
    """
    log_info(f"Processing {len(chemical_systems)} chemical systems across {size} MPI ranks", 
             current_rank=rank)
    
    comm.barrier()
    # Distribute workload among MPI ranks
    local_systems = [system for i, system in enumerate(chemical_systems) if i % size == rank]
    
    log_info(f"Rank {rank} processing {len(local_systems)} chemical systems", 
             current_rank=rank, rank=rank)
    
    local_results = []
    failed_systems = []
    
    # Process local chemical systems with progress bar (only show on rank 0)
    for elements in tqdm(local_systems, disable=(rank != 0), 
                        desc=f"Rank {rank} extracting phases"):
        try:
            docs, stable_entries, phase_diagram = query_mp_with_retry(elements, api_key)
            
            # Create mapping from material_id to doc and entry for efficient lookup
            doc_dict = {doc.material_id.string: doc for doc in docs}
            entry_dict = {entry.data.get('material_id'): entry for entry in stable_entries}
            
            # Extract phase data for each stable entry
            for entry in stable_entries:
                mp_id = entry.data.get('material_id')
                if mp_id in doc_dict:
                    phase_data = extract_phase_data(doc_dict[mp_id], entry, phase_diagram)
                    if phase_data is not None:
                        local_results.append(phase_data)
                    
        except Exception as e:
            failed_systems.append((elements, str(e)))
            log_warning(f"Failed to process system {elements}: {e}", current_rank=rank, rank=rank)
    
    log_info(f"Rank {rank} extracted {len(local_results)} phases", current_rank=rank, rank=rank)

    if failed_systems:
        log_warning(f"Rank {rank} failed to process {len(failed_systems)} systems", 
                   current_rank=rank, rank=rank)

    # Synchronize all ranks before gathering
    comm.barrier()
    log_info(f"Rank {rank} reached synchronization barrier", current_rank=rank, rank=rank)

    # Gather results from all MPI ranks
    all_results = comm.gather(local_results, root=0)
    all_failures = comm.gather(failed_systems, root=0)
    
    # Combine and process results on rank 0
    if rank == 0:
        return combine_phase_results(all_results, all_failures, chemical_systems, system_to_compounds)
    else:
        return None





def combine_phase_results(all_results: List[List[Dict]], 
                         all_failures: List[List[Tuple]],
                         chemical_systems: List[List[str]],
                         system_to_compounds: Dict[str, pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine phase results from all MPI ranks and create final DataFrame.
    
    Args:
        all_results: List of result lists from each MPI rank
        all_failures: List of failure lists from each MPI rank
        chemical_systems: Original list of chemical systems to verify completeness
        system_to_compounds: Dictionary mapping chemical system string to DataFrame of compounds
        
    Returns:
        Tuple of (phases_df, failed_compounds_df):
            - phases_df: Combined DataFrame with competing phase data
            - failed_compounds_df: DataFrame with failed compounds (original input format)
    """
    # Flatten results from all ranks
    flat_results = [item for rank_results in all_results for item in rank_results]
    flat_failures = [item for rank_failures in all_failures for item in rank_failures]
    
    log_info(f"Combined {len(flat_results)} phases from all ranks")
    
    # Verify completeness
    failed_systems_set = {tuple(sorted(system)) for system, _ in flat_failures}
    total_systems = len(chemical_systems)
    failed_count = len(failed_systems_set)
    success_count = total_systems - failed_count
    
    log_info(f"Successfully processed {success_count}/{total_systems} chemical systems")
    
    if flat_failures:
        log_warning(f"Failed to process {failed_count} chemical systems:")
        for system, error in flat_failures[:10]:  # Show first 10 failures
            log_warning(f"  {'-'.join(system)}: {error}")
        if len(flat_failures) > 10:
            log_warning(f"  ... and {len(flat_failures) - 10} more")
    
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
    
    # Create DataFrame for failed compounds (using original input format)
    if flat_failures and system_to_compounds is not None:
        failed_compounds_list = []
        failed_systems_set = set()
        
        for system, error in flat_failures:
            system_key = '-'.join(system)
            if system_key not in failed_systems_set and system_key in system_to_compounds:
                failed_systems_set.add(system_key)
                # Get all compounds from this failed system
                compounds_df = system_to_compounds[system_key]
                failed_compounds_list.append(compounds_df)
        
        if failed_compounds_list:
            failed_compounds_df = pd.concat(failed_compounds_list, ignore_index=False)
            # Remove duplicates based on index
            failed_compounds_df = failed_compounds_df[~failed_compounds_df.index.duplicated(keep='first')]
            log_info(f"Created failed compounds DataFrame with {len(failed_compounds_df)} compounds from {len(failed_systems_set)} failed systems")
        else:
            # If no compounds found in mapping, create empty DataFrame
            failed_compounds_df = pd.DataFrame()
    elif flat_failures and system_to_compounds is None:
        # Fallback: create simple error report if mapping not provided
        log_warning("system_to_compounds mapping not provided, creating simplified error report")
        failed_systems_data = []
        for system, error in flat_failures:
            failed_systems_data.append({
                'chemical_system': '-'.join(system),
                'elements': str(system),
                'error_message': error
            })
        failed_compounds_df = pd.DataFrame(failed_systems_data)
        failed_compounds_df = failed_compounds_df.drop_duplicates(subset=['chemical_system'])
    else:
        # No failures - create empty DataFrame
        failed_compounds_df = pd.DataFrame()
    
    return phases_df, failed_compounds_df

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
    parser.add_argument("--failed_systems_output", type=str, default=None,
                       help="Output CSV file path for failed chemical systems (optional)")
    
    args = parser.parse_args()
    
    try:
        # Print MPI configuration
        print_mpi_info()
        
        # Load input database only on rank 0, then broadcast
        if rank == 0:
            log_info(f"Loading compound database from {args.database_candidate}...")
            
            if not os.path.exists(args.database_candidate):
                raise FileNotFoundError(f"Database file not found: {args.database_candidate}")
            
            compound_db = pd.read_csv(args.database_candidate, index_col=0)
            log_info(f"Loaded {len(compound_db)} compounds")
            
            # Validate input data
            validate_input_database(compound_db, [args.composition_column])
            
            # Extract unique chemical systems
            log_info("Extracting unique chemical systems...")
            chemical_systems, system_to_compounds = extract_unique_chemical_systems(
                compound_db, 
                composition_column=args.composition_column
            )
        else:
            chemical_systems = None
            system_to_compounds = None
        
        # Broadcast chemical systems and compound mapping from rank 0 to all other ranks
        chemical_systems = comm.bcast(chemical_systems, root=0)
        system_to_compounds = comm.bcast(system_to_compounds, root=0)
        log_info(f"Rank {rank} received {len(chemical_systems)} chemical systems", current_rank=rank)
        
        # Extract competing phases using MPI
        log_info("Extracting competing phases from Materials Project database...", current_rank=rank)
        result = extract_competing_phases(chemical_systems, api_key=args.api_key, 
                                         system_to_compounds=system_to_compounds)
        
        # Save results (only on rank 0)
        if rank == 0:
            competing_phases_db, failed_systems_db = result
            
            if competing_phases_db is not None and not competing_phases_db.empty:
                log_info(f"Saving results to {args.output}...")
                competing_phases_db.to_csv(args.output, index=False)
                log_info("Competing phases extraction completed successfully!")
                
                # Print summary statistics
                print("\n=== Summary Statistics ===")
                print(f"Input compounds: {len(compound_db)}")
                print(f"Unique chemical systems: {len(chemical_systems)}")
                print(f"Unique competing phases found: {len(competing_phases_db)}")
                print(f"Failed chemical systems: {len(failed_systems_db)}")
                
            else:
                log_warning("No competing phases found - output file will be empty")
                pd.DataFrame().to_csv(args.output, index=False)
            
            # Save failed compounds if requested
            if args.failed_systems_output:
                if not failed_systems_db.empty:
                    log_info(f"Saving failed compounds to {args.failed_systems_output}...")
                    # Save with index to maintain consistency with input format
                    failed_systems_db.to_csv(args.failed_systems_output, index=True)
                    log_info(f"Saved {len(failed_systems_db)} failed compounds")
                else:
                    log_info("No failed compounds to save")
                    # Create empty DataFrame with same structure as input
                    pd.DataFrame().to_csv(args.failed_systems_output, index=True)
        
        return 0
        
    except Exception as e:
        log_warning(f"Error: {str(e)}", current_rank=rank)
        import traceback
        if rank == 0:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

  



