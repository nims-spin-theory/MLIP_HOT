"""
Convex Hull Phase Preparation for ML-predicted Formation Energies

This script prepares competing phase data for convex hull analysis by extracting
stable phases from the QMPY database for each unique chemical system found in
the input compound database. 

This script is based on qmpy_rester:
https://github.com/mohanliu/qmpy_rester

Thus, the installation of QMPY to local device is not required.

The script uses MPI for parallel processing to efficiently handle large datasets
and multiple chemical systems.

Example usage:
    mpirun -n 4 python get_convex_hull_compounds_qmpy_rester.py -d compounds.csv -o convex_phases.csv
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
from ase.formula import Formula
from mpi4py import MPI
from pymatgen.core import Lattice, Structure
import qmpy_rester as qr
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


def oqmd_entry_to_structure(entry: Dict) -> Structure:
    """
    Convert an OQMD-style entry dict (e.g., from QMPYRester/OQMD API) to a pymatgen Structure.

    Expected keys in `entry`:
      - "unit_cell": 3x3 list of lattice vectors in Å (fractional basis assumed for coords below)
      - "sites": list of site specs in either format:
          * "Fe @ 0 0 0" (element @ frac_x frac_y frac_z)
          * {"element": "Fe", "frac_coords": [0, 0, 0]}  # also accepts "coords" alias
    Optional (ignored for structure build): "spacegroup", "composition", etc.

    Returns
    -------
    Structure
        A pymatgen Structure with the given lattice and fractional coordinates.
    """
    if "unit_cell" not in entry or "sites" not in entry:
        raise KeyError("Entry must contain 'unit_cell' and 'sites'.")

    # 1) Lattice
    lattice = Lattice(entry["unit_cell"])

    # 2) Parse sites
    species: List[str] = []
    frac_coords: List[List[float]] = []

    def _parse_site(site: Union[str, Dict]) -> Tuple[str, List[float]]:
        if isinstance(site, str):
            # format: "Fe @ 0 0 0"
            elem, sep, xyz = site.partition("@")
            if not sep:
                raise ValueError(f"Site string missing '@': {site}")
            elem = elem.strip()
            coords = [float(v) for v in xyz.strip().replace(",", " ").split()]
            if len(coords) != 3:
                raise ValueError(f"Expected 3 fractional coords, got {coords} in site: {site}")
            return elem, coords

        elif isinstance(site, dict):
            # format: {"element": "Fe", "frac_coords": [0,0,0]} or {"element":"Fe","coords":[...]}
            if "element" not in site:
                raise KeyError(f"Site dict missing 'element': {site}")
            elem = str(site["element"]).strip()

            # Prefer explicit frac coords; fallback to generic "coords" (assumed fractional)
            coords_key = "frac_coords" if "frac_coords" in site else "coords"
            if coords_key not in site:
                raise KeyError(f"Site dict missing 'frac_coords' or 'coords': {site}")

            coords_raw = site[coords_key]
            if len(coords_raw) != 3:
                raise ValueError(f"Expected 3 fractional coords, got {coords_raw} in site: {site}")
            coords = [float(v) for v in coords_raw]
            return elem, coords

        else:
            raise TypeError(f"Unsupported site type: {type(site)}; value: {site}")

    for s in entry["sites"]:
        elem, coords = _parse_site(s)
        species.append(elem)
        frac_coords.append(coords)

    # 3) Build structure (coords are fractional by construction)
    struct = Structure(lattice, species, frac_coords, coords_are_cartesian=False)

    return struct


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


def extract_phase_data(structure) -> Optional[Dict[str, Any]]:
    """
    Extract relevant data from a QMPY structure object.
    
    Args:
        structure: QMPY structure object
        
    Returns:
        Dictionary containing extracted phase data, or None if extraction fails
    """
    try:
        # Parse structure from POSCAR
        pymatgen_structure = oqmd_entry_to_structure(structure)
        
        phase_data = {
            'composition': str(pymatgen_structure.composition.reduced_formula),
            'cell': str(pymatgen_structure.lattice.matrix.tolist()),
            'positions': str(pymatgen_structure.frac_coords.tolist()),
            'numbers': str(list(pymatgen_structure.atomic_numbers)),
            'entry_id': structure['entry_id'],
            'calculation_id': structure['calculation_id'],
            'OQMD Formation energy (eV/atom)': structure['delta_e'],
            'OQMD Hull distance (eV/atom)': structure['stability'],
            'natoms': structure['natoms'],
            'spacegroup': structure['spacegroup']
        }
        
        return phase_data
        
    except Exception as e:
        log_warning(f"Failed to extract data from structure {structure}: {e}")
        return None


def query_qmpy_with_retry(elements: List[str], max_retries: int = 4, 
                          initial_delay: float = 3.0) -> List[Dict]:
    """
    Query QMPY database with exponential backoff retry logic.
    
    Args:
        elements: List of element symbols for the chemical system
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        
    Returns:
        List of structure data from QMPY
        
    Raises:
        Exception: If all retries fail
    """
    elements_str = '-'.join(elements)
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            with qr.QMPYRester() as q:
                kwargs = {
                    "composition": elements_str,
                    "stability": "0",
                    "natom": "<10",
                }

                result = q.get_oqmd_phases(verbose=False, **kwargs)
                return result.get('data', [])
                
        except Exception as e:
            last_exception = e
            if attempt < max_retries :
                delay = initial_delay * (2 ** attempt)
                log_warning(
                    f"Rank {rank}: Query failed for {elements_str} (attempt {attempt + 1}/{max_retries}). "
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


def extract_competing_phases(chemical_systems: List[List[str]], 
                            system_to_compounds: Dict[str, pd.DataFrame] = None) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract competing phases from QMPY database for given chemical systems.
    
    Uses MPI to distribute workload across multiple processes for efficiency.
    
    Args:
        chemical_systems: List of chemical systems, each as a list of element symbols
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
            list_of_data = query_qmpy_with_retry(elements)

            for structure in list_of_data:
                phase_data = extract_phase_data(structure)
                if phase_data:
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
        Tuple of (phases_df, failed_systems_df):
            - phases_df: Combined DataFrame with competing phase data
            - failed_systems_df: DataFrame with failed compounds (original input format)
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
    
    # Remove duplicates based on calculation ID
    if 'calculation_id' in phases_df.columns:
        initial_count = len(phases_df)
        phases_df = phases_df.drop_duplicates(subset=['calculation_id'])
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
        mpirun -n 4 python get_convex_hull_compounds_qmpy_rester.py -d compounds.csv -o competing_phases.csv
    
    Single process:
        python get_convex_hull_compounds_qmpy_rester.py -d compounds.csv -o competing_phases.csv
    
    Custom composition column:
        mpirun -n 8 python get_convex_hull_compounds_qmpy_rester.py -d data.csv -o phases.csv \\
            --composition_column "formula"
    """
    
    parser = argparse.ArgumentParser(
        description="Extract competing phases from QMPY database for convex hull analysis. "
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
        log_info("Extracting competing phases from QMPY database using Rester...", current_rank=rank)
        result = extract_competing_phases(chemical_systems, system_to_compounds)
        
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

  



