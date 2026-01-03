"""
Convex Hull Distance Evaluator for ML-predicted Formation Energies

This script calculates the distance to the convex hull (hull distance) for 
compounds using ML-predicted formation energies. The hull distance indicates
thermodynamic stability - compounds on the hull (distance = 0) are stable,
while positive distances indicate metastable compounds.

The script uses MPI for parallel processing to efficiently handle large datasets.

Example usage:
    mpirun -n 4 python MLIP_hull.py -i candidates.csv -c convex_phases.csv -o results.csv
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from ase.formula import Formula
from mpi4py import MPI
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.core.composition import Composition
from tqdm import tqdm

# Configuration constants
DEFAULT_COLUMNS = {
    'candidate_formula': 'composition',
    'convex_formula': 'name',
    'candidate_formation_energy': 'ML_formE',
    'convex_formation_energy': 'ML_formE',
    'hull_distance': 'ML_hull'
}

REQUIRED_CANDIDATE_COLUMNS = ['composition', 'ML_formE']
REQUIRED_CONVEX_COLUMNS = ['name', 'ML_formE']


def log_info(message: str, rank: int = 0, current_rank: int = 0) -> None:
    """Log information message only from specified rank."""
    if current_rank == rank:
        print(f"[INFO] {message}")


def log_warning(message: str, rank: int = 0, current_rank: int = 0) -> None:
    """Log warning message only from specified rank."""
    if current_rank == rank:
        print(f"[WARNING] {message}")


def print_mpi_info(rank: int, size: int) -> None:
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


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
            log_warning(f"{dataframe_name} has {null_count} null values in column '{col}'", 
                       current_rank=rank)

def create_energy_dictionary(dataframe: pd.DataFrame, key_column: str = 'name', 
                           value_column: str = 'ML_formE') -> Dict[str, float]:
    """
    Create a dictionary mapping compound names to their formation energies.
    
    Args:
        dataframe: DataFrame containing competing phase data
        key_column: Column name containing compound formulas
        value_column: Column name containing formation energy values
        
    Returns:
        Dictionary mapping compound formulas to formation energies
        
    Raises:
        KeyError: If specified columns don't exist in the DataFrame
        ValueError: If there are duplicate compounds or invalid data
    """
    if key_column not in dataframe.columns:
        raise KeyError(f"Column '{key_column}' not found in DataFrame")
    if value_column not in dataframe.columns:
        raise KeyError(f"Column '{value_column}' not found in DataFrame")
    
    # Check for duplicates
    if dataframe[key_column].duplicated().any():
        duplicates = dataframe[dataframe[key_column].duplicated()][key_column].tolist()
        log_warning(f"Duplicate compounds found in convex hull database: {duplicates[:5]}", 
                   current_rank=rank)
    
    return dict(zip(dataframe[key_column], dataframe[value_column]))


def extract_elements_from_formula(formula: str) -> List[str]:
    """
    Extract element symbols from a chemical formula.
    
    Args:
        formula: Chemical formula string (e.g., 'Li2O', 'NaCl')
        
    Returns:
        List of unique element symbols
    """
    # Match element symbols (capital letter followed by optional lowercase)
    elements = re.findall(r'[A-Z][a-z]?', formula)
    return list(set(elements))  # Remove duplicates


def create_phase_diagram_entries(compositions: List[str], 
                               energies: List[float]) -> List[PDEntry]:
    """
    Create PDEntry objects for phase diagram construction.
    
    Args:
        compositions: List of chemical formula strings
        energies: List of total energies (not per atom)
        
    Returns:
        List of PDEntry objects for phase diagram construction
        
    Raises:
        ValueError: If composition cannot be parsed or lists have different lengths
    """
    if len(compositions) != len(energies):
        raise ValueError(f"Compositions ({len(compositions)}) and energies ({len(energies)}) "
                        "must have the same length")
    
    pd_entries = []
    failed_entries = []
    
    for composition, energy in zip(compositions, energies):
        try:
            pd_entry = PDEntry(Composition(composition), energy)
            pd_entries.append(pd_entry)
        except Exception as e:
            failed_entries.append((composition, str(e)))
    
    if failed_entries:
        log_warning(f"Failed to create PDEntry for {len(failed_entries)} compositions", 
                   current_rank=rank)
    
    return pd_entries

def calculate_hull_distance(composition: str, formation_energy_per_atom: float, 
                           convex_phases: Dict[str, float]) -> Tuple[Any, float]:
    """
    Calculate the distance to the convex hull for a given compound.
    
    Args:
        composition: Chemical formula of the compound
        formation_energy_per_atom: Formation energy per atom of the compound
        convex_phases: Dictionary mapping competing phase formulas to formation energies
        
    Returns:
        Tuple of (decomposition, hull_distance)
        - decomposition: Dictionary of competing phases and their fractions
        - hull_distance: Distance to convex hull (0 = stable, >0 = metastable)
        
    Raises:
        ValueError: If composition cannot be parsed or phase diagram construction fails
    """
    try:
        # Parse composition to get elements and atom counts
        element_counts = Formula(composition).count()
        elements = list(element_counts.keys())
        total_atoms = sum(element_counts.values())
        
        # Pure elements are stable by definition (hull distance = 0)
        if len(elements) == 1:
            return {composition: 1.0}, 0.0
        
        # Filter competing phases to include only those with relevant elements
        relevant_phases = {}
        for phase_formula, phase_energy in convex_phases.items():
            try:
                phase_elements = extract_elements_from_formula(phase_formula)
                # Include phase if all its elements are in the target composition
                if set(phase_elements).issubset(set(elements)):
                    relevant_phases[phase_formula] = phase_energy
            except Exception as e:
                log_warning(f"Failed to parse competing phase formula '{phase_formula}': {e}", 
                           current_rank=rank)
                continue
        
        if not relevant_phases:
            log_warning(f"No relevant competing phases found for {composition}", 
                       current_rank=rank)
            return {composition: 1.0}, float('inf')  # Cannot construct hull
        
        # Prepare compositions and total energies for phase diagram
        phase_compositions = list(relevant_phases.keys())
        phase_total_energies = []
        
        for phase_formula in phase_compositions:
            try:
                phase_element_counts = Formula(phase_formula).count()
                phase_total_atoms = sum(phase_element_counts.values())
                total_energy = relevant_phases[phase_formula] * phase_total_atoms
                phase_total_energies.append(total_energy)
            except Exception as e:
                log_warning(f"Failed to calculate total energy for phase '{phase_formula}': {e}", 
                           current_rank=rank)
                continue
        
        # Add the target composition to the phase diagram
        phase_compositions.append(composition)
        target_total_energy = formation_energy_per_atom * total_atoms
        phase_total_energies.append(target_total_energy)
        
        # Create phase diagram entries
        pd_entries = create_phase_diagram_entries(phase_compositions, phase_total_energies)
        
        if len(pd_entries) < 2:
            log_warning(f"Insufficient valid phases for hull calculation of {composition}", 
                       current_rank=rank)
            return {composition: 1.0}, float('inf')
        
        # Construct phase diagram and calculate hull distance
        phase_diagram = PhaseDiagram(pd_entries)
        target_entry = pd_entries[-1]  # Last entry is the target composition
        
        decomposition, hull_distance = phase_diagram.get_decomp_and_e_above_hull(target_entry)
        
        return decomposition, hull_distance
        
    except Exception as e:
        log_warning(f"Error calculating hull distance for {composition}: {e}", 
                   current_rank=rank)
        return {composition: 1.0}, float('inf')

def calculate_hull_distances_parallel(dataframe: pd.DataFrame, convex_phases: Dict[str, float],
                                     formula_column: str = 'composition', 
                                     formation_energy_column: str = 'ML_formE',
                                     output_column: str = 'ML_hull') -> Optional[pd.DataFrame]:
    """
    Calculate hull distances for all compounds using MPI parallelization.
    
    Args:
        dataframe: DataFrame containing candidate compounds
        convex_phases: Dictionary mapping competing phase formulas to formation energies
        formula_column: Column name containing chemical formulas
        formation_energy_column: Column name containing formation energies
        output_column: Column name for storing hull distances
        
    Returns:
        DataFrame with hull distances added (only on rank 0), None on other ranks
    """
    log_info(f"Calculating hull distances for {len(dataframe)} compounds using {size} MPI ranks", 
             current_rank=rank)
    
    # Distribute workload among MPI ranks
    local_indices = list(range(rank, len(dataframe), size))
    local_results = []
    failed_calculations = []
    
    log_info(f"Rank {rank} processing {len(local_indices)} compounds", current_rank=rank)
    
    # Process local compounds with progress tracking
    progress_interval = max(1, len(local_indices) // 10)  # Report every 10%
    
    for progress, idx in enumerate(local_indices):
        try:
            row = dataframe.iloc[idx]
            composition = row[formula_column]
            formation_energy = row[formation_energy_column]
            
            # Skip if data is missing
            if pd.isna(composition) or pd.isna(formation_energy):
                local_results.append((idx, float('nan')))
                continue
            
            # Calculate hull distance
            _, hull_distance = calculate_hull_distance(composition, formation_energy, convex_phases)
            local_results.append((idx, hull_distance))
            
            # Progress reporting for rank 0
            if rank == 0 and progress % progress_interval == 0:
                percentage = (progress / len(local_indices)) * 100
                log_info(f"Progress: {progress}/{len(local_indices)} ({percentage:.1f}%)")
                
        except Exception as e:
            log_warning(f"Failed to calculate hull distance for row {idx}: {e}", current_rank=rank)
            local_results.append((idx, float('nan')))
            failed_calculations.append((idx, composition))
    
    log_info(f"Rank {rank} completed {len(local_results)} calculations", current_rank=rank)
    
    if failed_calculations:
        log_warning(f"Rank {rank} failed to process {len(failed_calculations)} compounds", 
                   current_rank=rank)
    
    # Gather results from all MPI ranks
    all_results = comm.gather(local_results, root=0)
    all_failures = comm.gather(failed_calculations, root=0)
    
    # Combine results on rank 0
    if rank == 0:
        return combine_hull_results(dataframe, all_results, all_failures, output_column)
    else:
        return None


def combine_hull_results(dataframe: pd.DataFrame, all_results: List[List[Tuple]], 
                        all_failures: List[List[Tuple]], output_column: str) -> pd.DataFrame:
    """
    Combine hull distance results from all MPI ranks.
    
    Args:
        dataframe: Original DataFrame
        all_results: List of result lists from each MPI rank
        all_failures: List of failure lists from each MPI rank
        output_column: Column name for storing hull distances
        
    Returns:
        DataFrame with hull distances added
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = dataframe.copy()
    
    # Flatten results from all ranks
    flat_results = [item for rank_results in all_results for item in rank_results]
    flat_failures = [item for rank_failures in all_failures for item in rank_failures]
    
    log_info(f"Processing {len(flat_results)} hull distance results")
    
    # Apply results to DataFrame
    for idx, hull_distance in flat_results:
        result_df.loc[idx, output_column] = hull_distance
    
    # Report statistics
    valid_hulls = result_df[output_column].dropna()
    if len(valid_hulls) > 0:
        stable_count = (valid_hulls == 0.0).sum()
        metastable_count = (valid_hulls > 0.0).sum()
        
        log_info("Hull distance calculation completed!")
        log_info(f"Results summary:")
        log_info(f"  Total compounds: {len(result_df)}")
        log_info(f"  Successful calculations: {len(valid_hulls)}")
        log_info(f"  Stable compounds (hull = 0): {stable_count}")
        log_info(f"  Metastable compounds (hull > 0): {metastable_count}")
        
        if len(valid_hulls) > 0:
            log_info(f"  Mean hull distance: {valid_hulls.mean():.4f}")
            log_info(f"  Max hull distance: {valid_hulls.max():.4f}")
    
    if flat_failures:
        log_warning(f"Failed to process {len(flat_failures)} compounds")
    
    return result_df


def main() -> int:
    """
    Main function to orchestrate the hull distance calculation process.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    examples = """
Examples:
    Basic usage with MPI:
        mpirun -n 4 python MLIP_hull.py -i candidates.csv -c convex_phases.csv -o results.csv
    
    Single process:
        python MLIP_hull.py -i candidates.csv -c convex_phases.csv -o results.csv
    
    Custom column names:
        mpirun -n 8 python MLIP_hull.py -i data.csv -c phases.csv -o hull_results.csv \
            --composition_column_input "formula" \
            --formE_column_input "formation_energy"
    """
    
    parser = argparse.ArgumentParser(
        description="Calculate convex hull distances for compounds using ML-predicted "
                   "formation energies. Hull distance indicates thermodynamic stability - "
                   "compounds on the hull (distance = 0) are stable, while positive "
                   "distances indicate metastable compounds.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Path to CSV file containing candidate compounds for hull evaluation")
    parser.add_argument("-c", "--database_convex", type=str, required=True,
                       help="Path to CSV file containing competing phases with formation energies")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output CSV file path for results with hull distances")
    
    # Optional arguments for column names
    parser.add_argument("--composition_column_input", type=str, default="optimized_formula",
                       help="Column name containing compound compositions in input database")
    parser.add_argument("--composition_column_convex", type=str, default="composition",
                       help="Column name containing compound compositions in convex phases database")
    parser.add_argument("--formE_column_input", type=str, default="Formation Energy (eV/atom)",
                       help="Column name containing formation energies in input database")
    parser.add_argument("--formE_column_convex", type=str, default="Formation Energy (eV/atom)",
                       help="Column name containing formation energies in convex phases database")
    parser.add_argument("--out_column", type=str, default="Hull Distance (eV/atom)",
                       help="Column name for storing calculated hull distances")
    
    args = parser.parse_args()
    
    # Pre-flight info
    log_info(f"Compound input file: {args.input}", current_rank=rank)
    log_info(f"Convex hull file: {args.database_convex}", current_rank=rank)
    log_info(f"Output file: {args.output}", current_rank=rank)
    log_info(f"From input file, composition and formation energy are loaded from columns:", current_rank=rank)
    log_info(f"    '{args.composition_column_input}',{args.formE_column_input}", current_rank=rank)
    log_info(f"From elements file, composition and energy are loaded from columns:", current_rank=rank)
    log_info(f"    '{args.composition_column_convex}',{args.formE_column_convex}", current_rank=rank)
    log_info(f"The calculated formation energy will be stored in column:", current_rank=rank)
    log_info(f"    '{args.out_column}'", current_rank=rank)

    try:
        # Print MPI configuration
        print_mpi_info(rank, size)
        
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Candidate database file not found: {args.input}")
        
        candidates_db = pd.read_csv(args.input)
        log_info(f"Loaded {len(candidates_db)} candidate compounds", current_rank=rank)
        
        # Validate candidate database
        required_candidate_cols = [args.composition_column_input, args.formE_column_input]
        validate_dataframe(candidates_db, required_candidate_cols, "Candidate compounds database")
        
        
        if not os.path.exists(args.database_convex):
            raise FileNotFoundError(f"Convex phases database file not found: {args.database_convex}")
        
        convex_db = pd.read_csv(args.database_convex, index_col=0)
        log_info(f"Loaded {len(convex_db)} competing phases on convex hull", current_rank=rank)
        
        # Validate convex phases database
        required_convex_cols = [args.composition_column_convex, args.formE_column_convex]
        validate_dataframe(convex_db, required_convex_cols, "Competing phases database")
        
        # Create convex phases energy dictionary
        log_info("Creating competing phases energy dictionary...", current_rank=rank)
        convex_phases = create_energy_dictionary(
            convex_db,
            key_column=args.composition_column_convex,
            value_column=args.formE_column_convex
        )
        log_info(f"Competing phases dictionary contains {len(convex_phases)} entries", 
                current_rank=rank)
        
        # Calculate hull distances using MPI
        log_info("Calculating hull distances...", current_rank=rank)
        result_db = calculate_hull_distances_parallel(
            candidates_db,
            convex_phases,
            formula_column=args.composition_column_input,
            formation_energy_column=args.formE_column_input,
            output_column=args.out_column
        )
        
        # Save results (only on rank 0)
        if rank == 0:
            if result_db is not None:
                log_info(f"Saving results to {args.output}...")
                output_dir = os.path.dirname(os.path.abspath(args.output))
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    log_info(f"Created output directory: {output_dir}")
                result_db.to_csv(args.output, index=False)
                log_info("Hull distance calculation completed successfully!")
            else:
                log_warning("No results to save")
        
        return 0
        
    except Exception as e:
        log_warning(f"Error: {str(e)}", current_rank=rank)
        return 1


if __name__ == "__main__":
    exit(main())









