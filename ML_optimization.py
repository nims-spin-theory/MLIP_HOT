"""
Machine Learning Force Field Optimization Script

This script performs structure optimization using various ML-FF models
with support for parallel processing via MPI. It supports multiple ML 
models including CHGNet, SevenNet, MatterSim, eqV2, and HIENet.

Features:
- Multi-stage optimization with decreasing step sizes
- Stagnation detection to prevent infinite loops
- Symmetry preservation during optimization
- MPI parallelization for large datasets
- Support for Heusler compound conventional cell conversion
- Strain application for finding global minima

Usage:
    mpirun -n <num_processes> python ML_optimization.py -d database.csv -m model_name -o output_dir -s size -r rank

Author: [Author Name]
Date: 2025
"""

import argparse
import ast
import os
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import spglib
from mpi4py import MPI

# ASE imports
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator

# PyMatGen imports
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.ase import AseAtomsAdaptor

# Constants
DEFAULT_TOLERANCE = 1e-10
DEFAULT_SYMPREC = 0.01
DEFAULT_FMAX = 0.0001
DEFAULT_MAX_STEPS = 200

# Supported ML models
SUPPORTED_MODELS = {
    'chgnet', '7net-0', '7net-l3i5', '7net-mf-ompa', 'mattersim', 
    'hienet', 'esen_30m_oam'
}

# Models that support specific patterns
EQIV2_MODELS = {'eqV2_31M', 'eqV2_86M', 'eqV2_153M'}

def str_to_2d_array(string: str) -> Optional[np.ndarray]:
    """
    Convert a string representation of a 2D array to a NumPy array.
    
    Args:
        string: String representation of a 2D array
        
    Returns:
        NumPy array if conversion is successful, None otherwise
    """
    if ',' not in string:
        string = string.replace(' ', ',')
    try:
        list_of_lists = ast.literal_eval(string)
        return np.array(list_of_lists)
    except (ValueError, SyntaxError):
        return None


def clean_matrix(matrix: np.ndarray, decimals: int = 6) -> np.ndarray:
    """
    Clean matrix by rounding to specified decimal places.
    
    Args:
        matrix: Input matrix to clean
        decimals: Number of decimal places to round to
        
    Returns:
        Cleaned matrix
    """
    return np.round(matrix, decimals=decimals)
    
def symmetrize_structure(structure: Structure, symprec: float = DEFAULT_SYMPREC) -> Tuple[Structure, Optional[str]]:
    """
    Convert non-primitive to primitive structure.
    The primitive from this function is 'standard' in our database.
    
    Args:
        structure: Input pymatgen Structure
        symprec: Symmetry precision tolerance
        
    Returns:
        Tuple of (primitive structure, spacegroup symbol)
    """
    cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
    try:
        lattice, scaled_positions, numbers = spglib.standardize_cell(
            cell, 
            to_primitive=True, 
            no_idealize=False, 
            symprec=symprec
        )
        spacegroup_symbol = spglib.get_spacegroup(cell, symprec=symprec)
        return Structure(Lattice(lattice), numbers, scaled_positions), spacegroup_symbol
    except Exception as e:
        print(f"Warning: Could not symmetrize structure: {e}")
        return structure, None

def get_structure(system: pd.Series) -> Tuple[Structure, str]:
    """
    Convert pandas Series data to pymatgen Structure object.
    
    Args:
        system: pandas Series containing structure data (cell, positions, numbers)
        
    Returns:
        Tuple of (Structure object, spacegroup symbol)
    """
    cell = str_to_2d_array(system['cell'])
    positions = str_to_2d_array(system['positions'])
    
    try:
        numbers = str_to_2d_array(system['numbers'])
    except (KeyError, ValueError):
        numbers = str_to_2d_array(system['numbers'].replace(' ', ','))
    
    if cell is None or positions is None or numbers is None:
        raise ValueError("Could not parse structure data from system")
    
    lattice = Lattice(cell)
    structure = Structure(lattice, numbers, positions)
    structure, spacegroup_symbol = symmetrize_structure(structure)
    
    if spacegroup_symbol is None:
        return structure, "Unknown"
    
    return structure, spacegroup_symbol.split()[0]

def get_conven_structure(structure: Structure, spacegroup_symbol: str) -> Structure:
    """
    Convert primitive cell to conventional supercell (2 formula units).
    
    Args:
        structure: Input primitive structure
        spacegroup_symbol: Spacegroup symbol string
        
    Returns:
        Conventional structure
    """
    # Define scaling matrices for different space groups
    tetragonal_groups = ['I-4m2', 'I4/mmm']
    cubic_groups = ['F-43m', 'Fm-3m', 'Fd-3m']
    simple_cubic = ['Pm-3m']
    
    if spacegroup_symbol in tetragonal_groups:
        scaling_matrix = np.array([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]
        ])
        structure.make_supercell(scaling_matrix=scaling_matrix)
        
    elif spacegroup_symbol in cubic_groups:
        scaling_matrix = np.array([
            [0., 0., 1.],
            [1., -1., 0.],
            [1., 1., -1.]
        ])
        structure.make_supercell(scaling_matrix=scaling_matrix)

        # Apply 45-degree rotation
        theta = np.radians(45)
        rotation_matrix = [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix, [0, 0, 0])
        structure.apply_operation(symmop)

        # Clean and recreate structure
        cleaned_lattice = clean_matrix(structure.lattice.matrix.copy())
        cleaned_coords = clean_matrix(structure.frac_coords.copy())
        structure = Structure(Lattice(cleaned_lattice), structure.species, cleaned_coords)
        
    elif spacegroup_symbol in simple_cubic:
        pass  # No transformation needed
        
    else:
        print(f'Warning: Unsupported spacegroup {spacegroup_symbol}')

    # Validate lattice properties
    lattice_matrix = structure.lattice.matrix
    if np.count_nonzero(lattice_matrix - np.diag(np.diagonal(lattice_matrix))) != 0:
        print('Warning: Lattice not diagonal.')
    elif lattice_matrix[0, 0] != lattice_matrix[1, 1]:
        print('Warning: a != b')

    return structure

class StagnationFIRE(FIRE):
    """
    FIRE optimizer with stagnation detection.
    
    Stops optimization when force changes remain below threshold for a given window.
    """
    
    def __init__(self, atoms: Atoms, window: int = 10, delta: float = 1e-5, **kwargs):
        """
        Initialize StagnationFIRE optimizer.
        
        Args:
            atoms: ASE Atoms object
            window: Number of recent steps to track for stagnation detection
            delta: Tolerance for detecting stagnation
            **kwargs: Additional arguments passed to FIRE
        """
        super().__init__(atoms, **kwargs)
        self.window = window
        self.delta = delta
        self.history: List[float] = []

    def converged(self, forces: Optional[np.ndarray] = None) -> bool:
        """
        Check if optimization has converged or stagnated.
        
        Args:
            forces: Force array (optional)
            
        Returns:
            True if converged or stagnated, False otherwise
        """
        super().converged()
        
        if forces is None:
            forces = self.optimizable.get_forces()

        result = self.optimizable.converged(forces, self.fmax)
        
        if not result:
            fmax = sqrt((forces ** 2).sum(axis=1).max())
            self.history.append(fmax)

            if len(self.history) > self.window:
                self.history.pop(0)
                changes = np.abs(np.diff(self.history))
                
                if np.all(changes < self.delta):
                    print(f"Stopping FIRE due to stagnation in fmax. "
                          f"delta={self.delta}, window={self.window}")
                    result = True
                    
        return result
        
def opt_with_symmetry_mod(
    atoms_in: Atoms,
    calculator: Calculator,
    fix_symmetry: bool = False
) -> Atoms:
    """
    Perform structure optimization with optional symmetry constraints.
    
    Uses a multi-stage optimization approach with decreasing step sizes.
    
    Args:
        atoms_in: Input ASE Atoms object
        calculator: ML calculator to use
        fix_symmetry: Whether to apply symmetry constraints
        
    Returns:
        Optimized Atoms object
    """
    atoms = atoms_in.copy()
    atoms.calc = calculator

    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])

    ecf = FrechetCellFilter(atoms, hydrostatic_strain=False)
    
    # Multi-stage optimization with decreasing step sizes
    optimization_stages = [
        {'maxstep': 0.1, 'downhill_check': True, 'delta': 1e-4},
        {'maxstep': 0.01, 'downhill_check': True, 'delta': 1e-4},
        {'maxstep': 0.001, 'downhill_check': False, 'delta': 1e-5}
    ]
    
    for stage in optimization_stages:
        opt = StagnationFIRE(
            ecf, 
            logfile=None, 
            maxstep=stage['maxstep'],
            downhill_check=stage['downhill_check'],
            Nmin=5,
            window=10, 
            delta=stage['delta']
        )
        opt.run(fmax=DEFAULT_FMAX, steps=DEFAULT_MAX_STEPS)
    
    return atoms
    
def create_calculator(model: str) -> Calculator:
    """
    Create and return the appropriate ML calculator based on model name.
    
    Args:
        model: Name of the ML model to use
        
    Returns:
        Initialized calculator object
        
    Raises:
        ValueError: If model is not supported
        ImportError: If required dependencies are not available
    """
    try:
        if model == 'chgnet':
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator(use_device='cpu')
        
        elif model in ['7net-0', '7net-l3i5']:
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model=model, device='cpu')
        
        elif model == '7net-mf-ompa':
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model=model, device='cpu', modal='mpa')
        
        elif model == 'mattersim':
            from mattersim.forcefield import MatterSimCalculator
            return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cpu")
        
        elif 'eqV2' in model or 'esen_30m_oam' in model:
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator
            checkpoint_path = f'./fairchem_checkpoints/{model}.pt'
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            return OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)
        
        elif model == 'hienet':
            from hienet.hienet_calculator import HIENetCalculator
            return HIENetCalculator()
        
        else:
            raise ValueError(f"Unsupported model: {model}")
            
    except ImportError as e:
        raise ImportError(f"Could not import calculator for model '{model}': {e}") from e


def opt_loop_row(local_data: List, model: str, strain: List[float], heusler: bool = False) -> List:
    """
    Perform optimization loop on local data structures.
    
    Args:
        local_data: List of structure data to optimize
        model: Name of ML model to use
        strain: Strain values to apply [x, y, z]
        heusler: Whether to convert Heusler compounds to conventional cell (2 f.u.)
        
    Returns:
        List of optimization results for each structure
    """
    calc = create_calculator(model)
    local_results = []
    
    for row in local_data:
        try:
            structure, spacegroup_symbol = get_structure(row)
            
            # Convert to conventional cell for Heusler compounds
            if heusler:
                structure = get_conven_structure(structure, spacegroup_symbol)
            
            # Apply strain and re-symmetrize
            structure.apply_strain(strain)
            structure, spacegroup_symbol = symmetrize_structure(structure)
            
            # Store initial state after strain
            init_cell = str(structure.lattice.matrix.tolist())
            init_positions = str(structure.frac_coords.tolist())
            init_numbers = str(list(structure.atomic_numbers))
            
            # Convert to ASE and optimize
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms_opt = opt_with_symmetry_mod(atoms, calc, fix_symmetry=True)
            final_structure = AseAtomsAdaptor.get_structure(atoms_opt)
            
            # Extract results
            ml_cell = str(final_structure.lattice.matrix.tolist())
            ml_energy = atoms_opt.get_total_energy() / atoms_opt.get_global_number_of_atoms()
            
            local_results.append([init_cell, init_positions, init_numbers, ml_cell, ml_energy])
            
        except Exception as e:
            print(f"Error processing structure: {e}")
            # Add placeholder result to maintain list structure
            local_results.append([None, None, None, None, None])
    
    return local_results 

def chunk_dataframe(df: pd.DataFrame, size: int, rank: int) -> pd.DataFrame:
    """
    Splits a DataFrame into nearly equal-sized chunks and returns the specified rank-th chunk.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    size (int): The number of chunks to split the DataFrame into.
    rank (int): The rank (index) of the chunk to return (0-based index).

    Returns:
    pd.DataFrame: The chunk corresponding to the given rank.
    """
    if size <= 0:
        raise ValueError("Size must be greater than zero.")
    if rank < 0 or rank >= size:
        raise ValueError("Rank must be between 0 and size-1.")

    n = len(df)
    indices = np.linspace(0, n, size + 1, dtype=int)  # Compute chunk boundaries
    return df.iloc[indices[rank]:indices[rank + 1]]


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total number of processes


def scatter_dataframe(df: pd.DataFrame) -> List:
    """
    Evenly distribute DataFrame rows across MPI processes.
    
    Args:
        df: DataFrame to distribute
        
    Returns:
        List of data records for current process
    """
    data = df.to_dict(orient="records")
    total_rows = len(data)
    
    # Determine chunk sizes
    base_chunk_size = total_rows // size
    remainder = total_rows % size
    
    # Create chunk assignments
    chunks = []
    start_idx = 0
    
    for i in range(size):
        extra_row = 1 if i < remainder else 0
        end_idx = start_idx + base_chunk_size + extra_row
        chunks.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    return comm.scatter(chunks, root=0)

def main():
    """Main function to run the ML-FF optimization workflow."""
    parser = argparse.ArgumentParser(
        description="Structure optimization using different ML-FF models with MPI parallelization."
    )
    
    parser.add_argument(
        "-d", "--database_csv", 
        type=str, 
        required=True, 
        help="Path to the database CSV file."
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        required=True, 
        help="ML-FF model name (chgnet, 7net-0, 7net-l3i5, 7net-mf-ompa, "
             "mattersim, eqV2_31M/86M/153M_omat, eqV2_31M/86M/153M_omat_mp_salex, "
             "esen_30m_oam, hienet)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True, 
        help="Output directory path."
    )
    parser.add_argument(
        "-s", "--size", 
        type=int, 
        required=True, 
        help="Number of chunks for parallel processing (size > 0)."
    )
    parser.add_argument(
        "-r", "--rank", 
        type=int, 
        required=True, 
        help="Rank of chunk for this job (0 <= rank <= size-1)."
    )
    parser.add_argument(
        "--heusler2fu", 
        type=bool, 
        default=False, 
        help="Convert Heusler compounds to conventional cell with 2 f.u. before applying strain."
    )
    parser.add_argument(
        "--strain", 
        nargs=3, 
        type=float, 
        default=[0.0, 0.0, 0.0],
        help="Directional strain as three floats (e.g., 0.01 -0.02 0.00 for x/y/z). "
             "Default is no strain."
    )

    args = parser.parse_args()

    # Load and process database
    print(f"Loading database from {args.database_csv}")
    db = pd.read_csv(args.database_csv, index_col=0)
    print(f"Database shape: {db.shape}")

    # Chunk data for this process
    db_chunk = chunk_dataframe(db, args.size, args.rank)
    print(f"Process {rank}/{size}: Processing chunk of size {db_chunk.shape}")
    
    # Distribute data among MPI processes
    local_data = scatter_dataframe(db_chunk)
    
    # Perform optimization
    print(f"Starting optimization with model: {args.model}")
    local_results = opt_loop_row(local_data, args.model, args.strain, args.heusler2fu)
    
    # Gather results from all processes
    gathered_results = comm.gather(local_results, root=0)

    # Process results on root process
    if rank == 0:
        print("Gathering and saving results...")
        # Flatten results
        all_results = []
        for sublist in gathered_results:
            all_results.extend(sublist)

        # Update dataframe with results
        result_columns = ['init_cell', 'init_positions', 'init_numbers', 'ML_cell', 'ML_e']
        db_chunk[result_columns] = all_results

        # Save results
        db_name = os.path.splitext(os.path.basename(args.database_csv))[0]
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f'{db_name}_{args.size}_{args.rank}.csv')
        db_chunk.to_csv(output_file)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()



