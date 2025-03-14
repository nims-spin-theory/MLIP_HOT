from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, FrechetCellFilter, StrainFilter
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator

from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np
import pandas as pd
import re, os, collections, ast, subprocess
import spglib
from tqdm import tqdm

from chgnet.model import CHGNet

def str_to_2d_array(string):
    if ',' not in string:
        string = string.replace(' ', ',')
    try:
        # Convert the string to a list of lists using ast.literal_eval
        list_of_lists = ast.literal_eval(string)

        # Convert the list of lists to a NumPy array and return
        return np.array(list_of_lists)
    except ValueError:
        # Handle the error if conversion is not possible
        return None

def clean_matrix(matrix, tol=1e-10):
    # matrix[np.abs(matrix) < tol] = 0
    matrix = np.round(matrix, decimals=6)
    return matrix
    
def symmetrize_structure(structure, symprec=0.01):
    '''
    convert non-premitive to premitive
    The premitive from this function is 'standard' in our databse
    '''
    cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers )
    try:
        lattice, scaled_positions, numbers = spglib.standardize_cell(cell, 
                                                                     to_primitive=True, 
                                                                     no_idealize=False, 
                                                                     symprec=symprec)
        spacegroup_symbol = spglib.get_spacegroup(cell, symprec=symprec)
        # return Atoms(numbers=numbers,cell=lattice,scaled_positions=scaled_positions), spacegroup_symbol
        return Structure(Lattice(lattice), numbers, scaled_positions), spacegroup_symbol
    except:
        return structure, None

def get_structure(system):
    '''
    system: pandas.Series of one compound ("row").
    return pymatgen Structure object.
    '''
    cell = str_to_2d_array(system['cell'])
    posi = str_to_2d_array(system['positions'])    
    try:
        atom = str_to_2d_array(system['numbers'] )
    except: 
        atom = str_to_2d_array(system['numbers'].replace(' ', ',') )
    lattice = Lattice(cell)
    structure = Structure(lattice, atom, posi)
    structure,spacegroup_symbol = symmetrize_structure(structure)
    
    return structure, spacegroup_symbol.split()[0]

def get_conven_structure(structure, spacegroup_symbol):
    '''
    convert primitive cell to conventional supercell (2fu).
    '''
    # if SpacegroupAnalyzer(structure).get_crystal_system() == "tetragonal":
    if spacegroup_symbol in ['I-4m2', 'I4/mmm']:
        scaling_matrix = np.array([[ 0.,  1.,  1.],
                                   [ 1.,  0.,  1.],
                                   [ 1.,  1.,  0.]])
        structure.make_supercell(scaling_matrix=scaling_matrix )
    elif spacegroup_symbol in ['F-43m', 'Fm-3m']:
        scaling_matrix = np.array([[ 0.,  0.,  1.],
                                   [ 1., -1.,  0.],
                                   [ 1.,  1., -1.]] )
        structure.make_supercell(scaling_matrix=scaling_matrix )

        # a = np.sqrt(2)/2
        # rotation_matrix = np.array([[a,a,0,],[-a,a,0],[0,0,1]])
        theta = np.radians(45)  # 45 degrees in radians
        rotation_matrix = [
            [ np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix, [0, 0, 0])
        structure.apply_operation(symmop)

        # Clean lattice
        cleaned_lattice = clean_matrix(structure.lattice.matrix.copy())
        cleaned_coords  = clean_matrix(structure.frac_coords.copy())
        
        # Recreate structure with clean lattice
        structure = Structure(Lattice(cleaned_lattice), 
                                      structure.species, 
                                      cleaned_coords)
        
    else:
        print('error,', spacegroup_symbol)

    LM = structure.lattice.matrix
    if np.count_nonzero(LM - np.diag(np.diagonal(LM)), )!=0:
        print('Lattice not diagonal.')
    elif LM[0,0]!=LM[1,1]:
        print('a!=b')

    return structure

def prepare_db(db):

    indexes = db[db['space group symbol']=='Fm-3m'].index
    db.loc[indexes, 'phase'] = 'cubic'
    indexes = db[db['space group symbol']=='F-43m'].index
    db.loc[indexes, 'phase'] = 'cubic'
    indexes = db[db['space group symbol']=='I4/mmm'].index
    db.loc[indexes, 'phase'] = 'tetra'
    indexes = db[db['space group symbol']=='I-4m2'].index
    db.loc[indexes, 'phase'] = 'tetra'
    # print(db['phase'].value_counts())

    cols = ['composition', 'type', 'labels', 'space group symbol',
            'cell', 'positions', 'numbers', 'UUID', 'phase', 
            'energy (eV/atom)', 
            'total magnetization (muB/f.u.)', 
            'local magnetization',
            ]
    
    db = db[cols].copy()
    return db

def opt_with_symmetry_mod(
    atoms_in: Atoms,
    calculator: Calculator,
    fix_symmetry: bool = False
) -> Atoms:

    atoms = atoms_in.copy()
    atoms.calc = calculator

    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])

    # ecf = FrechetCellFilter(atoms, hydrostatic_strain=False)
    ecf = StrainFilter(atoms)
    opt = FIRE(ecf, logfile=None, maxstep=0.01, downhill_check=True, Nmin=20)
    # opt = FIRE(ecf, maxstep=0.01, downhill_check=True, Nmin=20)
    opt.run(fmax=0.001, steps=200)

    return atoms
    
def opt_loop_row(row, model):

#    row, model = row_model

    # input model
    if model=='chgnet':
        from chgnet.model.dynamics import CHGNetCalculator
        calc   = CHGNetCalculator(use_device='cpu')
    elif model=='7net-0':
        from sevenn.calculator import SevenNetCalculator
        calc   = SevenNetCalculator(model=model, device='cpu')
    elif model=='7net-l3i5':
        from sevenn.calculator import SevenNetCalculator
        calc   = SevenNetCalculator(model=model, device='cpu')
    elif model=='mattersim':
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cpu")
    elif 'eqV2' in model:
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        calc = OCPCalculator(checkpoint_path=f'./fairchem_checkpoints/{model}.pt', cpu=True) 
    # get conventional cell with 2 fu
    structure, spacegroup_symbol = get_structure(row)
    structure = get_conven_structure(structure, spacegroup_symbol)
    
    # get DFT c/a/ca ratio
    DFT_a  = structure.lattice.matrix[0,0]
    DFT_c  = structure.lattice.matrix[2,2]
    DFT_ca = structure.lattice.matrix[2,2]/structure.lattice.matrix[0,0]

    structure.apply_strain([0.1,0.1,0.1])  # apply strain to test more realistic performance.
    
    atoms  = AseAtomsAdaptor.get_atoms(structure)
    # do relaxation
    atoms_opt = opt_with_symmetry_mod(atoms, calc, True)
    final_structure = AseAtomsAdaptor.get_structure(atoms_opt)

    # get ML c/a/ca ratio
    ML_a  = final_structure.lattice.matrix[0,0]
    ML_c  = final_structure.lattice.matrix[2,2]
    ML_ca = final_structure.lattice.matrix[2,2]/final_structure.lattice.matrix[0,0]
    ML_cell = str(final_structure.lattice.matrix.tolist())

    ML_e = atoms_opt.get_total_energy()/atoms_opt.get_global_number_of_atoms()
    # predict local mom by CHGNet
    prediction = chgnet.predict_structure(final_structure)
    ML_m = prediction['m']
    ML_m = str([float(i) for i in ML_m])

    # test_db.loc[ind, ['ML_e', 'ML_m']] = [ML_e, ML_m]
    
    return [ML_cell, ML_a, ML_c, ML_ca, DFT_a, DFT_c, DFT_ca, ML_e, ML_m]

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


import argparse
import multiprocessing as mp
from mpi4py import MPI
chgnet = CHGNet.load()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total number of processes

def scatter_dataframe(df):
    """Evenly distribute DataFrame rows across MPI processes."""
    data = df.to_dict(orient="records")
    total_rows = len(data)
    
    # Determine chunk sizes
    base_chunk_size = total_rows // size  # Minimum rows per process
    remainder = total_rows % size  # Extra rows to distribute

    # Create chunk assignments (each process gets base_chunk_size, plus one extra if remainder > 0)
    chunks = []
    start_idx = 0

    for i in range(size):
        extra_row = 1 if i < remainder else 0  # First 'remainder' processes get 1 extra row
        end_idx = start_idx + base_chunk_size + extra_row
        chunks.append(data[start_idx:end_idx])
        start_idx = end_idx

    return comm.scatter(chunks, root=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that test structure optmization via different ML-FF models.")
    
    parser.add_argument("-d", "--database_csv", type=str, required=True, help="Path to the db csv file.")
    parser.add_argument("-t", "--type",         type=str, required=True, help="type of compounds. all/full/inverse/half")
    parser.add_argument("-p", "--phase",        type=str, required=True, help="phase of compounds. all/cubic/tetra")
    parser.add_argument("-m", "--model",        type=str, required=True, help="ML-FF model  chgnet/7net-0/7net-l3i5/mattersim/\
                                                                               eqV2_31M_omat/eqV2_86M_omat/eqV2_153M_omat")
    parser.add_argument("-o", "--output",       type=str, required=True, help="output dir")
    parser.add_argument("-s", "--size",         type=int, required=True, help="The number of chunks. size>0")
    parser.add_argument("-r", "--rank",         type=int, required=True, help="The rank of chunk selected in this job.\
                                                                               0 <= rank <= size-1")
    args = parser.parse_args()

    db = pd.read_csv(args.database_csv, index_col=0)
    db = prepare_db(db)

    print('database shape: ', db.shape)
    if args.type!='all':
        db = db[db['type']==args.type].copy()
    if args.phase!='all':
        db = db[db['phase']==args.phase].copy()
    print('test database shape: ', db.shape)

    db_test   = db.copy()
    # db_test = db.sample(100).copy()

    db_test = chunk_dataframe(db_test, args.size, args.rank)
    print('size: ', args.size, 'rank: ', args.rank, )
    print('chunk length: ', db_test.shape)
    
    local_data = scatter_dataframe(db_test)  

    local_results = [opt_loop_row(row, args.model) for row in local_data]

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten list and store in DataFrame
        results = []
        for sublist in gathered_results:
            for item in sublist:
                results.append(item)

        # update db
        db_test[['ML_cell','ML_a', 'ML_c', 'ML_ca', 'DFT_a', 'DFT_c', 'DFT_ca', 'ML_e', 'ML_m']] \
             = results

        # save db 
        db_name = args.database_csv.split("/")[-1].split('.')[0]
        os.makedirs(args.output, exist_ok=True)
        db_test.to_csv(f'./{args.output}/{db_name}_{args.type}_{args.phase}_{args.size}_{args.rank}.csv')



