import os
import pymysql
pymysql.install_as_MySQLdb()

os.environ["qmdb_v1_1_name"] = "oqmd__v1_6"
os.environ["qmdb_v1_1_user"] = "enda"
os.environ["qmdb_v1_1_host"] = "localhost"
os.environ["qmdb_v1_1_pswd"] = "password"

import pandas as pd
from ase.formula import Formula
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from qmpy import PhaseSpace
from tqdm import tqdm
import argparse
import re

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_unique_elements_list(db):
    unique_elements_list = []
    for ind, row in db.iterrows():
        composition = row['composition']
        elem_counts = Formula(composition).count()
        unique_elements = list(elem_counts.keys())
        unique_elements_list.append(unique_elements)
    print('All compounds: ', len(unique_elements_list))

    unique_elements_list = set(tuple(sorted(sublist)) for sublist in unique_elements_list)
    unique_elements_list = [list(t) for t in unique_elements_list]
    print('Unique phase space: ', len(unique_elements_list))
    
    return unique_elements_list

def get_db_convex(unique_elements_list):
    db_convex = pd.DataFrame()

    # Scatter workload among ranks
    local_list = [el for i, el in enumerate(unique_elements_list) if i % size == rank]
    
    local_results = []
    for unique_elements in tqdm(local_list, disable=(rank != 0)):
        phase = PhaseSpace(unique_elements)
        for structure in phase.stable:
            data = {
                'name': structure.name,
                'cell': str(Structure.from_str(structure.calculation.POSCAR, fmt='poscar').lattice.matrix.tolist()),
                'positions': str(Structure.from_str(structure.calculation.POSCAR, fmt='poscar').frac_coords.tolist()),
                'numbers': str(list(Structure.from_str(structure.calculation.POSCAR, fmt='poscar').atomic_numbers)),
                'calculation.id': structure.calculation.id,
                'calculation.energy': structure.calculation.energy,
                'formation.delta_e': structure.formation.delta_e,
                'element': structure.natoms == 1
            }
            local_results.append(data)

    # Gather data from all processes
    all_data = comm.gather(local_results, root=0)

    if rank == 0:
        flat_data = [item for sublist in all_data for item in sublist]
        db_convex = pd.DataFrame(flat_data)
        print('Competing phases in all phase space: ', db_convex.shape[0])
        db_convex = db_convex.drop_duplicates()
        print('After removing duplicates: ', db_convex.shape[0])
        return db_convex
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use ML formation energy of compounds and competing phases to evaluate Hull distance. Energy is in energy per atom."
    )
    
    parser.add_argument("-d", "--database_candidate",  type=str, required=True, help="compounds db csv file containing compounds for hull evaluation. ")
    
    parser.add_argument("-o", "--output",              type=str, required=True, help="Output, convex hull competing phase compounds joblist csv file name.")
        
    args = parser.parse_args()

    db = pd.read_csv(args.database_candidate, index_col=0)
    
    unique_elements_list = get_unique_elements_list(db)
    db_convex = get_db_convex(unique_elements_list)

    if rank == 0: db_convex.to_csv(args.output)

  



