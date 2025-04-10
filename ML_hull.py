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


def get_unique_elements_list(db):
    unique_elements_list = []
    for ind, row in db.iterrows():
        composition = row['composition']
        elem_counts = Formula(composition).count()
        unique_elements = list(elem_counts.keys())
        unique_elements_list.append(unique_elements)
    print('All compounds: ', len(unique_elements_list))

    unique_elements_list = list(set(tuple(sublist) for sublist in unique_elements_list))
    unique_elements_list = [list(t) for t in unique_elements_list]
    print('Unique phase space: ', len(unique_elements_list))
    
    return unique_elements_list

def get_db_convex(unique_elements_list):
    db_convex = pd.DataFrame()

    ind = 0 
    for unique_elements in tqdm(unique_elements_list):
        phase = PhaseSpace(unique_elements)
        for structure in phase.stable:
            db_convex.loc[ind, 'name'] = structure.name

            cell = Structure.from_str(structure.calculation.POSCAR, fmt='poscar')
            db_convex.loc[ind, 'cell']      = str(cell.lattice.matrix.tolist())
            db_convex.loc[ind, 'positions'] = str(cell.frac_coords.tolist())
            db_convex.loc[ind, 'numbers']   = str(list(cell.atomic_numbers))

            db_convex.loc[ind, 'calculation.id']     = structure.calculation.id 
            db_convex.loc[ind, 'calculation.energy'] = structure.calculation.energy 
            db_convex.loc[ind, 'formation.delta_e'] = structure.formation.delta_e 
            
            if structure.natoms==1:
                db_convex.loc[ind, 'element']= True
            else:
                db_convex.loc[ind, 'element']= False
            ind = ind + 1
            # break
        # break
        
    print('Competing phases in all phase space: ', db_convex.shape[0])
    db_convex = db_convex.drop_duplicates()
    print('After removing duplicates: ', db_convex.shape[0])
    
    return db_convex


def get_dict_energy(db, key='element', value='ML_e'):
    return dict(zip(db[key], db[value]))


def make_pdentry_list(list_compositions, list_energies):
    pdentry_list = []
    for name, energy in zip(list_compositions, list_energies):
        pdentry = PDEntry(Composition(name), energy)
        pdentry_list.append(pdentry)
    return pdentry_list

def get_hull_distance(composition, energy_per_atom, dict_convex):
    elem_counts = Formula(composition).count()
    unique_elements = list(elem_counts.keys())
    natom = sum(list(elem_counts.values()))

    if len(unique_elements) == 1:
        return 0.0

    phase = PhaseSpace(unique_elements)

    list_compositions = []
    list_formation_energies = []
    for structure in phase.stable:
        list_compositions.append(structure.name)
        # list_formation_energies.append(structure.formation.delta_e * structure.natoms)
        # list_formation_energies.append(dict_convex[structure.calculation.id] * structure.natoms)
        list_formation_energies.append(dict_convex[structure.name] * structure.natoms)

    list_compositions.append(composition)
    list_formation_energies.append(energy_per_atom * natom)

    pdentries = make_pdentry_list(list_compositions, list_formation_energies)
    phasediagram = PhaseDiagram(pdentries)
    decomp, e_above_hull = phasediagram.get_decomp_and_e_above_hull(pdentries[-1])

    return decomp, e_above_hull

def update_hull_distance(db, dict_convex, col_formula='composition', col_E='ML_formE'):
    for ind, row in tqdm(db.iterrows(), total=len(db)):
        composition     = row[col_formula]
        energy_per_atom = row[col_E]
        _, hull = get_hull_distance(composition, energy_per_atom, dict_convex)
        db.loc[ind, 'ML_hull'] = hull
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use ML formation energy of compounds and competing phases to evaluate Hull distance. Energy is in energy per atom."
    )
    
    parser.add_argument("-m", "--model", type=str, required=True, help="model: prepare/evaluate")
    
    parser.add_argument("-d", "--database_candidate",  type=str, required=True, help="compounds db csv file containing compounds for hull evaluation. [prepare/evaluate]")
    
    parser.add_argument("-c", "--database_convex",    type=str, required=False,        help="Competing phases formation energy db csv file. [evaluate]")  
    parser.add_argument("--formula_column_candidate", type=str, default="composition", help="Column name in compound database containing the formulas. (default: composition)  [evaluate]" ) 
    parser.add_argument("--formula_column_convex",    type=str, default="name",        help="Column name in convex database containing the formulas. (default: name)  [evaluate]" ) 
    parser.add_argument("--formation_energy_column",  type=str, default="ML_formE",    help="Column name in database containing the ML formation energies (default: ML_formE)  [evaluate]" ) 

    parser.add_argument("-o", "--output",              type=str, required=True, help="Output, convex hull competing phase compounds joblist csv file name. [prepare] \\ \
                                                                                      Output, hull distance csv file name. [evaluate]")
        
    args = parser.parse_args()

    db = pd.read_csv(args.database_candidate, index_col=0)
    
    if args.model=='prepare': 
        unique_elements_list = get_unique_elements_list(db)
        db_convex = get_db_convex(unique_elements_list)

        db_convex.to_csv(args.output)

    elif args.model=='evaluate': 
        db_convex   = pd.read_csv(args.database_convex, index_col=0)
        dict_convex = get_dict_energy(db_convex, key=args.formula_column_convex, value=args.formation_energy_column)
        print("dict_convex len: ", len(dict_convex))
            
        db = update_hull_distance(db, dict_convex, col_formula=args.formula_column_candidate, col_E=args.formation_energy_column)
        db.to_csv(args.output)









