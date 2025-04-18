import pandas as pd
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
from pymatgen.core import Composition

'''
Through this script, energy is energy per atom.
'''

def get_formE(formula, E, dic_terminal):
    comp     = Composition(formula)
    expanded = [str(el) for el, amt in comp.items() for _ in range(int(amt))]
    E_terminal = [dic_terminal[el] for el in expanded]
    E_form_ML  = E - np.mean(E_terminal)
    return E_form_ML


def get_dict_energy(db, key='element', value='ML_e'):
    return dict(zip(db[key], db[value]))


def update_formation_energy(db, dic_terminal, col_formula='composition', col_E='ML_e'):
    for ind, row in tqdm(db.iterrows(), total=len(db)):
        formula = row[col_formula]
        E       = row[col_E]
        E_form_ML = get_formE(formula, E, dic_terminal)
        db.loc[ind, 'ML_formE'] = E_form_ML
    return db


def load_db_csv(folder='./convex_eqV2_31M_omat_mp_salex/', name="db*convex*.csv"):
    csv_files = glob.glob(os.path.join(folder, name))
    db = pd.concat((pd.read_csv(file, index_col=0) for file in csv_files), ignore_index=False)
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use ML energy of compounds and terminal elements to evaluate formation energy. Energy is in energy per atom."
    )

    parser.add_argument("-f", "--database_folder",   type=str, required=True, help="Path to the db csv file folder.")
    parser.add_argument("-n", "--database_name",     type=str, required=True, help="Keyword in db csv file name for glob purpose.")
    parser.add_argument("-t", "--database_terminal", type=str, required=True, help="Terminal elements energy db csv file.")
    parser.add_argument("-o", "--output",            type=str, required=True, help="Output csv file name.")
    
    parser.add_argument("--formula_column_compound", type=str, default="composition", help="Column name in compound database containing the formula. (default: composition)" ) 
    parser.add_argument("--formula_column_terminal", type=str, default="element",     help="Column name in terminal database containing the formula. (default: element)" ) 
    parser.add_argument("--energy_column",           type=str, default="ML_e",        help="Column name in database containing the ML energies (default: ML_e)" ) 
    
    args = parser.parse_args()

    db_terminal   = pd.read_csv(args.database_terminal, index_col=0)
    dict_terminal = get_dict_energy(db_terminal, key=args.formula_column_terminal, value=args.energy_column)

    db = load_db_csv(folder=args.database_folder, name=args.database_name)
    db = update_formation_energy(db, dict_terminal, col_formula=args.formula_column_compound, col_E=args.energy_column)
    db = db.sort_index()

    db.to_csv(args.output)
