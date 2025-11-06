# Example Data and Results

This directory contains example input files for testing the MLIP-HOT toolkit and following the examples provided in the main README. Pre-computed results are also included to help verify your installation and compare outputs.

## Directory Contents

### Input Files

#### `example_data.csv`
- **Description**: A dataset of 100 Heusler alloy compounds used for demonstration purposes
- **Source**: Obtained from the [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/).
- **Structure**: Contains columns defining crystal structures:
  - `composition`: Chemical composition of the compound
  - `cell`: Lattice vectors as a 3×3 matrix
  - `positions`: Fractional atomic coordinates
  - `numbers`: Atomic numbers of elements

#### `terminal_elements_oqmd.csv`
- **Description**: Ground state crystal structures for 89 elements from the OQMD database
- **Purpose**: Initial structures for calculating energies of elements, which are then used for formation energy calculation
- **Source**: DFT-optimized structures from OQMD

#### `terminal_elements_mp.csv`
- **Description**: Ground state crystal structures for 88 elements from the Materials Project database
- **Purpose**: Initial structures for calculating energies of elements, which are then used for formation energy calculation
- **Source**: DFT-optimized structures from Materials Project

### Output Files (`example_result/` directory)

This subdirectory contains pre-computed results from running the complete MLIP-HOT workflow on the example dataset. These files demonstrate the expected output format and can be used to verify your installation.

#### `example_data_result_oqmd.csv`
- **Description**: Optimized structures, formation energies, and hull distances. The reference energies for formation energy and hull distance calculations are obtained by optimizations initialized using structures from the OQMD database
- **Columns**:
  - `optimized_formula`: Chemical formula of optimized structure
  - `optimized_cell`: Optimized lattice vectors
  - `optimized_positions`: Optimized atomic positions
  - `optimized_numbers`: Atomic numbers (may differ if structure changed)
  - `Energy (eV/atom)`: Energy of optimized structure
  - `Formation Energy (eV/atom)`: Formation energy
  - `Hull Distance (eV/atom)`: Distance to convex hull

#### `example_data_result_mp.csv`
- **Description**: Optimized structures, formation energies, and hull distances. The reference energies for formation energy and hull distance calculations are obtained by optimizations initialized using structures from the MP database
- **Columns**: Same as `example_data_result_oqmd.csv`

#### `convex_hull_compounds_oqmd.csv`
- **Description**: Competing phases retrieved from OQMD database
- **Purpose**: Validates the usage of the corresponding script

#### `convex_hull_compounds_mp.csv`
- **Description**: Competing phases retrieved from Materials Project database
- **Purpose**: Validates the usage of the corresponding script

### Extra File
#### `example_data_dft.csv`
- **Description**: DFT reference results for comparison with MLIP predictions
- **Source**: Data obtained from [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/)
- **Purpose**: Enables validation and benchmarking of MLIP predictions against DFT calculations
- **Contains**: DFT-optimized structures, formation energies, and hull distances
- **Note**: Formation energies and hull distances were computed using the QMPY package with data from the OQMD database


## File Format Notes

All CSV files use the following structure format:
- **cell**: 3×3 matrix as list `[[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]]`
- **positions**: Fractional coordinates as list of vectors
- **numbers**: List of atomic numbers corresponding to each position

This format is compatible with ASE (Atomic Simulation Environment) and can be easily converted to other structure formats.




