# Convex Hull Compounds Scripts

## Get Competing Phases from OQMD

To construct the convex hull of a chemical system using the same MLIP, obtain DFT structures from the OQMD or Materials Project database and use them as initial structures for MLIP optimization and formation energy calculation. With the convex hull compounds' formation energies and the formation energies of screened compounds, use the `MLIP_hull.py` script to calculate hull distances.

The example below uses DFT structures from the OQMD database via QMPY Rester (`qmpy_rester`). First, we retrieve the DFT structures of competing phases, which will serve as initial configurations for subsequent MLIP optimization.

**Note:** QMPY Rester may encounter network connection issues. To address this, the script automatically makes 4 attempts for each compound, which succeeds for all compounds in most cases. However, if any compounds fail after all retry attempts, they will be saved to the file specified by `--failed_systems_output`. This file can then be used as input to rerun the same script, and the newly retrieved results can be merged with the original output.


```bash
# Get competing phases from OQMD database
mpirun -np 10 python ../scripts/get_convex_hull_compounds_qmpy_rester.py \
    -d example_data.csv \
    -o convex_hull_compounds.csv \
    --failed_systems_output test_fail.csv
```


``` bash
# Optimize convex hull phases and calculate formation energy
mpirun -np 10 python ../scripts/MLIP_optimize.py \
    -d ./convex_hull_compounds.csv \
    -m "mattersim" \
    -o "convex_hull_compounds_energy"

python ../scripts/MLIP_form.py \
    -i convex_hull_compounds_energy/convex_hull_compounds.csv \
    -t terminal_elements_energy/terminal_elements.csv \
    -o convex_hull_compounds_formation_energy.csv

# Calculate distance to convex hull
mpirun -np 4 python ../scripts/MLIP_hull.py \
    -d example_data_result_formation_energy.csv \
    -c convex_hull_compounds_formation_energy.csv \
    -o example_data_result_hull.csv

# Flags:
#   -d: CSV file of screening compounds with formation energies
#   -c: CSV file of convex hull compounds with formation energies
#   -o: Output file containing hull distance (eV/atom)
```


## Get Competing Phases from Materials Project

This requires the `mp-api` module. Please add it with the command below when the conda environment is activated.


``` bash
pip install mp-api
```


This requires an API key, which can be obtained here: <https://materialsproject.org/api>


```bash
mpirun -np 5 python ../scripts/get_convex_hull_compounds_mp_rester.py \
    -d example_data.csv \
    -o convex_hull_compounds.csv \
    --api_key='your_api_key_here'

# Flags:
#   --api_key: Materials Project API key
```

**Note 1:** This method is faster and more stable than the OQMD Rester approach. However, avoid using too many parallel processes, as this may exceed the Materials Project API rate limits and trigger an error:

```python
REST query returned with error status code 429. Content: b'{"error": "Your IP has been (temporarily) blocked due to rate limits.
```

To avoid rate limiting, use a smaller number of parallel processes (e.g., `-np 5` or lower) when querying the Materials Project API.

**Note 2:** Since Materials Project and OQMD are different databases with distinct data curation procedures, the convex hull compounds retrieved from each database may differ. Consequently, this can lead to different computed hull distance values for the same target compound.

## Using OQMD Database on Local Machine

The OQMD database can be installed on a local machine following the instructions here: <https://static.oqmd.org/static/docs/getting_started.html>

**Recommendation:** Using the local database script is significantly faster and supports highly parallel execution with a large number of processes. Therefore, installing the local OQMD database is recommended for high-throughput workflows, despite requiring some initial setup time.


```bash
mpirun -np 4 python ../scripts/get_convex_hull_compounds_qmpy.py \
    -d example_data.csv \
    -o convex_hull_compounds.csv
```

To use this script, please configure the local database connection part in the script:


```python
DEFAULT_DB_CONFIG = {
    'name': 'oqmd__v1_6',
    'user': 'user',
    'host': 'localhost',
    'password': 'password'  
}
```

**Expected Output Files:**

- `example_data_global_min.csv`: Ground state structures with lowest energies
- `example_data_formation_energy.csv`: Formation energies (eV/atom) for all compounds
- `example_data_final_results.csv`: Final results including hull distances (eV/atom)

**Key Points:**

- Using multiple strain strategies increases the chance of finding true global minima
- Dividing datasets into chunks (`-s` and `-r` flags) enables parallel processing across different compute nodes
- The convex hull analysis requires target compounds, terminal elements, and competing phases to all be optimized with the same MLIP model
- In practice, convex hull compounds can be generated first, then optimization of screened compounds, elements, and convex hull compounds can run simultaneously
