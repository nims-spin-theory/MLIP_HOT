# Convex Hull Compounds Scripts

To construct the convex hull of a chemical system using the same MLIP, obtain DFT structures from the OQMD or Materials Project database and use them as initial structures for MLIP optimization and formation energy calculation. With the convex hull compounds' formation energies and the formation energies of screened compounds, we can obtain distance above convex hull.

### Get Competing Phases from OQMD via API

The example below get DFT structures from the OQMD database via QMPY Rester (`qmpy_rester`). 

**Note:** QMPY Rester may encounter network connection issues. To address this, the script automatically makes 4 attempts for each compound, which succeeds for all compounds in most cases. However, if any compounds fail after all retry attempts, they will be saved to the file specified by `--failed_systems_output`. This file can then be used as input to rerun the same script, and the newly retrieved results can be merged with the original output.

```bash
# Get competing phases from OQMD database
mpirun -np 10 python ../scripts/get_convex_hull_compounds_qmpy_rester.py \
    -d example.csv \
    -o convex_hull_compounds.csv \
    --failed_systems_output test_fail.csv
```

### Get Competing Phases from Materials Project

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

### Using OQMD Database on Local Machine

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
