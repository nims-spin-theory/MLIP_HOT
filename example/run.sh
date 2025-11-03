#!/bin/bash
export OMP_NUM_THREADS=1

# mpirun -np 4 python ../ML_optimization.py \
#     -d ./example_data.csv \
#     -m "chgnet" \
#     -o "result_test" 

# mpirun -np 10 python ../ML_optimization.py \
#     -d ./example_data.csv \
#     -m "chgnet" \
#     -o "result_test" \
#     -s 1 -r 0 \
#     --strain 0.1 

# mpirun -np 10 python ../ML_optimization.py \
#     -d ./example_data.csv \
#     -m "7net-mf-ompa" \
#     -o "result_test" \
#     -s 3 -r 0 \
#     --strain "[[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]" 

#####################################

# for ((n = 0; n < 3; n++)); do
#     mpirun -np 10 python ../ML_optimization.py \
#         -d ./example_data.csv \
#         -m "mattersim" \
#         -o "result_test1" \
#         -s 3 -r $n \
#         --strain 0.1 
# done

# python ../concat_csv.py -f "./result_test1" \
#         -p "example_data_*.csv" -o example_data_result_test1.csv


# for ((n = 0; n < 3; n++)); do
#     mpirun -np 10 python ../ML_optimization.py \
#         -d ./example_data.csv \
#         -m "mattersim" \
#         -o "result_test2" \
#         -s 3 -r $n \
#         --strain 0.2 
# done

# python ../concat_csv.py -f "./result_test2" \
#         -p "example_data_*.csv" -o example_data_result_test2.csv

#####################################

# python ../find_global_minimun.py \
#         -i example_data_result_test1.csv \
#            example_data_result_test2.csv \
#         -o example_data_result_global_min.csv

# python ../find_global_minimun.py \
#         -i example_data_result_test1.csv \
#            example_data_result_test2.csv \
#         --labels test1 test2 \
#         -o example_data_result_global_min.csv

#####################################

# mpirun -np 10 python ../ML_optimization.py \
#     -d ./terminal_elements.csv \
#     -m "mattersim" \
#     -o "terminal_elements_energy" 

# python ../ML_formE.py -i example_datbash a_result_global_min.csv -t terminal_elements_energy/terminal_elements.csv -o example_data_result_formation_energy.csv

#####################################

# mpirun -np 10 python ../ML_optimization.py \
#     -d ./convex_hull_phase.csv \
#     -m "mattersim" \
#     -o "convex_hull_phase" 

# mpirun -np 10 python ../ML_hull_prepare_qmpy_rester.py \
#        -d example_data_10.csv -o tmp_competing_phases.csv
    
mpirun -np 2 python ../get_convex_hull_compounds_mp_rester.py \
       -d example_data_10.csv -o tmp_competing_phases.csv \
       --api_key='pJx41Nd1Bl2zMSeN8JPeo9a65TJG0dAY'
    


