# get competing phases list and structures from OQMD
# the ML energy of these compounds are then evaluated.
# Then, ML hull distance can be evaluated.

models=("eqV2_31M_omat_mp_salex" "eqV2_86M_omat_mp_salex" "eqV2_153M_omat_mp_salex")

for model in "${models[@]}"; do

mpirun -np 10 python ../dev/src/ML_hull_evaluate.py \
  --database_candidate  ./formE_results_models/formE_$model.csv \
  --database_convex     ../5_competing_phase/formE_results/formE_$model.csv \
  --output              ./hull_results_models/hull_$model.csv \
  --formula_column_candidate  composition \
  --formula_column_convex     name \
  --formE_column_candidate   ML_formE  \
  --formE_column_convex      ML_formE

done  


