# get formation energy of competing phases
# since the competing phases are obtained from OQMD,
# to keep consistence, the composition col name is name. 

#models="chgnet"
#set="chgnet"

models=$1
set=$1
mkdir results_formE_$set

keys=("tetra_p0p1" "tetra_n0p1" "tetra_p0p2" "tetra_n0p2" "tetra_p0p3" "tetra_n0p3" 
      "tetra_p0p4" "tetra_n0p4" "tetra_p0p5" "tetra_n0p5")
for model in "${models[@]}"; do
    for key in "${keys[@]}"; do

    python ../dev/src/ML_formE.py \
      -f ./jobs_$set/result_$model"_"$key/ \
      -n "DXMag_heusler_test*.csv" \
      -t ../4_terminal_elements/result_$model/terminal_elements_1_0.csv \
      -o ./results_formE_$set/formE_$model"_"$key.csv \
      --formula_column_compound composition \
      --formula_column_terminal element \
      --energy_column  ML_e
    done
done


keys=("cubic_p0p1" "cubic_n0p1" "cubic_p0p3" "cubic_n0p3")
for model in "${models[@]}"; do
    for key in "${keys[@]}"; do

    python ../dev/src/ML_formE.py \
      -f ./jobs_$set/result_$model"_"$key/ \
      -n "DXMag_heusler_test*.csv" \
      -t ../4_terminal_elements/result_$model/terminal_elements_1_0.csv \
      -o ./results_formE_$set/formE_$model"_"$key.csv \
      --formula_column_compound composition \
      --formula_column_terminal element \
      --energy_column  ML_e
    done
done


