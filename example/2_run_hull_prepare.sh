# get competing phases list and structures from OQMD
# the ML energy of these compounds are then evaluated.
# Then, ML hull distance can be evaluated.

# this code can only be ran on dxmac where the oqmd is installed
# conda activate qmpy_env

mpirun -np 10 python ./src/ML_hull_prepare.py \
  --database_candidate ./joblist/joblist_quaternary.csv \
  --output             ./joblist_ch/joblist_quaternary_ch.csv \


  
  
  
  
  