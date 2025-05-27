# get competing phases list and structures from OQMD
# the ML energy of these compounds are then evaluated.
# Then, ML hull distance can be evaluated.


mpirun -np 10 python ./stg_src/ML_hull_prepare.py \
  --database_candidate ./data/DXMag_heusler_ground.csv \
  --output             ./data/db_convex_joblist.csv \


  
  
  