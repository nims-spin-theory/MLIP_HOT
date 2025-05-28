#!/bin/bash
#$ -cwd
#$ -V -S /bin/bash
#$ -N  jobs_chg_SIZE_RANK
#$ -pe smp 20

########$ -q  40core-stg #######

module load intel/2023.2
module load intelmpi/2021.10
export OMP_NUM_THREADS=1

conda activate MLFFopt
#conda activate MLFFopt_mattersim
#conda activate MLFFopt_eqV2
#conda activate MLFFopt_fairchem
#conda activate MLFFopt_HIENet

#model=eqV2_31M_omat_mp_salex
#model=eqV2_86M_omat_mp_salex
#model=eqV2_153M_omat_mp_salex
#model=esen_30m_oam
#model=hienet
model=chgnet

declare -A strains=(
  [tetra_p0p1]="0.0 0.0  0.1"
  [tetra_n0p1]="0.0 0.0 -0.1"
  [tetra_p0p3]="0.0 0.0  0.3"
  [tetra_n0p3]="0.0 0.0 -0.3"
  [cubic_p0p1]=" 0.1  0.1  0.1"
  [cubic_n0p1]="-0.1 -0.1 -0.1"
  [cubic_p0p3]=" 0.3  0.3  0.3"
  [cubic_n0p3]="-0.3 -0.3 -0.3"
  [tetra_p0p5]="0.0 0.0  0.5"
  [tetra_n0p5]="0.0 0.0 -0.5"
)


for key in "${!strains[@]}"; do
  strain="${strains[$key]}"
  mpirun -np 20 python ./src/ML_optimization.py \
    -d ./data/DXMag_heusler_test.csv \
    -m "$model" \
    -o "result_${model}_$key" \
    -s SIZE -r RANK \
    --heusler2fu True \
    --strain $strain
done




declare -A strains=(
  [tetra_p0p2]="0.0 0.0  0.2"
  [tetra_n0p2]="0.0 0.0 -0.2"
  [tetra_p0p4]="0.0 0.0  0.4"
  [tetra_n0p4]="0.0 0.0 -0.4"
)


for key in "${!strains[@]}"; do
  strain="${strains[$key]}"
  mpirun -np 20 python ./src/ML_optimization.py \
    -d ./data/DXMag_heusler_test.csv \
    -m "$model" \
    -o "result_${model}_$key" \
    -s SIZE -r RANK \
    --heusler2fu True \
    --strain $strain
done


