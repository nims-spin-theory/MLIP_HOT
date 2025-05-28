#!/bin/bash

# Check for input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job_script_template>"
    echo "Usage: $1 <parallel SIZE>"
    exit 1
fi

JOB_SCRIPT_TEMPLATE=$1
SIZE=$2

for ((n = 0; n < SIZE; n++)); do
    # Define the new job script name
    JOB_SCRIPT="tmp_${JOB_SCRIPT_TEMPLATE%.sh}_${n}.sh"
    
    # Copy the template script
    cp "$JOB_SCRIPT_TEMPLATE" "$JOB_SCRIPT"
    
    # Replace 'RANK' with the current value of n
    sed -i "s/SIZE/$SIZE/g" "$JOB_SCRIPT"
    sed -i "s/RANK/$n/g" "$JOB_SCRIPT"
 
    # Submit the job (assuming sbatch for Slurm, change if needed)
    qsub "$JOB_SCRIPT"
done
