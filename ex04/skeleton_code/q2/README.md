# Modules
module load gcc/6.4.0 (or newer)

module load python/3.6.4 (for Question 1c)

module load ffmpeg/3.2.4 (for Question 1c)


# Python Dependencies
pip install --user matplotlib (for Question 1c)

# Interactive Shell 
bsub -n 12 -W 00:30 --Is bash

# Make commands

**Compilation**

make

**Run simulation** (for Question 1c) 

make run

**Visualize output** (for Question 1c)

make plot

**Collect statistics** (for Question 1g)

make stat

make statjob (alternative)

make statjobfull (alternative, request full node, may take long)

**Plot statistics** (for Question 1g)

make plotstat

**Cleanup**

make clean
