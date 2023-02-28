#!/bin/bash
#
#
# resource allocation (typical parameters)
#
# --nodes: number of nodes among which to distribute allocation
#          only use this if you need whole nodes because you need
#          all cpus on each node, or you need all memory on each node;
#          otherwise, you are unnecessarily taking away useful
#          resources from your collaborators; default 1
#
# --tasks-per-node: number of discrete job segments, each of which
#                   is running an independent executable, likely in
#                   concert with all other executables communicating
#                   via MPI; set to number of CPUs per node to get
#                   full-node allocation; default 1
#
# --mem-per-cpu: amount of memory required PER CPU in this job
#       use suffix of M or G for Megabytes or Gigabytes, for example
#       --mem-per-cpu=500M
#
SBATCH --nodes=4
SBATCH --tasks-per-node=32
SBATCH --mem-per-cpu=500M
#
#
# -C: constraints required for node selection
#
# This will tell SLURM the job needs all of its nodes
# to be on the same Infiniband switch.
#SBATCH -C "[ib1|ib2|ib3|ib4]"
#
# -t: walltime specification
#
# Time Formats, where DD=days, HH=hours, MM=minutes, SS=seconds:
# "MM"
# "MM:SS"
# "HH:MM:SS"
# "DD-HH"
# "DD-HH:MM"
# "DD-HH:MM:SS"
#
#SBATCH -t 30
#
#
# -J: job identifier (friendly name)
#
#SBATCH -J vasprun
#
#
# -o: send output to file (default is named "slurm-<JOBID>.out")
# -e: send errors to file (default is to combine with output)
#
# Filename Formats:
# %j     jobid of the running job.
# %N     short hostname. This will create a separate IO file per node.
# %s     stepid of the running job.
# %t     task identifier (rank) relative to current job. This will create a separate IO file per task.
# %u     User name.
# %x     Job name.
#
#SBATCH -o %x.out.%N.%j
#SBATCH -e %x.err.%N.%j
#
#
# --mail-type, --mail-user: e-mail notification configuration
#
# Notification Types:
# NONE
# BEGIN          job start
# END            job completion
# FAIL
# REQUEUE
# ALL (equivalent to BEGIN, END, FAIL, and REQUEUE)
# TIME_LIMIT
# TIME_LIMIT_90  reached 90 percent of time limit
# TIME_LIMIT_80  reached 80 percent of time limit
# TIME_LIMIT_50  reached 50 percent of time limit
# ARRAY_TASKS    send emails for each array task
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myemail@seas.upenn.edu


# configure environment
#

# on Chestnut Cluster, mpirun is an alias to
#    'mpirun -x PATH -x LD_LIBRARY_PATH -x MPICH_HOME -x MPI_ROOT'
# which will pass along those environment variables to each MPI rank
#
# PLEASE NOTE: mpirun is integrated with SLURM, so you should NOT use
#  the -np or -hostfile arguments
#
python3 babylm_syllables.py