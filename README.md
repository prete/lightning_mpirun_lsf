# Create environment

```bash
# create venv
python3 -m venv env

# activate venv
source env/bin/activate

# update pip and setuptools
pip install -U pip setuptools

# load cuda module before installing anything else
module load cuda-12.1.1 

# install requirements (pytorch/lightning/nvidia_cuda_xxx)
pip install -r requirements.txt

```

### ⚠️ mpi4p dependency

**The package `mpi4py` is required because we'll use 'mpirun' to use an MPIEnvironment in lightning.**

For more info see [lightning.ai MPIEnvironmnet docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.environments.MPIEnvironment.html). For other projects, when you create a new environment you'll have install it:
```
pip install mpi4py
```

# Run

Wrapped it up like a burrito. 🌯 Now let's hope it doesn't fall apart

LSF script
```bash
#!/bin/bash
#BSUB -q gpu-lotfollahi                             # name of the queue
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request gpus per host
#BSUB -o %J.out                                     # output file
#BSUB -e %J.err                                     # error file
#BSUB -M 8G                                         # RAM memory per host
#BSUB -R "select[mem>8G] rusage[mem=8G]"            # same as above
#BSUB -n 6                                          # number of cores in total
#BSUB -R "span[ptile=3]"                            # split X cores per host

## This exmaple requests 6 cores in total. The spread of those across hosts is 3 cores each.
## So in total we requested for 2 hosts, each host with 3 CPUs, 2 GPUs, and 8G of RAM.

set -eo pipefail

module load cuda-12.1.1
module load ISG/openmpi

export NCCL_DEBUG=INFO
# disable infiniband to prevent annoying errors
export NCCL_IB_DISABLE=1
export UCX_IB_MLX5_DEVX=n

# activate environment with the right dependencies
source env/bin/activate

# get some variables out of LSF that OpenMPI needs in the next step
NUM_HOSTS=$(sed 's/ /\n/g' <<< $LSB_HOSTS  | sort | uniq | wc -l)   # Total hosts for the job
NUM_GPUS=$(bjobs -noheader -o 'gpu_num' $LSB_JOBID)                 # Total GPUs for the job
GPU_PER_HOST=$((NUM_GPUS / NUM_HOSTS))                              # How many GPU devices per host we have (asumes all hosts will have the same)


# for this example we want:
#    a) 4 MPI processes [-n] in total, because 2 hosts * 2 GPUs on each one
#    b) we'll group those [--map-by] 2 processes per host, because that's the number of GPUs on each
# now run the script with mpirun for those values
mpirun \
    -n ${NUM_GPUS} \
    --map-by "ppr:${GPU_PER_HOST}:node" \
    --oversubscribe \
    --display-allocation python testing.py

echo "Done 🧙🧙"
```
