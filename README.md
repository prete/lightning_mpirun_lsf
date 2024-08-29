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

### mpi4p dependency

The package `mpi4py` is required because we'll use 'mpirun' to use an MPIEnvironment.
For more info see [lightning.ai MPIEnvironmnet docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.environments.MPIEnvironment.html)
For your own environment:
```
pip install mpi4py
```

# Run

Wrapped it up like a burrito. 🌯 Now let's hope it doesn't fall apart

LSF script
```bash
#!/bin/bash
#BSUB -q gpu-lotfollahi                             # name of the queue gpu-lotfollahi
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request for exclusive access to gpu
#BSUB -o %J.out                                     # output file
#BSUB -e %J.err                                     # error file
#BSUB -M 8G                                         # 8G RAM memory per host
#BSUB -R "select[mem>8G] rusage[mem=8G]"            # same as above
#BSUB -n 4                                          # number of cores in total
#BSUB -R "span[ptile=2]"                            # split 2 core per host

set -eo pipefail

module load cuda-12.1.1
module load ISG/openmpi

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export UCX_IB_MLX5_DEVX=n

source env/bin/activate
mpirun --display-allocation python testing.py

echo "Done"
```
