
# create env
python3 -m venv env
source env/bin/activate

# update pip and setuptools
pip install -U pip setuptools

# load cuda module before installing anything else
module load cuda-12.1.1 

# install requirements
pip install -r requirements.txt

# install mpi4py — required because we'll use 'mpirun' to use an MPIEnvironment
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.environments.MPIEnvironment.html
pip install mpi4py

