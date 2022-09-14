# File that is called by the wandb sweep process
# Simply repackages the arguments and calls "python main.py fit ..."
import sys
import os

args = ' '.join(sys.argv[1:])
logdir = os.environ['SCRATCH']
datadir = os.environ['SCRATCH']
os.system('/home/mila/l/leo.gagnon/columns/venv/bin/python ~/columns/src/main.py fit --config config.yaml' + ' --logdir=' + logdir + ' --data.data_dir=' + datadir + ' ' + args)
