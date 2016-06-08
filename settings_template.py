####################
# settings for the cnntools app
####################

# Absolute path to the Caffe root directory
CAFFE_ROOT = ''

# This seed is used to initialize the solvers for reproducible training
# experiments
CAFFE_SEED = 123

# Set it to True, if you want to use the GPU by default
CAFFE_GPU = False

# List absolute paths which should be added to the python path before training starts.
# For example if you have a python layer, you might want to import the module
# which defines it, so it's on the python path before the Caffe is searching
# for the layer definition.
TRAINING_EXTRA_PYTHON_PATH = [
]

