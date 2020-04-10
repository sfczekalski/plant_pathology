from imports import *

SEED = 42
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 64
SIZE = 512
data_dir = 'data/'
device = torch.device('cuda:0')

transforms_valid = None
transforms_train = None
transforms_test = None
