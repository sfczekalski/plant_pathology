import albumentations as A
from albumentations.pytorch import ToTensorV2
from train import *

# Define training parameters
# Size of input image
SIZE = 512

N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 64

# Transforms
transforms_train = A.Compose([
    A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    # Pixels
    A.OneOf([
        A.IAAEmboss(p=1.0),
        A.IAASharpen(p=1.0),
        A.Blur(p=1.0),
    ], p=0.5),

    # Affine
    A.OneOf([
        A.ElasticTransform(p=1.0),
        A.IAAPiecewiseAffine(p=1.0)
    ], p=0.5),

    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

training_loop(N_FOLDS=N_FOLDS, N_EPOCHS=N_EPOCHS, BATCH_SIZE=BATCH_SIZE,
              transforms_train=transforms_train, transforms_valid=transforms_valid)
