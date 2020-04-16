import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from model import Model
from dataset import PlantDataset
from loss import CrossEntropy


def train_one_fold(i_fold, model, criterion, optimizer, N_EPOCHS, dataloader_train, dataloader_valid, device):
    """
    Train one fold. Data has already been split

    :param i_fold: index of the current fold
    :param model: the classifier
    :param criterion: criterion to optimize
    :param optimizer: optimizer
    :param dataloader_train: training data for the current fold
    :param dataloader_valid: validation data for the current fold

    :return:
    """
    train_fold_results = []

    for epoch in range(N_EPOCHS):

        print(f'  Epoch {epoch + 1}/{N_EPOCHS}')
        print('  ' + ('-' * 20))

        model.train()
        tr_loss = 0

        # iterate over training data, batch by batch
        for step, batch in enumerate(dataloader_train):
            images = batch[0]
            labels = batch[1]

            # put the data into the model
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            # get model's output, calculate the loss
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1))
            loss.backward()

            # store the loss
            tr_loss += loss.item()

            # backpropagation
            optimizer.step()
            optimizer.zero_grad()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):

            images = batch[0]
            labels = batch[1]

            # Store the labels
            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            # don't store gradient this time!
            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels.squeeze(-1))
                val_loss += loss.item()

                preds = torch.softmax(outputs, dim=1).data.cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat([val_preds, preds], dim=0)

        train_fold_results.append({
            'fold': i_fold,
            'epoch': epoch,
            'train_loss': tr_loss / len(dataloader_train),
            'valid_loss': val_loss / len(dataloader_valid),
            'valid_score': roc_auc_score(val_labels, val_preds, average='macro'),
        })

    return val_preds, train_fold_results


def training_loop(N_FOLDS=5, N_EPOCHS=10, BATCH_SIZE=64, transforms_train=None, transforms_valid=None, data_dir='data/',
                  device=torch.device('cuda:0')):
    """
    Function training the model.
    It uses cross-validation to train separate models and predicts on test set.
    :param N_FOLDS: Number of cross-validation splits
    :param N_EPOCHS: Number of training epochs at each split
    :param BATCH_SIZE: Batch size
    :param transforms_train: Transformations of images that should be applied on training set
    :param transforms_valid: Transformations of images that should be applied on validation / test set
    :param data_dir: Path to directory with data
    :param device: Device to use, by default it uses cuda gpu
    :return: training results
    """

    # Read training data, get the labels to split the folds appropriately (the classes are imbalanced)
    train_df = pd.read_csv(data_dir + 'train.csv')

    train_labels = train_df.iloc[:, 1:].values
    train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros((train_df.shape[0], 4))
    train_results = []

    # Test dataloader
    submission_df = pd.read_csv(data_dir + 'sample_submission.csv')
    submission_df.iloc[:, 1:] = 0
    dataset_test = PlantDataset(df=submission_df, data_dir=data_dir, transforms=transforms_valid)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    submissions = None

    # Train - iterate over folds
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
        print(f'Fold {i_fold+1} / {N_FOLDS}')

        # get validation and training sets for the fold
        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)

        dataset_valid = PlantDataset(df=valid, data_dir=data_dir, transforms=transforms_valid)
        dataset_train = PlantDataset(df=train, data_dir=data_dir, transforms=transforms_train)

        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

        # initialize a new model, send it to GPU
        model = Model()
        model.to(device)

        # criterion, optimizer, learning rate
        criterion = CrossEntropy()
        optimizer = optim.Adam(model.parameters(), lr=5e-5)

        # train on folds, validate on oof instances
        # get validation predictions and results
        val_preds, train_fold_results = train_one_fold(i_fold, model, criterion, optimizer, N_EPOCHS,
                                                       dataloader_train, dataloader_valid,  device)
        oof_preds[valid_idx, :] = val_preds.numpy()
        train_results = train_results + train_fold_results

        # Evaluate on test set
        model.eval()
        test_preds = None

        for step, batch in enumerate(dataloader_test):
            # Get the images
            images = batch[0]
            images = images.to(device, dtype=torch.float)

            # Predict their class
            with torch.no_grad():
                outputs = model(images)

                if test_preds is None:
                    test_preds = outputs.data.cpu()
                else:
                    test_preds = torch.cat([test_preds, outputs.data.cpu()], dim=0)

        # Save predictions per fold
        submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)
        submission_df.to_csv(f'submissions/submission_fold_{i_fold}.csv', index=False)

        # Each model predicts on test set, the predictions are averaged
        if submissions is None:
            submissions = test_preds / N_FOLDS
        else:
            submissions += test_preds / N_FOLDS

    print(f"Validation score: {round(roc_auc_score(train_labels, oof_preds, average='macro'), 3)}")

    # All models trained
    # Aggregate the predictions, get probabilities
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)
    submission_df.to_csv('submissions/Plants_submission.csv', index=False)

    return train_results


def training_loop_single(N_EPOCHS, BATCH_SIZE, transforms_train=None, transforms_valid=None, data_dir='data/',
                         device=torch.device('cuda:0')):
    """
    Split the data to training and validation set and train one model.
    :param N_EPOCHS: Number of training epochs at each split
    :param BATCH_SIZE: Batch size
    :param transforms_train: Transformations of images that should be applied on training set
    :param transforms_valid: Transformations of images that should be applied on validation / test set
    :param data_dir: Path to directory with data
    :param device: Device to use, by default it uses cuda gpu
    :return: trained model, training results
    """
    # Read DataFrame
    df = pd.read_csv(data_dir + 'train.csv')

    # Train - validation split
    n = int(0.8 * df.shape[0])
    np.random.seed(42)
    train_idx = np.random.choice(df.index, n)
    train_df = df.loc[train_idx]
    train_df.reset_index(drop=True, inplace=True)

    valid_idx = np.setdiff1d(df.index, train_idx)
    valid_df = df.loc[valid_idx]
    valid_df.reset_index(drop=True, inplace=True)

    # Prepare Datasets and DataLoaders
    train_dataset = PlantDataset(df=train_df, data_dir=data_dir, transforms=transforms_train)
    valid_dataset = PlantDataset(df=valid_df, data_dir=data_dir, transforms=transforms_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    # initialize model, optimzer and loss
    model = Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropy()

    train_results = []

    for i in range(N_EPOCHS):
        print(f'Epoch: {i+1} / {N_EPOCHS}')

        # Train
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            images = batch[0]
            labels = batch[1]

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1))

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validate
        model.eval()
        valid_loss = 0.0
        valid_labels = None
        valid_preds = None
        for step, batch in enumerate(valid_dataloader):
            images = batch[0]
            labels = batch[1]

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            if valid_labels is None:
                valid_labels = labels.clone().squeeze(-1)
            else:
                valid_labels = torch.cat([valid_labels, labels.squeeze(-1)], dim=0)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(-1))
                valid_loss += loss.item()

                preds = torch.softmax(outputs, dim=1).data.cpu()

                if valid_preds is None:
                    valid_preds = preds
                else:
                    valid_preds = torch.cat([valid_preds, preds], dim=0)

        train_results.append({
            'epoch': i,
            'train_loss': train_loss / len(train_dataloader),
            'valid_loss': valid_loss / len(valid_dataloader),
            'valid_score': roc_auc_score(valid_labels.data.cpu(), valid_preds.data.cpu(), average='macro')
        })

    print(f'Validation score: {roc_auc_score(valid_labels.data.cpu(), valid_preds.data.cpu(), average="macro")}')

    return model, train_results
