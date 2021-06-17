#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils import get_extracted_xvectors, get_mean_scores
from sklearn.model_selection import KFold, LeaveOneOut
import argparse
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.preprocessing import StandardScaler


class LinearModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(512, 1)
  
  def forward(self, x):
    return self.linear(x)


class NonlinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnet-name', default="xvector", help='nnet name')
    args = parser.parse_args()

    xvectors = get_extracted_xvectors("scp:xvectors/{}/pathologic_voices/xvector.scp".format(args.nnet_name))
    scores_LARY = get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'effort', 'intell', 'overall']]
    scores_PARE = get_mean_scores("/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")[['utt', 'effort', 'intell', 'overall']]
    scores_CTRL = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'effort', 'intell', 'overall']]
    df = xvectors.merge(pd.concat([scores_LARY, scores_CTRL, scores_PARE]), on='utt')


    sel_group_combinations = [['LARY'], ['PARE'], ['LARY', 'PARE'], ['CTRL']]
    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True)


    print("=================================================")
    print("               LinearModel")
    print("=================================================")
    for crit in ['overall', 'intell','effort']:
        for sel_speaker_groups in sel_group_combinations:

            num_epochs = 100
            learningRate = 1e-5
            
            dataset = df.loc[df.speaker_group.isin(sel_speaker_groups)]
            X = np.vstack(dataset.embedding)
            y = dataset[crit].values.reshape((-1, 1))
            
            fold_results = {'mse': {}, 'mae': {}}
            for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
                xscaler, yscaler = StandardScaler(), StandardScaler()
                X_train = Tensor( xscaler.fit_transform( X[train_ids] ) )
                y_train = Tensor( yscaler.fit_transform( y[train_ids] ) ).view(-1, 1)
                X_test = Tensor( xscaler.transform( X[test_ids] ) )
                y_test = Tensor( yscaler.transform( y[test_ids] ) ).view(-1, 1)
            
                net = LinearModel()
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
                
                # Run the training loop for defined number of epochs
                for epoch in range(0, num_epochs):
                    loss = criterion(net(X_train), y_train)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Evaluation for this fold
                total_loss, total = 0, 0
                net.eval()
                with torch.no_grad():
                    y_pred = net(X_test)
                    loss = criterion(y_pred, y_test)
                    mae_fn = nn.L1Loss()
                    mae = mae_fn(y_pred, y_test)

                fold_results['mse'][fold] = loss.item()
                fold_results['mae'][fold] = mae.item()


            print(f'{sel_speaker_groups} {crit}: {k_folds}-Fold cross validation')
            sum = 0.0
            for key, value in fold_results['mse'].items():
                sum += value
            print(f'Average MSE:', sum/len(fold_results['mse'].items()))

            sum = 0.0
            for key, value in fold_results['mae'].items():
                sum += value
            print(f'Average MAE:', sum/len(fold_results['mae'].items()))
            print()




    print("=================================================")
    print("           Feed-forward model")
    print("=================================================")
    for crit in ['overall', 'intell','effort']:
        for sel_speaker_groups in sel_group_combinations:
            num_epochs = 10
            learningRate = 1e-5
            kfold = KFold(n_splits=k_folds, shuffle=True)

            dataset = df.loc[df.speaker_group.isin(sel_speaker_groups)]
            X = np.vstack(dataset.embedding)
            y = dataset[crit].values.reshape((-1, 1))

            fold_results = {'mse': {}, 'mae': {}}
            for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
                xscaler, yscaler = StandardScaler(), StandardScaler()
                X_train = Tensor( xscaler.fit_transform( X[train_ids] ) )
                y_train = Tensor( yscaler.fit_transform( y[train_ids] ) ).view(-1, 1)
                X_test = Tensor( xscaler.transform( X[test_ids] ) )
                y_test = Tensor( yscaler.transform( y[test_ids] ) ).view(-1, 1)
                
                net = NonlinearRegressionModel()
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
                
                # Run the training loop for defined number of epochs
                for epoch in range(0, num_epochs):
                    loss = criterion(net(X_train), y_train)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Evaluation for this fold
                total_loss, total = 0, 0
                net.eval()
                with torch.no_grad():
                    y_pred = net(X_test)
                    loss = criterion(y_pred, y_test)
                    mae_fn = nn.L1Loss()
                    mae = mae_fn(y_pred, y_test)

                fold_results['mse'][fold] = loss.item()
                fold_results['mae'][fold] = mae.item()


            print(f'{sel_speaker_groups} {crit}: {k_folds}-Fold cross validation')
            sum = 0.0
            for key, value in fold_results['mse'].items():
                sum += value
            print(f'Average MSE:', sum/len(fold_results['mse'].items()))

            sum = 0.0
            for key, value in fold_results['mae'].items():
                sum += value
            print(f'Average MAE:', sum/len(fold_results['mae'].items()))
            print()
