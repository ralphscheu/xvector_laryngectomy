#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import argparse
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
from torch.nn.modules.linear import Linear
from utils import get_extracted_xvectors, get_mean_scores
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
from datetime import datetime
import torch
from torch.nn import functional as F

RESULTS_COLUMNS = ['model', 'num_epochs', 'lr', 'crit', 'speaker_group', 'avg_mse', 'avg_mae', 'pearson_r', 'p_value', 'r2']
SEL_GROUP_COMBINATIONS = [['LARY'], ['PARE'], ['LARY', 'PARE'], ['CTRL']]


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


def run_pytorch_model(title, df, model, criterion=torch.nn.L1Loss(), lr=1e-5, num_epochs=100, k_folds=10, print_results=False):
    """ fit and evaluate a pytorch regression model """

    print(f"running {title}...")

    # this will hold crossvalidated evaluation results
    results = pd.DataFrame(columns=RESULTS_COLUMNS)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    for crit in ['overall', 'intell','effort']:
        for sel_speaker_groups in SEL_GROUP_COMBINATIONS:

            dataset = df.loc[df.speaker_group.isin(sel_speaker_groups)]
            X = np.vstack(dataset.embedding)
            y = dataset[crit].values#.reshape((-1, 1))

            y_preds = []
            y_tests = []
            fold_results = {'mse': {}, 'mae': {}}
            for fold, (train_ids, test_ids) in enumerate( LeaveOneOut().split(X) ):
                # generate train and test portion, scale X_train, X_test according to X_train mean+std
                xscaler = StandardScaler()
                X_train = torch.Tensor( xscaler.fit_transform( X[train_ids] ) )
                X_test =  torch.Tensor( xscaler.transform( X[test_ids] ) )
                y_train = torch.Tensor( y[train_ids] ).view(-1, 1)
                y_test =  torch.Tensor( y[test_ids] ).view(-1, 1)

                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

                # Run the training loop for defined number of epochs
                for epoch in range(0, num_epochs):
                    loss = criterion(model(X_train), y_train)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # evaluate on test
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test)

                y_preds.append( y_pred.ravel() )
                y_tests.append( y_test.ravel() )

                mse = mean_squared_error(y_pred, y_test)
                mae = mean_absolute_error(y_pred, y_test)
                
                fold_results['mse'][fold] = mse
                fold_results['mae'][fold] = mae
            
            y_preds = np.concatenate(y_preds)
            y_tests = np.concatenate(y_tests)

            # pearson correlation
            r, pvalue = pearsonr( y_preds , y_tests)

            sum_mae = 0.0
            for _, value in fold_results['mae'].items():
                sum_mae += value

            sum_mse = 0.0
            for _, value in fold_results['mse'].items():
                sum_mse += value

            r2 = r2_score(y_preds, y_tests)

            # save results
            results = results.append({
                'model': title,
                'num_epochs': num_epochs, 'lr': lr,
                'crit': crit, 'speaker_group': sel_speaker_groups, 
                'avg_mse': sum_mse / len(fold_results['mse'].items()), 
                'avg_mae': sum_mae / len(fold_results['mae'].items()), 
                'pearson_r': r, 'p_value': pvalue, 'r2': r2
                }, ignore_index=True)

            if print_results:
                # print header
                print("=================================================")
                print("=", title)
                print("=================================================")

                # evaluate crossvalidation results
                print(f'LinearRegression - LOO cross validation')
                print(f"{sel_speaker_groups} {crit} - ( Pearson r {r}, p-value: {pvalue})")
                print(f'Avg MAE:', sum_mae / len(fold_results['mae'].items()))
                print(f'Avg MSE:', sum_mse / len(fold_results['mse'].items()))
                print(f'Avg R-squared:', r2)
    
    return results


def run_sklearn_model(title, model, df, print_results=False):
    """ fit and evaluate a sklearn regressor """

    print(f"running {title}...")

    # this will hold crossvalidated evaluation results
    results = pd.DataFrame(columns=RESULTS_COLUMNS)

    for crit in ['overall', 'intell','effort']:
        for sel_speaker_groups in SEL_GROUP_COMBINATIONS:

            dataset = df.loc[df.speaker_group.isin(sel_speaker_groups)]
            X = np.vstack(dataset.embedding)
            y = dataset[crit].values#.reshape((-1, 1))

            y_preds = []
            y_tests = []
            fold_results = {'mse': {}, 'mae': {}}
            for fold, (train_ids, test_ids) in enumerate( LeaveOneOut().split(X) ):
                X_train = X[train_ids]
                y_train = y[train_ids]

                X_test = X[test_ids]
                y_test = y[test_ids]

                regr = model
                regr.fit(X_train, y_train)

                y_pred = regr.predict(X_test)

                y_preds.append( y_pred.ravel() )
                y_tests.append( y_test.ravel() )

                mse = mean_squared_error(y_pred, y_test)
                mae = mean_absolute_error(y_pred, y_test)
                
                fold_results['mse'][fold] = mse
                fold_results['mae'][fold] = mae
            
            y_preds = np.concatenate(y_preds)
            y_tests = np.concatenate(y_tests)

            # pearson correlation
            r, pvalue = pearsonr( y_preds , y_tests)

            sum_mae = 0.0
            for _, value in fold_results['mae'].items():
                sum_mae += value

            sum_mse = 0.0
            for _, value in fold_results['mse'].items():
                sum_mse += value

            r2 = r2_score(y_preds, y_tests)

            # save results
            results = results.append({
                'model': title,
                'crit': crit, 'speaker_group': sel_speaker_groups, 
                'avg_mse': sum_mse / len(fold_results['mse'].items()), 
                'avg_mae': sum_mae / len(fold_results['mae'].items()), 
                'pearson_r': r, 'p_value': pvalue, 'r2': r2
                }, ignore_index=True)

            if print_results:
                # print header
                print("=================================================")
                print("=", title)
                print("=================================================")

                # evaluate crossvalidation results
                print(f'LinearRegression - LOO cross validation')
                print(f"{sel_speaker_groups} {crit} - ( Pearson r {r}, p-value: {pvalue})")
                print(f'Avg MAE:', sum_mae / len(fold_results['mae'].items()))
                print(f'Avg MSE:', sum_mse / len(fold_results['mse'].items()))
                print(f'Avg R-squared:', r2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnet-name', default="xvector", help='nnet name')
    args = parser.parse_args()

    # load data
    xvectors = get_extracted_xvectors("scp:xvectors/{}/pathologic_voices/xvector.scp".format(args.nnet_name))
    scores_LARY = get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'effort', 'intell', 'overall']]
    scores_PARE = get_mean_scores("/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")[['utt', 'effort', 'intell', 'overall']]
    scores_CTRL = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'effort', 'intell', 'overall']]
    df = xvectors.merge(pd.concat([scores_LARY, scores_CTRL, scores_PARE]), on='utt')

    all_results = pd.DataFrame(columns=RESULTS_COLUMNS)
    

    ### sklearn models

    all_results = all_results.append( run_sklearn_model(
        "Linear Regression", sklearn.linear_model.LinearRegression(), df), ignore_index=True)

    all_results = all_results.append( run_sklearn_model(
        "SVR [linear]", sklearn.svm.SVR(kernel='linear'), df), ignore_index=True)
    all_results = all_results.append( run_sklearn_model(
        "SVR [poly]", sklearn.svm.SVR(kernel='poly'), df), ignore_index=True)
    all_results = all_results.append( run_sklearn_model(
        "SVR [rbf]", sklearn.svm.SVR(kernel='rbf'), df), ignore_index=True)
    
    all_results = all_results.append( run_sklearn_model(
        "Random Forest Regressor", sklearn.ensemble.RandomForestRegressor(), df), ignore_index=True)


    ### PyTorch models

    for _num_epochs in [200, 500, 1000]:
        for lr in [1e-3, 1e-4]:

            all_results = all_results.append( run_pytorch_model(
                "LinearModel", df, 
                LinearModel(), criterion=torch.nn.L1Loss(), lr=lr, num_epochs=_num_epochs
                ), ignore_index=True)

            all_results = all_results.append( run_pytorch_model(
                "Feed-forward", df, 
                NonlinearRegressionModel(), criterion=torch.nn.L1Loss(), lr=lr, num_epochs=_num_epochs
                ), ignore_index=True)
    

    # save results
    filename = 'xvectors/{}/predict_scoring_results__{}.pkl'.format(args.nnet_name, datetime.utcnow().strftime(r'%Y%m%dT%H%M%s'))
    print(f"Saving results to {filename}...")
    all_results.to_pickle(filename)
    print("Done.")
