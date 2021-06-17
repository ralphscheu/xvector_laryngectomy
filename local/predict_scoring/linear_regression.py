#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_validate
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import argparse
from utils import get_extracted_xvectors, get_mean_scores
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn


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
    kfold = KFold(n_splits=k_folds)
    loo = LeaveOneOut()

    print("=================================================")
    print("        Linear Regression (sklearn)")
    print("=================================================")
    for crit in ['overall', 'intell','effort']:
        for sel_speaker_groups in sel_group_combinations:

            dataset = df.loc[df.speaker_group.isin(sel_speaker_groups)]
            X = np.vstack(dataset.embedding)
            y = dataset[crit].values.reshape((-1, 1))

            fold_results = {'mse': {}, 'mae': {}, 'r2': {}}
            for fold, (train_ids, test_ids) in enumerate( kfold.split(X) ):
                X_train = X[train_ids]
                y_train = y[train_ids]

                X_test = X[test_ids]
                y_test = y[test_ids]

                regr = LinearRegression()
                regr.fit(X_train, y_train)

                y_pred = regr.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                fold_results['mse'][fold] = mse
                fold_results['mae'][fold] = mae
                fold_results['r2'][fold] = r2
        
            # pearson r correlation over dataset
            r, pvalue = pearsonr(
                StandardScaler().fit_transform( X ).mean(axis=1),
                StandardScaler().fit_transform( y ).mean(axis=1),
                )
            print(f"{sel_speaker_groups} {crit} - ( Pearson r {r}, p-value: {pvalue})")

            # evaluate crossvalidation results
            print(f'LinearRegression - {k_folds}-fold cross validation')
            sum = 0.0
            for key, value in fold_results['mse'].items():
                sum += value
            print(f'Avg MSE:', sum/len(fold_results['mse'].items()))

            sum = 0.0
            for key, value in fold_results['mae'].items():
                sum += value
            print(f'Avg MAE:', sum/len(fold_results['mae'].items()))

            sum = 0.0
            for key, value in fold_results['r2'].items():
                sum += value
            print(f'Avg R-squared:', sum/len(fold_results['r2'].items()))
            print()

        print("-----------\n")
