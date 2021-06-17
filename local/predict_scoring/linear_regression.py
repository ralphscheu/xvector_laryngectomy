#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_validate
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import argparse
from utils import get_extracted_xvectors, get_mean_scores
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
    loo = LeaveOneOut()

    print("=================================================")
    print("        Linear Regression (sklearn)")
    print("=================================================")
    for crit in ['overall', 'intell','effort']:
        for sel_speaker_groups in sel_group_combinations:

            dataset = df.loc[df.speaker_group.isin(sel_speaker_groups)]
            X = np.vstack(dataset.embedding)
            y = dataset[crit].values.reshape((-1, 1))

            fold_results = {'mse': {}, 'mae': {}}
            fold = 0
            for train_ids, test_ids in loo.split(X):
                xscaler, yscaler = StandardScaler(), StandardScaler()
                X_train = xscaler.fit_transform( X[train_ids] )
                y_train = yscaler.fit_transform( y[train_ids] )

                X_test = xscaler.transform( X[test_ids] )
                y_test = yscaler.transform( y[test_ids] )

                regr = LinearRegression()
                regr.fit(X_test, y_test)

                y_pred = regr.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                fold_results['mse'][fold] = mse
                fold_results['mae'][fold] = mae

                fold += 1
        
            # evaluate crossvalidation results
            r, pvalue = pearsonr(
                StandardScaler().fit_transform( X ).mean(axis=1),
                StandardScaler().fit_transform( y ).mean(axis=1),
                )
            print("Pearson Correlation -  r: {}, p-value: {}".format(r, pvalue))

            print(f'{sel_speaker_groups} {crit}: LeaveOneOut cross validation')
            sum = 0.0
            for key, value in fold_results['mse'].items():
                sum += value
            print(f'Average MSE:', sum/len(fold_results['mse'].items()))

            sum = 0.0
            for key, value in fold_results['mae'].items():
                sum += value
            print(f'Average MAE:', sum/len(fold_results['mae'].items()))
            print()

    # ###

    # print("\n== laryng: intell ==")
    # X, y = np.vstack(df.loc[df.speaker_group == "LARY"].embedding.values), df.loc[df.speaker_group == "LARY"].intell.values
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))

    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # r, pvalue = pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1))

    # print("Pearson Correlation -  r: {}, p-value: {}".format(r, pvalue))
    # print(cvscores.describe().loc[['mean', 'std']])

    # ###

    # print("\n== laryng: effort ==")
    # X, y = np.vstack(df.loc[df.speaker_group == "LARY"].embedding.values), df.loc[df.speaker_group == "LARY"].effort.values
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))

    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # r, pvalue = pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1))

    # print("Pearson Correlation -  r: {}, p-value: {}".format(r, pvalue))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\n===========================")
    

    # print("\nPARE group - overall")
    # X, y = np.vstack(df.loc[df.speaker_group == "PARE"].embedding.values), df.loc[df.speaker_group == "PARE"].overall.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\nPARE group - intell")
    # X, y = np.vstack(df.loc[df.speaker_group == "PARE"].embedding.values), df.loc[df.speaker_group == "PARE"].intell.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\nPARE group - effort")
    # X, y = np.vstack(df.loc[df.speaker_group == "PARE"].embedding.values), df.loc[df.speaker_group == "PARE"].effort.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])








    # print("\n===========================")
    # lary_pare = pd.concat([df.loc[df.speaker_group == "PARE"], df.loc[df.speaker_group == "LARY"]])

    # print("\nPARE/LARY group - overall")
    # X, y = np.vstack(lary_pare.embedding.values), lary_pare.overall.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\nPARE/LARY group - intell")
    # X, y = np.vstack(lary_pare.embedding.values), lary_pare.intell.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\nPARE/LARY group - effort")
    # X, y = np.vstack(lary_pare.embedding.values),lary_pare.effort.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])









    # print("\n===========================")


    # print("\nCTRL group - overall")
    # X, y = np.vstack(df.loc[df.speaker_group == "CTRL"].embedding.values), df.loc[df.speaker_group == "CTRL"].overall.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\nCTRL group - intell")
    # X, y = np.vstack(df.loc[df.speaker_group == "CTRL"].embedding.values), df.loc[df.speaker_group == "CTRL"].intell.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])

    # print("\nCTRL group - effort")
    # X, y = np.vstack(df.loc[df.speaker_group == "CTRL"].embedding.values), df.loc[df.speaker_group == "CTRL"].effort.values
    # cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    # print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    # print(cvscores.describe().loc[['mean', 'std']])
