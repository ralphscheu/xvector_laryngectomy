#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import argparse
from utils import get_extracted_xvectors, get_mean_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnet-name', default="xvector", help='nnet name')
    args = parser.parse_args()

    xvectors = get_extracted_xvectors("scp:xvectors/{}/pathologic_voices/xvector.scp".format(args.nnet_name))
    scores_LARY = get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'effort', 'intell', 'overall']]
    scores_PARE = get_mean_scores("/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")[['utt', 'effort', 'intell', 'overall']]
    scores_CTRL = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'effort', 'intell', 'overall']]
    df = xvectors.merge(pd.concat([scores_LARY, scores_CTRL, scores_PARE]), on='utt')
    

    print("\n== laryng: overall ==")
    X, y = np.vstack(df.loc[df.speaker_group == "LARY"].embedding.values), df.loc[df.speaker_group == "LARY"].overall.values
    X_scaled , y_scaled = StandardScaler().fit_transform(X) , StandardScaler().fit_transform(y.reshape(-1, 1))

    cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    r, pvalue = pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1))

    print("Pearson Correlation -  r: {}, p-value: {}".format(r, pvalue))
    print(cvscores.describe().loc[['mean', 'std']])

    ###

    print("\n== laryng: intell ==")
    X, y = np.vstack(df.loc[df.speaker_group == "LARY"].embedding.values), df.loc[df.speaker_group == "LARY"].intell.values
    X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))

    cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    r, pvalue = pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1))

    print("Pearson Correlation -  r: {}, p-value: {}".format(r, pvalue))
    print(cvscores.describe().loc[['mean', 'std']])

    ###

    print("\n== laryng: effort ==")
    X, y = np.vstack(df.loc[df.speaker_group == "LARY"].embedding.values), df.loc[df.speaker_group == "LARY"].effort.values
    X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))

    cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    r, pvalue = pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1))

    print("Pearson Correlation -  r: {}, p-value: {}".format(r, pvalue))
    print(cvscores.describe().loc[['mean', 'std']])

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
