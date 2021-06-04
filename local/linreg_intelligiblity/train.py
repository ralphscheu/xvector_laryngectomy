import sklearn
import pandas as pd
import numpy as np
from kaldiio import ReadHelper
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
import argparse


def get_mean_scores(f_crits, f_scores):
    criteria = pd.read_table(f_crits, header=None, encoding="iso-8859-1")[0].values
    mean_scores = pd.read_table(f_scores, ",", names=criteria)
    mean_scores.rename(columns={'Untersucher': 'rater', 'PatientNr': 'utt', 'file': 'utt', 'Anstr': 'effort', 'Verst': 'intell', 'Gesamt': 'overall'}, inplace=True)
    mean_scores = mean_scores.groupby(["utt"]).mean().reset_index()  # compute mean score across all raters
    return mean_scores

def get_extracted_xvectors(scp_input_string):
    xvectors_df = pd.DataFrame(columns=["utt", "speaker_group", "embedding"])
    with ReadHelper(scp_input_string) as reader:
        for key, mat in reader:
            xvectors_df = xvectors_df.append({
                'utt': key.split('_', maxsplit=1)[1], 
                'speaker_group': key.split('_', maxsplit=1)[0], 
                'embedding': mat}, ignore_index=True)
    return xvectors_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnet_name', default="xvector", help='nnet name')
    args = parser.parse_args()

    xvectors = get_extracted_xvectors("scp:xvectors/{}/pathologic_voices/xvector.scp".format(args.nnet_name))
    scores_LARY = get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'overall']]
    scores_PARE = get_mean_scores("/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.crits", "/mnt/speechdata/pathologic_voices/teilres85/labels/teilres85.experts1.scores")[['utt', 'overall']]
    scores_CTRL = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'overall']]
    df = xvectors.merge(pd.concat([scores_LARY, scores_CTRL, scores_PARE]), on='utt')
    

    print("\nLARYNG group")
    X, y = np.vstack(df.loc[df.speaker_group == "LARY"].embedding.values), df.loc[df.speaker_group == "LARY"].overall.values
    cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    print(cvscores.describe().loc[['mean', 'std']])
    

    print("\nPARE group")
    X, y = np.vstack(df.loc[df.speaker_group == "PARE"].embedding.values), df.loc[df.speaker_group == "PARE"].overall.values
    cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    print(cvscores.describe().loc[['mean', 'std']])


    print("\nCTRL group")
    X, y = np.vstack(df.loc[df.speaker_group == "CTRL"].embedding.values), df.loc[df.speaker_group == "CTRL"].overall.values
    cvscores = pd.DataFrame(cross_validate(LinearRegression(), X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    X_scaled, y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(y.reshape(-1, 1))
    print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))
    print(cvscores.describe().loc[['mean', 'std']])
