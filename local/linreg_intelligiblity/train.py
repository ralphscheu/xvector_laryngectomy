import sklearn
import pandas as pd
import numpy as np
from kaldiio import ReadHelper
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr


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
    xvectors = get_extracted_xvectors("scp:xvectors/pathologic_voices_CTRL_LARY/xvector.scp")
    scores_LARY = get_mean_scores("/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.crits", "/mnt/speechdata/pathologic_voices/laryng41/labels/laryng41.raters5.scores")[['utt', 'overall']]
    scores_CTRL = get_mean_scores("/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.crits", "/mnt/speechdata/pathologic_voices/altersstimme110_cut/labels/altersstimme110_cut.logos.scores")[['utt', 'overall']]

    df = xvectors.merge(pd.concat([scores_LARY, scores_CTRL]), on='utt')
    df = df.loc[df.speaker_group == "LARY"]
    X, y = np.vstack(df.embedding.values), df.overall.values

    X_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_scaled, y_scaled = X_scaler.fit_transform(X), y_scaler.fit_transform(y.reshape(-1, 1))
    print("Pearson Correlation (r, p-value):", pearsonr(X_scaled.mean(axis=1), y_scaled.mean(axis=1)))

    # y = StandardScaler().fit_transform(y.reshape(-1,1))
    # y = (y-5)/5

    linreg = LinearRegression()
    cvscores = pd.DataFrame(cross_validate(linreg, X, y, cv=LeaveOneOut(), scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))).drop(columns=['fit_time', 'score_time'])
    print("\n=== LOO - laryng")
    print(cvscores.describe().loc[['mean', 'std']])


