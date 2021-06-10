#!/usr/bin/env python3
import pandas as pd
from kaldiio import ReadHelper


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
