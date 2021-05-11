#!/usr/bin/env python
import re

utts_LA = []
utts_CTRL = []
with open("data/pathologic_voices_CTRL_LARY_no_sil/wav.scp") as f:
    for line in f:
        if re.search(r'(\w+)_', line.split()[0])[1] == 'LA':
            utts_LA.append(line.split()[0])
        else:
            utts_CTRL.append(line.split()[0])

trials = []

for utt in utts_LA:
    for other in utts_CTRL:
        trials.append("{} {} nontarget".format( utt, other ))

    other_utts_in_group = set(utts_LA) - set(utt) 
    for o_utt in other_utts_in_group:
        trials.append("{} {} target".format( utt, o_utt ))

for utt in utts_CTRL:
    for other in utts_LA:
        trials.append("{} {} nontarget".format( utt, other ))

    other_utts_in_group = set(utts_CTRL) - set(utt) 
    for o_utt in other_utts_in_group:
        trials.append("{} {} target".format( utt, o_utt ))

# write to trials file
with open("data/pathologic_voices_CTRL_LARY_no_sil/trials", "w") as f:
    for trial_line in trials:
        f.write("{}\n".format( trial_line ))
