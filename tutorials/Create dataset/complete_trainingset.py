import pandas as pd
import numpy as np
import h5py
import os

## We need to read the half life time features
hl_feature = pd.read_csv('HL_normalized.csv', index_col=0)
print(np.shape(hl_feature))

## We need to read the sequences
file = h5py.File('TSS_onehot2.h5', 'r')
seq, geneIDs = file['TSS'], file['geneIDs']
seq = np.asarray(seq)
geneIDs = np.asarray(geneIDs).astype('U30')
file.close()

print(np.shape(seq))

## Intersect such that the order of the genes is the same
_, idx_hl, idx_seq = np.intersect1d(np.asarray(hl_feature.index, dtype=str), geneIDs, return_indices=True)
X_halflife = hl_feature.values[idx_hl]
seq = seq[idx_seq]
geneNames = geneIDs[idx_seq]

print(len(geneNames))

## Save datasets
os.chdir('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/predict_expression/M1_data/human')

geneNames = np.asarray(geneNames).astype('S30')


compress_args = {'compression': 'gzip', 'compression_opts': 1}

hf = h5py.File('complete.h5', 'w')
hf.create_dataset('data', data=X_halflife, **compress_args)
hf.create_dataset('promoter', data=seq, **compress_args)
hf.create_dataset('geneName', data=geneNames, **compress_args)
hf.close()





