import pandas as pd
import numpy as np

x = pd.read_csv('gencode.v22.annotation.gtf', skiprows=5, sep='\t', header=None)
idx = np.where((x[0] != 'chrM') & (x[0] != 'chrY'))[0]
x = x.iloc[idx]
idx = np.where(x[2] != 'gene')[0]
x = x.iloc[idx]
x = x.drop(columns=[1,5,7])
x = x.rename(columns={0: "chr", 2: "type",
                 3: "start", 4: "end",
                 6: "strand", 8: "rest"})
print(x.iloc[:5])
x['gene'] = x['rest'].str.split(';').str[0].str.split(' ').str[1]
x['genename'] = x['rest'].str.split(';').str[4].str.split(' ').str[2]
x['transcriptID'] = x['rest'].str.split(';').str[1].str.split(' ').str[2]
x = x.drop(columns='rest')
print(x.iloc[:5])
idx_cds = np.where(x['type'] == 'CDS')[0]
x_cds = x.iloc[idx_cds]
transcript_cds = np.unique(x_cds['transcriptID'])
tokeep = np.isin(x['transcriptID'], transcript_cds)
print(np.sum(tokeep)/len(tokeep))
x_CDS = x.iloc[tokeep]
x_CDS.to_csv('gencode.v22.annotation.CDS.csv')