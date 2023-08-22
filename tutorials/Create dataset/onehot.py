import pandas as pd
import numpy as np
import h5py

genes = pd.read_csv('genes_withTSS2.csv', index_col=0)
genes = genes.astype({"TSS": int})

def read_fasta( fasta_path, genes ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    genesused=[]
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                chrom_loc = line.replace('>', '').strip().split('-')[0]
#                 print(chrom_loc)
                chrom1, chrom2, loc = chrom_loc.split('_')
#                 print(chrom)
#                 print(loc)
                chrom = chrom1 + '_' + chrom2
                idx = np.where((genes['Chr'] == chrom) & (genes['TSS'] == int(loc)))[0]
                geneid = genes.index[idx[0]]
                if np.isin(geneid, genesused):
                    geneid = genes.index[idx[1]]
                    if np.isin(geneid, genesused):
                        geneid = genes.index[idx[2]]
                        if np.isin(geneid, genesused):
                            geneid = genes.index[idx[3]]
                            if np.isin(geneid, genesused):
                                geneid = genes.index[idx[4]]
                sequences[ geneid ] = ''
                genesused.append(geneid)
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ geneid ] += ''.join( line.split() ).upper()
    return sequences
    
TSS_all = read_fasta('TSS_all2.fa', genes)   
print('Finished reading')

seqlen = 95001
features_all = np.zeros((len(TSS_all),seqlen,4))
alphabet = 'ACTG'
ohdict = dict((c, i) for i, c in enumerate(alphabet))
geneIDs = []
j=0

for geneID, seq in TSS_all.items():
    if len(seq) != 95001:
        print(geneID)
        continue

    if(j%1000 == 0):
        print(j)
    geneIDs.append(geneID)
    features = np.zeros((seqlen, len(ohdict)), dtype=np.float32)
    for i in range(seqlen):
        if(seq[i] == 'N'):
            features[i] = 1/4
        elif(seq[i] == 'Y'):
            features[i,1] = 1/2
            features[i,2] = 1/2
        elif(seq[i] == 'R'):
            features[i,0] = 1/2
            features[i,3] = 1/2
        elif(seq[i] == 'K'):
            features[i,2] = 1/2
            features[i,3] = 1/2
        elif(seq[i] == 'M'):
            features[i,1] = 1/2
            features[i,0] = 1/2
        elif(seq[i] == 'S'):
            features[i,1] = 1/2
            features[i,3] = 1/2
        elif(seq[i] == 'W'):
            features[i,0] = 1/2
            features[i,2] = 1/2
        elif(seq[i] == 'B'):
            features[i,1] = 1/3
            features[i,2] = 1/3
            features[i,3] = 1/3
        elif(seq[i] == 'D'):
            features[i,0] = 1/3
            features[i,2] = 1/3
            features[i,3] = 1/3
        elif(seq[i] == 'H'):
            features[i,1] = 1/3
            features[i,2] = 1/3
            features[i,0] = 1/3
        elif(seq[i] == 'V'):
            features[i,1] = 1/3
            features[i,0] = 1/3
            features[i,3] = 1/3
        else:
            features[i, ohdict[seq[i]]] = 1
    features_all[j] = features
    j=j+1

print(np.sum(features_all)/len(TSS_all))

compress_args = {'compression': 'gzip', 'compression_opts': 1}

geneIDs = np.asarray(geneIDs).astype('S35')

hf = h5py.File('TSS_onehot2.h5', 'w')
hf.create_dataset('TSS', data=features_all, **compress_args)
hf.create_dataset('geneIDs', data=geneIDs, **compress_args)
hf.close()