python -u run_scEP_cp.py --dir='Zenodo/human/' --train_file='human_seq_hl.h5' \
--label_dir='Zenodo/human/M1/subclass' --label_file='subclass.csv' \
--cols='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19' \
--upstream=7000 --downstream=3500 --startfold=0 --endfold=20 \
--numgenes=18138 --output='output/' --numruns=5 --numepochs=40