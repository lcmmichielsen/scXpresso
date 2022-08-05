# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:24:26 2021

@author: lcmmichielsen
"""

import argparse
import os
import shutil
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

from scEP_cp import scEPdata, CNN1D, train, test

# Check if GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("[*] Selected device: ", device)

# Parse all arguments
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dir',            dest='dir',             default='/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/predict_expression/TM_data/general_datasets/',
                    help='Directory with the sequences and half-life time')
parser.add_argument('--train_file',     dest='train_file',      default='complete.h5',
                    help='Filename of the sequences and half-life time')
parser.add_argument('--label_dir',      dest='label_dir',       default='/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/predict_expression/TM_data/muscle_10X/',
                    help='Directory with the labels')
parser.add_argument('--label_file',     dest='label_file',      default='logmean.csv',
                    help='Filename of the labels (pseudobulk expression values)')
parser.add_argument('--cols',           dest='cols',            default='1,2,3',
                    help='Index of columns used to train the model (separated by commas)')
parser.add_argument('--upstream',       dest='upstream',        type=int,       default=7000,
                    help='Number of nucleotides upstream of TSS')
parser.add_argument('--downstream',     dest='downstream',      type=int,       default=3500,
                    help='Number of nucleotides downstream of TSS')
parser.add_argument('--startfold',      dest='startfold',       type=int,       default=0,
                    help='During 20fold CV, first fold to train here (if you want to parellize the training)')
parser.add_argument('--endfold',        dest='endfold',         type=int,       default=20,
                    help='During 20fold CV, last fold to train here (if you want to parellize the training)')
parser.add_argument('--numgenes',       dest='numgenes',        type=int,       default=20467,
                    help='Number of genes in the dataset')
parser.add_argument('--output',         dest='output',          type=str,       default='output',
                    help='Directory to save the model and predictions')
parser.add_argument('--numruns',        dest='numruns',         type=int,       default=5,
                    help='Number of models trained')
parser.add_argument('--numepochs',      dest='numepochs',       type=int,       default=40,
                    help='Number of epochs to train the model')

args = parser.parse_args()

general_dir = args.dir
train_file = args.train_file
label_dir = args.label_dir
label_fn = args.label_file
cols = np.asarray(args.cols.split(','), dtype=int)
upstream = args.upstream
downstream = args.downstream
startfold = args.startfold
numfold = args.endfold
numgenes = args.numgenes
output = args.output
numruns = args.numruns
numepochs = args.numepochs

# Find number of cell pop. in the data
os.chdir(label_dir)
num_ct = len(cols)

# 20 fold CV
idx_all = np.arange(0,numgenes)
kf = KFold(n_splits=20, shuffle=True, random_state=1)
counter=0

for train_val_idx, idx_test in kf.split(idx_all):
    if counter == startfold:
        os.chdir(label_dir)
       
        # Split the genes NOT in the test set in a training and validation set
        np.random.seed(startfold)
        idx_rest = np.random.permutation(numgenes-len(idx_test))
        idx_val = train_val_idx[idx_rest[:1000]]
        idx_train = train_val_idx[idx_rest[1000:]]
        
        # Create dataloaders for the train, validation, and test set
        input_data = general_dir + train_file
        train_set = scEPdata(input_data, label_fn, cols, idx_train, idx_train,
                             upstream=upstream, downstream=downstream)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        train_loader_eval = DataLoader(train_set, batch_size=1, shuffle=False)
        
        val_set = scEPdata(input_data, label_fn, cols, idx_val, idx_train,
                           upstream=upstream, downstream=downstream)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        
        test_set = scEPdata(input_data, label_fn, cols, idx_test, idx_train,
                            upstream=upstream, downstream=downstream)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        
        # Create output directory
        ckpt_dir_ = 'ckpt_dir'
        logs_dir_ = 'logs_dir'
        output_dir = output + str(startfold)
        print('Output directory: ', output_dir)
        
        try:
            os.mkdir(output_dir)
        except:
            print('Output already exists')    
        
        os.chdir(output_dir)
        
        # Train the five models
        i = 0
        
        while i < numruns:
            
            net = CNN1D(num_ct, upstream=upstream, downstream=downstream).to(device)
        
            print("[*] Initialize model successfully")
            print(net)
            print("[*] Number of model parameters:")
            print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        
            ckpt_dir = ckpt_dir_ + str(i)
            logs_dir = logs_dir_ + str(i)
        
            try:
                os.mkdir(ckpt_dir)
                os.mkdir(logs_dir)
            except:
                print('Dir already exists')
                
            original_stdout = sys.stdout
            
            with open(logs_dir + '/model.txt', 'w') as tf:
                sys.stdout = tf
                print(net)
                sys.stdout = original_stdout
            
            criterion = torch.nn.MSELoss().to(device)
            init_lr = 0.0005
            lr_sched = True
            
            train(device=device, net=net, criterion=criterion,
                      learning_rate=init_lr, lr_sched= lr_sched, num_epochs=numepochs,
                      train_loader=train_loader, train_loader_eval=train_loader_eval, 
                      valid_loader=val_loader, ckpt_dir=ckpt_dir, logs_dir=logs_dir)
            
            model_file = ckpt_dir + '/model_epoch' + str(numepochs) + '.pth.tar'
            save_file = logs_dir + '/results_testdata.pkl'
            test(device=device, net=net, criterion=criterion, model_file=model_file,
                        test_loader=test_loader, save_file=save_file)
            
            model_file = ckpt_dir + '/model_best.pth.tar'
            save_file = logs_dir + '/results_testdata_best.pkl'
            test(device=device, net=net, criterion=criterion, model_file=model_file,
                        test_loader=test_loader, save_file=save_file)
        
            model_file = ckpt_dir + '/model_best.pth.tar'
            save_file = logs_dir + '/results_valdata_best.pkl'
            test(device=device, net=net, criterion=criterion, model_file=model_file,
                        test_loader=val_loader, save_file=save_file)
            
            model_file = ckpt_dir + '/model_best.pth.tar'
            save_file = logs_dir + '/results_traindata_best.pkl'
            test(device=device, net=net, criterion=criterion, model_file=model_file,
                        test_loader=train_loader_eval, save_file=save_file)
            
            i += 1
            
            y = pd.read_pickle(save_file)
            y_pred = np.asarray(y['y_pred'])
            
            # Sometimes the model gets stuck in a local optimum & predicts
            # same value for every gene --> check if this is the case
            # If so, redo the run
            if np.var(y_pred[:,0]) < 1e-5:
                i -= 1
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                shutil.rmtree(logs_dir, ignore_errors=True)

        
        numfold = numfold - 1
        if numfold > 0:
            startfold = startfold + 1
    
    counter=counter+1



