# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:02:13 2021

@author: lcmmichielsen
"""

import pickle
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from tensorboardX import SummaryWriter
from sklearn.preprocessing import StandardScaler

def load_checkpoint(net, optimizer=None, scheduler=None, filename='model_last.pth.tar'):
    start_epoch = 0
    try:
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("\n[*] Loaded checkpoint at epoch %d" % start_epoch)
    except:
        print("[!] No checkpoint found, start epoch 0")

    return start_epoch

def evaluate_mut(device, net, seq, halflife):
    # Eval each sample
    net.eval()
    with torch.no_grad():   # set all 'requires_grad' to False
        # Get current batch and transfer to device
        x_seq = torch.from_numpy(seq).to(device, dtype=torch.float)
        x_hl = torch.from_numpy(halflife).to(device, dtype=torch.float)

        # Forward pass
        outputs = net(x_seq, x_hl)
        y_pred = outputs.cpu().numpy().squeeze()

    return y_pred


def evaluate(device, net, criterion, eval_loader):
    # Eval each sample
    net.eval()
    avg_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():   # set all 'requires_grad' to False
        for data in eval_loader:
            # Get current batch and transfer to device
            labels = data['label'].to(device, dtype=torch.float)
            x_seq = data['seq'].to(device, dtype=torch.float)
            x_hl = data['hl'].to(device, dtype=torch.float)
            
            # Forward pass
            outputs = net(x_seq, x_hl)
            current_loss = criterion(outputs.squeeze(), labels.squeeze())
            avg_loss += current_loss.item() / len(eval_loader)
            y_true.append(labels.cpu().numpy().squeeze())
            y_pred.append(outputs.cpu().numpy().squeeze())

    return avg_loss, y_true, y_pred


def train(device, net, criterion, learning_rate, lr_sched, num_epochs, 
          train_loader, train_loader_eval, valid_loader, ckpt_dir, logs_dir,
         evaluate_train = True, save_step = 10):
    
    best_valid_loss=100
    
    logger = SummaryWriter(logs_dir)
    
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.1, patience=5)

    start_epoch = load_checkpoint(net, optimizer, scheduler, 
                                 filename = ckpt_dir+'/model_last.pth.tar')
    
    # Evaluate validation set before start training
    print("[*] Evaluating epoch %d..." % start_epoch)
    avg_valid_loss, _, _ = evaluate(device, net, criterion, valid_loader)
    print("--- Average valid loss:                  %.4f" % avg_valid_loss)

    # Training epochs
    for epoch in range(start_epoch, num_epochs):
        net.train()
        # Print current learning rate
        print("[*] Epoch %d..." % (epoch + 1))
        for param_group in optimizer.param_groups:
            print('--- Current learning rate: ', param_group['lr'])

        for data in train_loader:
            # Get current batch and transfer to device
            labels = data['label'].to(device, dtype=torch.float)
            x_seq = data['seq'].to(device, dtype=torch.float)
            x_hl = data['hl'].to(device, dtype=torch.float)

            with torch.set_grad_enabled(True):  # no need to specify 'requires_grad' in tensors
                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass
                outputs = net(x_seq, x_hl)
                current_loss = criterion(outputs.squeeze(), labels.squeeze())
                # print(current_loss)
                # Backward pass and optimize
                current_loss.backward()
                optimizer.step()

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(state, ckpt_dir + '/model_last.pth.tar')

        # Save model at epoch
        if (epoch + 1) % save_step == 0:
            print("[*] Saving model epoch %d..." % (epoch + 1))
            torch.save(state, ckpt_dir + '/model_epoch%d.pth.tar' % (epoch + 1))
        
        # Evaluate all training set and validation set at epoch
        print("[*] Evaluating epoch %d..." % (epoch + 1))
        if evaluate_train:
            avg_train_loss, _, _ = evaluate(device, net, criterion, train_loader_eval)
            print("--- Average train loss:                  %.4f" % avg_train_loss)
            
            logger.add_scalar('train_loss_epoch', avg_train_loss, epoch + 1)

        avg_valid_loss, _, _ = evaluate(device, net, criterion, valid_loader)
        print("--- Average valid loss:                  %.4f" % avg_valid_loss)
        
        # Check if best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, ckpt_dir + '/model_best.pth.tar')

        
        logger.add_scalar('valid_loss_epoch', avg_valid_loss, epoch + 1)
        
        # LR scheduler on plateau (based on validation loss)
        if lr_sched:
            scheduler.step(avg_valid_loss)

    print("[*] Finish training.")

def test(device, net, criterion, model_file, test_loader, save_file=None):
    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)
    
    # Evaluate model
    avg_test_loss, y_true, y_pred_sigm = evaluate(device, net, criterion, test_loader)

    # Save predictions
    if save_file is not None:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred_sigm}, open(save_file, 'wb'))

    # Display evaluation metrics
    print("--- Average test loss:                  %.4f" % avg_test_loss)


class scEPdata(Dataset):
    def __init__(self, general_fn, exp_fn, cols, idx, idx_train, upstream=7000, downstream=3500):
                
        upstreamidx=70000-upstream
        downstreamidx=70000+downstream
                                
        hf = h5py.File(general_fn, 'r', libver='latest', swmr=True)
        self.sequences = np.asarray(hf['promoter'])[idx,upstreamidx:downstreamidx,:]
        self.hl = np.array(hf['data'])[idx]
        gn = np.asarray(hf['geneName']).astype('U30')[idx]
        gn_train = np.asarray(hf['geneName']).astype('U30')[idx_train]
        hf.close()
        print('Finished reading')
          
        all_labels = pd.read_csv(exp_fn, index_col=0)
        train_labels = all_labels.loc[gn_train].values[:,cols]
        labels = all_labels.loc[gn].values[:,cols]
        scaler = StandardScaler()
        scaler.fit(train_labels)
        
        self.labels = np.squeeze(scaler.transform(labels))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        seq = np.transpose(self.sequences[index]).astype(np.float32)
        hl = self.hl[index]           
        labels = self.labels[index].astype(np.float32)
        
        sample = {"seq": seq, "hl": hl, "label": labels}
        return sample

class CNN1D(nn.Module):
    def __init__(self, num_ct, upstream=7000, downstream=3500):
        super(CNN1D, self).__init__()
        
        # Calculate shape of output of CNN
        l = upstream+downstream 
        l = np.floor((l-6+4)/1)+1
        l = np.floor((l-30+2)/30)+1
        l = np.floor((l-9+8)/1)+1
        l = np.floor((l-10)/10)+1
        
        fc_in = int(l*32)

        # Convolutional layers
        conv_layers = []
        conv_layers.append(nn.Conv1d(4, 128, kernel_size = 6, padding = 2))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(30, padding = 1))
        conv_layers.append(nn.Conv1d(128, 32, kernel_size = 9, padding = 4))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(10))        
        conv_layers.append(nn.Flatten())
        
        self.conv = nn.Sequential(*conv_layers)
        
        #Fully connected layers
        fc_layers = []
        fc_layers.append(nn.Linear(fc_in, 64))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(p=0.00099))
        fc_layers.append(nn.Linear(64, num_ct))

        self.fc = nn.Sequential(*fc_layers)

                
    def forward(self, seq, hl):           
        
        x1 = self.conv(seq)
        
        x = self.fc(x1)
        
        return x
    

