from __future__ import print_function
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import maskedMSE, maskedMSETest

from traj_data_set import trajDatasetMAY
from torch.utils.data import DataLoader
import torchvision
# from m_traj_1T import highwayNet_2T
import math
import time
# command line arguments
def parse_args(cmd_args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('-m', '--modeltype', type=str, default='33CNN_321', help="Type of the model to be trained: SL-single lstm, 33CNN-sqrcnn, W33CNN-sqrcnn with learned coefficients")

    parser.set_defaults(render=False)
    return parser.parse_args(cmd_args)

# Parse arguments
cmd_args = sys.argv[1:]
cmd_args = parse_args(cmd_args)

if cmd_args.modeltype =='SL':
    from m_traj_1T import highwayNet_2T as Net
    print('training Single LSTM')
elif cmd_args.modeltype =='9FC':
    from m_traj_2T_fc import highwayNet_2T as Net
    print('training 9 Fully Connected')
elif cmd_args.modeltype =='1T33CNN':
    from m_traj_1T_cnn import highwayNet_2T as Net
    print('training 1 Tube CNN')
elif cmd_args.modeltype =='33CNN':
    from m_traj_2T import highwayNet_2T as Net
    print('training 33 SQR CNN')
elif cmd_args.modeltype =='W33CNN':
    from m_traj_weight_2T import highwayNet_2T as Net
    print('training 33 SQR CNN with learning coefficients')
elif cmd_args.modeltype =='33CNN_321':
    from m_traj_2T_cnn321 import highwayNet_2T as Net
    print('training 33 SQR CNN 321')
elif cmd_args.modeltype =='1T33CNN_321':
    from m_traj_1T_cnn321 import highwayNet_2T as Net
    print('training interaction only 33 SQR CNN 321')
elif cmd_args.modeltype =='33CNN_31':
    from m_traj_2T_cnn31 import highwayNet_2T as Net
    print('training 33 SQR CNN 31')
elif cmd_args.modeltype =='33CNN_no1x1':
    from m_traj_2T_no1x1 import highwayNet_2T as Net
    print('training 33 SQR CNN without 1x1 convs')
else:
    print('\n choose a proper model type to train. \n')

## Network Arguments
args = {}
args['use_cuda'] = False

args['encoder_size'] = 32 #64
args['decoder_size'] = 64 #128
args['in_length'] = 16
args['out_length'] = 10
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32 # 32
args['input_embedding_size'] = 16 # 32
args['rondom_seed'] = 1
args['train_epoches'] =  20 #20
args['batch_size'] = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args['device'] = device
# device = 'cpu'
print('\ndevice {}\n'.format(device))
## set random seeds
torch.manual_seed(args['rondom_seed'])
if device != 'cpu':
    print('running on {}'.format(device))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args['rondom_seed'])
    torch.cuda.manual_seed_all(args['rondom_seed'])
    print('seed setted! {}'.format(args['rondom_seed']))
random.seed(args['rondom_seed'])
np.random.seed(args['rondom_seed'])
# torch.manual_seed(seed)

# Initialize network
net = Net(args)
net.to(device)
# print(net)
## Initialize optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=0.001) # lr 0.0035, batch_size=4 or 8.


full_train_set = trajDatasetMAY(hist_path= 'Hist_LC_train.npy', fut_path='Fut_LC_train.npy') # HIST_1w FUT_1w HIST FUT
val_set = trajDatasetMAY(hist_path= 'Hist_LC_test.npy', fut_path='Fut_LC_test.npy') # HIST_1w FUT_1w HIST FUT

print('totally {} data pieces for training'.format(len(full_train_set) + len(val_set)))
print('train \t{} / {}'.format(len(full_train_set), len(full_train_set) + len(val_set)))
print('val \t{} / {}'.format(len(val_set), len(full_train_set) + len(val_set)))

trDataloader = DataLoader(full_train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4,collate_fn=full_train_set.collate_fn)
valDataloader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4,collate_fn=val_set.collate_fn)

tic = time.time()
for ep in range(args['train_epoches']):
    running_loss = 0.0
    for i, data in enumerate(trDataloader):

        hh, ff = data
        hh = hh[:,::2,:].to(device)
        ff = ff[:,4::5,:].to(device)

        ## zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        fut_pred = net(hh)
        op_mask = torch.ones(ff.shape)
        l = maskedMSE(fut_pred, ff, op_mask)
        l.backward()
        # a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # print statistics
        running_loss += l.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('ep {}, {} batches, loss - {}'.format( ep + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
    

    ###_____________________________________________________________________________________________________________________________________________
    ## no validation
## save model

save_model_to_PATH = './trained_models/{}_EP{}BS{}_{}.tar'.format(cmd_args.modeltype, ep+1, args['batch_size'], 0)
if not os.path.exists('./trained_models/'):
    os.mkdir('./trained_models/')
torch.save(net.state_dict(), save_model_to_PATH)

###_____________________________________________________________________________________________________________________________________________
## test
lossVals = torch.zeros(10)
counts = torch.zeros(10)

test_net = Net(args)
test_net.to(device)
test_net.load_state_dict(torch.load(save_model_to_PATH))
with torch.no_grad():
    print('Testing no grad')
    for i, data in enumerate(valDataloader):
        hh, ff = data
        hh = hh[:,::2,:].to(device)
        ff = ff[:,4::5,:].to(device)

        fut_pred = test_net(hh)
        fut_pred = fut_pred.permute(1,0,2)
        ff = ff.permute(1,0,2)
        op_mask = torch.ones(ff.shape)
        l, c = maskedMSETest(fut_pred, ff, op_mask)

        lossVals +=l.detach()
        counts += c.detach()

print(torch.pow(lossVals / counts,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters

testLoss = torch.pow(lossVals / counts,0.5)*0.3048
torch.save(testLoss, './trained_models/loss{}.pt'.format(cmd_args.modeltype))
tac = time.time()
print('Finished Training for {} epoches in {} minutes'.format(args['train_epoches'], round((tac-tic)/60, 2)))
save_model_to_PATH = './trained_models/{}_EP{}BS{}_final.tar'.format(cmd_args.modeltype, ep+1, args['batch_size'])
torch.save(net.state_dict(), save_model_to_PATH)
