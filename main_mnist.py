import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import os
import model_mnist as model
from trainer import Trainer
from dataloader import get_train_valid_loader, get_test_loader

parser = argparse.ArgumentParser(description='MNIST task')
parser.add_argument('--data', type=str, default='data/mnist/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--ninp', type=int, default=1,
                    help='size of each pixel')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.002,
                    help='initial learning rate')
parser.add_argument('--final_lr', type=float, default=0.001)
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping') ############## it seems that there is no clipping
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay applied to all weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--norm', type=str, default='')
parser.add_argument('--method', type=str, default='')
parser.add_argument('--pmnist', default=False, action='store_true', help='If set, it uses permutated-MNIST dataset')
parser.add_argument('--shared', default=False, action='store_true', help='If set, it uses shared mean and var stats for all time steps')
parser.add_argument('--quantize', default=False, action='store_true')


args = parser.parse_args()
# args.quantize = True
os.makedirs('checkpoints', exist_ok=True)
savename = 'checkpoints/new_mnist_' + str(args.nhid) +'_lr_' + str(args.lr) + '_finallr_' + \
           str(args.final_lr) + '_clip_' + str(args.clip) + '_bsz_' + str(args.batchsize) + \
    '_optimizer_' + args.optimizer
tmp = args.method if args.quantize else ''
savename += '_' + tmp
tmp = args.norm if args.norm else ''
savename += '_' + tmp
tmp = '_perm' if args.pmnist else ''
args.save = savename + tmp


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### permuted mnist
if args.pmnist:
    perm = torch.randperm(784)
else:
    perm = torch.arange(0, 784).long()

train_loader, valid_loader = get_train_valid_loader(args.data, args.batchsize, perm, shuffle=True)
test_loader = get_test_loader(args.data, args.batchsize, perm)


model = model.mnistModel(args.model, args.ninp, args.nhid, args.nlayers, args, quantize=args.quantize)
model.to(device)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
params_fp = []
params_invariant = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'full_precision' in name:
            params_fp.append(param)
        else:
            params_invariant.append(param)

params = params_fp + params_invariant
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)


# Learning Rate
lr = args.lr
final_lr = args.final_lr
args.lr_decay = (final_lr / lr) ** (1. / args.epochs)



trainer = Trainer(optimizer, criterion,params_invariant, args, )
trainer.train(model, train_loader, valid_loader, test_loader)

