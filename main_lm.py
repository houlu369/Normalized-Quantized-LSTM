import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import os
import data
import model_lm as model
import io
from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--lr_decay', type=float,  default=1.2)
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropouto', type=float, default=0.4,
                    help='dropout applied to output layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')

parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')

parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--test', action='store_true')
parser.add_argument('--norm', type=str, default='')
parser.add_argument('--method', type=str, default='')
parser.add_argument('--char',action='store_true',)
parser.add_argument('--tied', action='store_true')
parser.add_argument('--keep-lr', action='store_true')
parser.add_argument('--no-warm-start', action='store_true')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--shared', default=False, action='store_true', help='If set, it uses shared mean and var stats for all time steps')



args = parser.parse_args()
args.quantize = True
os.makedirs('checkpoints', exist_ok=True)
dataset = args.data.split('/')[1]
need_quantized = '' if args.quantize else '_no_quantized'
keeplrlr = 'keeplr_' if args.keep_lr else ''
args.savefile = 'new_' + dataset + '_' + args.norm + 'norm_' + keeplrlr + args.method + '_' + \
                args.save + '_lr_{}_'.format(args.lr) + \
                need_quantized + '_lrdecay_{}'.format(args.lr_decay) + '_{}.log'.format(args.nhid)
args.savefile = os.path.join('checkpoints', args.savefile)
args.save = os.path.join('checkpoints', args.save)
args.resume = os.path.join('checkpoints', args.resume) if args.resume else args.resume

if args.test and not args.resume:
    print("You must specify resume file at test mode")
    exit()
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
if args.method == 'binary_connect' or args.method == 'ternary_connect':
    if args.keep_lr or args.char:
        lr_scale = 1.
    else:
        lr_scale = 2 * math.sqrt(args.nhid)
else:
    lr_scale = 1.

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 64
test_batch_size = 64
if not args.augment:
    train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)




criterion = None
optimizer = None
ntokens = len(corpus.dictionary)
print('ntokens: {}'.format(ntokens))
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args, args.dropouto, args.dropoute,
                           args.dropouth, args.dropouti, args.tied, args.quantize)

###
if not criterion:
    criterion = nn.CrossEntropyLoss()

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
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

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden, optimizer)
        total_loss += len(data) * criterion(output, targets).data
        if args.no_warm_start:
            hidden = model.init_hidden(batch_size)
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(train_data):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    return_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        seq_len = args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, optimizer, return_h=True)
        if args.no_warm_start:
            hidden = model.init_hidden(args.batch_size)
        raw_loss = criterion(output, targets)

        loss = raw_loss
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params_invariant, args.clip)
        if args.quantize:
            model.optim_grad(optimizer)
        optimizer.step()

        total_loss += raw_loss.data
        return_loss += raw_loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:6.3f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return return_loss.item()/(batch - 1)

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000


if not args.test:
    try:
        if not optimizer:
            # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD([
                    {'params':params_fp, 'lr': args.lr * lr_scale},
                    {'params':params_invariant, 'lr': args.lr}
                ], lr=args.lr, weight_decay=args.wdecay)
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam([
                    {'params':params_fp, 'lr': args.lr * lr_scale},
                    {'params':params_invariant, 'lr': args.lr}
                ], lr=args.lr, weight_decay=args.wdecay)
        if os.path.exists(args.savefile):
            os.system('rm {}'.format(args.savefile))
        for epoch in range(1, args.epochs+1):
            if args.augment:
                offset = np.random.randint(args.bptt)
                train_data = batchify(corpus.train[offset:], args.batch_size, args)

            epoch_start_time = time.time()
            train_loss = train(train_data)

            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:6.3f}s | valid loss {:6.3f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            # Run on test data.
            test_loss = evaluate(test_data, test_batch_size)
            print('=' * 89)
            print('| End of epoch {:3d} | test loss {:6.3f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(epoch,
                test_loss, math.exp(test_loss), test_loss / math.log(2)))
            print('=' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss
                is_best = True
            else:
                is_best = False
                if args.optimizer == 'sgd':
                    print('divide learning rate by {}'.format(args.lr_decay))
                    optimizer.param_groups[0]['lr'] /= args.lr_decay  ####
                    optimizer.param_groups[1]['lr'] /= args.lr_decay
            if args.optimizer == 'adam':
                if epoch > 10 :
                    if args.data.split('/')[1] != 'text8':
                        print('multiply learning rate by 0.98 ')
                        optimizer.param_groups[0]['lr'] *= 0.98
                    else:
                        print('keep learning rate as lu said')
            with io.open(args.savefile, 'a', newline='\n', encoding='utf8', errors='ignore') as tgt:
                msg = 'epoch {:3d} train_loss {:6.3f} train_ppl {:8.2f} train_bpc {:8.3f}\n'.format(
                    epoch, train_loss,math.exp(train_loss), train_loss / math.log(2) )
                msg = msg + '          valid_loss {:6.3f} valid_ppl {:8.2f} valid_bpc {:8.3f}\n'.format(
                    val_loss, math.exp(val_loss), val_loss / math.log(2))
                msg = msg + '           test_loss {:6.3f}  test_ppl {:8.2f}  test_bpc {:8.3f}\n'.format(test_loss, math.exp(test_loss), test_loss / math.log(2))
                msg = msg + "epoch {:3d} ".format(epoch) +('new best**************\n' if is_best else 'no best\n' )
                tgt.write(msg)
            best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if args.test:
    test_loss = evaluate(test_data, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)
