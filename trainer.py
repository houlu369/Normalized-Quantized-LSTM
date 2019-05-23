import torch
import os
import time
import io

class Trainer(object):

    def __init__(self, optimizer, criterion, params_invariant ,args, ):
        self._optimizer = optimizer
        self._epoch = 1
        self.args = args
        self.criterion = criterion
        self.params_invariant = params_invariant
        self.validloss = 1000000
        self.testloss = 0

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer(self):
        return self._epoch

    def save_cktpt(self, model, fn):
        state_dict = {
            'optimizer_state' : self._optimizer.state_dict(),
            'epoch' : self._epoch,
            'model' : model.state_dict(),
            'validloss' : self.validloss,
            'testloss' : self.testloss,
        }
        torch.save(state_dict, fn)
        print('save checkpint for epoch {}'.format(self._epoch))

    def load_cktpt(self, model, fn):
        if not os.path.exists(fn):
            print('{} does not exist. This model will train from scratch'.format(fn))
            return False
        else:
            statedict = torch.load(fn)
            model.load_state_dict(statedict['model'])
            self._optimizer.load_state_dict(statedict['optimizer_state'])
            self._epoch = statedict['epoch'] + 1
            self.validloss = statedict['validloss']
            self.testloss = statedict['testloss']
            print("load checkpoint {}".format(statedict['epoch']))
            return True

    def evaluate_step(self, model, loader, test=False):
        model.eval()
        total_loss = 0
        true_case = 0
        total_case =0
        for i, sample in enumerate(loader):
            images, labels = sample
            images = images.transpose(0, 1) # seq length, bsz, input = 784, 100, 1
            images = images.cuda()
            labels = labels.cuda()
            output = model(images, self._optimizer)
            labels = labels.view(labels.size(0))
            raw_loss = self.criterion(output, labels)
            total_loss += raw_loss.item() * labels.shape[0]
            total_case += labels.shape[0]
            if test:
                argmax = output.max(dim=1)[1]
                equal_case = torch.eq(argmax, labels)
                true_case += int(equal_case.float().sum().item())
        if test:
            return (total_loss / total_case , float(true_case) / total_case)
        else:
            return total_loss / total_case

    def train_step(self, model, train_loader):
        total_loss = 0
        total_cases = 0
        start_time = time.time()
        model.train()
        for i, sample in enumerate(train_loader):
            j = i + 1
            images, labels = sample
            images = images.cuda()
            labels = labels.cuda()
            images = images.transpose(0, 1)  # seq length, bsz, input = 784, 100, 1
            self._optimizer.zero_grad()
            output = model(images, self._optimizer)
            raw_loss = self.criterion(output, labels)
            raw_loss.backward()
            total_loss += raw_loss.item() * labels.shape[0]
            total_cases += labels.shape[0]
            torch.nn.utils.clip_grad_norm_(self.params_invariant, self.args.clip)
            if self.args.quantize:
                model.optim_grad(self._optimizer)
            self._optimizer.step()
            if j % 10 == 0 :
                elapsed = time.time() - start_time
                print('{:03d} loss {:8.6f} elaped time {:6.3f}  learning rate {:08.6f}'.format(j * self.args.batchsize, raw_loss.item(), elapsed, self._optimizer.param_groups[0]['lr']))
                start_time = time.time()
        return total_loss / total_cases

    def train(self, model, trainloader, validloader, testloader):
        if self.load_cktpt(model, self.args.save):
            test_loss, acc = self.evaluate_step(model, testloader, test=True)
            print('=' * 89)
            print('| End of epoch {:3d} | acc {:6.3f} | test loss {:6.3f} |'.format(self._epoch - 1, acc,
                                                                                    test_loss, ))
            print('=' * 89)
        for epoch in range(self._epoch, self.args.epochs+1):
            epoch_start_time = time.time()
            trainloss = self.train_step(model, trainloader)
            val_loss = self.evaluate_step(model, validloader)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:6.3f}s | valid loss {:6.3f} | '.format(
              epoch, (time.time() - epoch_start_time), val_loss, ))
            print('-' * 89)
            # Run on test data.
            test_loss, acc = self.evaluate_step(model, testloader, test=True)
            print('=' * 89)
            print('| End of epoch {:3d} | acc {:6.3f} | test loss {:6.3f} |'.format(epoch, acc,
                test_loss, ))
            print('=' * 89)

            if val_loss < self.validloss:
                self.validloss = val_loss
                is_best = True
                print('new best')
            else:
                is_best = False
            if self.args.optimizer == 'adam':
                    self._optimizer.param_groups[0]['lr'] *= self.args.lr_decay
            with io.open(self.args.save +  '.log', 'a', newline='\n', encoding='utf8', errors='ignore') as tgt:
                msg = 'epoch {:03d} train_loss {:010.7f} valid_loss {:010.7f} test_loss {:010.7f} accuracy {:06.4f}\n'.format(
                    self._epoch, trainloss, val_loss, test_loss, acc
                )
                msg += 'new best**************\n' if is_best else 'no best\n'
                tgt.write(msg)
            self.save_cktpt(model, self.args.save)

            self._epoch += 1