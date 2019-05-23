"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
from torch.nn import   Parameter
import torch.nn.functional as F
import numpy as np
class SeparatedBatchNorm1d(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return F.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))




def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)

class WeightNorm(nn.Module):

    def __init__(self, weight):
        # suppose that your shape of weight is like (out_dim, in_dim)
        super(WeightNorm, self).__init__()
        self.out_dim = weight.shape[0]
        self.g = nn.Parameter(_norm(weight, 0).data)

    def forward(self, v):
        eps = 1e-8
        v = v / (_norm(v, 0) + eps)
        return self.g * v


class LSTM_quantized_cell(nn.Module):

    def __init__(self, input_size, hidden_size,norm,args, bias=True):
        super(LSTM_quantized_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.norm = norm
        self.args = args
        gate_size = 4 * hidden_size
        self._all_weights = []
        layer = 0
        w_ih = Parameter(torch.Tensor(gate_size, input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih,)
        suffix = ''
        param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
        if bias:
            param_names += ['bias_ih_l{}{}', ]
        param_names = [x.format(layer, suffix) for x in param_names]
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        w_ih = Parameter(torch.Tensor(gate_size, input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        layer_params1 = (w_ih, w_hh,)
        suffix = ''
        param_names1 = ['weight_ih_l{}{}_full_precision', 'weight_hh_l{}{}_full_precision']
        param_names1 = [x.format(layer, suffix) for x in param_names1]
        for name, param in zip(param_names1, layer_params1):
            setattr(self, name, param)
        self._all_weights = param_names + param_names1
        if self.norm == 'layer':
            self.layernorm_iih = nn.LayerNorm(hidden_size)
            self.layernorm_ihh = nn.LayerNorm(hidden_size)
            self.layernorm_fih = nn.LayerNorm(hidden_size)
            self.layernorm_fhh = nn.LayerNorm(hidden_size)
            self.layernorm_aih = nn.LayerNorm(hidden_size)
            self.layernorm_ahh = nn.LayerNorm(hidden_size)
            self.layernorm_oih = nn.LayerNorm(hidden_size)
            self.layernorm_ohh = nn.LayerNorm(hidden_size)
            self.layernorm_ct = nn.LayerNorm(hidden_size)
            self.reset_parameters()
            self.reset_LN_parameters() # g=0.1 leanrns faster than g=1
        elif self.norm == 'weight':
            self.reset_parameters()
            self.weightnorm_ih = WeightNorm(self.weight_ih_l0_full_precision) # this will cause each row becomes 1.
            self.weightnorm_hh = WeightNorm(self.weight_hh_l0_full_precision)
        elif self.norm == 'batch':
            if self.args.shared:
                self.reset_parameters()
                self.batchnorm_ih = nn.BatchNorm1d(4 * hidden_size)
                self.batchnorm_hh = nn.BatchNorm1d(4 * hidden_size)
            else:
                max_length= 784
                self.batchnorm_ih = SeparatedBatchNorm1d(num_features=4 * hidden_size, max_length=max_length)
                self.batchnorm_hh = SeparatedBatchNorm1d(num_features=4 * hidden_size, max_length=max_length)
                self.batchnorm_c = SeparatedBatchNorm1d(num_features=hidden_size, max_length=max_length)
                self.reset_parameters()
                self.reset_BN_parameters()
        else:
            self.reset_parameters()
            self.weight_ih_l0.data.copy_(self.weight_ih_l0_full_precision.data)
            self.weight_hh_l0.data.copy_(self.weight_hh_l0_full_precision.data)

    def reset_parameters(self):
        def orthogonal(shape):
            # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
            """ benanne lasagne ortho init (faster than qr approach)"""
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return q[:shape[0], :shape[1]]
        tmp = np.concatenate([
            np.eye(self.hidden_size),
            orthogonal((self.hidden_size,
                        3 * self.hidden_size)), ], axis=1).astype(np.float32)
        tmp = np.transpose(tmp)
        weight_hh = np.zeros((4 * self.hidden_size, self.hidden_size),dtype=np.float32)
        weight_hh[:2 * self.hidden_size, :] = tmp[self.hidden_size: 3 * self.hidden_size, :]
        weight_hh[2 * self.hidden_size:3* self.hidden_size, :] = tmp[ :self.hidden_size, :]
        weight_hh[3* self.hidden_size:, :] = tmp[3* self.hidden_size:, :]
        weight_hh_data = torch.from_numpy(weight_hh)
        self.weight_hh_l0_full_precision.data.copy_(weight_hh_data)
        tmp = np.transpose(orthogonal((1, 4 * self.hidden_size)))
        weight_ih_data = torch.from_numpy(tmp)
        self.weight_ih_l0_full_precision.data.copy_(weight_ih_data)
        nn.init.constant_(self.bias_ih_l0.data, val=0)
        self.bias_ih_l0.data[self.hidden_size:2*self.hidden_size,] = 1.

    def reset_BN_parameters(self):
        if self.norm == 'batch':
            self.batchnorm_ih.reset_parameters()
            self.batchnorm_hh.reset_parameters()
            self.batchnorm_c.reset_parameters()
            self.batchnorm_ih.bias.data.fill_(0)
            self.batchnorm_hh.bias.data.fill_(0)
            self.batchnorm_c.bias.data.fill_(0)
            self.batchnorm_ih.weight.data.fill_(0.1)
            self.batchnorm_hh.weight.data.fill_(0.1)
            self.batchnorm_c.weight.data.fill_(0.1)

    def reset_LN_parameters(self):
        if self.norm == 'layer':
            self.layernorm_iih.bias.data.fill_(0)
            self.layernorm_ihh.bias.data.fill_(0)
            self.layernorm_fih.bias.data.fill_(0)
            self.layernorm_fhh.bias.data.fill_(0)
            self.layernorm_aih.bias.data.fill_(0)
            self.layernorm_ahh.bias.data.fill_(0)
            self.layernorm_oih.bias.data.fill_(0)
            self.layernorm_ohh.bias.data.fill_(0)
            self.layernorm_ct.bias.data.fill_(0)
            self.layernorm_iih.weight.data.fill_(0.1)
            self.layernorm_ihh.weight.data.fill_(0.1)
            self.layernorm_fih.weight.data.fill_(0.1)
            self.layernorm_fhh.weight.data.fill_(0.1)
            self.layernorm_aih.weight.data.fill_(0.1)
            self.layernorm_ahh.weight.data.fill_(0.1)
            self.layernorm_oih.weight.data.fill_(0.1)
            self.layernorm_ohh.weight.data.fill_(0.1)
            self.layernorm_ct.weight.data.fill_(0.1)

    def weight_forward(self):
        if self.norm == 'weight':
            self.weight_ih = self.weightnorm_ih(self.weight_ih_l0)
            self.weight_hh = self.weightnorm_hh(self.weight_hh_l0)

    def forward(self, x, hidden, time):
        # size of x should be like (batch_size, input_dim)
        # if hidden exist, size should be like (batch_size, hidden_dim)
        h, c = hidden
        if self.norm == 'weight':
            pre_ih = F.linear(x, self.weight_ih, )
            pre_hh = F.linear(h, self.weight_hh, )
        if self.norm == 'layer' or self.norm == 'batch' or not self.norm:
            pre_ih = F.linear(x, self.weight_ih_l0, )
            pre_hh = F.linear(h, self.weight_hh_l0, )
        if self.norm == 'batch':
            if self.args.shared:
                pre_ih = self.batchnorm_ih(pre_ih)
                pre_hh = self.batchnorm_hh(pre_hh)
            else:
                pre_ih = self.batchnorm_ih(pre_ih, time=time)
                pre_hh = self.batchnorm_hh(pre_hh, time=time)
        if self.norm == 'layer':
            ii, fi, ai, oi = torch.split(pre_ih, self.hidden_size, dim=1)
            ih, fh, ah, oh = torch.split(pre_hh, self.hidden_size, dim=1)
            ib, fb, ab, ob = torch.split(self.bias_ih_l0, self.hidden_size, dim=0)
            ii, fi, ai, oi = self.layernorm_iih(ii), self.layernorm_fih(fi), self.layernorm_aih(ai), self.layernorm_oih(oi)
            ih, fh, ah, oh = self.layernorm_ihh(ih), self.layernorm_fhh(fh), self.layernorm_ahh(ah), self.layernorm_ohh(oh)
            i, f, a, o = ii + ih + ib, fi + fh + fb, ah + ai +ab, oh + oi + ob
        else:
            i, f, a, o = torch.split(pre_ih + pre_hh + self.bias_ih_l0,self.hidden_size, dim=1)
        i = i.sigmoid()
        f = f.sigmoid()
        a = a.tanh()
        o = o.sigmoid()
        c_t = i * a + f * c
        tmp = c_t
        h_t = o * (tmp.tanh())
        return h_t, (h_t, c_t)

    def Quantization(self, tensor, acc, method='bwn'):

        if method == 'bwn':
            s = tensor.data.size()
            m = tensor.data.norm(p=1).div(tensor.data.nelement())
            quan_tensor = tensor.data.sign().mul(m.expand(s))
        elif method == 'binary_connect':
            quan_tensor = tensor.data.sign() ## need to change to hard sigmoid
        elif method == 'lab':
            s = tensor.data.size()
            D = acc.sqrt() + 1e-8
            # Wb = T.cast(T.switch(Wb, 1., -1.)
            m = (D * tensor.data).abs().sum() / D.sum()
            quan_tensor = tensor.data.sign().mul(m.expand(s))
        elif method == 'ternary_connect':
            m = tensor.data.norm(p=1).div(tensor.data.nelement())
            thres = 0.7 * m
            pos = (tensor > thres).float()
            neg = (tensor < -thres).float()
            quan_tensor = pos - neg
        elif method == 'twn':
            m = tensor.data.norm(p=1).div(tensor.data.nelement())
            thres = 0.7 * m
            pos = (tensor > thres).float()
            neg = (tensor < -thres).float()
            mask = (tensor.abs() > thres).float()
            alpha = (mask * tensor).abs().sum() / mask.sum()
            quan_tensor = alpha * pos - alpha * neg  # use the same scaling for positive and negative weights
        elif method == 'lat': # approximate loss-aware weight ternarization
            D = acc.sqrt() + 1e-8
            b = tensor.data.sign()
            # compute the threshold, converge within 10 iterations
            alpha = (b * D * tensor.data).abs().sum() / (b * D).abs().sum()
            b = (tensor.data > alpha/2).float() - (tensor.data < -alpha/2).float()
            alpha_old=100.
            while (alpha_old - alpha).abs() > 1e-6:
                # minimize alpha
                alpha_old = alpha
                alpha = (b * D * tensor.data).abs().sum() / (b * D).abs().sum()
                # minimize b
                b = (tensor.data > alpha/2).float() - (tensor.data < -alpha/2).float()
            quan_tensor = alpha*b
        return quan_tensor

    def quantize(self, optimizer):
        if self.args.method == 'lab' or self.args.method == 'lat':
            D_ih_l0 = optimizer.state[self.weight_ih_l0]['exp_avg_sq']  # second moment in adam optimizer
            D_hh_l0 = optimizer.state[self.weight_hh_l0]['exp_avg_sq']
        else:
            # if len(optimizer.state[optimizer.param_groups[0]['params'][0]])==0:
            D_ih_l0 = torch.ones_like(self.weight_ih_l0.data)
            D_hh_l0 = torch.ones_like(self.weight_hh_l0.data)
        method = self.args.method  # currently also supports twn, rowwise_bwn and row-wise_twn
        if self.weight_ih_l0.shape[1]==1:
            self.weight_ih_l0.data[:,:] = self.weight_ih_l0_full_precision.data
        else:
            self.weight_ih_l0.data[:self.hidden_size, :] = self.Quantization(
                self.weight_ih_l0_full_precision[:self.hidden_size, :], D_ih_l0[:self.hidden_size, :], method=method)
            self.weight_ih_l0.data[self.hidden_size: 2 * self.hidden_size, :] = self.Quantization(
                self.weight_ih_l0_full_precision[self.hidden_size: 2 * self.hidden_size, :], D_ih_l0[self.hidden_size: 2 * self.hidden_size, :], method=method)
            self.weight_ih_l0.data[2 * self.hidden_size: 3 * self.hidden_size, :] = self.Quantization(
                self.weight_ih_l0_full_precision[2 * self.hidden_size: 3 * self.hidden_size, :], D_ih_l0[2 * self.hidden_size: 3 * self.hidden_size, :], method=method)
            self.weight_ih_l0.data[3 * self.hidden_size:, :] = self.Quantization(
                self.weight_ih_l0_full_precision[3 * self.hidden_size:, :], D_ih_l0[3 * self.hidden_size:, :], method=method)
        self.weight_hh_l0.data[:self.hidden_size, :] = self.Quantization(
            self.weight_hh_l0_full_precision[:self.hidden_size, :], D_hh_l0[:self.hidden_size, :], method=method)
        self.weight_hh_l0.data[self.hidden_size: 2 * self.hidden_size, :] = self.Quantization(
            self.weight_hh_l0_full_precision[self.hidden_size: 2 * self.hidden_size, :], D_hh_l0[self.hidden_size: 2 * self.hidden_size, :], method=method)
        self.weight_hh_l0.data[2 * self.hidden_size: 3 * self.hidden_size, :] = self.Quantization(
            self.weight_hh_l0_full_precision[2 * self.hidden_size: 3 * self.hidden_size, :], D_hh_l0[2 * self.hidden_size: 3 * self.hidden_size, :], method=method)
        self.weight_hh_l0.data[3 * self.hidden_size:, :] = self.Quantization(
            self.weight_hh_l0_full_precision[3 * self.hidden_size:, :], D_hh_l0[3 * self.hidden_size:, :], method=method)

    def optim_grad(self, optimizer):
        self.weight_ih_l0_full_precision.grad=self.weight_ih_l0.grad
        self.weight_hh_l0_full_precision.grad=self.weight_hh_l0.grad

class LSTM_quantized(nn.Module):

    def __init__(self, input_size, hidden_size,norm, args, num_layers=1, bias=True, dropout=0.):
        super(LSTM_quantized, self).__init__()
        self.lstm_cell = LSTM_quantized_cell(input_size, hidden_size,norm ,args, bias)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.num_layers = num_layers
        self.norm = norm
        self.h0 = Parameter(torch.zeros(self.hidden_size).cuda(), requires_grad=True)
        self.c0 = Parameter(torch.zeros(self.hidden_size).cuda(), requires_grad=True)

    @staticmethod
    def _forward_rnn(cell, x, hidden):
        max_time = x.shape[0]
        outputs = []
        for time in range(max_time):
            _, hidden = cell(x[time], hidden, time)
            outputs.append(hidden[0])
        h = hidden[0].unsqueeze(0)
        c = hidden[1].unsqueeze(0)
        return torch.stack(outputs, dim=0), (h, c)

    def forward(self, x, hidden=None):
        self.lstm_cell.weight_forward()

        if self.norm == 'layer' or self.norm == 'batch':
            h, c = (self.h0.repeat(x.shape[1], 1) + self.h0.data.new(x.shape[1], self.hidden_size).normal_(0, 0.1),
                    self.c0.repeat(x.shape[1], 1) + self.c0.data.new(x.shape[1], self.hidden_size).normal_(0, 0.1))
        else:
            h, c = (self.h0.repeat((x.shape[1], 1)), self.c0.repeat((x.shape[1], 1)))
        hidden = (h, c)
        return LSTM_quantized._forward_rnn(self.lstm_cell, x, hidden)

    def __repr__(self):
        return 'LSTM({}, {})'.format(self.input_size, self.hidden_size)

    def optim_grad(self, optimizer):
        self.lstm_cell.optim_grad(optimizer)

    def quantize(self, optimizer):
        self.lstm_cell.quantize(optimizer)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class mnistModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type,ninp, nhid, nlayers,args, quantize=False):
        super(mnistModel, self).__init__()
        self.args = args
        self.quantize = quantize
        self.norm = args.norm
        self.rnns = [LSTM_quantized(ninp, nhid, self.norm, args, num_layers=1, dropout=0)]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, 10) # there are as a total of 10 digits
        self.init_weights()
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns] #why we need this?

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def quantize_layers(self, optimizer):
        for rnn in self.rnns:
            rnn.quantize(optimizer)

    def optim_grad(self, optimizer):
        for rnn in self.rnns:
            rnn.optim_grad(optimizer)

    def forward(self, input, optimizer, return_h=False):
        if self.quantize:
            self.quantize_layers(optimizer)
        _, output = self.rnns[0](input)
        result = output[0].view(-1, output[0].size(2))
        result = self.decoder(result)
        return result