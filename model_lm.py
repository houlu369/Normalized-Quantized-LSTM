import torch
import torch.nn as nn
from torch.nn import Parameter
import math
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

	def __init__(self, input_size, hidden_size,norm,args,  bias=True):
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
		# b_hh = Parameter(torch.Tensor(gate_size))
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
		# b_ih = Parameter(torch.Tensor(gate_size))
		# b_hh = Parameter(torch.Tensor(gate_size))
		layer_params1 = (w_ih, w_hh,)
		suffix = ''
		param_names1 = ['weight_ih_l{}{}_full_precision', 'weight_hh_l{}{}_full_precision']
		param_names1 = [x.format(layer, suffix) for x in param_names1]

		for name, param in zip(param_names1, layer_params1):
			setattr(self, name, param)
		self._all_weights = param_names + param_names1


		if self.norm == 'layer':
			self.reset_parameters()
			self.layernorm_iih = nn.LayerNorm(hidden_size)
			self.layernorm_ihh = nn.LayerNorm(hidden_size)
			self.layernorm_fih = nn.LayerNorm(hidden_size)
			self.layernorm_fhh = nn.LayerNorm(hidden_size)
			self.layernorm_aih = nn.LayerNorm(hidden_size)
			self.layernorm_ahh = nn.LayerNorm(hidden_size)
			self.layernorm_oih = nn.LayerNorm(hidden_size)
			self.layernorm_ohh = nn.LayerNorm(hidden_size)
			self.layernorm_ct = nn.LayerNorm(hidden_size)
		elif self.norm == 'weight':
			self.reset_parameters()
			self.weight_iih = WeightNorm(self.weight_ih_l0[:self.hidden_size, :])
			self.weight_ihh = WeightNorm(self.weight_hh_l0[:self.hidden_size, :])
			self.weight_fih = WeightNorm(self.weight_ih_l0[self.hidden_size:2 * self.hidden_size, :])
			self.weight_fhh = WeightNorm(self.weight_hh_l0[self.hidden_size:2 * self.hidden_size, :])
			self.weight_aih = WeightNorm(self.weight_ih_l0[2 * self.hidden_size:3 * self.hidden_size, :])
			self.weight_ahh = WeightNorm(self.weight_hh_l0[2 * self.hidden_size:3 * self.hidden_size, :])
			self.weight_oih = WeightNorm(self.weight_ih_l0[3 * self.hidden_size:, :])
			self.weight_ohh = WeightNorm(self.weight_hh_l0[3 * self.hidden_size:, :])
		elif self.norm == 'batch':
			if self.args.shared:
				self.reset_parameters()
				self.batchnorm_iih = nn.BatchNorm1d(hidden_size)
				self.batchnorm_ihh = nn.BatchNorm1d(hidden_size)
				self.batchnorm_fih = nn.BatchNorm1d(hidden_size)
				self.batchnorm_fhh = nn.BatchNorm1d(hidden_size)
				self.batchnorm_aih = nn.BatchNorm1d(hidden_size)
				self.batchnorm_ahh = nn.BatchNorm1d(hidden_size)
				self.batchnorm_oih = nn.BatchNorm1d(hidden_size)
				self.batchnorm_ohh = nn.BatchNorm1d(hidden_size)
				self.batchnorm_ct = nn.BatchNorm1d(hidden_size)
			else:
				self.reset_parameters()
				if self.args.char:
					if 'text8' in self.args.data:
						max_length = 180
					else:
						max_length = 100
				else:
					max_length = 35

				self.batchnorm_ih = SeparatedBatchNorm1d(num_features=4 * hidden_size, max_length=max_length)
				self.batchnorm_hh = SeparatedBatchNorm1d(num_features=4 * hidden_size, max_length=max_length)
				self.batchnorm_c = SeparatedBatchNorm1d(num_features=hidden_size, max_length=max_length)
		else:
			self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		# for weight in self.parameters():
		#     weight.data.uniform_(-stdv, stdv)
		for name, weight in self.named_parameters():
			weight.data.uniform_(-stdv, stdv)


	def weight_forward(self):
		if self.norm == 'weight':
			weight_ih0 = self.weight_iih(self.weight_ih_l0[:self.hidden_size, :])
			weight_ih1 = self.weight_fih(self.weight_ih_l0[self.hidden_size:2 * self.hidden_size, :])
			weight_ih2 = self.weight_aih(self.weight_ih_l0[2 * self.hidden_size:3 * self.hidden_size, :])
			weight_ih3 = self.weight_oih(self.weight_ih_l0[3 * self.hidden_size:, :])
			self.weight_ih = torch.cat([weight_ih0, weight_ih1,weight_ih2, weight_ih3], 0)
			weight_hh0 = self.weight_ihh(self.weight_hh_l0[:self.hidden_size, :])
			weight_hh1 = self.weight_fhh(self.weight_hh_l0[self.hidden_size:2 * self.hidden_size, :])
			weight_hh2 = self.weight_ahh(self.weight_hh_l0[2 * self.hidden_size:3 * self.hidden_size, :])
			weight_hh3 = self.weight_ohh(self.weight_hh_l0[3 * self.hidden_size:, :])
			self.weight_hh = torch.cat([weight_hh0, weight_hh1, weight_hh2, weight_hh3], 0)

	def forward(self, x, hidden, time):
		# size of x should be like (batch_size, input_dim)
		# if hidden exist, size should be like (batch_size, hidden_dim)

		h, c = hidden

		if not self.norm:   # full-precision, no normalization
			pre_ih = F.linear(x, self.weight_ih_l0, )
			pre_hh = F.linear(h, self.weight_hh_l0, )
			i, f, a, o = torch.split(pre_ih + pre_hh + self.bias_ih_l0, self.hidden_size, dim=1)

		if self.norm == 'weight':
			pre_ih = F.linear(x, self.weight_ih, )
			pre_hh = F.linear(h, self.weight_hh, )
			i, f, a, o = torch.split(pre_ih + pre_hh + self.bias_ih_l0,self.hidden_size, dim=1)

		if self.norm == 'batch':
			pre_ih = F.linear(x, self.weight_ih_l0, )
			pre_hh = F.linear(h, self.weight_hh_l0, )
			if self.args.shared:
				ii, fi, ai, oi = torch.split(pre_ih, self.hidden_size, dim=1)
				ih, fh, ah, oh = torch.split(pre_hh, self.hidden_size, dim=1)
				ib, fb, ab, ob = torch.split(self.bias_ih_l0, self.hidden_size, dim=0)
				ii, fi, ai, oi = self.batchnorm_iih(ii), self.batchnorm_fih(fi), self.batchnorm_aih(ai), self.batchnorm_oih(oi)
				ih, fh, ah, oh = self.batchnorm_ihh(ih), self.batchnorm_fhh(fh), self.batchnorm_ahh(ah), self.batchnorm_ohh(oh)
				i, f, a, o = ii + ih + ib, fi + fh + fb, ah + ai +ab, oh + oi + ob
			else:
				pre_ih = self.batchnorm_ih(pre_ih, time=time)
				pre_hh = self.batchnorm_hh(pre_hh, time=time)
				i, f, a, o = torch.split(pre_ih + pre_hh + self.bias_ih_l0,self.hidden_size, dim=1)

		if self.norm == 'layer':
			pre_ih = F.linear(x, self.weight_ih_l0, )
			pre_hh = F.linear(h, self.weight_hh_l0, )
			ii, fi, ai, oi = torch.split(pre_ih, self.hidden_size, dim=1)
			ih, fh, ah, oh = torch.split(pre_hh, self.hidden_size, dim=1)
			ib, fb, ab, ob = torch.split(self.bias_ih_l0, self.hidden_size, dim=0)
			ii, fi, ai, oi = self.layernorm_iih(ii), self.layernorm_fih(fi), self.layernorm_aih(ai), self.layernorm_oih(oi)
			ih, fh, ah, oh = self.layernorm_ihh(ih), self.layernorm_fhh(fh), self.layernorm_ahh(ah), self.layernorm_ohh(oh)
			i, f, a, o = ii + ih + ib, fi + fh + fb, ah + ai +ab, oh + oi + ob

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
		# if self.args.method == 'lab' or self.args.method == 'lat':
		# 	D_ih_l0 = optimizer.state[self.weight_ih_l0]['exp_avg_sq']  # second moment in adam optimizer
		# 	D_hh_l0 = optimizer.state[self.weight_hh_l0]['exp_avg_sq']
		# else:
		# 	# if len(optimizer.state[optimizer.param_groups[0]['params'][0]])==0:
		# 	D_ih_l0 = torch.ones_like(self.weight_ih_l0.data)
		# 	D_hh_l0 = torch.ones_like(self.weight_hh_l0.data)

		if len(optimizer.state[optimizer.param_groups[0]['params'][0]]) == 0:
			D_ih_l0 = torch.ones_like(self.weight_ih_l0.data)
			D_hh_l0 = torch.ones_like(self.weight_hh_l0.data)
		else:
			D_ih_l0 = optimizer.state[self.weight_ih_l0]['exp_avg_sq']  # second moment in adam optimizer
			D_hh_l0 = optimizer.state[self.weight_hh_l0]['exp_avg_sq']


		method = self.args.method  # currently also supports twn, rowwise_bwn and row-wise_twn
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
		for name in self.lstm_cell._all_weights:
			setattr(self, name, getattr(self.lstm_cell, name, None))

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
		if hidden:
			h, c = hidden
			h = h.view(h.shape[1], -1)
			c = c.view(c.shape[1], -1)
		else:
			h, c = (torch.zeros(x.shape[1], self.hidden_size).cuda(),
					torch.zeros(x.shape[1], self.hidden_size).cuda())

		hidden = (h, c)
		return LSTM_quantized._forward_rnn(self.lstm_cell, x, hidden)
		# for step in torch.unbind(x, dim=0):
		# 	_, hidden = self.lstm_cell(step, hidden)
		# 	outputs.append(hidden[0])
		# h = hidden[0].view(1, hidden[0].shape[0], hidden[0].shape[1])
		# c = hidden[1].view(1, hidden[0].shape[0], hidden[0].shape[1])
		# return torch.stack(outputs, dim=0), (h, c)

	def __repr__(self):
		return 'LSTM({}, {})'.format(self.input_size, self.hidden_size)

	def optim_grad(self, optimizer):
		self.lstm_cell.optim_grad(optimizer)

	def quantize(self, optimizer):
		self.lstm_cell.quantize(optimizer)






class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,args, dropouto=0.5, dropoute=0.5, dropouth=0.5, dropouti=0.5, tie_weights=False, quantize=False):
		super(RNNModel, self).__init__()
		self.idrop = nn.Dropout(dropouti)
		self.hdrop = nn.Dropout(dropouth)
		self.odrop = nn.Dropout(dropouto)
		self.args = args
		if args.char:
			ninp = ntoken
			self.encoder = nn.Embedding(ntoken,ntoken)
			self.encoder.weight.data = torch.eye(ntoken)
			self.encoder.weight.requires_grad = False
		else:
			self.encoder = nn.Embedding(ntoken, ninp)
		self.quantize = quantize
		self.norm = args.norm
		assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
		if rnn_type == 'LSTM':
			if quantize or args.norm:
				self.rnns = [LSTM_quantized(ninp if l == 0 else nhid,
										   nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),self.norm,args, num_layers=1, dropout=0)
							 for l in range(nlayers)]
			else:
				self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid,
										   nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0)
							 for l in range(nlayers)]
			# self.rnns = [LSTM_quantized(ninp if l == 0 else nhid,
			#                             nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),self.norm,args, num_layers=1, dropout=0)
			#              for l in range(nlayers)]
		else:
			print('sorry for not implementation')
		print(self.rnns)
		self.rnns = torch.nn.ModuleList(self.rnns)
		self.decoder = nn.Linear(nhid, ntoken)

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			#if nhid != ninp:
			#    raise ValueError('When using the tied flag, nhid must be equal to emsize')
			self.decoder.weight = self.encoder.weight

		self.init_weights()

		self.rnn_type = rnn_type
		self.ninp = ninp
		self.nhid = nhid
		self.nlayers = nlayers
		self.dropouto = dropouto
		self.dropouti = dropouti
		self.dropouth = dropouth
		self.dropoute = dropoute
		self.tie_weights = tie_weights

	def reset(self):
		if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

	def init_weights(self):
		initrange = 0.1
		if not self.args.char:
			self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.fill_(0)
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden, optimizer, return_h=False):
		# emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0) # randomly drop some word embeddings, i.e. keep only 0.9|V|* emb_size, others 0
		emb = self.encoder(input)
		emb = self.idrop(emb)
		# emb = self.lockdrop(emb, self.dropouti) # shape: (seq_len, batch_size, emb_size) => (seq_len, randomm drop, random drop)
		if self.quantize:
			self.quantize_layers(optimizer)

		raw_output = emb
		new_hidden = []
		#raw_output, hidden = self.rnn(emb, hidden)
		raw_outputs = []
		outputs = []
		for l, rnn in enumerate(self.rnns):
			# current_input = raw_output
			raw_output, new_h = rnn(raw_output, hidden[l])
			new_hidden.append(new_h)
			raw_outputs.append(raw_output)
			if l != self.nlayers - 1:
				raw_output = self.hdrop(raw_output)
				# raw_output = self.lockdrop(raw_output, self.dropouth)
				outputs.append(raw_output)
		hidden = new_hidden

		# output = self.lockdrop(raw_output, self.dropouto)
		output = self.odrop(raw_output)
		outputs.append(output)

		result = output.view(output.size(0)*output.size(1), output.size(2))
		result = self.decoder(result)
		if return_h:
			return result, hidden, raw_outputs, outputs
		return result, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
					weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
					for l in range(self.nlayers)]
		elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
			return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
					for l in range(self.nlayers)]

	def quantize_layers(self, optimizer):
		for rnn in self.rnns:
			rnn.quantize(optimizer)

	def optim_grad(self, optimizer):
		for rnn in self.rnns:
			rnn.optim_grad(optimizer)
