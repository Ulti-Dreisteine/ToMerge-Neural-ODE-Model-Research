# -*- coding: utf-8 -*-
"""
Created on 2020/6/11 3:47 下午

@File: ode_net.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from torch.nn import init
from torch import nn
import torch
import copy


class ODENet(nn.Module):
	"""
	ODE Net, aiming for calculating derivatives with X_{t_i} and t_i:
		dX_{t_i} / dt_i = ODENet(input), where input = (X_{t_i} + t_i)
	"""
	
	def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
		super(ODENet, self).__init__()
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.output_size = output_size
		
		self.bn_in = nn.BatchNorm1d(self.input_size)
		
		self.fc_0 = nn.Linear(self.input_size, self.hidden_sizes[0])
		self._init_layer(self.fc_0)
		self.bn_0 = nn.BatchNorm1d(self.hidden_sizes[0])
		
		self.fcs = []
		self.bns = []
		for i in range(len(hidden_sizes) - 1):
			fc_i = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
			setattr(self, 'fc_{}'.format(i + 1), fc_i)
			self._init_layer(fc_i)
			self.fcs.append(fc_i)
			bn_i = nn.BatchNorm1d(self.hidden_sizes[i + 1])
			self.bns.append(bn_i)
		
		self.fc_out = nn.Linear(self.hidden_sizes[-1], self.output_size)
		self.prelu = nn.PReLU()
		self._init_layer(self.fc_out)
		self.bn_out = nn.BatchNorm1d(self.output_size)
	
	def _init_layer(self, layer):
		init.normal_(layer.weight)                                              # 使用这种初始化方式能降低过拟合
		init.normal_(layer.bias)
	
	def forward(self, x):
		x = self.bn_in(x)
		x = self.fc_0(x)
		x = self.bn_0(x)
		x = torch.sigmoid(x)
		
		for i in range(len(self.fcs)):
			x = self.fcs[i](x)
			x = torch.sigmoid(x)
		
		x = self.fc_out(x)
		x = self.bn_out(x)
		x = self.prelu(x)
		
		return x


def ODESolver(X0_tensor: torch.Tensor, integ_t_tensor: torch.Tensor, dt_tensor: torch.Tensor, ode_net: ODENet) -> (torch.Tensor,
                                                                                                                   torch.Tensor):
	"""
	ODE integration solver.
	:param X0_tensor: torch.Tensor, initial value tensor
	:param integ_t_tensor: torch.Tensor, integration time table tensor
	:param dt_tensor: torch.Tensor, integration time step tensor
	:param ode_net: ODENet, for calculating partial derivatives
	"""
	X = X0_tensor                                                               # shape = (n_samples, n_vars)
	X_integ_records = X.unsqueeze(dim = 0)                                      # add new dim 0, shape = (1, n_samples, n_vars)
	
	for step in range(integ_t_tensor.shape[1] - 1):
		input = torch.cat((X, integ_t_tensor[:, step: step + 1]), dim = 1)      # generate input for the ODE Net
		derives = ode_net(input)
		delta_X = torch.mul(derives, dt_tensor)                                 # ** dot product
		
		X = torch.add(X, delta_X)
		X_integ_records = torch.cat((X_integ_records, X.unsqueeze(0)), dim = 0)
	
	return X, X_integ_records


if __name__ == '__main__':
	import pandas as pd
	import sys, os
	
	sys.path.append('../..')
	
	from lib import proj_dir
	from lib import VARIABLES_BOUNDS
	from lib.train_ode_net import normalize_cols, build_samples
	
	# ============ Testing Params ============
	data = pd.read_csv(os.path.join(proj_dir, 'data/total_obs_data.csv'))
	# show_all_fields(data, legend_label = 'field')
	
	# ============ Normalize the Data ============
	data = normalize_cols(data, cols_bounds = VARIABLES_BOUNDS)                 # include "time"
	
	# ============ Build Samples ============
	X0_tensor, X1_tensor, dt_tensor, integ_t_tensor = build_samples(data)
	
	print('\n' + '=' * 12 + ' Build Samples ' + '=' * 12)
	print('X0_tensor.shape: \t{}'.format(X0_tensor.shape))
	print('X1_tensor.shape: \t{}'.format(X1_tensor.shape))
	print('dt_tensor.shape: \t{}'.format(dt_tensor.shape))
	print('integ_t_tensor.shape: \t{}'.format(integ_t_tensor.shape))
	
	# ============ Initialize an ODE Net ============
	input_size, output_size = X0_tensor.shape[1] + 1, X0_tensor.shape[1]        # input_size = n_vars + n_time, where n_time = 1
	hidden_sizes = [2 * input_size, 4 * input_size, 4 * output_size, 2 * output_size]
	ode_net = ODENet(input_size, hidden_sizes, output_size)
	
	# ============ Test ODESolver ============
	X, X_integ_records = ODESolver(X0_tensor, integ_t_tensor, dt_tensor, ode_net)
	
	print('\n' + '=' * 12 + ' Testing ODESolver ' + '=' * 12)
	print('X.shape: \t{}'.format(X.shape))
	print('X_integ_records.shape: \t{}'.format(X_integ_records.shape))
	



