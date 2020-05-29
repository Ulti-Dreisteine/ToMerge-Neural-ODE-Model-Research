# -*- coding: utf-8 -*-
"""
Created on 2020/5/29

@File: ode_net

@Department: AI Lab, Rockontrol, Chengdu

@Author: Luo Lei

@Email: dreisteine262@163.com

@Describe: ODE Network components
"""

from torch.nn import init
from torch import nn
import numpy as np
import torch
import copy


class PartialDeriveNet(nn.Module):
	"""
	Inner layer model of the ODE Net: a network for calculating partial derivatives.
	"""

	def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
		super(PartialDeriveNet, self).__init__()
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

	@staticmethod
	def _init_layer(layer):
		init.normal_(layer.weight)  # initiating layers with normal weights can overcome the overfitting
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


def ODESolver(x0_tensor: torch.Tensor, integrate_t_tensor: torch.Tensor, dt_tensor: torch.Tensor,
              pd_net: PartialDeriveNet) -> (torch.Tensor, torch.Tensor):
	"""
	ODE solver.
	:param x0_tensor: torch.tensor, shape = (points, dim_x)
	:param integrate_t_tensor: torch.tensor, a time table for determining the start and end time step for each
			integration，shape = (points, discrete_t_steps)
	:param dt_tensor: torch.tensor, shape = (points, dim_x)
	:param pd_net: PatialDeriveNet, partial derive network obtained
	:return: x: torch.tensor, final integrated x, shape = (points, dim_x)
	:return: x_records: torch.tensor, records of x, shape = (time, points, dim_x)
	"""
	x = copy.deepcopy(x0_tensor)                                        # initial value
	x_records = x.unsqueeze(0)
	for step in range(integrate_t_tensor.shape[1] - 1):                 # **note the number of times of integration
		input_ = torch.cat((x, integrate_t_tensor[:, step: step + 1]), dim = 1)
		partial_derives = pd_net(input_)

		delta_x = torch.mul(partial_derives, dt_tensor)                 # **dot product
		x = torch.add(x, delta_x)
		x_records = torch.cat((x_records, x.unsqueeze(0)), dim = 0)
	return x, x_records


def integrate(x0: torch.Tensor, t0: float, t1: float, dt: float, pd_net: PartialDeriveNet) -> \
		(torch.Tensor, torch.Tensor):
	"""
	Calculate the solutions for the IVP.
	:param x0: torch.tensor, shape = (points, dim_x)
	:param t0: float, initial time
	:param t1: float, final time
	:param dt: float, time interval len for integration
	:param pd_net: torch.nn object, partial derive net obtained
	:return: x: torch.tensor, final integrated x, shape = (points, dim_x)
	:return: x_records: torch.tensor, records of x, shape = (time, points, dim_x)
	"""
	# Calculate out the integration time table.
	integrate_t_arr = np.arange(t0, t1, dt).reshape(1, -1)
	integrate_t_tensor = torch.from_numpy(integrate_t_arr.astype(np.float32))

	dt = np.array([dt]).reshape(-1, 1)
	dt = torch.from_numpy(dt.astype(np.float32))
	dt_tensor = dt.mm(torch.ones(1, 2))

	# Integrate.
	x, x_records = ODESolver(x0, integrate_t_tensor, dt_tensor, pd_net)

	return x, x_records
