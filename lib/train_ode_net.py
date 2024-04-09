# -*- coding: utf-8 -*-
"""
Created on 2020/6/11 3:40 下午

@File: train_ode_net.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: train the ODE Network model
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import copy
import sys, os

sys.path.append('../')

from lib import proj_dir, proj_cmap
from lib import VARIABLES, VARIABLES_BOUNDS, VARIABLES_N, DISCRETE_T_STEPS, LR, EPOCHS
from mod.tool.normalize_and_denoise import normalize_cols
from mod.neural_network.ode_net import ODENet, ODESolver
from mod.neural_network.loss_criterion import criterion
# from mod.tool.show_figure import show_all_fields


def _get_integrate_t_arr(t0: np.ndarray, dt: np.ndarray) -> np.ndarray:
	"""
	Get integration time table.
	:param t0: np.ndarray, initial time array
	:param dt: np.ndarray, final time array
	"""
	t, integrate_t_arr = copy.deepcopy(t0), copy.deepcopy(t0)
	for step in range(DISCRETE_T_STEPS):
		t += dt
		integrate_t_arr = np.hstack((integrate_t_arr, t))
	return integrate_t_arr


def _vstack_arr(arr: np.ndarray, arr_sub: np.ndarray) -> np.ndarray:
	if arr is None:
		arr = arr_sub
	else:
		arr = np.vstack((arr, arr_sub))
	return arr


@time_cost
def build_samples(data: pd.DataFrame) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
	"""
	Build samples for training the ODE Net model.
	
	Notes:
	according to the mathematical description in README.md:
		* the ODE Net input samples contain the info from: X_0, t_0, t_1, dt
		* the ODE Net output target is X_1
	"""
	try:
		assert 'label' in data.columns
	except Exception:
		raise ValueError('Field "label" is not in data.columns')
	
	labels = data['label'].drop_duplicates().tolist()
	
	# Build sample arrays.
	X0_arr, X1_arr, dt_arr, integ_t_arr = None, None, None, None
	for label in labels:
		_data_sub = data[data['label'] == label].copy()                     # type: pd.DataFrame
		
		_X0_arr_sub = _data_sub.iloc[:-1][VARIABLES].to_numpy()             # initial value of X, X0
		_X1_arr_sub = _data_sub.iloc[1:][VARIABLES].to_numpy()              # final value of X, X1
		
		_t0_arr_sub = _data_sub.iloc[:-1][['time']].to_numpy()              # initial value of t, t0
		_t1_arr_sub = _data_sub.iloc[1:][['time']].to_numpy()               # final value of t, t1
		
		_dt_arr_sub = (_t1_arr_sub - _t0_arr_sub) / DISCRETE_T_STEPS        # time step for each Euler integration
		_integ_t_arr_sub = _get_integrate_t_arr(_t0_arr_sub, _dt_arr_sub)   # get integration time table
		
		# Concatenate data and time tables.
		X0_arr = _vstack_arr(X0_arr, _X0_arr_sub)
		X1_arr = _vstack_arr(X1_arr, _X1_arr_sub)
		dt_arr = _vstack_arr(dt_arr, _dt_arr_sub)
		integ_t_arr = _vstack_arr(integ_t_arr, _integ_t_arr_sub)
	
	# Convert arrays to tensors.
	X0_tensor = torch.from_numpy(X0_arr.astype(np.float32))
	X1_tensor = torch.from_numpy(X1_arr.astype(np.float32))
	dt_tensor = torch.from_numpy(dt_arr.astype(np.float32)).mm(torch.ones(1, VARIABLES_N))
	integ_t_tensor = torch.from_numpy(integ_t_arr.astype(np.float32))
	
	return X0_tensor, X1_tensor, dt_tensor, integ_t_tensor


@time_cost
def split_train_verify_set(X0_tensor, X1_tensor, dt_tensor, integ_t_tensor):
	n_samples = X0_tensor.shape[0]
	
	# Shuffle.
	idxs = np.random.permutation(range(n_samples))
	X0_tensor = X0_tensor[idxs, :]
	X1_tensor = X1_tensor[idxs, :]
	dt_tensor = dt_tensor[idxs, :]
	integ_t_tensor = integ_t_tensor[idxs, :]
	
	# Split.
	train_ratio = 0.8
	train_data_set = {
		'X0_tensor': X0_tensor[: int(n_samples * train_ratio), :],
		'X1_tensor': X1_tensor[: int(n_samples * train_ratio), :],
		'dt_tensor': dt_tensor[: int(n_samples * train_ratio), :],
		'integ_t_tensor': integ_t_tensor[: int(n_samples * train_ratio), :]
	}
	vevify_data_set = {
		'X0_tensor': X0_tensor[int(n_samples * train_ratio):, :],
		'X1_tensor': X1_tensor[int(n_samples * train_ratio):, :],
		'dt_tensor': dt_tensor[int(n_samples * train_ratio):, :],
		'integ_t_tensor': integ_t_tensor[int(n_samples * train_ratio):, :]
	}
	
	return train_data_set, vevify_data_set
	

if __name__ == '__main__':
	use_cuda = False  # torch.cuda.is_available()
	print('\nUse Cuda: {}'.format(use_cuda))
	
	# ============ Load Data ============
	data = pd.read_csv(os.path.join(proj_dir, 'data/total_obs_data.csv'))
	# show_all_fields(data, legend_label = 'field')
	
	# ============ Normalize ============
	data = normalize_cols(data, cols_bounds = VARIABLES_BOUNDS)             # include "time"
	
	# ============ Build Samples ============
	X0_tensor, X1_tensor, dt_tensor, integ_t_tensor = build_samples(data)
	train_data_set, vevify_data_set = split_train_verify_set(X0_tensor, X1_tensor, dt_tensor, integ_t_tensor)
	
	print('\nX0_tensor.shape: \t{}'.format(X0_tensor.shape))
	print('X1_tensor.shape: \t{}'.format(X1_tensor.shape))
	print('dt_tensor.shape: \t{}'.format(dt_tensor.shape))
	print('integ_t_tensor.shape: \t{}'.format(integ_t_tensor.shape))
	
	# ============ Build the Network ============
	input_size, output_size = X0_tensor.shape[1] + 1, X0_tensor.shape[1]    # input_size = n_vars + n_time, where n_time = 1
	hidden_sizes = [2 * input_size, 4 * input_size, 4 * output_size, 2 * output_size]
	ode_net = ODENet(input_size, hidden_sizes, output_size)
	
	# Set the optimizer.
	optimizer = torch.optim.Adam(ode_net.parameters(), lr = LR)
	
	# Set GPU status and empty cache.
	if use_cuda:
		torch.cuda.empty_cache()
		X0_tensor = X0_tensor.cuda()
		X1_tensor = X1_tensor.cuda()
		dt_tensor = dt_tensor.cuda()
		integ_t_tensor = integ_t_tensor.cuda()
		ode_net = ode_net.cuda()
	
	# ============ Train the Model ============
	print('\nStart training the model')
	train_loss_records, verify_loss_records = [], []
	plt.figure('Training Process of the ODE Net', figsize = [12, 5])
	for epoch in range(EPOCHS):
		# Train.
		ode_net.train()
		X1_train_pred, _ = ODESolver(
			train_data_set['X0_tensor'], train_data_set['integ_t_tensor'], train_data_set['dt_tensor'], ode_net
		)
		train_loss = criterion(train_data_set['X1_tensor'], X1_train_pred)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()

		train_loss_records.append(train_loss)
		
		# Verify.
		ode_net.eval()
		with torch.no_grad():
			X1_verify_pred, _ = ODESolver(
				vevify_data_set['X0_tensor'], vevify_data_set['integ_t_tensor'], vevify_data_set['dt_tensor'], ode_net
			)
			verify_loss = criterion(vevify_data_set['X1_tensor'], X1_verify_pred)
			verify_loss_records.append(verify_loss)

		if (epoch + 1) % 5 == 0:
			print(epoch, train_loss, verify_loss)
		
		if (epoch + 1) % 50 == 0:
			X1_verify_true = vevify_data_set['X1_tensor'].cpu().numpy()
			X1_verify_pred = X1_verify_pred.detach().cpu().numpy()
			plt.clf()
			plt.subplot(1, 2, 1)
			plt.title('phase portrait')
			plt.scatter(X1_verify_true[:, 0], X1_verify_true[:, 1], c = proj_cmap['blue'], s = 3, label = 'true')
			plt.scatter(X1_verify_pred[:, 0], X1_verify_pred[:, 1], c = proj_cmap['orange'], s = 3, label = 'pred')
			plt.legend(loc = 'lower left')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.0])
			plt.pause(1.0)
		



