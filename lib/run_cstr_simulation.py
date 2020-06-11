# -*- coding: utf-8 -*-
"""
Created on 2020/5/29

@File: run_cstr_simulation

@Department: AI Lab, Rockontrol, Chengdu

@Author: Luo Lei

@Email: dreisteine262@163.com

@Describe: simulate a CSTR model and generate samples for the model
"""

import logging

logging.basicConfig(level = logging.INFO)

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('../')

from lib import proj_dir, proj_cmap
from lib import VARIABLES, VARIABLES_BOUNDS
from lib import steps, dt, init_states_n, obs_n, ca_0, T_0, q
from mod.chemical_reaction.reaction_model import anisothermal_reaction


def _convert_arr2dataframe(total_df: pd.DataFrame, arr: np.ndarray, label: int):
	columns = ['time'] + VARIABLES
	df = pd.DataFrame(arr, columns = columns)
	df['label'] = df.apply(lambda x: label, axis = 1)

	if total_df is None:
		total_df = df
	else:
		total_df = pd.concat([total_df, df], axis = 0)

	total_df = total_df.reset_index(drop = True)
	return total_df


def generate_samples(ca_list: list, T_list: list, t: np.ndarray, op_params: list) -> (pd.DataFrame, pd.DataFrame):
	"""
	Generate simulation samples set.
	:param ca_list: list, concent A values list
	:param T_list: list, temperature values list
	:param t: np.ndarray, integration time array
	:param op_params: list, operating params
	"""
	total_data = None										# total data set
	total_obs_data = None									# observed data set

	for i in range(init_states_n):
		ca, T = ca_list[i], T_list[i]
		arr, _ = odeint(anisothermal_reaction, (ca, T), t, (op_params,), full_output = True)

		# Add corresponding time records.
		arr = np.hstack((t.reshape(-1, 1), arr))

		# Get obs data.
		obs_locs = sorted(np.random.permutation(np.arange(steps)).tolist()[: obs_n])        # sort
		obs_arr = arr[obs_locs, :]

		# Record data.
		total_data = _convert_arr2dataframe(total_data, arr, i)
		total_obs_data = _convert_arr2dataframe(total_obs_data, obs_arr, i)                 # data of the sample label are from
																							# the same init state

	return total_data, total_obs_data


def draw_phase_portrait(ca_range: list, T_range: list, rand_init_states_n: int, t: np.ndarray, op_params: list):
	"""
	Draw the phase portrait of the system.
	:param ca_range: list, variation range of ca
	:param T_range: list, variation range of T
	:param rand_init_states_n: int, number of random init states
	:param t: np.array, time array for integretion
	:param op_params: list like [ca_0, T_0, q], vector list of op params
	"""
	for i in range(rand_init_states_n):
		ca, T = np.random.uniform(ca_range[0], ca_range[1]), np.random.uniform(T_range[0], T_range[1])

		# Integrate.
		outputs, _ = odeint(anisothermal_reaction, (ca, T), t, (op_params,), full_output = True)
		outputs = outputs[: steps, :]

		# Show the phase portrait.
		if outputs[-1, 1] < 380: 						# set manually
			c = proj_cmap['grey']
		else:
			c = proj_cmap['grey']

		plt.plot(outputs[:, 0], outputs[:, 1], '--', c = c, linewidth = 0.5)

	plt.xlabel('ca', fontsize = 10)
	plt.ylabel('T', fontsize = 10)
	plt.xlim(ca_range),
	plt.ylim(T_range)
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)
	plt.title('Phase Portrait')
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	# ============ Integration ============
	t = np.arange(0, steps * dt, dt)

	# Generate multiple initial values for ca & T.
	op_params = [ca_0, T_0, q]
	ca_list = list(np.random.permutation(np.linspace(0.13, 0.3, init_states_n)))
	T_list = list(np.random.permutation(np.linspace(410, 450, init_states_n)))
	total_data, total_obs_data = generate_samples(ca_list, T_list, t, op_params)

	# Save data as observed samples for the latter use.
	total_obs_data.to_csv(os.path.join(proj_dir, 'data/total_obs_data.csv'), index = False)

	# ============ Draw Data Figures ============
	# It's obvious that the system has 2 stable steady states.
	plt.figure('output time series', figsize = [6, 8])
	plt.suptitle('Temporal Outputs')
	for v in VARIABLES:
		plt.subplot(len(VARIABLES), 1, VARIABLES.index(v) + 1)
		for i in range(init_states_n):
			plt.plot(
				list(total_data[total_data['label'] == i]['time']),
				list(total_data[total_data['label'] == i][v]),
				'--',
				c = proj_cmap['grey'],
				linewidth = 0.6
			)
			plt.scatter(
				list(total_obs_data[total_obs_data['label'] == i]['time']),
				list(total_obs_data[total_obs_data['label'] == i][v]),
				s = 3,
				c = proj_cmap['blue']
			)
			plt.xlabel('t', fontsize = 10)
			plt.ylabel(v, fontsize = 10)
			plt.xticks(fontsize = 6)
			plt.yticks(fontsize = 6)
			plt.tight_layout()
	plt.subplots_adjust(top = 0.95)
	plt.savefig(os.path.join(proj_dir, 'img/temporal_variations.png'), dpi = 450)
	# plt.close()

	# ============ Draw the Phase Portraits ============
	ca_range, T_range = VARIABLES_BOUNDS['ca'], VARIABLES_BOUNDS['T']
	rand_init_states_n = 500
	plt.figure('phase portrait', figsize = [6, 6])
	draw_phase_portrait(ca_range, T_range, rand_init_states_n, t, op_params)
	plt.scatter(total_obs_data.loc[:, 'ca'], total_obs_data.loc[:, 'T'], s = 3, c = proj_cmap['blue'])
	plt.savefig(os.path.join(proj_dir, 'img/phase_portraits.png'), dpi = 450)
	# plt.close()
