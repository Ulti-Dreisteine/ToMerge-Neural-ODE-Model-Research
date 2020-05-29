# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 上午9:25

@Project -> File: ode-neural-network -> reaction_model.py

@Author: luolei

@Describe: anisothermal reaction funcs
"""

import numpy as np
import sys

sys.path.append('../')

from mod.chemical_reaction import V, E_div_R, k_0, delta_H, ro, roc, Cp, Cpc, qc, hA, Tc_0


def anisothermal_reaction(X, t, params) -> np.ndarray:
	"""
	Reaction ODEs
	:param X: list, [ca, T]
	:param t: time
	:param params: list, [ca_0, T_0, q]
	:return: derivs, np.array([dca_dt, dT_dt])
	"""
	ca, T = X[0], X[1]
	ca_0, T_0, q = params[0], params[1], params[2]

	const = np.exp(-E_div_R / T)
	dca_dt = q / V * (ca_0 - ca) - k_0 * ca * const
	dT_dt = q / V * (T_0 - T) - (-delta_H / (ro * Cp)) * k_0 * ca * const + \
			roc * Cpc / (ro * Cp * V) * qc * (1 - np.exp(-hA / (qc * roc * Cpc))) * (Tc_0 - T)

	derives = np.array([dca_dt, dT_dt])
	return derives
