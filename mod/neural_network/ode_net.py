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

from torch import nn


class ODENet(nn.Module):
	"""
	ODE Net, aiming for calculating partial derivs with X_{t_i} and t_i:
		dX_{t_i} / dt_i = ODENet(X_{t_i}, t_i)
	"""
	
	def __init__(self):
		...


def ODESolver(X_t0, t0, t1, ode_net):
	...


if __name__ == '__main__':
	pass



