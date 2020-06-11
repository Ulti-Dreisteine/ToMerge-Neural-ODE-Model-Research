# -*- coding: utf-8 -*-
"""
Created on 2020/5/29

@File: __init__

@Department: AI Lab, Rockontrol, Chengdu

@Author: Luo Lei

@Email: dreisteine262@163.com

@Describe: initialize and set config
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys
import os

sys.path.append('../')

from mod.config.config_loader import config

proj_dir, proj_cmap = config.proj_dir, config.proj_cmap

env_params = config.env_conf
model_params = config.conf
test_params = config.test_params

# ============ env params ============

# ============ model params ============
VARIABLES = model_params['VARIABLES']
VARIABLES_BOUNDS = model_params['VARIABLES_BOUNDS']

DISCRETE_T_STEPS = model_params['DISCRETE_T_STEPS']

VARIABLES_N = len(VARIABLES)

# ============ test params ============
dt = test_params['dt']
steps = test_params['steps']
obs_n = test_params['obs_n']
init_states_n = test_params['init_states_n']

ca_0 = test_params['ca_0']
T_0 = test_params['T_0']
q = test_params['q']

